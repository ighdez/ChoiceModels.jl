using Test
using ChoiceModels
using DataFrames
using Random

# The two things worth pinning about a latent-class model are (a) that the panel
# likelihood mixes classes ONCE per individual — `Σ_c π_c Π_t P_c(j_t)`, not
# `Π_t Σ_c π_c P_c(j_t)` — and (b) that it collapses to a plain MNL when the
# classes are made identical. Both are asserted below against values computed
# here from scratch, never from the implementation's own intermediate results.

# Closed-form MNL probability of alternative j, honouring availability.
function mnl_probs(V::Vector{<:Vector}, avail::Vector{<:AbstractVector{Bool}}, n::Int)
    J = length(V)
    e = [avail[j][n] ? exp(V[j][n]) : 0.0 for j in 1:J]
    return e ./ sum(e)
end

@testset "LatentClass.jl" begin

    # ---- shared fixture: 2 classes, 3 alternatives, 5 individuals × 4 obs -----
    Random.seed!(7)
    # 40 individuals, not 5. The panel likelihood returns one contribution per
    # INDIVIDUAL, so the score matrix `G` is built from is I × K — with I = 5 and
    # K = 5 free parameters it was square and rank-deficient, and
    # `bhhh_matrix_status` (correctly) flagged the whole suite on every run. The
    # hand-rolled references below are all computed from `params` and `df`, so
    # they follow the fixture size automatically.
    I_ind, T_obs, J = 40, 4, 3
    ids = repeat(1:I_ind, inner=T_obs)
    N = length(ids)

    df = DataFrame(ID=ids,
                   x1=randn(N), x2=randn(N), x3=randn(N),
                   choice=rand(1:J, N))
    avail = [trues(N), trues(N), [n != 3 for n in 1:N]]   # alt 3 unavailable for obs 3
    df.choice[3] = 1   # nobody may choose an unavailable alternative (log(0) = -Inf)

    Y = zeros(Bool, N, J)
    for n in 1:N
        Y[n, df.choice[n]] = true
    end

    # Class-specific tastes plus a class-membership logit.
    b1 = Parameter(:b1, value=-1.0)
    b2 = Parameter(:b2, value=0.5)
    a1 = Parameter(:a1, value=0.3)
    a2 = Parameter(:a2, value=-0.2)
    delta_1 = Parameter(:delta_1, value=0.4)
    delta_2 = Parameter(:delta_2, value=0.0, fixed=true)

    m1 = LogitModel([b1 * Variable(:x1) + a1, b1 * Variable(:x2), b1 * Variable(:x3)];
                    data=df, availability=avail)
    m2 = LogitModel([b2 * Variable(:x1) + a2, b2 * Variable(:x2), b2 * Variable(:x3)];
                    data=df, availability=avail)

    π1 = exp(delta_1) / (exp(delta_1) + exp(delta_2))
    π2 = exp(delta_2) / (exp(delta_1) + exp(delta_2))
    lc_expr = π1 * m1 + π2 * m2

    params = Dict(:b1 => -1.0, :b2 => 0.5, :a1 => 0.3, :a2 => -0.2,
                  :delta_1 => 0.4, :delta_2 => 0.0)

    # Independent reconstruction of the two classes' utilities and weights.
    w1 = exp(params[:delta_1]) / (exp(params[:delta_1]) + exp(params[:delta_2]))
    w2 = exp(params[:delta_2]) / (exp(params[:delta_1]) + exp(params[:delta_2]))
    V_c1 = [params[:b1] .* df.x1 .+ params[:a1], params[:b1] .* df.x2, params[:b1] .* df.x3]
    V_c2 = [params[:b2] .* df.x1 .+ params[:a2], params[:b2] .* df.x2, params[:b2] .* df.x3]

    p_chosen_c1 = [mnl_probs(V_c1, avail, n)[df.choice[n]] for n in 1:N]
    p_chosen_c2 = [mnl_probs(V_c2, avail, n)[df.choice[n]] for n in 1:N]

    @testset "cross-sectional likelihood is the per-observation mixture" begin
        model = LatentClassModel(lc_expr; data=df)   # no idvar
        ll = loglikelihood(model, Y; parameters=params)

        @test length(ll) == N
        expected = [log(w1 * p_chosen_c1[n] + w2 * p_chosen_c2[n]) for n in 1:N]
        @test ll ≈ expected rtol=1e-10
    end

    @testset "panel likelihood mixes classes once per individual" begin
        model = LatentClassModel(lc_expr; data=df, idvar=:ID)
        ll = loglikelihood(model, Y; parameters=params)

        @test length(ll) == I_ind

        # Σ_c π_c Π_t P_c(j_t) — the correct panel latent-class likelihood.
        expected = map(1:I_ind) do i
            rows = findall(==(i), ids)
            seq1 = prod(p_chosen_c1[rows])
            seq2 = prod(p_chosen_c2[rows])
            log(w1 * seq1 + w2 * seq2)
        end
        @test ll ≈ expected rtol=1e-10

        # The test must be able to tell the two formulations apart: mixing per
        # observation and then summing within the individual (the old, incorrect
        # behaviour) gives a materially different number.
        per_obs = map(1:I_ind) do i
            rows = findall(==(i), ids)
            sum(log(w1 * p_chosen_c1[n] + w2 * p_chosen_c2[n]) for n in rows)
        end
        @test !isapprox(sum(ll), sum(per_obs); rtol=1e-3)
    end

    @testset "identical classes collapse to a plain MNL" begin
        # With both classes sharing the same parameters, Σ_c π_c Π_t P_c = Π_t P,
        # so the latent-class panel LL must equal the MNL LL exactly — a structural
        # check that holds whatever the class weights are.
        same = Dict(:b1 => -1.0, :b2 => -1.0, :a1 => 0.3, :a2 => 0.3,
                    :delta_1 => 0.4, :delta_2 => 0.0)

        model = LatentClassModel(lc_expr; data=df, idvar=:ID)
        lc_ll = sum(loglikelihood(model, Y; parameters=same))

        mnl = LogitModel([b1 * Variable(:x1) + a1, b1 * Variable(:x2), b1 * Variable(:x3)];
                         data=df, availability=avail)
        mnl_ll = sum(loglikelihood(mnl, df.choice; parameters=same))

        @test lc_ll ≈ mnl_ll rtol=1e-10
    end

    @testset "identical classes at the starting values are flagged" begin
        # The shared fixture seeds the classes apart, so it must stay silent —
        # otherwise the check is just noise on every well-formed model.
        @test_logs LatentClassModel(lc_expr; data=df, idvar=:ID)

        # All class parameters at the same value: the classes coincide at θ₀.
        # Equal weights too (delta_1 = delta_2 = 0), which is the invariant-subspace
        # case — the gradients coincide exactly and BFGS cannot separate them.
        z_b1, z_b2 = Parameter(:b1, value=0.0), Parameter(:b2, value=0.0)
        z_a1, z_a2 = Parameter(:a1, value=0.0), Parameter(:a2, value=0.0)
        z_d1 = Parameter(:delta_1, value=0.0)
        z_d2 = Parameter(:delta_2, value=0.0, fixed=true)
        z_m1 = LogitModel([z_b1 * Variable(:x1) + z_a1, z_b1 * Variable(:x2), z_b1 * Variable(:x3)];
                          data=df, availability=avail)
        z_m2 = LogitModel([z_b2 * Variable(:x1) + z_a2, z_b2 * Variable(:x2), z_b2 * Variable(:x3)];
                          data=df, availability=avail)
        z_expr = (exp(z_d1) / (exp(z_d1) + exp(z_d2))) * z_m1 +
                 (exp(z_d2) / (exp(z_d1) + exp(z_d2))) * z_m2

        msg = (:warn,)
        @test_logs msg match_mode=:any LatentClassModel(z_expr; data=df, idvar=:ID)

        # It warns rather than throwing: the spec is valid, the start is just bad,
        # and both LC examples used to escape it. A usable model must come back.
        m = (@test_logs msg match_mode=:any LatentClassModel(z_expr; data=df, idvar=:ID))
        @test m isa LatentClassModel
        @test isfinite(sum(loglikelihood(m, Y; parameters=Dict(
            :b1 => 0.0, :b2 => 0.0, :a1 => 0.0, :a2 => 0.0, :delta_1 => 0.0, :delta_2 => 0.0))))

        # Escape hatch.
        @test_logs LatentClassModel(z_expr; data=df, idvar=:ID, check_class_separation=false)

        # Detection is on the class PROBABILITIES, not the parameter values: these
        # classes share every parameter value but the parameters enter different
        # utilities, so the classes are genuinely distinct and must NOT be flagged.
        s_b = Parameter(:sb, value=0.5)
        s_m1 = LogitModel([s_b * Variable(:x1), s_b * Variable(:x2), s_b * Variable(:x3)];
                          data=df, availability=avail)
        s_m2 = LogitModel([s_b * Variable(:x2), s_b * Variable(:x1), s_b * Variable(:x3)];
                          data=df, availability=avail)
        s_expr = (exp(z_d1) / (exp(z_d1) + exp(z_d2))) * s_m1 +
                 (exp(z_d2) / (exp(z_d1) + exp(z_d2))) * s_m2
        @test_logs LatentClassModel(s_expr; data=df, idvar=:ID)
    end

    @testset "estimate reports the fields summarize_results needs" begin
        model = LatentClassModel(lc_expr; data=df, idvar=:ID)
        results = estimate(model, :choice; verbose=false)

        # These two were missing from the returned NamedTuple, which made
        # `summarize_results` throw `FieldError: no field N`.
        @test results.N == N
        @test isfinite(results.null_loglikelihood)
        @test results.null_loglikelihood ≈ -(N * log(3) - log(3) + log(2)) rtol=1e-10

        @test isfinite(results.loglikelihood)
        @test haskey(results.parameters, :delta_1)
        @test results.parameters[:delta_2] == 0.0     # fixed parameter passes through

        preds = predict(model, results)
        @test size(preds) == (N, 3)
        @test all(abs.(sum(preds, dims=2) .- 1.0) .< 1e-10)
        @test all(preds[3, 3] .== 0.0)                # unavailable alternative
    end

    @testset "class weights must be a valid distribution" begin
        # Unnormalized weights: exp(delta_c) without dividing by the sum. This
        # evaluates, optimizes and "converges" perfectly happily, so it has to be
        # caught structurally — the resulting number is not a log-likelihood.
        bad_expr = exp(delta_1) * m1 + exp(delta_2) * m2
        @test_throws ErrorException LatentClassModel(bad_expr; data=df, idvar=:ID)

        # The message must say what is wrong, not just that something is.
        err = try
            LatentClassModel(bad_expr; data=df, idvar=:ID)
            nothing
        catch e
            e
        end
        @test occursin("sum to", sprint(showerror, err))

        # Negative weights are rejected too (they would silently be floored to
        # 1e-30 inside the panel likelihood).
        neg = Parameter(:neg_w, value=-0.5)
        pos = Parameter(:pos_w, value=1.5, fixed=true)
        @test_throws ErrorException LatentClassModel(neg * m1 + pos * m2; data=df, idvar=:ID)

        # A correctly normalized spec passes, cross-sectional or panel.
        @test LatentClassModel(lc_expr; data=df, idvar=:ID) isa LatentClassModel
        @test LatentClassModel(lc_expr; data=df) isa LatentClassModel

        # And the check can be switched off deliberately.
        @test LatentClassModel(bad_expr; data=df, idvar=:ID,
                               check_class_weights=false) isa LatentClassModel
    end

    @testset "panel estimation rejects a non-decomposable expression" begin
        # A sum whose second term is not `weight * model`, so the class structure
        # the panel likelihood needs cannot be recovered — must fail loudly, not
        # silently fall back to the per-observation mixture.
        odd_expr = π1 * m1 + b1
        bad = LatentClassModel(odd_expr; data=df, idvar=:ID)
        @test_throws ErrorException loglikelihood(bad, Y; parameters=params)

        # Without idvar the same expression is fine (cross-sectional path).
        ok = LatentClassModel(odd_expr; data=df)
        @test length(loglikelihood(ok, Y; parameters=params)) == N

        # A single `weight * model` term IS decomposable (one-class model). Its
        # weight must still be 1, or the class-weight check rejects it.
        unit = Parameter(:unit_weight, value=1.0, fixed=true)
        one_class = LatentClassModel(unit * m1; data=df, idvar=:ID)
        @test length(loglikelihood(one_class, Y;
                                   parameters=merge(params, Dict(:unit_weight => 1.0)))) == I_ind
    end

end
