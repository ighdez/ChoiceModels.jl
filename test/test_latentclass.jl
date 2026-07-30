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

    alts = [1, 2, 3]
    m1 = LogitModel(alts; utilities=[b1 * Variable(:x1) + a1, b1 * Variable(:x2), b1 * Variable(:x3)],
                    availability=avail, data=df)
    m2 = LogitModel(alts; utilities=[b2 * Variable(:x1) + a2, b2 * Variable(:x2), b2 * Variable(:x3)],
                    availability=avail, data=df)

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

        mnl = LogitModel(alts; utilities=[b1 * Variable(:x1) + a1, b1 * Variable(:x2), b1 * Variable(:x3)],
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
        z_m1 = LogitModel(alts; utilities=[z_b1 * Variable(:x1) + z_a1, z_b1 * Variable(:x2), z_b1 * Variable(:x3)],
                          data=df, availability=avail)
        z_m2 = LogitModel(alts; utilities=[z_b2 * Variable(:x1) + z_a2, z_b2 * Variable(:x2), z_b2 * Variable(:x3)],
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
        s_m1 = LogitModel(alts; utilities=[s_b * Variable(:x1), s_b * Variable(:x2), s_b * Variable(:x3)],
                          data=df, availability=avail)
        s_m2 = LogitModel(alts; utilities=[s_b * Variable(:x2), s_b * Variable(:x1), s_b * Variable(:x3)],
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

        # Such an expression also has no class models to inherit the alternatives
        # from, so it must be given them explicitly.
        @test_throws ErrorException LatentClassModel(odd_expr; data=df, idvar=:ID)

        bad = LatentClassModel(odd_expr; alternatives=alts, data=df, idvar=:ID)
        @test_throws ErrorException loglikelihood(bad, Y; parameters=params)

        # Without idvar the same expression is fine (cross-sectional path).
        ok = LatentClassModel(odd_expr; alternatives=alts, data=df)
        @test length(loglikelihood(ok, Y; parameters=params)) == N

        # A single `weight * model` term IS decomposable (one-class model). Its
        # weight must still be 1, or the class-weight check rejects it.
        unit = Parameter(:unit_weight, value=1.0, fixed=true)
        one_class = LatentClassModel(unit * m1; data=df, idvar=:ID)
        @test length(loglikelihood(one_class, Y;
                                   parameters=merge(params, Dict(:unit_weight => 1.0)))) == I_ind
    end

    # The `collect_*` walkers used to have an explicit `expr isa LogitModel` arm
    # and nothing else, so a Mixed, Nested or Latent Class model nested in a
    # latent-class expression contributed NO parameters at all. That failed
    # silently: `estimate` would optimize over an empty vector and "converge"
    # instantly at the starting values, reporting a fit it never searched for.
    @testset "walkers descend into every nested model type" begin
        pname(e) = sort([p.name for p in ChoiceModels.collect_parameters(e)])

        bx  = Parameter(:b_nested, value=-0.5)
        lam = Parameter(:lam_nested, value=0.7)
        w1  = Parameter(:unit_w, value=1.0, fixed=true)
        U2  = [bx * Variable(:x1), bx * Variable(:x2), bx * Variable(:x3)]

        # `alts` is the plain-vector form, so the alternatives are named alt1..alt3
        # and the nesting tree refers to them by those names.
        inner_mnl = LogitModel(alts; utilities=U2, data=df)
        inner_mxl = MixedLogitModel(alts; utilities=U2, data=df, idvar=:ID, R=4)
        inner_nl  = NestedLogitModel(alts; utilities=U2,
                                     tree=[:alt1, Nest(lam, [:alt2, :alt3])], data=df)
        inner_lc  = LatentClassModel(w1 * inner_mnl; data=df, idvar=:ID)

        # Every model type contributes its parameters when wrapped in an expression.
        @test pname(1.0 * inner_mnl) == [:b_nested]
        @test pname(1.0 * inner_mxl) == [:b_nested]
        @test pname(1.0 * inner_lc)  == [:b_nested, :unit_w]

        # A nested logit is the one whose parameters are NOT all in `utilities` —
        # λ lives in the tree. Walking `.utilities` alone drops it, which reads as
        # a converged model whose λ never left its starting value.
        @test pname(1.0 * inner_nl) == [:b_nested, :lam_nested]

        # The other two walkers descend as well. `collect_draws` reaching into a
        # class is what lets a latent class discover its Mixed Logit classes'
        # draw dimensions.
        @test sort(ChoiceModels.collect_variables([1.0 * inner_nl])) == [:x1, :x2, :x3]
        @test ChoiceModels.collect_draws([1.0 * inner_mxl]) == Symbol[]
        mxl_rnd = MixedLogitModel(alts; data=df, idvar=:ID, R=4,
                                  utilities=[(bx + Draw(:z)) * Variable(:x1),
                                             bx * Variable(:x2), bx * Variable(:x3)])
        @test ChoiceModels.collect_draws([1.0 * mxl_rnd]) == [:z]

        # Only a nested logit declares a log-scale estimation space, and it must
        # survive being nested — otherwise a latent class searches λ unconstrained.
        @test ChoiceModels.collect_log_scale_parameters(1.0 * inner_nl) == Set([:lam_nested])
        @test isempty(ChoiceModels.collect_log_scale_parameters(1.0 * inner_mnl))
    end

    # An NL class must be estimated with λ on a log scale, exactly as a standalone
    # nested logit is. Two things are pinned: λ̂ stays strictly positive (which an
    # unconstrained search does not guarantee — it dies as a NaN log-likelihood
    # the first time the line search steps past zero), and the reported value and
    # its standard error are in λ space, not log λ space.
    @testset "nested logit as a latent class" begin
        # A DEDICATED fixture. The shared one above draws `choice` at random, which
        # is fine for the hand-rolled likelihood checks but leaves a two-class
        # NL/MNL mixture completely unidentified: λ̂ collapses to 9e-5 and the BHHH
        # matrix is (correctly) singular, so no covariance matrix is computed at
        # all. Suspect the fixture before the code — see the fixture warning under
        # "Known issues" in CLAUDE.md. Here the choices are simulated FROM the
        # model, so its parameters are genuinely recoverable.
        Random.seed!(20260731)
        I2, T2 = 200, 6
        ids2 = repeat(1:I2, inner=T2)
        N2 = length(ids2)
        df2 = DataFrame(ID=ids2, x1=randn(N2), x2=randn(N2), x3=randn(N2))

        λ_true, b_nl_true = 0.6, -1.2
        b_mnl_true, a_mnl_true, π_true = 0.9, 0.4, 0.65

        # Class membership is drawn ONCE per individual, which is what makes this
        # a latent class rather than a per-observation mixture.
        in_class1 = rand(I2) .< π_true
        choices2 = Vector{Int}(undef, N2)
        for n in 1:N2
            if in_class1[ids2[n]]
                # Nested logit: alt1 alone against a nest of {alt2, alt3}.
                V = (b_nl_true * df2.x1[n], b_nl_true * df2.x2[n], b_nl_true * df2.x3[n])
                e2, e3 = exp(V[2] / λ_true), exp(V[3] / λ_true)
                iv = λ_true * log(e2 + e3)
                if rand() < exp(V[1]) / (exp(V[1]) + exp(iv))
                    choices2[n] = 1
                else
                    choices2[n] = rand() < e2 / (e2 + e3) ? 2 : 3
                end
            else
                V = (b_mnl_true * df2.x1[n] + a_mnl_true,
                     b_mnl_true * df2.x2[n], b_mnl_true * df2.x3[n])
                p = exp.(V) ./ sum(exp.(V))
                u = rand()
                choices2[n] = u < p[1] ? 1 : (u < p[1] + p[2] ? 2 : 3)
            end
        end
        df2.choice = choices2
        Y2 = zeros(Bool, N2, J)
        for n in 1:N2
            Y2[n, choices2[n]] = true
        end

        bx  = Parameter(:b_nl_lc, value=-1.0)
        lam = Parameter(:lam_nl_lc, value=0.8)
        bm  = Parameter(:b_mnl_lc, value=0.7)
        am  = Parameter(:a_mnl_lc, value=0.2)
        e1  = Parameter(:e_1, value=0.5)
        e2p = Parameter(:e_2, value=0.0, fixed=true)
        U2  = [bx * Variable(:x1), bx * Variable(:x2), bx * Variable(:x3)]

        nl_class = NestedLogitModel(alts; utilities=U2,
                                    tree=[:alt1, Nest(lam, [:alt2, :alt3])], data=df2)
        mnl_class = LogitModel(alts; utilities=[bm * Variable(:x1) + am,
                                                bm * Variable(:x2), bm * Variable(:x3)],
                               data=df2)

        w1 = exp(e1) / (exp(e1) + exp(e2p))
        w2 = exp(e2p) / (exp(e1) + exp(e2p))
        lc = LatentClassModel(w1 * nl_class + w2 * mnl_class; data=df2, idvar=:ID)
        res = estimate(lc, :choice; verbose=false)

        @test res.converged
        # Reported in MODEL space: λ̂ > 0 by construction of the transform. An
        # unconstrained search does not guarantee this — it dies as a NaN
        # log-likelihood the first time the line search steps past zero.
        @test res.parameters[:lam_nl_lc] > 0
        # λ was actually searched. The silent-failure mode this testset exists for
        # is λ̂ sitting exactly where it started, because the walker never found it.
        @test res.parameters[:lam_nl_lc] != 0.8

        # A covariance matrix exists, and λ's SE is finite and in λ space.
        @test res.std_errors !== nothing
        @test isfinite(res.std_errors[:lam_nl_lc])

        # Recovery, in standard-error units rather than a fixed tolerance — the
        # same convention as test_nestedlogit.jl, since these parameters differ a
        # lot in precision.
        @test abs(res.parameters[:lam_nl_lc] - λ_true) < 3 * res.std_errors[:lam_nl_lc]
        @test abs(res.parameters[:b_nl_lc] - b_nl_true) < 3 * res.std_errors[:b_nl_lc]

        # Feeding the reported estimates back in must reproduce the reported
        # log-likelihood. A λ left in θ space would not: exp(θ̂) ≠ θ̂.
        @test sum(loglikelihood(lc, Y2; parameters=res.parameters)) ≈ res.loglikelihood
    end

end
