using Test
using ChoiceModels
using DataFrames
using Random
using Statistics

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

    # ---------------------------------------------------------------------------
    # Mixed Logit classes (LC-MMNL).
    #
    # The likelihood is `Σ_c π_c · (1/R) Σ_r Π_t P_c(j_t | β_cr)`: the product over
    # an individual's observations is INNERMOST, then the draws are averaged, then
    # the classes are mixed. That ordering is Apollo's (apollo_mnl → panelProd →
    # avgInterDraws → lc), and getting it wrong gives a different model rather than
    # a less accurate one — the same class of error as mixing classes per
    # observation, one level further in.
    # ---------------------------------------------------------------------------
    @testset "mixed logit classes: likelihood ordering" begin
        Random.seed!(31415)
        I3, T3, J3, R3 = 15, 4, 2, 9
        ids3 = repeat(1:I3, inner=T3)
        N3 = length(ids3)
        df3 = DataFrame(ID=ids3, x1=randn(N3), x2=randn(N3), choice=rand(1:J3, N3))
        Y3 = zeros(Bool, N3, J3)
        for n in 1:N3
            Y3[n, df3.choice[n]] = true
        end

        # σ deliberately large: it is what separates the correct ordering from the
        # wrong ones, and a small σ makes the three nearly coincide.
        mu = Parameter(:mu3, value=-0.5); sg = Parameter(:sg3, value=1.5)
        bd = Parameter(:bd3, value=0.6)
        e1 = Parameter(:e1_3, value=0.3); e2 = Parameter(:e2_3, value=0.0, fixed=true)
        w1 = exp(e1) / (exp(e1) + exp(e2))
        w2 = exp(e2) / (exp(e1) + exp(e2))
        alts3 = (a=1, b=2)

        mxl = MixedLogitModel(alts3; data=df3, idvar=:ID, R=R3, draw_scheme=:halton,
                              utilities=(a=(mu + sg*Draw(:z3)) * Variable(:x1),
                                         b=(mu + sg*Draw(:z3)) * Variable(:x2)))
        mnl = LogitModel(alts3; utilities=(a=bd*Variable(:x1), b=bd*Variable(:x2)), data=df3)
        lc3 = LatentClassModel(w1*mxl + w2*mnl; data=df3, idvar=:ID,
                               check_class_separation=false)

        pars3 = Dict(:mu3=>-0.5, :sg3=>1.5, :bd3=>0.6, :e1_3=>0.3, :e2_3=>0.0)
        got = sum(loglikelihood(lc3, Y3; parameters=pars3))

        # One contribution per INDIVIDUAL, as for every panel latent class.
        @test length(loglikelihood(lc3, Y3; parameters=pars3)) == I3

        # ---- independent brute force, from the draw matrix rather than logit_prob.
        # Two-alternative probabilities in the stable logistic form; exp(V)/Σexp(V)
        # would reproduce a softmax-underflow bug and pass vacuously.
        zz = ChoiceModels._lc_classes(lc3.expr)[1][2].draws[:z3]   # N × R
        pchosen(Vc, Vo) = 1 / (1 + exp(Vo - Vc))
        π1 = exp(0.3) / (exp(0.3) + 1)
        π2 = 1 / (exp(0.3) + 1)
        mixed_one(n, r) = begin
            b = -0.5 + 1.5 * zz[n, r]
            df3.choice[n] == 1 ? pchosen(b*df3.x1[n], b*df3.x2[n]) :
                                 pchosen(b*df3.x2[n], b*df3.x1[n])
        end
        det_one(n) = df3.choice[n] == 1 ? pchosen(0.6*df3.x1[n], 0.6*df3.x2[n]) :
                                          pchosen(0.6*df3.x2[n], 0.6*df3.x1[n])
        rows_of(i)   = findall(==(i), ids3)
        mixed_obs(n) = sum(mixed_one(n, r) for r in 1:R3) / R3

        reference = sum(log(π1 * (sum(prod(mixed_one(n, r) for n in rows_of(i)) for r in 1:R3) / R3)
                            + π2 * prod(det_one(n) for n in rows_of(i))) for i in 1:I3)
        @test got ≈ reference atol=1e-10

        # ---- and it must NOT equal either wrong ordering. Without these the test
        # would pass under an implementation that averages or mixes at the wrong
        # level, exactly as the panel testset above would have passed under the
        # per-observation mixture.
        wrong_avg_first = sum(log(π1 * prod(mixed_obs(n) for n in rows_of(i))
                                  + π2 * prod(det_one(n) for n in rows_of(i))) for i in 1:I3)
        wrong_mix_obs = sum(log(π1 * mixed_obs(n) + π2 * det_one(n)) for n in 1:N3)
        @test !isapprox(got, wrong_avg_first; atol=1e-4)
        @test !isapprox(got, wrong_mix_obs;   atol=1e-4)
    end

    # Two structural anchors, in the spirit of "identical classes collapse to a
    # plain MNL": make the new machinery degenerate and it must reproduce a model
    # the package already computes another way.
    @testset "mixed logit classes: degenerate cases" begin
        Random.seed!(2718)
        I4, T4, J4 = 20, 3, 2
        ids4 = repeat(1:I4, inner=T4)
        N4 = length(ids4)
        df4 = DataFrame(ID=ids4, x1=randn(N4), x2=randn(N4), choice=rand(1:J4, N4))
        Y4 = zeros(Bool, N4, J4)
        for n in 1:N4
            Y4[n, df4.choice[n]] = true
        end
        Y4_3d = repeat(Y4, outer=(1, 1, 6))   # MixedLogit's own loglikelihood wants N × J × R
        alts4 = (a=1, b=2)

        # (1) σ ≡ 0 removes the randomness, so the Mixed Logit class is a plain
        #     logit and the whole model must equal the corresponding LC-MNL.
        mu = Parameter(:mu4, value=-0.7)
        s0 = Parameter(:s0_4, value=0.0, fixed=true)
        bd = Parameter(:bd4, value=0.5)
        g1 = Parameter(:g1_4, value=0.25); g2 = Parameter(:g2_4, value=0.0, fixed=true)
        v1 = exp(g1) / (exp(g1) + exp(g2))
        v2 = exp(g2) / (exp(g1) + exp(g2))

        mxl0 = MixedLogitModel(alts4; data=df4, idvar=:ID, R=6, draw_scheme=:halton,
                               utilities=(a=(mu + s0*Draw(:z4)) * Variable(:x1),
                                          b=(mu + s0*Draw(:z4)) * Variable(:x2)))
        mnl0 = LogitModel(alts4; utilities=(a=mu*Variable(:x1), b=mu*Variable(:x2)), data=df4)
        other = LogitModel(alts4; utilities=(a=bd*Variable(:x1), b=bd*Variable(:x2)), data=df4)

        p4 = Dict(:mu4=>-0.7, :s0_4=>0.0, :bd4=>0.5, :g1_4=>0.25, :g2_4=>0.0)
        lc_mxl = LatentClassModel(v1*mxl0 + v2*other; data=df4, idvar=:ID,
                                  check_class_separation=false)
        lc_mnl = LatentClassModel(v1*mnl0 + v2*other; data=df4, idvar=:ID,
                                  check_class_separation=false)
        @test sum(loglikelihood(lc_mxl, Y4; parameters=p4)) ≈
              sum(loglikelihood(lc_mnl, Y4; parameters=p4))

        # (2) A single Mixed Logit class at weight 1 is just that Mixed Logit.
        #     This is the check that the draw average is taken at the right level:
        #     a per-observation average would not reproduce the standalone model.
        unit = Parameter(:unit4, value=1.0, fixed=true)
        sg = Parameter(:sg4, value=0.8)
        solo = MixedLogitModel(alts4; data=df4, idvar=:ID, R=6, draw_scheme=:halton,
                               utilities=(a=(mu + sg*Draw(:z4)) * Variable(:x1),
                                          b=(mu + sg*Draw(:z4)) * Variable(:x2)))
        lc_solo = LatentClassModel(unit * solo; data=df4, idvar=:ID)
        p5 = Dict(:mu4=>-0.7, :sg4=>0.8, :unit4=>1.0)
        # The latent class rebuilt the class onto its own shared draws, so compare
        # against a Mixed Logit holding those same draws.
        shared = ChoiceModels._lc_classes(lc_solo.expr)[1][2]
        @test sum(loglikelihood(lc_solo, Y4; parameters=p5)) ≈
              sum(loglikelihood(shared, Y4_3d; parameters=p5))
    end

    # Draw generation is taken over by the latent class so that a draw dimension
    # named in two classes really is the same draw. Generating per class is not
    # merely redundant — `generate_draws` assigns Halton bases by the POSITION of
    # each dimension in the symbol list, so two classes each generating one
    # dimension independently both receive base 2 and their random coefficients
    # come out perfectly correlated across classes. Measured: corr = 1.0 before,
    # ≈ 0 after. Same failure mode as the MLHS shuffle bug in Draws.jl.
    @testset "mixed logit classes: draws are owned by the latent class" begin
        Random.seed!(161803)
        I5, T5 = 30, 3
        ids5 = repeat(1:I5, inner=T5)
        N5 = length(ids5)
        df5 = DataFrame(ID=ids5, x1=randn(N5), x2=randn(N5), choice=rand(1:2, N5))
        alts5 = (a=1, b=2)
        sa = Parameter(:sa5, value=0.3); sb = Parameter(:sb5, value=0.3)
        h1 = Parameter(:h1_5, value=0.2); h2 = Parameter(:h2_5, value=0.0, fixed=true)
        u1 = exp(h1) / (exp(h1) + exp(h2))
        u2 = exp(h2) / (exp(h1) + exp(h2))

        mk(sym, par) = MixedLogitModel(alts5; data=df5, idvar=:ID, R=64, draw_scheme=:halton,
                                       utilities=(a=(par*Draw(sym))*Variable(:x1),
                                                  b=(par*Draw(sym))*Variable(:x2)))

        # DIFFERENT draw names: independent dimensions, which is only true if one
        # generator saw both of them.
        ca, cb = mk(:za5, sa), mk(:zb5, sb)
        @test cor(vec(ca.draws[:za5]), vec(cb.draws[:zb5])) ≈ 1.0   # the trap, per class
        lc5 = LatentClassModel(u1*ca + u2*cb; data=df5, idvar=:ID,
                               check_class_separation=false)
        cls5 = ChoiceModels._lc_classes(lc5.expr)
        @test abs(cor(vec(cls5[1][2].draws[:za5]), vec(cls5[2][2].draws[:zb5]))) < 0.05

        # SAME draw name: one shared matrix, so the symbol means one thing.
        cc, cd = mk(:zs5, sa), mk(:zs5, sb)
        lc6 = LatentClassModel(u1*cc + u2*cd; data=df5, idvar=:ID,
                               check_class_separation=false)
        cls6 = ChoiceModels._lc_classes(lc6.expr)
        @test cls6[1][2].draws[:zs5] == cls6[2][2].draws[:zs5]

        # Settings are inherited from the classes and must agree; there is no
        # defensible way to choose for the analyst.
        wrongR = MixedLogitModel(alts5; data=df5, idvar=:ID, R=32, draw_scheme=:halton,
                                 utilities=(a=(sb*Draw(:zs5))*Variable(:x1),
                                            b=(sb*Draw(:zs5))*Variable(:x2)))
        @test_throws ErrorException LatentClassModel(u1*cc + u2*wrongR; data=df5, idvar=:ID,
                                                     check_class_separation=false)
        wrongS = MixedLogitModel(alts5; data=df5, idvar=:ID, R=64, draw_scheme=:mlhs,
                                 utilities=(a=(sb*Draw(:zs5))*Variable(:x1),
                                            b=(sb*Draw(:zs5))*Variable(:x2)))
        @test_throws ErrorException LatentClassModel(u1*cc + u2*wrongS; data=df5, idvar=:ID,
                                                     check_class_separation=false)

        # A Mixed Logit class makes `idvar` mandatory: both the class membership
        # and the random coefficients are drawn once per individual, so there is
        # no coherent per-observation reading of the model.
        @test_throws ErrorException LatentClassModel(u1*cc + u2*cd; data=df5,
                                                     check_class_separation=false)
    end

end
