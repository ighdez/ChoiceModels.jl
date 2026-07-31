using Test
using ChoiceModels
using DataFrames
using Random
using Statistics

# ---------------------------------------------------------------------------
# `evaluate(expressions, model, results)` — derived quantities (WTP, ratios)
# with delta-method standard errors.
#
# It was implemented twice, identically, for `LogitModel` and `NestedLogitModel`
# and was an `error("not implemented yet")` stub for `MixedLogitModel` and
# `LatentClassModel`. It is now one shared `delta_method` in `Utils.jl` that all
# four call, so these tests are written to run the SAME assertions against all
# four models rather than testing the two new ones only.
#
# Two things are worth knowing about how this is checked.
#
# The reference gradients here are supplied ANALYTICALLY (`d(a/b)/da = 1/b`,
# `d(a/b)/db = -a/b²`) and contracted with the covariance matrix by hand, so the
# reference shares no code with the implementation, which takes its gradient by
# ForwardDiff. A reference that also called ForwardDiff would only be checking
# ForwardDiff against itself.
#
# The sharpest assertion is the degenerate one: the derived quantity `g(θ) = θ_k`
# must reproduce parameter k's own estimate AND its own reported standard error,
# because ∇g is then the k-th unit vector and `∇g' V ∇g` is exactly `V[k,k]`.
# That pins the alignment between the gradient's coordinates and the covariance
# matrix's — the one thing that silently breaks if the free-parameter ordering
# used here ever drifts from the one `estimate` used. It is also why this file
# matters to the traversal-order change in `Utils.jl`: a permuted θ layout that
# was not applied consistently would fail here and nowhere else.
# ---------------------------------------------------------------------------

# Var(g) = ∇g' V ∇g, with ∇g given as name => partial derivative.
function hand_delta_se(grad::Dict{Symbol,Float64}, free_names, V)
    g = [get(grad, n, 0.0) for n in free_names]
    return sqrt(g' * V * g)
end

free_names_of(params) = [p.name for p in params if !p.fixed]

@testset "Delta method" begin

    # ---------------------------------------------------------------------
    # Multinomial logit
    # ---------------------------------------------------------------------
    @testset "LogitModel" begin
        Random.seed!(20260731)
        N = 800
        df = DataFrame(t1 = 2 .+ randn(N), t2 = 2 .+ randn(N),
                       c1 = 2 .+ randn(N), c2 = 2 .+ randn(N))
        v1 = -1.0 .* df.t1 .- 0.5 .* df.c1
        v2 = -1.0 .* df.t2 .- 0.5 .* df.c2
        p1 = 1 ./ (1 .+ exp.(v2 .- v1))
        df.choice = [rand() < p1[n] ? 1 : 2 for n in 1:N]

        β_time = Parameter(:β_time, value = -0.5)
        β_cost = Parameter(:β_cost, value = -0.2)
        asc    = Parameter(:asc,    value = 0.0, fixed = true)

        V1 = asc + β_time * Variable(:t1) + β_cost * Variable(:c1)
        V2 =       β_time * Variable(:t2) + β_cost * Variable(:c2)

        model = LogitModel((a = 1, b = 2); utilities = (a = V1, b = V2), data = df)
        results = estimate(model, :choice; verbose = false)

        free = free_names_of(ChoiceModels.collect_parameters(model.utilities))

        # A derived quantity that is just one parameter must return that
        # parameter's estimate and its own standard errors, exactly.
        out = evaluate(Dict(:only_time => β_time), model, results)
        @test out[:only_time].value ≈ results.parameters[:β_time]
        @test out[:only_time].std_error ≈ results.std_errors[:β_time]
        @test out[:only_time].robust_std_error ≈ results.rob_std_errors[:β_time]

        # WTP: a ratio, against an analytic delta method.
        bt = results.parameters[:β_time]
        bc = results.parameters[:β_cost]
        wtp = evaluate(Dict(:wtp => β_time / β_cost), model, results)
        grad = Dict(:β_time => 1 / bc, :β_cost => -bt / bc^2)

        @test wtp[:wtp].value ≈ bt / bc
        @test wtp[:wtp].std_error ≈ hand_delta_se(grad, free, results.vcov)
        @test wtp[:wtp].robust_std_error ≈ hand_delta_se(grad, free, results.rob_vcov)

        # A fixed parameter contributes no variance: it is not in θ, so an
        # expression that merely mentions it has the same SE as one that does not.
        with_fixed = evaluate(Dict(:shifted => β_time + asc), model, results)
        @test with_fixed[:shifted].value ≈ bt
        @test with_fixed[:shifted].std_error ≈ results.std_errors[:β_time]

        # Several expressions in one call are independent of each other.
        both = evaluate(Dict(:wtp => β_time / β_cost, :only_time => β_time), model, results)
        @test both[:wtp].std_error ≈ wtp[:wtp].std_error
        @test both[:only_time].std_error ≈ out[:only_time].std_error
    end

    # ---------------------------------------------------------------------
    # Nested logit — the interesting case is λ, which is ESTIMATED as log λ but
    # reported as λ. `estimate` converts H and G to λ space before building the
    # covariance matrices, so the delta method needs no further correction; the
    # degenerate expression `λ` is what proves the two spaces did not get crossed.
    # ---------------------------------------------------------------------
    @testset "NestedLogitModel" begin
        Random.seed!(20260801)
        N = 900
        df = DataFrame(x1 = randn(N), x2 = randn(N), x3 = randn(N))
        b = -0.8
        v = [b .* df.x1, b .* df.x2 .+ 0.3, b .* df.x3 .+ 0.1]

        # Simulate from the nested structure itself (λ = 0.6 on {two, three}),
        # not from a plain MNL. Drawing from an MNL would make the true λ equal 1,
        # and λ̂ then lands above 1 about half the time and trips
        # `_warn_lambda_above_one` — permanent noise in the suite from a fixture
        # that does not match the model being fitted. Same lesson as the separable
        # fixtures noted in CLAUDE.md: suspect the fixture, not the code.
        λ_true = 0.6
        df.choice = map(1:N) do n
            iv = λ_true * log(exp(v[2][n] / λ_true) + exp(v[3][n] / λ_true))
            p_nest = exp(iv) / (exp(v[1][n]) + exp(iv))
            if rand() > p_nest
                1
            else
                p2 = exp(v[2][n] / λ_true) / (exp(v[2][n] / λ_true) + exp(v[3][n] / λ_true))
                rand() < p2 ? 2 : 3
            end
        end

        β  = Parameter(:β,  value = -0.5)
        a2 = Parameter(:a2, value = 0.0)
        a3 = Parameter(:a3, value = 0.0)
        λ  = Parameter(:λ_nest, value = 1.0)

        utils = (one = β * Variable(:x1),
                 two = β * Variable(:x2) + a2,
                 three = β * Variable(:x3) + a3)
        model = NestedLogitModel((one = 1, two = 2, three = 3);
                                 utilities = utils, data = df,
                                 tree = [:one, Nest(λ, [:two, :three])])
        results = estimate(model, :choice; verbose = false)

        free = free_names_of(ChoiceModels._nl_parameters(model))

        # λ̂ and SE(λ̂) round-trip through the delta method in MODEL space.
        out = evaluate(Dict(:lam => λ), model, results)
        @test out[:lam].value ≈ results.parameters[:λ_nest]
        @test out[:lam].std_error ≈ results.std_errors[:λ_nest]
        @test out[:lam].robust_std_error ≈ results.rob_std_errors[:λ_nest]

        # A ratio involving λ, analytically.
        bh = results.parameters[:β]
        lh = results.parameters[:λ_nest]
        r = evaluate(Dict(:ratio => β / λ), model, results)
        grad = Dict(:β => 1 / lh, :λ_nest => -bh / lh^2)
        @test r[:ratio].value ≈ bh / lh
        @test r[:ratio].std_error ≈ hand_delta_se(grad, free, results.vcov)
    end

    # ---------------------------------------------------------------------
    # Mixed logit — previously `error("not implemented yet")`.
    # ---------------------------------------------------------------------
    @testset "MixedLogitModel" begin
        Random.seed!(20260802)
        I_ind, T_obs = 60, 5
        ids = repeat(1:I_ind, inner = T_obs)
        N = length(ids)
        df = DataFrame(ID = ids, t1 = 2 .+ randn(N), t2 = 2 .+ randn(N),
                       c1 = 2 .+ randn(N), c2 = 2 .+ randn(N))
        bt_i = repeat(-1.0 .+ 0.4 .* randn(I_ind), inner = T_obs)
        v1 = bt_i .* df.t1 .- 0.5 .* df.c1
        v2 = bt_i .* df.t2 .- 0.5 .* df.c2
        p1 = 1 ./ (1 .+ exp.(v2 .- v1))
        df.choice = [rand() < p1[n] ? 1 : 2 for n in 1:N]

        mu_t    = Parameter(:mu_t,    value = -0.8)
        sigma_t = Parameter(:sigma_t, value = 0.3)
        β_cost  = Parameter(:β_cost,  value = -0.4)

        b_time = mu_t + sigma_t * Draw(:d_t)
        V1 = b_time * Variable(:t1) + β_cost * Variable(:c1)
        V2 = b_time * Variable(:t2) + β_cost * Variable(:c2)

        model = MixedLogitModel((a = 1, b = 2); utilities = (a = V1, b = V2),
                                data = df, idvar = :ID, R = 40, draw_scheme = :halton)
        results = estimate(model, :choice; verbose = false)

        free = free_names_of(ChoiceModels.collect_parameters(model.utilities))

        # It runs at all — this is what used to throw.
        out = evaluate(Dict(:only_mu => mu_t), model, results)
        @test out[:only_mu].value ≈ results.parameters[:mu_t]
        @test out[:only_mu].std_error ≈ results.std_errors[:mu_t]
        @test out[:only_mu].robust_std_error ≈ results.rob_std_errors[:mu_t]

        # WTP at the MEAN of the taste distribution: a function of the
        # distribution's parameters, which are ordinary estimated parameters.
        mh = results.parameters[:mu_t]
        ch = results.parameters[:β_cost]
        wtp = evaluate(Dict(:wtp_mean => mu_t / β_cost), model, results)
        grad = Dict(:mu_t => 1 / ch, :β_cost => -mh / ch^2)
        @test wtp[:wtp_mean].value ≈ mh / ch
        @test wtp[:wtp_mean].std_error ≈ hand_delta_se(grad, free, results.vcov)
        @test wtp[:wtp_mean].robust_std_error ≈ hand_delta_se(grad, free, results.rob_vcov)
        @test isfinite(wtp[:wtp_mean].std_error)

        # The spread parameter is estimable like any other, so a quantile of the
        # taste distribution has a delta-method SE too.
        sh = results.parameters[:sigma_t]
        q90 = evaluate(Dict(:q90 => (mu_t + 1.2815515655446004 * sigma_t) / β_cost), model, results)
        grad90 = Dict(:mu_t => 1 / ch, :sigma_t => 1.2815515655446004 / ch,
                      :β_cost => -(mh + 1.2815515655446004 * sh) / ch^2)
        @test q90[:q90].value ≈ (mh + 1.2815515655446004 * sh) / ch
        @test q90[:q90].std_error ≈ hand_delta_se(grad90, free, results.vcov)

        # A `Draw` in the expression is REFUSED, not silently averaged over draws.
        # WTP as a ratio of random coefficients is three different quantities and
        # only the parameter-space one has a delta-method standard error at all;
        # for a normal-over-normal spec the mean of the ratio does not even exist.
        err = try
            evaluate(Dict(:wtp_random => b_time / β_cost), model, results)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("d_t", err.msg)
        @test occursin("DOES NOT EXIST", err.msg)
    end

    # ---------------------------------------------------------------------
    # Latent class — previously `error("not implemented yet")`.
    # ---------------------------------------------------------------------
    @testset "LatentClassModel" begin
        Random.seed!(20260803)
        I_ind, T_obs = 80, 4
        ids = repeat(1:I_ind, inner = T_obs)
        N = length(ids)
        df = DataFrame(ID = ids, t1 = 2 .+ randn(N), t2 = 2 .+ randn(N),
                       c1 = 2 .+ randn(N), c2 = 2 .+ randn(N))

        # Two genuinely different classes, membership drawn once per individual.
        in_class1 = repeat(rand(I_ind) .< 0.6, inner = T_obs)
        bt = ifelse.(in_class1, -1.5, -0.3)
        bc = ifelse.(in_class1, -0.3, -1.0)
        v1 = bt .* df.t1 .+ bc .* df.c1
        v2 = bt .* df.t2 .+ bc .* df.c2
        p1 = 1 ./ (1 .+ exp.(v2 .- v1))
        df.choice = [rand() < p1[n] ? 1 : 2 for n in 1:N]

        # Seeded apart: identical classes are an invariant subspace for BFGS.
        bt1 = Parameter(:bt1, value = -1.2)
        bc1 = Parameter(:bc1, value = -0.4)
        bt2 = Parameter(:bt2, value = -0.4)
        bc2 = Parameter(:bc2, value = -0.9)
        d1  = Parameter(:d1,  value = 0.0)
        d2  = Parameter(:d2,  value = 0.0, fixed = true)

        alts = (a = 1, b = 2)
        m1 = LogitModel(alts; utilities = (a = bt1 * Variable(:t1) + bc1 * Variable(:c1),
                                           b = bt1 * Variable(:t2) + bc1 * Variable(:c2)),
                        data = df)
        m2 = LogitModel(alts; utilities = (a = bt2 * Variable(:t1) + bc2 * Variable(:c1),
                                           b = bt2 * Variable(:t2) + bc2 * Variable(:c2)),
                        data = df)

        π1 = exp(d1) / (exp(d1) + exp(d2))
        π2 = exp(d2) / (exp(d1) + exp(d2))
        model = LatentClassModel(π1 * m1 + π2 * m2; data = df, idvar = :ID)
        results = estimate(model, :choice; verbose = false)

        free = free_names_of(ChoiceModels.collect_parameters(model.expr))

        # It runs at all — this is what used to throw. `collect_parameters`
        # descends into the class models, so a class parameter is reachable.
        out = evaluate(Dict(:only_bt1 => bt1), model, results)
        @test out[:only_bt1].value ≈ results.parameters[:bt1]
        @test out[:only_bt1].std_error ≈ results.std_errors[:bt1]
        @test out[:only_bt1].robust_std_error ≈ results.rob_std_errors[:bt1]

        # Class-specific WTP, analytically.
        t1h = results.parameters[:bt1]
        c1h = results.parameters[:bc1]
        w1 = evaluate(Dict(:wtp_c1 => bt1 / bc1), model, results)
        grad1 = Dict(:bt1 => 1 / c1h, :bc1 => -t1h / c1h^2)
        @test w1[:wtp_c1].value ≈ t1h / c1h
        @test w1[:wtp_c1].std_error ≈ hand_delta_se(grad1, free, results.vcov)
        @test w1[:wtp_c1].robust_std_error ≈ hand_delta_se(grad1, free, results.rob_vcov)

        # Both classes at once, and the class-membership parameter is a parameter
        # like any other — a weighted average of the two class WTPs is writable as
        # an expression and gets its SE from the same covariance matrix, including
        # its covariance with the membership parameter.
        t2h = results.parameters[:bt2]
        c2h = results.parameters[:bc2]
        d1h = results.parameters[:d1]
        pooled = evaluate(Dict(:wtp_pooled => π1 * (bt1 / bc1) + π2 * (bt2 / bc2)),
                          model, results)
        w1h = exp(d1h) / (exp(d1h) + 1)
        @test pooled[:wtp_pooled].value ≈ w1h * (t1h / c1h) + (1 - w1h) * (t2h / c2h)
        @test isfinite(pooled[:wtp_pooled].std_error)
        # It is NOT the same as either class's own WTP, nor an unweighted mean —
        # without this the previous assertion would pass on a degenerate mixture.
        @test !isapprox(pooled[:wtp_pooled].value, t1h / c1h; rtol = 1e-3)
        @test !isapprox(pooled[:wtp_pooled].value,
                        0.5 * (t1h / c1h) + 0.5 * (t2h / c2h); rtol = 1e-3)
    end
end
