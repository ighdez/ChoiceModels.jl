using Test
using ChoiceModels
using DataFrames
using Random
using LinearAlgebra
using ForwardDiff

# Nested logit is checked against three independent things, never against a
# snapshot of its own output:
#
#  1. The MNL special case. Every λ fixed at 1 must reproduce the plain logit
#     log-likelihood EXACTLY, at any nesting depth. This is the structural
#     anchor — the analogue of "identical classes collapse to a plain MNL" in
#     test_latentclass.jl — and it catches a mis-built tree, a wrong path, or a
#     dropped inclusive-value term in one line.
#  2. Hand-rolled closed-form probabilities for a two- and a three-level tree,
#     written from the textbook formula rather than from the implementation.
#  3. For the standard errors, a Hessian taken directly in the MODEL space, with
#     no log transform anywhere. That is the only way to check the θ = log λ
#     machinery without re-using the very formula under test.
@testset "NestedLogit.jl" begin

    # Shared fixture: 4 alternatives, non-separable data.
    Random.seed!(20260730)
    N = 120
    df = DataFrame(x1 = randn(N), x2 = randn(N), x3 = randn(N), x4 = randn(N))
    df.choice = rand(1:4, N)

    alts = (c1 = 1, c2 = 2, c3 = 3, c4 = 4)
    b  = Parameter(:b,  value = -0.4)
    a2 = Parameter(:a2, value =  0.2)
    a3 = Parameter(:a3, value = -0.3)
    a4 = Parameter(:a4, value =  0.1)
    U = (c1 = b * Variable(:x1),
         c2 = a2 + b * Variable(:x2),
         c3 = a3 + b * Variable(:x3),
         c4 = a4 + b * Variable(:x4))

    base = Dict(:b => -0.4, :a2 => 0.2, :a3 => -0.3, :a4 => 0.1)
    Vmat = hcat(-0.4 .* df.x1,
                0.2 .- 0.4 .* df.x2,
                -0.3 .- 0.4 .* df.x3,
                0.1 .- 0.4 .* df.x4)

    # ------------------------------------------------------------------
    @testset "λ ≡ 1 collapses to a plain MNL" begin
        mnl = LogitModel(alts; utilities = U, data = df, parameters = base)
        ll_mnl = sum(loglikelihood(mnl, df.choice))

        # A fixed root-level nest, a deep tree, and a *free* Parameter sitting at
        # 1 must all give back the MNL likelihood — bitwise, not approximately.
        trees = [
            [:c1, Nest(1.0, [:c2, :c3, :c4])],
            [Nest(1.0, [Nest(1.0, [:c1, :c2]), :c3]), :c4],
            [:c1, Nest(Parameter(:lam, value = 1.0), [:c2, :c3, :c4])],
        ]
        for tree in trees
            p = merge(base, Dict(:lam => 1.0))
            nl = NestedLogitModel(alts; utilities = U, tree = tree, data = df, parameters = p)
            @test sum(loglikelihood(nl, df.choice)) == ll_mnl
        end
    end

    # ------------------------------------------------------------------
    @testset "two-level probabilities match the closed form" begin
        λ = 0.5
        p = merge(base, Dict(:lam => λ))
        tree = [:c1, Nest(Parameter(:lam, value = λ), [:c2, :c3, :c4])]
        nl = NestedLogitModel(alts; utilities = U, tree = tree, data = df, parameters = p)
        got = predict(nl, (parameters = p,))

        @test size(got) == (N, 4)

        for n in 1:N
            inner = [exp(Vmat[n, j] / λ) for j in 2:4]
            IV = λ * log(sum(inner))
            den = exp(Vmat[n, 1]) + exp(IV)
            @test got[n, 1] ≈ exp(Vmat[n, 1]) / den
            for (k, j) in enumerate(2:4)
                @test got[n, j] ≈ (exp(IV) / den) * (inner[k] / sum(inner))
            end
        end

        @test all(0.0 .<= got .<= 1.0)
        @test all(abs.(sum(got, dims = 2) .- 1.0) .< 1e-12)

        # The whole point: this is NOT the MNL answer.
        mnl = LogitModel(alts; utilities = U, data = df, parameters = base)
        @test maximum(abs.(got .- predict(mnl, (parameters = base,)))) > 0.01
    end

    # ------------------------------------------------------------------
    @testset "three-level probabilities match the closed form" begin
        # root -> [ A(λa) -> [ B(λb) -> [c1, c2], c3 ], c4 ]
        λa, λb = 0.7, 0.4
        p = merge(base, Dict(:la => λa, :lb => λb))
        tree = [Nest(Parameter(:la, value = λa),
                     [Nest(Parameter(:lb, value = λb), [:c1, :c2]), :c3]),
                :c4]
        nl = NestedLogitModel(alts; utilities = U, tree = tree, data = df, parameters = p)
        got = predict(nl, (parameters = p,))

        for n in 1:N
            # Inner nest B over {c1, c2}
            eb = [exp(Vmat[n, j] / λb) for j in 1:2]
            IVb = λb * log(sum(eb))
            # Nest A over {B, c3}, at scale λa
            ea = [exp(IVb / λa), exp(Vmat[n, 3] / λa)]
            IVa = λa * log(sum(ea))
            # Root over {A, c4}, at scale 1
            er = [exp(IVa), exp(Vmat[n, 4])]
            den = sum(er)

            pA = er[1] / den
            pB_given_A = ea[1] / sum(ea)
            @test got[n, 1] ≈ pA * pB_given_A * (eb[1] / sum(eb))
            @test got[n, 2] ≈ pA * pB_given_A * (eb[2] / sum(eb))
            @test got[n, 3] ≈ pA * (ea[2] / sum(ea))
            @test got[n, 4] ≈ er[2] / den
        end

        @test all(abs.(sum(got, dims = 2) .- 1.0) .< 1e-12)
    end

    # ------------------------------------------------------------------
    @testset "availability, including a nest with no available child" begin
        av2 = trues(N); av3 = trues(N); av4 = trues(N)
        av2[1:40] .= false; av3[1:40] .= false; av4[1:40] .= false
        avail = (c1 = trues(N), c2 = av2, c3 = av3, c4 = av4)
        choice = [i <= 40 ? 1 : df.choice[i] for i in 1:N]

        λ = 0.7
        p = merge(base, Dict(:lam => λ))
        nl = NestedLogitModel(alts; utilities = U,
                              tree = [:c1, Nest(Parameter(:lam, value = λ), [:c2, :c3, :c4])],
                              availability = avail, data = df, parameters = p)
        probs = predict(nl, (parameters = p,))

        @test !any(isnan, probs)
        # Where the whole nest is dead the only alternative left takes everything.
        @test all(probs[1:40, 1] .≈ 1.0)
        @test all(probs[1:40, 2:4] .== 0.0)
        @test all(abs.(sum(probs, dims = 2) .- 1.0) .< 1e-12)

        # The real trap this guards: an unavailable branch must not poison the
        # DERIVATIVES. Pushing -Inf through the division by λ gives an Inf partial
        # and then a NaN out of exp, even though the term contributes nothing.
        names = [:b, :a2, :a3, :a4, :lam]
        f = v -> -sum(loglikelihood(nl, choice;
                                    parameters = Dict{Symbol,Any}(names[i] => v[i] for i in 1:5)))
        v0 = [-0.4, 0.2, -0.3, 0.1, 0.7]
        @test all(isfinite, ForwardDiff.gradient(f, v0))
        @test all(isfinite, ForwardDiff.hessian(f, v0))
    end

    # ------------------------------------------------------------------
    @testset "tree validation" begin
        okλ = Parameter(:lam, value = 0.5)

        # An alternative in two nests at once.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [:c1, Nest(okλ, [:c1, :c2, :c3, :c4])], data = df)

        # An alternative placed nowhere.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [:c1, Nest(okλ, [:c2, :c3])], data = df)

        # A name that is not an alternative.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [:c1, :c2, :c3, :c4, :bogus], data = df)

        # A child that is neither a Symbol nor a Nest.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [1, :c2, :c3, :c4], data = df)

        # λ must be strictly positive at the starting values — estimation takes
        # its log, and λ < 0 inverts the model.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [:c1, Nest(Parameter(:l, value = 0.0), [:c2, :c3, :c4])], data = df)
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [:c1, Nest(-0.5, [:c2, :c3, :c4])], data = df)

        # The root is a plain Vector, never a Nest.
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = Nest(okλ, [:c1, :c2, :c3, :c4]), data = df)
        @test_throws ErrorException NestedLogitModel(
            alts; utilities = U, tree = [], data = df)

        # A nest needs children, and a Nest needs a Parameter/number scale.
        @test_throws ErrorException Nest(okλ, [])
        @test_throws ErrorException Nest(:not_a_parameter, [:c1, :c2])
    end

    @testset "a single-child nest warns rather than throwing" begin
        # λ is unidentified with nothing to choose between, but the model is
        # still estimable, so this warns (cf. `_check_class_separation`).
        @test_logs (:warn, r"single child") match_mode = :any NestedLogitModel(
            alts; utilities = U,
            tree = [:c1, :c2, :c3, Nest(Parameter(:l, value = 0.5), [:c4])],
            data = df)

        # A well-formed tree stays silent.
        @test_logs NestedLogitModel(
            alts; utilities = U,
            tree = [:c1, Nest(Parameter(:l, value = 0.5), [:c2, :c3, :c4])],
            data = df)
    end

    # ------------------------------------------------------------------
    @testset "the tree is by name, so argument order is irrelevant" begin
        λ = 0.6
        p = merge(base, Dict(:lam => λ))
        t1 = [:c1, Nest(Parameter(:lam, value = λ), [:c2, :c3, :c4])]
        # Same tree, every list written backwards.
        t2 = [Nest(Parameter(:lam, value = λ), [:c4, :c3, :c2]), :c1]

        m1 = NestedLogitModel(alts; utilities = U, tree = t1, data = df, parameters = p)
        m2 = NestedLogitModel(alts; utilities = U, tree = t2, data = df, parameters = p)
        @test predict(m1, (parameters = p,)) ≈ predict(m2, (parameters = p,))
        @test sum(loglikelihood(m1, df.choice)) ≈ sum(loglikelihood(m2, df.choice))
    end

    @testset "alternative codes are translated, not assumed to be 1:J" begin
        λ = 0.6
        p = merge(base, Dict(:lam => λ))
        tree = [:c1, Nest(Parameter(:lam, value = λ), [:c2, :c3, :c4])]

        # This testset ESTIMATES, so it needs choices actually generated by the
        # model. The shared fixture's `rand(1:4)` choices leave the ASCs and λ
        # unidentified, which makes `estimate` warn about a singular Hessian and
        # BHHH matrix on every run — a correct diagnosis of a bad fixture, and
        # permanent noise in the suite.
        Random.seed!(31337)
        M = 1200
        gdf = DataFrame(x1 = randn(M), x2 = randn(M), x3 = randn(M), x4 = randn(M))
        gdf.choice = ones(Int, M)
        gen = NestedLogitModel(alts; utilities = U, tree = tree, data = gdf, parameters = p)
        gprobs = predict(gen, (parameters = p,))
        for n in 1:M
            u = rand(); acc = 0.0
            for j in 1:4
                acc += gprobs[n, j]
                if u <= acc
                    gdf.choice[n] = j
                    break
                end
            end
        end

        # Same model and data, alternatives coded (7, 3, 9, 5) instead of 1:4.
        odd = (c1 = 7, c2 = 3, c3 = 9, c4 = 5)
        codes = [7, 3, 9, 5]
        df2 = copy(gdf)
        df2.choice = [codes[c] for c in gdf.choice]

        m_odd = NestedLogitModel(odd;  utilities = U, tree = tree, data = df2, parameters = p)
        m_ref = NestedLogitModel(alts; utilities = U, tree = tree, data = gdf, parameters = p)

        r_odd = estimate(m_odd, :choice; verbose = false)
        r_ref = estimate(m_ref, :choice; verbose = false)
        @test r_odd.loglikelihood ≈ r_ref.loglikelihood

        # A choice value no alternative claims must be refused.
        df3 = copy(df2); df3.choice[1] = 42
        m_bad = NestedLogitModel(odd; utilities = U, tree = tree, data = df3, parameters = p)
        @test_throws ErrorException estimate(m_bad, :choice; verbose = false)
    end

    # ------------------------------------------------------------------
    @testset "estimation recovers parameters and reports λ in model space" begin
        # Simulate from a known nested logit, then recover it.
        Random.seed!(4242)
        M = 3000
        sim = DataFrame(x1 = randn(M), x2 = randn(M), x3 = randn(M), x4 = randn(M))
        sim.choice = ones(Int, M)
        truth = Dict(:b => -0.8, :a2 => 0.4, :a3 => -0.5, :a4 => 0.3, :lam => 0.6)

        Usim = (c1 = Parameter(:b, value = -0.1) * Variable(:x1),
                c2 = Parameter(:a2, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x2),
                c3 = Parameter(:a3, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x3),
                c4 = Parameter(:a4, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x4))
        tree = [:c1, Nest(Parameter(:lam, value = 1.0), [:c2, :c3, :c4])]

        gen = NestedLogitModel(alts; utilities = Usim, tree = tree, data = sim, parameters = truth)
        probs = predict(gen, (parameters = truth,))
        for n in 1:M
            u = rand(); acc = 0.0
            for j in 1:4
                acc += probs[n, j]
                if u <= acc
                    sim.choice[n] = j
                    break
                end
            end
        end

        model = NestedLogitModel(alts; utilities = Usim, tree = tree, data = sim)
        res = estimate(model, :choice; verbose = false)

        @test res.converged
        @test res.hessian === :posdef
        # Recovery is asserted in STANDARD ERROR units, not against a fixed
        # absolute tolerance. These parameters have quite different precisions
        # (SE ranges over ~0.04–0.09 at this sample size), so any single atol is
        # either too loose for the sharp ones or flaky for the diffuse ones —
        # `a3` lands 1.5 SE from truth here, unremarkable sampling noise that
        # nonetheless breaks a 0.12 atol. Three SEs is ~99.7% coverage.
        for k in [:b, :a2, :a3, :a4, :lam]
            @test abs(Float64(res.parameters[k]) - truth[k]) < 3 * res.std_errors[k]
        end

        # λ is reported in the user's space, so it must be positive whatever the
        # optimizer did in θ space, and all SEs must be finite.
        @test res.parameters[:lam] > 0
        @test all(isfinite, values(res.std_errors))

        # THE transform check. A Hessian taken directly in model space — no log
        # anywhere — must reproduce the reported classical standard errors. This
        # is what pins `curvature_in_model_space`; re-deriving SE(λ) = λ·SE(θ)
        # here would just be the implementation checking itself.
        names = [:b, :a2, :a3, :a4, :lam]
        f = v -> -sum(loglikelihood(model, sim.choice;
                                    parameters = Dict{Symbol,Any}(names[i] => v[i] for i in 1:5)))
        vhat = [Float64(res.parameters[k]) for k in names]
        se_direct = sqrt.(diag(inv(ForwardDiff.hessian(f, vhat))))
        for (i, k) in enumerate(names)
            @test isapprox(res.std_errors[k], se_direct[i], rtol = 1e-6)
        end

        # The robust matrix must be in the same space, not left in θ.
        @test all(isfinite, values(res.rob_std_errors))
        @test res.free_parameters == 5
        @test res.N == M
    end

    # ------------------------------------------------------------------
    @testset "λ̂ > 1 warns but still returns estimates" begin
        # Data generated with a λ well above 1: valid arithmetic, but not
        # consistent with global RUM, so it is reported and flagged.
        Random.seed!(99)
        M = 1500
        sim = DataFrame(x1 = randn(M), x2 = randn(M), x3 = randn(M), x4 = randn(M))
        sim.choice = ones(Int, M)
        truth = Dict(:b => -0.8, :a2 => 0.0, :a3 => 0.0, :a4 => 0.0, :lam => 2.5)
        Usim = (c1 = Parameter(:b, value = -0.1) * Variable(:x1),
                c2 = Parameter(:a2, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x2),
                c3 = Parameter(:a3, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x3),
                c4 = Parameter(:a4, value = 0.0) + Parameter(:b, value = -0.1) * Variable(:x4))
        tree = [:c1, Nest(Parameter(:lam, value = 1.0), [:c2, :c3, :c4])]

        gen = NestedLogitModel(alts; utilities = Usim, tree = tree, data = sim, parameters = truth)
        probs = predict(gen, (parameters = truth,))
        for n in 1:M
            u = rand(); acc = 0.0
            for j in 1:4
                acc += probs[n, j]
                if u <= acc
                    sim.choice[n] = j
                    break
                end
            end
        end

        model = NestedLogitModel(alts; utilities = Usim, tree = tree, data = sim)
        res = @test_logs (:warn, r"above 1") match_mode = :any estimate(model, :choice; verbose = false)
        @test res.parameters[:lam] > 1
        @test res.converged
    end

    # ------------------------------------------------------------------
    @testset "show prints the tree" begin
        λ = 0.5
        nl = NestedLogitModel(alts; utilities = U,
                              tree = [:c1, Nest(Parameter(:λ_PT, value = λ), [:c2, :c3, :c4])],
                              data = df)
        out = sprint(show, MIME"text/plain"(), nl)
        @test occursin("NestedLogitModel with 4 alternatives", out)
        @test occursin("λ_PT", out)
        for alt in ["c1", "c2", "c3", "c4"]
            @test occursin(alt, out)
        end
    end

    @testset "an NL evaluates as a nested term, for use inside a latent class" begin
        λ = 0.5
        p = merge(base, Dict(:lam => λ))
        nl = NestedLogitModel(alts; utilities = U,
                              tree = [:c1, Nest(Parameter(:lam, value = λ), [:c2, :c3, :c4])],
                              data = df, parameters = p)
        probs = ChoiceModels.evaluate(nl, df, p)
        @test size(probs) == (N, 4)
        @test probs ≈ predict(nl, (parameters = p,))
    end
end
