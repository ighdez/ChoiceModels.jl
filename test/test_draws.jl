using Test
using ChoiceModels
using Random
using Statistics
using Distributions: Normal, quantile

# Draw quality is checked on TWO axes, because marginal checks alone are not
# enough: the MLHS shuffle bug left every dimension with perfect marginals
# (right values, right order-statistics) while making the dimensions perfectly
# rank-correlated with each other. Only the cross-dimensional correlation test
# below catches that class of bug.
@testset "Draws.jl" begin

    N, R = 200, 100
    dims = [:d1, :d2, :d3]
    schemes = (:normal, :uniform, :halton, :mlhs)

    @testset "marginals: $scheme" for scheme in schemes
        Random.seed!(987)
        d = generate_draws(dims, N, R; scheme=scheme)

        @test d.scheme == scheme
        @test d.R == R
        @test Set(keys(d.values)) == Set(dims)

        for s in dims
            m = d.values[s]
            @test size(m) == (N, R)
            @test all(isfinite, m)
            # Every scheme is standardized: mean 0, unit variance.
            @test abs(mean(m)) < 0.05
            @test abs(std(m) - 1.0) < 0.05
        end
    end

    @testset "cross-dimensional independence: $scheme" for scheme in schemes
        Random.seed!(987)
        d = generate_draws(dims, N, R; scheme=scheme)

        for (a, b) in ((:d1, :d2), (:d1, :d3), (:d2, :d3))
            ρ = cor(vec(d.values[a]), vec(d.values[b]))
            @test abs(ρ) < 0.1
        end
    end

    @testset "halton bases are distinct small primes in dimension order" begin
        # halton_draws(N, R, dim) must use the dim-th prime (2, 3, 5, …); a
        # repeated base would make two random coefficients perfectly correlated,
        # and a large base gives poor coverage at small R.
        R_small = 50
        h1 = ChoiceModels.halton_draws(1, R_small, 1)
        h2 = ChoiceModels.halton_draws(1, R_small, 2)
        @test h1 != h2
        # Base 2 is the first prime: its first Halton point is 1/2 → quantile 0.
        @test h1[1, 1] ≈ 0.0 atol=1e-12
        # Base 3: first point is 1/3.
        @test h2[1, 1] ≈ quantile(Normal(), 1/3) atol=1e-12
    end

    @testset "unsupported scheme errors" begin
        @test_throws ErrorException generate_draws([:d1], 10, 10; scheme=:not_a_scheme)
    end

end
