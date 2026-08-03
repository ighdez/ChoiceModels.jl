using Random, Distributions, Statistics

# Deliberately NOT exported. Both are internal machinery the model constructors
# drive — `MixedLogitModel` generates its own draws and `_lc_share_draws` takes
# that over for a latent class, so a user never calls `generate_draws` and never
# holds a `Draws` (it is unpacked into a plain `Dict` at construction). Exporting
# them would commit the signature under semver, and it has already changed shape
# once: Halton bases are assigned by a dimension's POSITION in `param_names`.
# Same treatment as `logit_prob`, `chosen_logprob`, `covariance_estimates`,
# `hessian_status` and `bhhh_matrix_status`; reach them as `ChoiceModels.foo`.

"""
Struct to hold simulation draws used for random parameters in Mixed Logit models.

# Fields
- `values::Dict`: mapping from each **draw** name — the `Symbol` of a `Draw(:z)`
  node, not a parameter name — to its `N × R` matrix, `N` being the number of
  individuals at this point (the model constructors expand these to observations)
- `scheme::Symbol`: name of the sampling scheme (`:normal`, `:uniform`, `:halton`, `:mlhs`)
- `R::Int`: number of draws per individual
"""
struct Draws
    values::Dict
    scheme::Symbol
    R::Int
end

"""
Generates one `N × R` matrix of simulation draws per random dimension.

All four schemes produce **standard normal** draws (`:uniform` produces a
standardized uniform, mean 0 and unit variance), so a specification writes its own
scale — `mu + sigma * Draw(:z)`.

**The order of `param_names` is load-bearing under `:halton`**: the `dim`-th name
receives the `dim`-th prime base, so it decides which random coefficient integrates
against which low-discrepancy sequence. Callers therefore pass the list in
first-seen traversal order (`collect_draws`, and `_lc_share_draws` for a latent
class), which is both reproducible across Julia versions and what Apollo does.

# Arguments
- `param_names::Vector{Symbol}`: the random dimensions, named by their `Draw`
  symbols. Order matters — see above
- `N::Int`: number of individuals
- `R::Int`: number of draws per individual
- `scheme::Symbol = :normal`: sampling scheme; one of `:normal` (pseudo-random),
  `:uniform`, `:halton`, `:mlhs`. Unknown schemes throw

# Returns
- `Draws`: object containing a dictionary of draw matrices and metadata
"""
function generate_draws(param_names::Vector{Symbol}, N::Int, R::Int; scheme::Symbol = :normal)
    values = Dict()

    for (dim, pname) in enumerate(param_names)
        if scheme == :normal
            values[pname] = rand(Normal(), N, R)

        elseif scheme == :uniform
            values[pname] = rand(Uniform(-√3, √3), N, R)

        elseif scheme == :halton
            values[pname] = halton_draws(N, R, dim)

        elseif scheme == :mlhs
            values[pname] = mlhs_draws(N, R, pname)

        else
            error("Unsupported sampling scheme: $scheme")
        end
    end

    return Draws(values, scheme, R)
end

"""
Generates Halton sequence draws for one random dimension.

The dimension is assigned the `dim`-th prime as its Halton base (2, 3, 5, 7, …),
matching Apollo's dimension-ordered small-prime bases; the sequence is then
transformed to standard normal via the Normal quantile function. Each individual
gets a consecutive block of `R` draws.

The base is chosen by **position, not by name**. It used to be
`hash(pname) % length(primes) + 1`, which handed out large primes (47, 41, 19, 79
on a four-dimension model) and could even give two dimensions the same base,
perfectly correlating their draws. Large-base Halton covers poorly at small `R` —
a base-79 block of 100 draws has mean ≈ −0.28 rather than ≈ 0 — which biased the
simulated integral. A burn-in was tested as an alternative and moved the fit
*further* from Apollo, so the plain consecutive block is deliberate.

Since `dim` is the dimension's position in the list `generate_draws` was given,
which base a random coefficient integrates against is fixed by `collect_draws`'
traversal order. See `generate_draws`.

# Arguments
- `N::Int`: number of individuals
- `R::Int`: number of draws per individual
- `dim::Int`: 1-based position of this dimension, which selects the prime base

# Returns
- `Matrix{Float64}`: an `N × R` matrix of standard normal Halton draws
"""
function halton_draws(N::Int, R::Int, dim::Int)
    function halton(n, base)
        f, r = 1.0, 0.0
        while n > 0
            f /= base
            r += f * (n % base)
            n ÷= base
        end
        return r
    end

    # Assign the dim-th prime (2, 3, 5, 7, …) to the dim-th random dimension,
    # matching Apollo's dimension-ordered small-prime bases. Large/hash-picked
    # bases give terrible low-discrepancy coverage at small R and can collide.
    base = Primes.prime(dim)

    draws = zeros(N, R)
    for i in 1:N
        for r in 1:R
            u = halton((i - 1) * R + r, base)
            draws[i, r] = quantile(Normal(), u)
        end
    end
    return draws
end

"""
Generates Modified Latin Hypercube Sampling (MLHS) draws for a specific parameter.

This method improves coverage of the sampling space by stratifying the uniform distribution,
and then applies the inverse CDF of the standard normal to each value. Each individual's
row is stratified and then **shuffled**, which is what keeps the dimensions independent
of one another — see the comment on the assignment, and the "cross-dimensional
independence" testset in `test/test_draws.jl` that pins it.

# Arguments
- `N::Int`: number of individuals
- `R::Int`: number of draws per individual
- `pname::Symbol`: the draw's name (not used — the row is stratified identically for
  every dimension and then independently permuted; the argument keeps the signature
  symmetric with `halton_draws`)

# Returns
- `Matrix{Float64}`: an `N × R` matrix of standard normal MLHS draws
"""
function mlhs_draws(N::Int, R::Int, pname::Symbol)
    draws = zeros(N, R)
    for i in 1:N
        u = ((0:R-1) .+ rand(R)) ./ R
        # Shuffle the fresh vector, THEN assign. `shuffle!(draws[i, :])` would
        # permute a throwaway copy (row indexing materialises a new array), leaving
        # the row in sorted order — which makes every dimension identically ordered
        # and thus perfectly rank-correlated, corrupting the joint draw.
        draws[i, :] .= shuffle!(quantile.(Normal(), u))
    end
    return draws
end
