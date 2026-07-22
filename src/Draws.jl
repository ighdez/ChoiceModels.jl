using Random, Distributions, Sobol, StatsBase

export Draws, generate_draws

"""
Struct to hold simulation draws used for random parameters in Mixed Logit models.

# Fields
- `values::Dict`: mapping from parameter names to draw matrices (`N × R`)
- `scheme::Symbol`: name of the sampling scheme (`:normal`, `:uniform`, `:halton`, `:mlhs`)
- `R::Int`: number of draws per individual
"""
struct Draws
    values::Dict
    scheme::Symbol
    R::Int
end

"""
Generates simulation draws for each parameter name provided.

# Arguments
- `param_names::Vector{Symbol}`: list of parameter names
- `N::Int`: number of individuals
- `R::Int`: number of draws per individual
- `scheme::Symbol = :normal`: sampling scheme; one of `:normal`, `:uniform`, `:halton`, `:mlhs`

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
Generates Halton sequence draws for a specific parameter.

Each parameter is assigned a prime base based on its name, then Halton sequences are generated
and transformed via the Normal quantile function.

# Arguments
- `N::Int`: number of individuals
- `R::Int`: number of draws
- `pname::Symbol`: parameter name (used to assign Halton base)

# Returns
- `Matrix{Float64}`: a `N × R` matrix of transformed Halton draws
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
and then applies the inverse CDF of the standard normal to each value.

# Arguments
- `N::Int`: number of individuals
- `R::Int`: number of draws per individual
- `pname::Symbol`: parameter name (not used directly, included for API symmetry)

# Returns
- `Matrix{Float64}`: a `N × R` matrix of MLHS draws (normally distributed)
"""
function mlhs_draws(N::Int, R::Int, pname::Symbol)
    draws = zeros(N, R)
    for i in 1:N
        u = ((0:R-1) .+ rand(R)) ./ R
        draws[i, :] .= quantile.(Normal(), u)
        shuffle!(draws[i, :])
    end
    return draws
end
