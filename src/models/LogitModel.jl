"""
Implementation of the LogitModel type and estimation methods.

This module defines the LogitModel struct and all associated functions required to estimate, update, and predict a multinomial logit model. Built on top of the symbolic utilities defined in Expressions.jl.
"""

using DataFrames

"""
Data structure for multinomial logit models.

Encapsulates the alternative set, symbolic utility functions, data, parameters
and availability constraints used in estimation and prediction.

# Fields
- `alternatives::NamedTuple`: alternative name => code in the choice column; defines
  the alternative set and the canonical order of every other per-alternative field
- `utilities::Vector{<:DCMExpression}`: utility expressions, in `alternatives` order
- `availability::Vector{<:AbstractVector{Bool}}`: availability vectors, in `alternatives` order
- `data::DataFrame`: input dataset
- `parameters::Dict`: parameter values (estimates or fixed)
"""
struct LogitModel <: DiscreteChoiceModel
    alternatives::NamedTuple
    utilities::Vector{<:DCMExpression}
    availability::Vector{<:AbstractVector{Bool}}
    data::DataFrame
    parameters::Dict
end

"""
Constructor for `LogitModel`.

# Arguments
- `alternatives`: the alternative set, as a NamedTuple mapping each alternative's
  name to the code identifying it in the choice column — `(car = 1, bus = 4, rail = 7)`.
  A plain vector of codes (`[1, 4, 7]`) is accepted for unnamed alternatives, which
  are then labelled `alt1, alt2, …`
- `utilities`: utility expressions, as a NamedTuple over the same names as
  `alternatives` (their order is irrelevant — they are matched by name)
- `availability`: boolean vectors marking which alternatives are available, likewise
  keyed by name (default: empty, meaning no availability restrictions are applied)
- `data`: `DataFrame` with explanatory variables
- `parameters`: initial/fixed values for model parameters (default: empty `Dict()`)

# Returns
- A `LogitModel` instance
"""
function LogitModel(
    alternatives;
    utilities,
    availability = [],
    data::DataFrame,
    parameters::Dict = Dict()
)
    alts, named = _resolve_alternatives(alternatives)
    utils = _check_utilities(_match_alternatives(alts, named, utilities, "utilities"))
    # An empty `availability` is the "no restrictions" default rather than a
    # mismatch against the alternatives, so it skips the matching entirely.
    avail = isempty(availability) ? AbstractVector{Bool}[] :
            _check_availability(_match_alternatives(alts, named, availability, "availability"), data)

    return LogitModel(alts, utils, avail, data, parameters)
end

"""
Computes choice probabilities under the Multinomial Logit model.

# Arguments
- `utilities`: vector of symbolic utility expressions
- `data`: `DataFrame` of observed variables
- `parameters`: dictionary mapping parameter names to values
- `availability`: vector of boolean vectors indicating available alternatives for each observation

# Returns
- `Vector{Vector{Float64}}`: each inner vector contains probabilities for all alternatives for one observation
"""
function logit_prob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    availability::Vector{<:AbstractVector{Bool}},
    parameters::Dict
)
    N = nrow(data)         # Number of observations
    J = length(utilities)  # Number of alternatives
    
    utils = Vector{Vector}(undef,J)
    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], data, parameters)
    end

    # Initialize arrays
    T = eltype(first(utils))
    
    # Utilities
    U = Array{T}(undef, N, J)
    @inbounds for j in 1:J
        @views U[:, j] .= utils[j]
    end
    
    # Apply availability conditions
    @inbounds for j in 1:J
        @views U[:, j] .= ifelse.(availability[j], U[:, j], -Inf)
    end
    
    # Calculate choice probabilities (stable softmax: subtract the per-row max
    # utility so the largest term is exp(0)=1 and extreme utilities can't under/
    # overflow; the shift cancels in the ratio). Unavailable alts are -Inf, so
    # they stay exp(-Inf)=0.
    m = maximum(U, dims=2)
    expU = exp.(U .- m)
    s_expU = sum(expU, dims=2)
    probs = expU ./ s_expU

    return probs
end

# Cross-sectional `evaluate` for a nested LogitModel term. Defined here rather than
# in Expressions.jl because `LogitModel`/`logit_prob` are only available after this
# file is loaded; dispatch resolves it alongside the DCMExpression node methods.
evaluate(e::LogitModel, data::DataFrame, params::AbstractDict) =
    logit_prob(e.utilities, data, e.availability, params)

"""
Computes predicted probabilities using estimated parameters.

# Arguments
- `model::LogitModel`: the model structure
- `results::NamedTuple`: output of `estimate`, must include `parameters`

# Returns
- `Vector{Vector{Float64}}`: predicted probabilities for each observation
"""
function predict(model::LogitModel,results::NamedTuple)
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        results.parameters,
    )
    return probs
end

"""
Computes the log-likelihood of the model given observed choices.

# Arguments
- `model::LogitModel`: model object with defined parameters
- `choices::Vector{Int}`: vector with indices of chosen alternatives for each observation

# Returns
- `Vector{Float64}`: log-likelihood contribution per observation
"""
function loglikelihood(model::LogitModel, choices::Vector{Int}; parameters::Dict = model.parameters)
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        parameters,
    )

    N, _ = size(probs)

    T = eltype(first(probs))
    loglik = zeros(T, N)

    # Loop over observations
    Threads.@threads for n in 1:N
        chosen = choices[n]
        p = probs[n, chosen]
        @inbounds loglik[n] = log(p)
    end

    return loglik
end

"""
Compute the log-likelihood of the null model, assuming equal choice probability
over available alternatives for each individual.

# Arguments
- `availability::Vector{<:AbstractVector{Bool}}`: availability pattern per individual.

# Returns
- `ll0::Float64`: log-likelihood of the null (equal-probability) model.
"""
function null_loglikelihood_mnl(availability::Vector{<:AbstractVector{Bool}})
    J = length(availability)
    N = length(availability[1])

    ll0 = 0.0
    for i in 1:N
        available = count(j -> availability[j][i], 1:J)
        if available == 0
            error("Observation $i has no available alternatives.")
        end
        ll0 -= log(available)
    end

    return ll0
end

"""
Estimates the parameters of a `LogitModel` via maximum likelihood.

Uses `Optim.jl` to minimize the negative log-likelihood. Computes standard errors via inverse Hessian.

# Arguments
- `model::LogitModel`: model specification
- `choicevar::Symbol`: name of the column in `model.data` that contains observed choices
- `verbose::Bool=true`: whether to print optimization progress

# Returns
- `NamedTuple` containing:
    - `parameters`: estimated values
    - `std_errors`: classical standard errors
    - `rob_std_errors`: robust standard errors (White)
    - `bhhh_std_errors`: BHHH/OPG standard errors — returned for completeness but
      deliberately never reported by `summarize_results` (see `covariance_estimates`)
    - `vcov`: classical variance-covariance matrix
    - `rob_vcov`: robust variance-covariance matrix
    - `bhhh_vcov`: BHHH/OPG variance-covariance matrix
    - `hessian`: `hessian_status` verdict for `H` (`:posdef`/`:indefinite`/`:singular`)
    - `bhhh_matrix`: `bhhh_matrix_status` verdict for `G` (`:posdef`/`:singular`)
    - `free_parameters`: number of free parameters
    - `loglikelihood`: log-likelihood at optimum
    - `iters`: number of iterations
    - `converged`: convergence status
    - `estimation_time`: time taken (in seconds)

Every covariance and standard error field above is `nothing` when `bhhh_matrix`
is `:singular` — no covariance matrix is computed at all in that case.
"""
function estimate(
    model::LogitModel,
    choicevar::Symbol;
    verbose::Bool = true
)

    choice_data = model.data[:,choicevar]

    # Translate the analyst's alternative codes into positions in `model.alternatives`
    # once, here; everything downstream indexes the probability matrix by position.
    choices = _recode_choices(choice_data, model.alternatives, choicevar)

    params = collect_parameters(model.utilities)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    # Get free/fixed masks
    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    # Initial guess only for free params
    θ0 = [init_values[n] for n in free_names]

    mutable_parameters = deepcopy(model.parameters)

    function f_obj_i(θ)
        @inbounds begin
            for (i, name) in enumerate(free_names)
                mutable_parameters[name] = θ[i]
            end
            
            for name in fixed_names
                mutable_parameters[name] = init_values[name]
            end
        end

        loglik = loglikelihood(model, choices; parameters=mutable_parameters)
        return -loglik
    end

    function f_obj(θ)
        @inbounds begin
            for (i, name) in enumerate(free_names)
                mutable_parameters[name] = θ[i]
            end
            
            for name in fixed_names
                mutable_parameters[name] = init_values[name]
            end
        end

        loglik = loglikelihood(model, choices; parameters=mutable_parameters)
        return -sum(loglik)
    end

    if verbose
        println("Warming-up automatic differentiation...")
    end
    
    # Warm-up automatic differentiation
    H = zeros(length(θ0), length(θ0))
    cfg = ForwardDiff.HessianConfig(f_obj, θ0)
    H = ForwardDiff.hessian!(H, f_obj, θ0, cfg)
    
    ForwardDiff.gradient(f_obj, θ0)
    
    scores = zeros(length(choice_data),length(θ0))
    ForwardDiff.jacobian!(scores,f_obj_i, θ0)

    ll0 = null_loglikelihood_mnl(model.availability)

    if verbose
        println("Starting optimization routine...")
        println("Init Log-likelihood: ", round(-f_obj(θ0); digits=2))
    end

    t_start = time()
    result = Optim.optimize(
            f_obj,
            θ0,
            Optim.BFGS(),
            Optim.Options(
                show_trace = verbose,
                iterations = 1000);autodiff=:forward)

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict{Symbol, Real}()

    for (i, name) in enumerate(free_names)
        estimated_params[name] = θ̂[i]
    end

    for name in fixed_names
        estimated_params[name] = init_values[name]
    end

    # Hessian
    if verbose
        println("Computing Standard Errors")
    end

    ForwardDiff.hessian!(H, f_obj, θ̂, cfg)

    # `covariance_estimates` builds all three estimators at once (classical,
    # sandwich, BHHH/OPG), so the score Jacobian must be ready before it is
    # called — not just before the robust block, as it used to be.
    ForwardDiff.jacobian!(scores,f_obj_i, θ̂)  # N × K
    G = scores' * scores  # K × K

    cov = covariance_estimates(H, G, free_names)

    t_end = time()

    return (
        result = result,
        parameters = estimated_params,
        std_errors = cov.std_errors,
        vcov = cov.vcov,
        rob_std_errors = cov.rob_std_errors,
        rob_vcov = cov.rob_vcov,
        bhhh_std_errors = cov.bhhh_std_errors,
        bhhh_vcov = cov.bhhh_vcov,
        hessian = cov.status,
        bhhh_matrix = cov.bhhh_matrix,
        free_parameters = length(free_names),
        null_loglikelihood = ll0,
        loglikelihood = -Optim.minimum(result),
        iters = Optim.iterations(result),
        converged = Optim.converged(result),
        estimation_time = t_end - t_start,
        N = nrow(model.data)
    )
end

"""
Evaluates derived expressions (e.g. WTP, elasticities) based on a fitted LogitModel.

Uses the Delta method to compute standard errors (both classical and robust) via gradient propagation.

# Arguments
- `expressions::Dict{Symbol, <:DCMExpression}`: expressions to evaluate, each identified by name
- `model::LogitModel`: estimated model (structure, utilities, data)
- `results::NamedTuple`: estimation results, must include `parameters`, `vcov`, and `rob_vcov`

# Returns
- `Dict{Symbol, NamedTuple}`: for each expression, returns a named tuple with:
    - `value`: estimated mean value across observations
    - `std_error`: standard error using classical variance
    - `robust_std_error`: standard error using robust variance
"""
function evaluate(
    expressions::Dict{Symbol, <:DCMExpression},
    model::LogitModel,
    results::NamedTuple
)
    # The delta method needs a covariance matrix; there is none when the BHHH
    # matrix was singular (see `covariance_estimates`), so fail with the reason
    # rather than on `g' * nothing * g`.
    if isnothing(results.vcov) || isnothing(results.rob_vcov)
        error("""
              Cannot evaluate derived expressions: no covariance matrix was computed for this \
              model (the BHHH matrix was singular at the optimum), and the standard error of a \
              derived quantity is obtained from it by the delta method. See the warning issued \
              during estimation for the unidentified parameter(s).
              """)
    end

    # 1. Extraer nombres de parámetros libres y su orden
    all_params = collect_parameters(model.utilities)
    free_params = filter(p -> !p.fixed, all_params)
    free_names = [p.name for p in free_params]
    θ̂ = [results.parameters[n] for n in free_names]

    output = Dict{Symbol, NamedTuple{(:value, :std_error, :robust_std_error), Tuple{Float64, Float64, Float64}}}()

    for (name, expr) in expressions
        # 2. Definir función escalar
        f_expr = θ -> begin
            param_dict = copy(results.parameters)
            for (i, pname) in enumerate(free_names)
                param_dict[pname] = θ[i]
            end
            mean(evaluate(expr, model.data, param_dict))
        end

        # 3. Evaluar función y gradiente
        val = f_expr(θ̂)
        g = ForwardDiff.gradient(f_expr, θ̂)

        # 4. Errores estándar
        se_normal  = sqrt(g' * results.vcov * g)
        se_robust  = sqrt(g' * results.rob_vcov * g)

        # 5. Almacenar resultados
        output[name] = (value = val, std_error = se_normal, robust_std_error = se_robust)
    end

    return output
end
