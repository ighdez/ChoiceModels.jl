"""
Implementation of the LogitModel type and estimation methods.

This module defines the LogitModel struct and all associated functions required to estimate, update, and predict a multinomial logit model. Built on top of the symbolic utilities defined in Expressions.jl.
"""

using DataFrames

"""
Data structure for multinomial logit models.

Encapsulates symbolic utility functions, data, parameters, and availability constraints
used in estimation and prediction.

# Fields
- `utilities::Vector{<:DCMExpression}`: utility expressions (one per alternative)
- `data::DataFrame`: input dataset
- `parameters::Dict`: parameter values (estimates or fixed)
- `availability::Vector{Vector{Bool}}`: list of availability vectors per alternative (optional)
"""
struct LogitModel <: DiscreteChoiceModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict
    availability::Vector{<:AbstractVector{Bool}}
end

"""
Constructor for `LogitModel`.

# Arguments
- `utilities`: vector of symbolic utility expressions
- `data`: `DataFrame` with explanatory variables
- `parameters`: initial/fixed values for model parameters (default: empty `Dict()`)
- `availability`: list of boolean vectors indicating which alternatives are available (default: all available)

# Returns
- A `LogitModel` instance
"""
function LogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    parameters::Dict = Dict(),
    availability::Vector{<:AbstractVector{Bool}} = []
)
    return LogitModel(utilities, data, parameters, availability)
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
    - `vcov`: classical variance-covariance matrix
    - `rob_vcov`: robust variance-covariance matrix
    - `loglikelihood`: log-likelihood at optimum
    - `iters`: number of iterations
    - `converged`: convergence status
    - `estimation_time`: time taken (in seconds)
"""
function estimate(
    model::LogitModel,
    choicevar::Symbol;
    verbose::Bool = true
)

    choice_data = model.data[:,choicevar]

    if any(ismissing, choice_data)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choice_data)
    
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
    
    vcov = try
        inv(H)
    catch
        pinv(H)
    end

    d = diag(vcov)
    bad = free_names[findall(<(0), d)]
    if !isempty(bad)
        @warn "Non-positive-definite Hessian: negative variance for $(bad); reporting NaN standard error(s). These parameters are likely not identified."
    end
    std_errors = [dᵢ < 0 ? NaN : sqrt(dᵢ) for dᵢ in d]

    se = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        se[name] = std_errors[i]
    end

    if verbose
        println("Computing Robust Standard Errors")
    end
    ForwardDiff.jacobian!(scores,f_obj_i, θ̂)  # N × K
    G = scores' * scores  # K × K    

    V_rob = try
        inv(H) * G * inv(H)
    catch
        pinv(H) * G * pinv(H)
    end

    d_rob = diag(V_rob)
    bad_rob = free_names[findall(<(0), d_rob)]
    if !isempty(bad_rob)
        @warn "Non-positive-definite robust covariance: negative variance for $(bad_rob); reporting NaN robust standard error(s). These parameters are likely not identified."
    end
    rob_std_errors = [dᵢ < 0 ? NaN : sqrt(dᵢ) for dᵢ in d_rob]

    rob_se = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        rob_se[name] = rob_std_errors[i]
    end

    t_end = time()

    return (
        result = result,
        parameters = estimated_params,
        std_errors = se,
        vcov = vcov,
        rob_std_errors = rob_se,
        rob_vcov = V_rob,
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
