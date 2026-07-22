using DataFrames

struct LatentClassModel <: DiscreteChoiceModel
    expr::DCMExpression
    data::DataFrame
    id::Union{Nothing, Tuple{Dict,Vector}}
    parameters::Dict
end

function LatentClassModel(
    expression::DCMExpression;
    data::DataFrame,
    idvar::Union{Nothing,Symbol}=nothing,
    parameters::Dict = Dict()
)

    if !isnothing(idvar)
        # Get id variable
        id = data[:,idvar]

        # 0. Ensure IDs are sorted
        @assert issorted(id) "The vector `id` must be sorted to ensure consistent draw assignment."

        # 2. : Identify unique individuals
        individuals = unique(id)
        id_index_map = Dict(pid => idx for (idx, pid) in enumerate(individuals))
    
        return LatentClassModel(
            expression,
            data,
            (id_index_map, id),
            parameters
        )
    else
        return LatentClassModel(
            expression,
            data,
            nothing,
            parameters
        )
    end
end

function loglikelihood(model::LatentClassModel, Y::Matrix{Bool}; parameters::Dict = model.parameters)
    probs = evaluate(model.expr, model.data, parameters)  # N × J
    N, J = size(probs)

    # Ensure numerical stability
    probs = max.(probs, 1e-30)

    # Extract probabilities of chosen alternatives
    chosen_probs = sum(probs .* Y, dims=2)[:, 1]  # N-vector
    log_chosen = log.(chosen_probs)

    if isnothing(model.id)
        return log_chosen  # Cross-sectional log-likelihood
    else
        id_map, id = model.id
        I = length(id_map)
        log_indiv = zeros(eltype(probs), I)

        for n in 1:N
            i = id_map[id[n]]
            log_indiv[i] += log_chosen[n]
        end

        return log_indiv  # Panel log-likelihood
    end
end

function estimate(model::LatentClassModel, choicevar::Symbol; verbose::Bool = true)

    # Parameter setup
    params = collect_parameters(model.expr)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    choice_data = model.data[:, choicevar]

    if any(ismissing, choice_data)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choice_data)
    
    J = size(evaluate(model.expr, model.data, init_values), 2)
    N = length(choices)

    # Build Y: N × J matrix (one-hot)
    Y = zeros(Bool, N, J)
    @inbounds for n in 1:N
        j = choices[n]
        if j > 0
            Y[n, j] = true
        end
    end

    θ0 = [init_values[n] for n in free_names]
    mutable_parameters = deepcopy(model.parameters)

    function f_obj_i(θ)
        @inbounds for (i, name) in enumerate(free_names)
            mutable_parameters[name] = θ[i]
        end
        for name in fixed_names
            mutable_parameters[name] = init_values[name]
        end
        loglikelihood(model, Y; parameters=mutable_parameters)
    end

    function f_obj(θ)
        @inbounds for (i, name) in enumerate(free_names)
            mutable_parameters[name] = θ[i]
        end
        for name in fixed_names
            mutable_parameters[name] = init_values[name]
        end
        -sum(loglikelihood(model, Y; parameters=mutable_parameters))
    end

    if verbose
        println("Warming-up automatic differentiation...")
    end
    
    # Warm-up automatic differentiation
    H = zeros(length(θ0), length(θ0))
    cfg = ForwardDiff.HessianConfig(f_obj, θ0)
    H = ForwardDiff.hessian!(H, f_obj, θ0, cfg)
    
    ForwardDiff.gradient(f_obj, θ0)
    
    scores = zeros(length(f_obj_i(θ0)),length(θ0))
    ForwardDiff.jacobian!(scores,f_obj_i, θ0)

    if verbose
        println("Starting optimization routine...")
        println("Init Log-likelihood: ", round(-f_obj(θ0); digits=2))
    end

    t_start = time()
    result = Optim.optimize(
        f_obj,
        θ0,
        Optim.BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(
            show_trace = verbose,
            iterations = 1000,
            f_abstol = 1e-6,
            g_abstol = 1e-8
        );
        autodiff = :forward
    )

    if verbose && Optim.converged(result)
        println("Converged")
    end

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        estimated_params[name] = θ̂[i]
    end
    for name in fixed_names
        estimated_params[name] = init_values[name]
    end

    # Hessian and standard errors
    if verbose
        println("Computing Standard Errors...")
    end

    ForwardDiff.hessian!(H, f_obj, θ̂, cfg)

    vcov = try inv(H) catch; pinv(H) end
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
        println("Computing Robust Standard Errors...")
    end

    ForwardDiff.jacobian!(scores,f_obj_i, θ̂)  # N × K
    G = scores' * scores

    V_rob = try inv(H) * G * inv(H) catch; pinv(H) * G * pinv(H) end
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
        loglikelihood = -Optim.minimum(result),
        iters = Optim.iterations(result),
        converged = Optim.converged(result),
        estimation_time = t_end - t_start
    )
end

function evaluate(
    expressions::Dict{Symbol, <:DCMExpression},
    model::LatentClassModel,
    results::NamedTuple
)
    error("Evaluate for LatentClassModel is not implemented yet")
end