using DataFrames

struct LatentClassModel <: DiscreteChoiceModel
    expr::DCMExpression
    data::DataFrame
    id::Union{Nothing, Tuple{Dict,Vector}}
    parameters::Dict
end

"""
Constructs a `LatentClassModel` from a symbolic latent-class expression.

# Arguments
- `expression`: the class mixture, normally `Σ_c π_c * model_c`
- `data`: `DataFrame` with observations
- `idvar`: optional ID column. Supplying it selects the **panel** likelihood, in
  which class membership is drawn once per individual (see `loglikelihood`)
- `parameters`: optional parameter values (leaf `Parameter` values take precedence)
- `check_class_weights`: verify at construction that the class weights form a
  valid probability distribution (default `true`); see `_check_class_weights`

# Returns
- `LatentClassModel` instance
"""
function LatentClassModel(
    expression::DCMExpression;
    data::DataFrame,
    idvar::Union{Nothing,Symbol}=nothing,
    parameters::Dict = Dict(),
    check_class_weights::Bool = true
)

    if check_class_weights
        # Evaluate the weights at the leaf `Parameter` values, which is what
        # `estimate` uses as θ₀ (falling back to `parameters` for anything not
        # on a leaf). A spec whose weights don't sum to 1 is a modelling error,
        # and catching it here is far cheaper than reading it off a converged
        # but meaningless log-likelihood.
        init = Dict(p.name => p.value for p in collect_parameters(expression))
        _check_class_weights(expression, data, merge(parameters, init),
                             "the initial parameter values")
    end

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

"""
Decomposes a canonical latent-class expression `Σ_c π_c * P_c` into its
`(weight_expression, class_model)` terms.

The panel likelihood cannot be computed from the collapsed `N × J` mixture alone:
class membership is drawn **once per individual**, so the per-class choice
probabilities must be multiplied within an individual *before* they are mixed.
That requires getting at the individual class terms again, which is what this
walker recovers from the user-built expression tree.

Returns `nothing` if the expression is not a sum of `weight * model` products
(either operand order is accepted).
"""
_lc_classes(::DCMExpression) = nothing

function _lc_classes(e::DCMSum)
    left  = _lc_classes(e.left)
    left === nothing && return nothing
    right = _lc_classes(e.right)
    right === nothing && return nothing
    return vcat(left, right)
end

function _lc_classes(e::DCMMult)
    e.right isa DiscreteChoiceModel && return [(e.left, e.right)]
    e.left  isa DiscreteChoiceModel && return [(e.right, e.left)]
    return nothing
end

"""
Verifies that the latent-class weights form a valid probability distribution —
non-negative, and summing to 1 for every observation — at the given parameter
values.

Nothing in the symbolic API enforces this: the user writes the weights by hand
(typically `exp(δ_c) / Σ_c exp(δ_c)`), and a spec that omits the normalisation
still evaluates, still optimizes, and still reports a converged log-likelihood.
That number is not a log-likelihood, though — the "probabilities" don't
integrate to one — so it is comparable to nothing, and neither the LL, the AIC,
the ρ², nor any likelihood-ratio test against it means anything. Hence a check
rather than a silent renormalisation: rescaling the user's weights behind their
back would hide a specification error instead of surfacing it.

Silently skipped when the expression is not a canonical `Σ_c weight * model`
sum, since there are then no class weights to identify.

# Arguments
- `expr`: the latent-class expression
- `data`: `DataFrame` the weights are evaluated against
- `parameters`: parameter values to evaluate at
- `when::AbstractString`: phrase naming those values, used in the message
- `strict::Bool = true`: `true` throws, `false` warns (used post-estimation,
  where the run has already completed and the results are worth showing)
- `atol::Real = 1e-6`: tolerance on the deviation from 1
"""
function _check_class_weights(
    expr::DCMExpression,
    data::DataFrame,
    parameters::AbstractDict,
    when::AbstractString;
    strict::Bool = true,
    atol::Real = 1e-6
)
    classes = _lc_classes(expr)
    classes === nothing && return nothing

    weights = [evaluate(w, data, parameters) for (w, _) in classes]
    total = sum(weights)

    problems = String[]

    negative = findall(w -> any(<(-atol), w), weights)
    if !isempty(negative)
        push!(problems, "class(es) $(negative) have negative membership probabilities " *
                        "(minimum $(minimum(minimum, weights[negative])))")
    end

    worst = maximum(abs.(total .- 1))
    if worst > atol
        push!(problems, "membership probabilities sum to between $(minimum(total)) and " *
                        "$(maximum(total)) across observations, not 1 " *
                        "(largest deviation $(worst))")
    end

    isempty(problems) && return nothing

    msg = """
          Latent class membership probabilities are not a valid distribution at $when: \
          $(join(problems, "; ")).
          Class weights must be non-negative and sum to 1 over classes for every observation \
          — e.g. `prob_c = exp(delta_c) / sum(exp(delta_1) + ... + exp(delta_C))` with one \
          delta fixed for identification. Otherwise the reported log-likelihood is not a \
          log-likelihood, and the AIC/BIC/rho-squared derived from it are meaningless.
          """

    strict ? error(msg) : @warn msg
    return nothing
end

"""
Computes the log-likelihood of a Latent Class model.

Two regimes, selected by whether the model was built with an `idvar`:

- **Cross-sectional** (`id === nothing`): each observation is its own decision
  maker, so the collapsed mixture `Σ_c π_c P_c(j)` evaluated per row is already
  the right thing; returns one contribution per observation.
- **Panel** (`id` present): class membership is fixed within an individual, so
  the likelihood is `Σ_c π_c Π_t P_c(j_t)` — the class-conditional probabilities
  are multiplied over the individual's observations *first*, then mixed. Mixing
  per observation instead (`Π_t Σ_c π_c P_c(j_t)`) is a different, incorrect
  model. Returns one contribution per individual.

The panel path is computed in logs throughout and mixed with a log-sum-exp, so
long choice sequences (whose sequence probabilities underflow `Float64` quickly)
stay representable.
"""
function loglikelihood(model::LatentClassModel, Y::Matrix{Bool}; parameters::Dict = model.parameters)
    if isnothing(model.id)
        probs = evaluate(model.expr, model.data, parameters)  # N × J
        chosen_probs = sum(probs .* Y, dims=2)[:, 1]          # N-vector
        return log.(max.(chosen_probs, 1e-30))
    end

    classes = _lc_classes(model.expr)
    if classes === nothing
        error("""
              Panel latent-class estimation needs the class structure, but the model
              expression is not a sum of `class_probability * class_model` terms
              (e.g. `prob_1 * model_1 + prob_2 * model_2`). Either write it in that
              form, or build the model without `idvar` to treat every row as an
              independent decision maker.
              """)
    end

    id_map, id = model.id
    I = length(id_map)
    C = length(classes)
    N = nrow(model.data)

    # Per class: log P_c(chosen) per observation, and the class weight per observation.
    log_chosen = [log.(max.(sum(evaluate(m, model.data, parameters) .* Y, dims=2)[:, 1], 1e-30))
                  for (_, m) in classes]
    weights = [evaluate(w, model.data, parameters) for (w, _) in classes]

    T = promote_type(eltype(first(log_chosen)), eltype(first(weights)))

    log_seq = zeros(T, I, C)   # log Π_t P_c(j_t) per individual
    log_w   = zeros(T, I, C)   # log π_c per individual

    for c in 1:C
        lp, π = log_chosen[c], weights[c]
        @inbounds for n in 1:N
            i = id_map[id[n]]
            log_seq[i, c] += lp[n]
            # Class membership is an individual-level quantity; a weight built from
            # individual-level covariates is constant within an individual, so any
            # of their rows gives the same value. We take the last one seen.
            log_w[i, c] = log(max(π[n], 1e-30))
        end
    end

    # Mix within individual: log Σ_c exp(log π_c + log Π_t P_c), via log-sum-exp.
    loglik = Vector{T}(undef, I)
    @inbounds for i in 1:I
        m = -T(Inf)
        for c in 1:C
            m = max(m, log_w[i, c] + log_seq[i, c])
        end
        s = zero(T)
        for c in 1:C
            s += exp(log_w[i, c] + log_seq[i, c] - m)
        end
        loglik[i] = m + log(s)
    end

    return loglik
end

"""
Predicts unconditional (class-mixed) choice probabilities for each observation.

# Arguments
- `model::LatentClassModel`: the model structure
- `results::NamedTuple`: estimation results, must include `parameters`

# Returns
- `Matrix{Float64}`: `N × J` matrix of `Σ_c π_c P_c(j)` probabilities
"""
function predict(model::LatentClassModel, results::NamedTuple)
    return evaluate(model.expr, model.data, results.parameters)
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

    # Null log-likelihood: equal probability over each observation's available
    # alternatives. The availability pattern lives on the nested class models, so
    # it is only reachable when the expression decomposes; otherwise report NaN
    # (`summarize_results` already degrades ρ² to NaN rather than failing).
    classes = _lc_classes(model.expr)
    ll0 = if classes === nothing || isempty(first(classes)[2].availability)
        NaN
    else
        null_loglikelihood_mnl(first(classes)[2].availability)
    end

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

    # The constructor checked the weights at θ₀; a spec can still leave the
    # simplex during optimization (e.g. free, unnormalized weight parameters that
    # happened to sum to 1 at the start). Warn rather than throw — the run is
    # finished and the user is better served seeing the estimates alongside the
    # reason not to trust the fit statistics.
    _check_class_weights(model.expr, model.data, estimated_params,
                         "the estimated parameter values"; strict=false)

    # Hessian and standard errors
    if verbose
        println("Computing Standard Errors...")
    end

    ForwardDiff.hessian!(H, f_obj, θ̂, cfg)

    # `covariance_estimates` builds all three estimators at once (classical,
    # sandwich, BHHH/OPG), so the score Jacobian must be ready before it is
    # called — not just before the robust block, as it used to be.
    ForwardDiff.jacobian!(scores,f_obj_i, θ̂)  # N × K
    G = scores' * scores

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

function evaluate(
    expressions::Dict{Symbol, <:DCMExpression},
    model::LatentClassModel,
    results::NamedTuple
)
    error("Evaluate for LatentClassModel is not implemented yet")
end