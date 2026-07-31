"""
Implementation of the NestedLogitModel type and estimation methods.

Defines the nesting tree (`Nest`), the flat plan the likelihood walks, the
log-space nested logit probabilities, and estimation on θ = log λ.
"""

using DataFrames

# ---------------------------------------------------------------------------
# The tree
#
# Apollo describes the tree as two cross-referencing dictionaries (`nlNests`
# name => λ, `nlStructure` name => children). That needs a name registry, and
# every nest appears twice, so the two halves can disagree. Here the tree is a
# single recursive object: the indentation *is* the structure.
# ---------------------------------------------------------------------------

"""
A nest in a nested logit tree.

# Fields
- `lambda`: the nest's scale parameter, either a `Parameter` (estimated) or a
  plain number (fixed). It doubles as the nest's *identity* — there is no
  separate name field, so λ flows through `collect_parameters` → `estimate` →
  `summarize_results` with no special-casing.
- `children::Vector`: the nest's children, each either an alternative's name
  (a `Symbol`, as declared in the model's `alternatives`) or another `Nest`.
  Nesting is arbitrarily deep.

An alternative sitting where a nest could go is simply a non-`Nest` child at
that level, so degenerate nests need no wrapper and no convention.

# Example
```julia
λ_PT = Parameter(:λ_PT, value=1.0)
tree = [:car, Nest(λ_PT, [:bus, :air, :rail])]   # root is a plain Vector
```
"""
struct Nest
    lambda::Union{DCMParameter, Real}
    children::Vector

    function Nest(lambda::Union{DCMParameter, Real}, children::AbstractVector)
        if isempty(children)
            error("""
                  A `Nest` needs at least one child, but one was given none. Children are \
                  alternative names (`Symbol`s from `alternatives`) or further `Nest`s.
                  """)
        end
        return new(lambda, collect(children))
    end
end

Nest(lambda, children) = error("""
    `Nest(lambda, children)` takes a `Parameter` or a number as its scale parameter and a \
    vector of children, but got a $(typeof(lambda)) and a $(typeof(children)). \
    For example: `Nest(Parameter(:λ_PT, value=1.0), [:bus, :rail])`.
    """)

# The root of a tree has λ ≡ 1 by normalization, so it is never written by the
# user; it is materialized here so that every alternative has a uniform path of
# nests up to a single top node.
const _NL_ROOT_LAMBDA = 1.0

_nl_lambda_label(l::DCMParameter) = string(l.name)
_nl_lambda_label(l::Real) = "λ = $l"

_nl_lambda_value(l::DCMParameter, parameters::AbstractDict) = parameters[l.name]
_nl_lambda_value(l::Real, ::AbstractDict) = l

function _nl_check_lambda_start(l::DCMParameter)
    if !(l.value > 0)
        error("""
              Nest parameter $(l.name) starts at $(l.value), but a nest scale parameter must be \
              strictly positive: `exp(V/λ)` inverts the model at λ < 0 and overflows as λ → 0. \
              Nested logit is estimated on θ = log λ, so the starting value has to admit a log.
              """)
    end
    return nothing
end

function _nl_check_lambda_start(l::Real)
    if !(l > 0)
        error("""
              A fixed nest scale parameter must be strictly positive, but got $l.
              """)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The flat plan
#
# The tree is walked exactly once, at construction, into arrays the likelihood
# can index. The hot loop never re-traverses the tree and never touches the
# heterogeneous `children` vector. Same idea as `_lc_classes`.
# ---------------------------------------------------------------------------

"""
Flattened nesting structure, built once by `_nl_plan`.

# Fields
- `lambdas`: scale parameter of each nest; the root is the last entry and is fixed at 1
- `children`: per nest, its children as `(is_nest, index)` pairs
- `path`: per alternative, the nest indices from its immediate parent up to the root
- `order`: nest indices in an order that visits children before parents
- `avail`: per nest, per observation, whether *any* alternative below it is available
- `root`: index of the root nest
"""
struct NLPlan
    lambdas::Vector{Union{DCMParameter, Real}}
    children::Vector{Vector{Tuple{Bool, Int}}}
    path::Vector{Vector{Int}}
    order::Vector{Int}
    avail::Vector{Vector{Bool}}
    root::Int
end

"""
Walks a nesting tree into an `NLPlan`, validating it on the way.

Validation performed here, all of it once per model rather than per likelihood
evaluation:
- every alternative in `alternatives` appears **exactly once** in the tree —
  a duplicated or an omitted alternative both throw, naming the alternative
- a child `Symbol` that is not a declared alternative throws
- a child that is neither a `Symbol` nor a `Nest` throws
- λ ≤ 0 at the starting values throws (see `_nl_check_lambda_start`)
- a nest with a single child **warns**: its λ cannot be identified, since there
  is no choice to be made inside it. This warns rather than throws, in the
  spirit of `_check_class_separation` — the model is still estimable.

# Arguments
- `tree::AbstractVector`: the root's children (the root itself is implicit, λ ≡ 1)
- `alternatives::NamedTuple`: the model's alternative set
- `availability`: availability vectors, in the alternatives' order
"""
function _nl_plan(tree, alternatives::NamedTuple, availability::AbstractVector)
    if !(tree isa AbstractVector)
        error("""
              `tree` must be a `Vector` holding the root's children — the root's scale parameter \
              is fixed at 1 by normalization, so there is nothing to supply for it and no `Nest` \
              wrapper around the whole tree. Got a $(typeof(tree)). \
              For example: `tree = [:car, Nest(λ_PT, [:bus, :rail])]`.
              """)
    end
    isempty(tree) && error("`tree` is empty: a nested logit needs at least one branch at the root.")

    labels = collect(keys(alternatives))
    position = Dict(k => j for (j, k) in enumerate(labels))
    J = length(labels)

    lambdas = Union{DCMParameter, Real}[]
    children = Vector{Tuple{Bool, Int}}[]
    parent_of_nest = Int[]
    parent_of_alt = zeros(Int, J)
    claimed = zeros(Int, J)

    function add_nest!(lambda, parent::Int)
        _nl_check_lambda_start(lambda)
        push!(lambdas, lambda)
        push!(children, Tuple{Bool, Int}[])
        push!(parent_of_nest, parent)
        return length(lambdas)
    end

    function walk(node, parent::Int)
        if node isa Nest
            m = add_nest!(node.lambda, parent)
            for c in node.children
                push!(children[m], walk(c, m))
            end
            if length(node.children) == 1
                @warn """
                      Nest $(_nl_lambda_label(node.lambda)) has a single child, so its scale \
                      parameter is not identified: with nothing to choose between inside the \
                      nest, the likelihood does not depend on λ. Either give the nest more \
                      children, or drop the nest and place the child at its parent's level \
                      (a degenerate nest needs no wrapper).
                      """
            end
            return (true, m)
        elseif node isa Symbol
            j = get(position, node, 0)
            if j == 0
                error("""
                      `tree` refers to `:$node`, which is not one of the model's alternatives \
                      ($(join(string.(labels), ", "))). Children are given by name; with \
                      `alternatives` supplied as a plain vector of codes the generated names are \
                      $(join(string.(labels), ", ")).
                      """)
            end
            claimed[j] += 1
            if claimed[j] > 1
                error("""
                      Alternative `:$node` appears more than once in `tree`. Each alternative \
                      belongs to exactly one nest.
                      """)
            end
            parent_of_alt[j] = parent
            return (false, j)
        else
            error("""
                  `tree` contains a $(typeof(node)), but every child must be either an \
                  alternative's name (a `Symbol`) or a `Nest`.
                  """)
        end
    end

    # The root is created first, so every nest is created after its parent and
    # therefore carries a strictly larger index. That is what makes `order`
    # below — plain reverse creation order — visit children before parents.
    root = add_nest!(_NL_ROOT_LAMBDA, 0)
    for c in tree
        push!(children[root], walk(c, root))
    end
    if length(tree) == 1
        @warn """
              The root of `tree` has a single child, so the model is degenerate: the top-level \
              choice is between one option, and that child's scale parameter is not identified.
              """
    end

    missing_ = labels[findall(iszero, claimed)]
    if !isempty(missing_)
        error("""
              `tree` does not place $(join(string.(missing_), ", ")) anywhere. Every alternative \
              in `alternatives` must appear exactly once in the tree; an alternative that belongs \
              to no nest is simply listed at the root.
              """)
    end

    M = length(lambdas)
    order = collect(M:-1:1)

    # Path from each alternative up to (and including) the root.
    path = Vector{Vector{Int}}(undef, J)
    for j in 1:J
        chain = Int[]
        p = parent_of_alt[j]
        while p != 0
            push!(chain, p)
            p = parent_of_nest[p]
        end
        path[j] = chain
    end

    # A nest is "available" for an observation when at least one alternative
    # below it is. Precomputed because it is fixed across the estimation, and
    # because deriving it at run time would mean comparing Dual numbers against
    # -Inf inside the hot loop.
    N = length(first(availability))
    avail = [Vector{Bool}(undef, N) for _ in 1:M]
    for m in order
        for n in 1:N
            a = false
            for (is_nest, idx) in children[m]
                a = is_nest ? avail[idx][n] : availability[idx][n]
                a && break
            end
            avail[m][n] = a
        end
    end

    bad = findfirst(!, avail[root])
    if bad !== nothing
        error("Observation $bad has no available alternatives.")
    end

    return NLPlan(lambdas, children, path, order, avail, root)
end

"""
The nest scale parameters that are `Parameter`s, in the plan's nest order.
The root's fixed λ ≡ 1 is a plain number and so never appears here.
"""
_nl_lambda_parameters(plan::NLPlan) = DCMParameter[l for l in plan.lambdas if l isa DCMParameter]

"""
Every parameter of a nested logit: those in the utilities, followed by the nest
scale parameters, de-duplicated by name with the first occurrence winning.

`estimate` and the delta-method `evaluate` must agree on this list and its order,
since it fixes the layout of θ and therefore of every covariance matrix.
"""
function _nl_parameters(model)
    params = collect_parameters(model.utilities)
    seen = Set(p.name for p in params)
    for l in _nl_lambda_parameters(model.plan)
        if !(l.name in seen)
            push!(params, l)
            push!(seen, l.name)
        end
    end
    return params
end

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

"""
Data structure for nested logit models.

# Fields
- `alternatives::NamedTuple`: alternative name => code in the choice column; defines
  the alternative set and the canonical order of every other per-alternative field
- `utilities::Vector{<:DCMExpression}`: utility expressions, in `alternatives` order
- `availability::Vector{<:AbstractVector{Bool}}`: availability vectors, in `alternatives` order
- `tree::Vector`: the nesting tree as written by the user (kept for `show`)
- `plan::NLPlan`: the flattened tree the likelihood walks
- `data::DataFrame`: input dataset
- `parameters::Dict`: parameter values (estimates or fixed)
"""
struct NestedLogitModel <: DiscreteChoiceModel
    alternatives::NamedTuple
    utilities::Vector{<:DCMExpression}
    availability::Vector{<:AbstractVector{Bool}}
    tree::Vector
    plan::NLPlan
    data::DataFrame
    parameters::Dict
end

"""
Constructor for `NestedLogitModel`.

# Arguments
- `alternatives`: the alternative set, as a NamedTuple mapping each alternative's
  name to the code identifying it in the choice column — `(car = 1, bus = 4, rail = 7)`
- `utilities`: utility expressions, as a NamedTuple over the same names as
  `alternatives` (their order is irrelevant — they are matched by name)
- `tree`: the nesting structure, a `Vector` of the root's children. Each child is
  either an alternative's name or a `Nest`; see `Nest`. The root's scale parameter
  is fixed at 1 by normalization and is not supplied.
- `availability`: boolean vectors keyed by name (default: all alternatives available)
- `data`: `DataFrame` with explanatory variables
- `parameters`: initial/fixed values for model parameters (default: empty `Dict()`)

# Convention
λ follows Apollo: `V_nest = λ · log Σ_j exp(V_j / λ)`, with λ ≤ 1 the
RUM-consistent range. Biogeme parameterizes the reciprocal scale μ = 1/λ ≥ 1, so
the two packages fit the same model but report reciprocal nest parameters.

# Example
```julia
λ_PT = Parameter(:λ_PT, value=1.0)
model = NestedLogitModel(
    (car = 1, bus = 2, air = 3, rail = 4);
    utilities    = (car = V_car, bus = V_bus, air = V_air, rail = V_rail),
    tree         = [:car, Nest(λ_PT, [:bus, :air, :rail])],
    availability = (car = av_car, bus = av_bus, air = av_air, rail = av_rail),
    data         = df,
)
```
"""
function NestedLogitModel(
    alternatives;
    utilities,
    tree,
    availability = [],
    data::DataFrame,
    parameters::Dict = Dict()
)
    alts, named = _resolve_alternatives(alternatives)
    utils = _check_utilities(_match_alternatives(alts, named, utilities, "utilities"))
    avail = isempty(availability) ? _default_availability(alts, data) :
            _check_availability(_match_alternatives(alts, named, availability, "availability"), data)

    plan = _nl_plan(tree, alts, avail)

    return NestedLogitModel(alts, utils, avail, collect(tree), plan, data, parameters)
end

# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

"""
Log choice probabilities under the nested logit model, as an `N × J` matrix.

Everything is computed in logs. Writing `W` for a node's value — a utility for an
alternative, an inclusive value `W_m = λ_m · log Σ_c exp(W_c / λ_m)` for a nest —
the conditional probability of a child within its parent is

    log P(c | m) = (W_c − W_m) / λ_m

because `W_m / λ_m` is exactly the log-sum-exp being normalized by. The
alternative's log probability is then the sum of that along its path to the root,
which is why the plan stores the path.

Numerically, each log-sum-exp subtracts the per-row maximum before exponentiating,
the same stabilization `logit_prob` needs — and it matters more here, since
dividing by a small λ stretches the exponent range.

Unavailable alternatives are masked **after** the division by λ rather than by
pushing `-Inf` through it. `-Inf / λ` is fine in value but its ForwardDiff partial
is `Inf · ∂λ`, which then makes `exp(...)` produce a `NaN` derivative even though
the term contributes nothing. Masking afterwards keeps the `-Inf` a constant with
zero partials. For the same reason a nest with no available children takes its
value from `plan.avail` instead of from a `-Inf` comparison.

# Arguments
- `utilities`: vector of symbolic utility expressions, in the alternatives' order
- `data`: `DataFrame` of observed variables
- `availability`: vector of boolean vectors, in the alternatives' order
- `plan::NLPlan`: the flattened nesting structure
- `parameters`: dictionary mapping parameter names to values (λ in model space)

# Returns
- `N × J` matrix of log probabilities; `-Inf` for unavailable alternatives
"""
function nl_logprob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    availability::Vector{<:AbstractVector{Bool}},
    plan::NLPlan,
    parameters::AbstractDict
)
    N = nrow(data)
    J = length(utilities)
    M = length(plan.lambdas)

    utils = Vector{Vector}(undef, J)
    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], data, parameters)
    end

    # λ values are read out of the tree once, here: the plan's `lambdas` field is
    # abstractly typed, so touching it inside the loops below would cost dispatch
    # on every observation.
    λraw = Any[_nl_lambda_value(l, parameters) for l in plan.lambdas]

    T = promote_type(
        mapreduce(eltype, promote_type, utils),
        mapreduce(typeof, promote_type, λraw)
    )
    λ = T[x for x in λraw]

    W_alt = Array{T}(undef, N, J)
    @inbounds for j in 1:J
        @views W_alt[:, j] .= utils[j]
    end

    # Inclusive values, children before parents.
    W_nest = Array{T}(undef, N, M)
    mx = Vector{T}(undef, N)
    s = Vector{T}(undef, N)

    @inbounds for m in plan.order
        lam = λ[m]
        kids = plan.children[m]

        fill!(mx, T(-Inf))
        for (is_nest, idx) in kids
            for n in 1:N
                if is_nest ? plan.avail[idx][n] : availability[idx][n]
                    z = (is_nest ? W_nest[n, idx] : W_alt[n, idx]) / lam
                    if z > mx[n]
                        mx[n] = z
                    end
                end
            end
        end

        # Where the nest is unavailable the shift is meaningless; zero it so no
        # -Inf ever reaches the arithmetic below.
        for n in 1:N
            plan.avail[m][n] || (mx[n] = zero(T))
        end

        fill!(s, zero(T))
        for (is_nest, idx) in kids
            for n in 1:N
                if is_nest ? plan.avail[idx][n] : availability[idx][n]
                    z = (is_nest ? W_nest[n, idx] : W_alt[n, idx]) / lam
                    s[n] += exp(z - mx[n])
                end
            end
        end

        for n in 1:N
            # An available nest always has an available child contributing
            # exp(0) = 1, so s ≥ 1 here and the log is safe. The unavailable
            # branch substitutes 1 so that `log` is never handed a zero at all,
            # rather than relying on `ifelse` to discard a NaN partial.
            avail_m = plan.avail[m][n]
            s_n = avail_m ? s[n] : one(T)
            W_nest[n, m] = avail_m ? lam * (mx[n] + log(s_n)) : zero(T)
        end
    end

    logP = Array{T}(undef, N, J)
    Threads.@threads for j in 1:J
        chain = plan.path[j]
        av = availability[j]
        @inbounds for n in 1:N
            if !av[n]
                logP[n, j] = T(-Inf)
                continue
            end
            acc = (W_alt[n, j] - W_nest[n, chain[1]]) / λ[chain[1]]
            for k in 2:length(chain)
                acc += (W_nest[n, chain[k - 1]] - W_nest[n, chain[k]]) / λ[chain[k]]
            end
            logP[n, j] = acc
        end
    end

    return logP
end

"""
Choice probabilities under the nested logit model, as an `N × J` matrix.
Thin wrapper over `nl_logprob`; the likelihood uses the log form directly.
"""
nl_prob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    availability::Vector{<:AbstractVector{Bool}},
    plan::NLPlan,
    parameters::AbstractDict
) = exp.(nl_logprob(utilities, data, availability, plan, parameters))

# Cross-sectional `evaluate` for a nested NestedLogitModel term, mirroring the
# `LogitModel` method — this is what lets an NL serve as a class inside a
# `LatentClassModel` expression.
# The symbolic children the `collect_*` walkers descend into when this model is
# nested inside another expression. See the traversal note in `Utils.jl`.
#
# A nested logit is the one model whose parameters are NOT all in `utilities`: the
# nest scale parameters live in the tree, reachable only through the flattened
# plan. Returning `utilities` alone would let an NL class inside a latent class
# contribute its taste parameters but silently drop every λ — which reads as a
# converged model whose λs never moved off their starting values. Utilities first,
# then λs, matching `_nl_parameters` so the two agree on order.
_children(m::NestedLogitModel) =
    vcat(Vector{DCMExpression}(m.utilities),
         Vector{DCMExpression}(_nl_lambda_parameters(m.plan)))

# λ is searched as log λ; see the estimation-space discussion in `estimate` below.
# A latent class containing this model reads this to build its own `log_scale` mask,
# so the inner model's estimation space survives being nested.
_log_scale_names(m::NestedLogitModel) = (l.name for l in _nl_lambda_parameters(m.plan))

evaluate(e::NestedLogitModel, data::DataFrame, params::AbstractDict) =
    nl_prob(e.utilities, data, e.availability, e.plan, params)

"""
Computes predicted probabilities using estimated parameters.

# Arguments
- `model::NestedLogitModel`: the model structure
- `results::NamedTuple`: output of `estimate`, must include `parameters`

# Returns
- `N × J` matrix of predicted probabilities
"""
predict(model::NestedLogitModel, results::NamedTuple) =
    nl_prob(model.utilities, model.data, model.availability, model.plan, results.parameters)

"""
Computes the log-likelihood of the model given observed choices.

# Arguments
- `model::NestedLogitModel`: model object with defined parameters
- `choices::Vector{Int}`: **positions** of the chosen alternatives (see `_recode_choices`)

# Returns
- `Vector`: log-likelihood contribution per observation
"""
function loglikelihood(model::NestedLogitModel, choices::Vector{Int}; parameters::Dict = model.parameters)
    logP = nl_logprob(model.utilities, model.data, model.availability, model.plan, parameters)

    N = size(logP, 1)
    T = eltype(logP)
    loglik = Vector{T}(undef, N)

    Threads.@threads for n in 1:N
        @inbounds loglik[n] = logP[n, choices[n]]
    end

    return loglik
end

# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

"""
Estimates the parameters of a `NestedLogitModel` via maximum likelihood.

**Estimation space.** Nest scale parameters are searched as θ = log λ, so λ = exp(θ)
is strictly positive by construction. An unconstrained search on λ itself fails as a
`NaN` log-likelihood somewhere inside a line search — which surfaces as a baffling
convergence failure rather than as a message about λ. The transform deliberately does
*not* impose λ ≤ 1: λ > 1 breaks no arithmetic, it is a diagnosis (inconsistent with
global RUM), so it is computed and **warned about** rather than made unreachable. It
also keeps λ = 1 attainable at the finite point θ = 0 — any smooth monotone map of ℝ
onto (0,1] would send the MNL special case to θ = ±∞, turning the very common "the
nest is not significant" outcome into a non-converging optimizer.

A bonus of θ = log λ: the null λ = 1 is exactly θ = 0, so an ordinary t-statistic
against zero in the estimation space is the test against MNL.

**Reporting space.** Everything returned is in the user's space: `parameters[:λ]` is
λ̂ = exp(θ̂), and the covariance matrices are converted by `curvature_in_model_space`
before any estimator is built, so all three (classical, robust, BHHH) are consistent
and `SE(λ̂) = λ̂ · SE(θ̂)` — the delta method done exactly.

# Arguments
- `model::NestedLogitModel`: model specification
- `choicevar::Symbol`: name of the column in `model.data` that contains observed choices
- `verbose::Bool=true`: whether to print optimization progress
- `hessian_method::Symbol=:ad`: how to compute the Hessian the covariance matrices are
  built from — `:ad` for exact ForwardDiff second derivatives, `:fd` for a finite-difference
  Jacobian of the exact gradient (Apollo's routine). See `model_hessian!`

# Returns
- `NamedTuple` with the same fields as `estimate(::LogitModel, …)`
"""
function estimate(
    model::NestedLogitModel,
    choicevar::Symbol;
    verbose::Bool = true,
    hessian_method::Symbol = :ad
)
    # Validate before optimizing, not after: a typo'd method should not cost a full
    # estimation run before it is reported.
    _check_hessian_method(hessian_method)

    choice_data = model.data[:, choicevar]

    # Translate the analyst's alternative codes into positions in `model.alternatives`
    # once, here; everything downstream indexes the probability matrix by position.
    choices = _recode_choices(choice_data, model.alternatives, choicevar)

    params = _nl_parameters(model)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    # Get free/fixed masks
    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    # Which free parameters the optimizer searches as logs. Fixed λs are written
    # straight into the parameter dict in model space and so never transformed.
    lambda_names = Set(l.name for l in _nl_lambda_parameters(model.plan))
    log_scale = [n in lambda_names for n in free_names]

    θ0 = [_estimation_space_value(init_values[n], log_scale[i], n) for (i, n) in enumerate(free_names)]

    # `Dict{Symbol,Any}`, not `deepcopy`: the objective closures below write
    # ForwardDiff `Dual`s into this dict, which a user-supplied `parameters` dict
    # typed as `Dict{Symbol,Float64}` cannot hold (it raised a `MethodError` from
    # `convert`). The value type has to admit Duals regardless of what was passed.
    mutable_parameters = Dict{Symbol,Any}(model.parameters)

    function f_obj_i(θ)
        @inbounds begin
            for (i, name) in enumerate(free_names)
                mutable_parameters[name] = _model_space_value(θ[i], log_scale[i])
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
                mutable_parameters[name] = _model_space_value(θ[i], log_scale[i])
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
    cfg = _hessian_config(f_obj, θ0)
    H = model_hessian!(H, f_obj, θ0, cfg, hessian_method)

    ForwardDiff.gradient(f_obj, θ0)

    scores = zeros(length(choice_data), length(θ0))
    ForwardDiff.jacobian!(scores, f_obj_i, θ0)

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
        estimated_params[name] = _model_space_value(θ̂[i], log_scale[i])
    end

    for name in fixed_names
        estimated_params[name] = init_values[name]
    end

    # Hessian
    if verbose
        println("Computing Standard Errors")
    end

    model_hessian!(H, f_obj, θ̂, cfg, hessian_method)

    # `covariance_estimates` builds all three estimators at once (classical,
    # sandwich, BHHH/OPG), so the score Jacobian must be ready before it is called.
    ForwardDiff.jacobian!(scores, f_obj_i, θ̂)  # N × K
    G = scores' * scores  # K × K

    # Convert the curvature into the space the estimates are reported in, so that
    # every estimator below — not just the classical one — is in λ rather than θ.
    Hλ, Gλ = curvature_in_model_space(H, G, θ̂, log_scale)

    cov = covariance_estimates(Hλ, Gλ, free_names)

    t_end = time()

    _nl_warn_lambda_above_one(model, estimated_params)

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
Warns when any estimated nest scale parameter exceeds 1.

λ > 1 is arithmetically fine — the likelihood is perfectly well defined — but it is
inconsistent with global random utility maximization, so it is a result to report and
flag rather than a region to make unreachable. Same instinct as refusing to substitute
BHHH into the classical column: give the number, and say why to distrust it.
"""
_nl_warn_lambda_above_one(model::NestedLogitModel, estimated_params::AbstractDict) =
    _warn_lambda_above_one((l.name for l in _nl_lambda_parameters(model.plan)), estimated_params)

# Takes the λ names rather than a `NestedLogitModel`, so a `LatentClassModel` whose
# classes are nested logits gets the same diagnosis. It reaches its λs through
# `collect_log_scale_parameters`, not through a `plan` it does not have.
function _warn_lambda_above_one(lambda_names, estimated_params::AbstractDict)
    over = [(n, estimated_params[n]) for n in lambda_names
            if haskey(estimated_params, n) && estimated_params[n] > 1]
    isempty(over) && return nothing

    @warn """
          Nest scale parameter(s) above 1 at the optimum: \
          $(join(("$n = $(round(Float64(v); digits=4))" for (n, v) in over), ", ")). \
          The estimates are valid maximum-likelihood estimates, but λ > 1 is inconsistent \
          with global random utility maximization (it implies a negative correlation between \
          the alternatives in the nest). Treat the nesting structure as suspect: check whether \
          λ is significantly different from 1, and whether the alternatives grouped together \
          really are the close substitutes.
          """
    return nothing
end

"""
Evaluates derived expressions (e.g. WTP, elasticities) based on a fitted NestedLogitModel.

Delegates to the shared `delta_method`; the only thing specific to this model is that
the free-parameter list includes the nest scale parameters. The covariance matrices it
reads are already in λ space (`estimate` converts H and G before `covariance_estimates`),
which is the same space `results.parameters` reports λ̂ in, so no further correction
applies here.
"""
evaluate(
    expressions::Dict{Symbol, <:DCMExpression},
    model::NestedLogitModel,
    results::NamedTuple
) = delta_method(expressions, _nl_parameters(model), model.data, results)

# ---------------------------------------------------------------------------
# Display
#
# "Is my tree what I think it is" should be a one-line check, not a re-reading of
# nested brackets.
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", model::NestedLogitModel)
    labels = collect(keys(model.alternatives))
    println(io, "NestedLogitModel with $(length(labels)) alternatives")
    _nl_show_children(io, model.tree, labels, "")
    return nothing
end

function Base.show(io::IO, nest::Nest)
    print(io, "Nest(", _nl_lambda_label(nest.lambda), ", ", length(nest.children), " children)")
    return nothing
end

function _nl_show_children(io::IO, children, labels, prefix::AbstractString)
    for (i, c) in enumerate(children)
        last = i == length(children)
        branch = last ? "└─ " : "├─ "
        if c isa Nest
            println(io, prefix, branch, _nl_lambda_label(c.lambda))
            _nl_show_children(io, c.children, labels, prefix * (last ? "   " : "│  "))
        else
            println(io, prefix, branch, c)
        end
    end
    return nothing
end
