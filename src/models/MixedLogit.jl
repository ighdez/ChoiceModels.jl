using DataFrames, StatsBase

"""
Structure for a Mixed Logit model, allowing for random taste variation.

Combines symbolic utility expressions (fixed and random parameters) with data and simulation draws.

# Fields
- `alternatives::NamedTuple`: alternative name => code in the choice column; defines
  the alternative set and the canonical order of every other per-alternative field
- `utilities::Vector{DCMExpression}`: utility expressions, in `alternatives` order
- `availability::Vector{Vector{Bool}}`: availability flags, in `alternatives` order
- `data::DataFrame`: dataset with individual choice observations
- `id::Tuple{Dict, Vector}`: the panel structure, as `(id => row-block index, the raw
  ID column)`. The likelihood uses it to accumulate an individual's observations
  inside the draw integral
- `parameters::Dict`: parameter values (means, standard deviations, etc.). Untyped
  because the objective closures write ForwardDiff `Dual`s into the working copy
- `draws::Dict`: simulation draws per draw name, each already expanded from
  per-individual to per-observation (size `N × R`, `N = nrow(data)`)
- `R::Int`: number of draws per individual
- `draw_scheme::Symbol`: the scheme they were generated with. Stored because a
  `LatentClassModel` inherits it when it takes over draw generation for its classes
  (see `_lc_share_draws`)
"""
struct MixedLogitModel <: DiscreteChoiceModel
    alternatives::NamedTuple                        # Alternative name => choice-column code
    utilities::Vector{DCMExpression}                # Utility expressions V_j
    availability::Vector{Vector{Bool}}              # Alternative availability
    data::DataFrame                                 # Dataset
    id::Tuple{Dict,Vector}                          # ID
    parameters::Dict                                # Initial parameter values (mu, sigma, etc.)
    draws::Dict                                     # Draws: N x R
    R::Int                                          # Number of simulations (draws)
    draw_scheme::Symbol                             # Scheme the draws were generated with
end

"""
Constructs a `MixedLogitModel` from symbolic utility expressions and model inputs.

Draws are generated for every `Draw` symbol found in the utilities: `R` per
**individual**, then expanded to one row per observation, so an individual's
repeated choices integrate against the same taste draw. The ID column must be
sorted, and this is asserted.

# Arguments
- `alternatives`: the alternative set, as a NamedTuple mapping each alternative's name
  to the code identifying it in the choice column — `(car = 1, bus = 4, rail = 7)`. A
  plain vector of codes is accepted for unnamed alternatives (labelled `alt1, alt2, …`)
- `utilities`: utility expressions, keyed by the same names as `alternatives` (matched
  by name, so their order is irrelevant)
- `availability`: availability vectors, likewise keyed by name
- `data`: DataFrame with observations
- `idvar`: column identifying the individual (panel structure and draw assignment).
  Required, and must be sorted
- `parameters`: dictionary with initial values (including means/sigmas). Values on
  the leaf `Parameter`s take precedence at estimation
- `R`: number of draws per individual (default `100`). Memory scales linearly in it
  — see the resource note in CLAUDE.md before raising it on a full dataset
- `draw_scheme`: sampling method — `:normal` (default), `:uniform`, `:halton` or
  `:mlhs`. `:halton` is what the examples use; the pseudo-random `:normal` default
  is noisy enough at small `R` to move the optimizer onto different local optima

# Returns
- `MixedLogitModel` instance with utility functions and simulation-ready data
"""
function MixedLogitModel(
    alternatives;
    utilities,
    availability = [],
    data::DataFrame,
    idvar::Symbol,
    parameters::Dict = Dict(),
    draw_scheme::Symbol = :normal,
    R::Int = 100
)

    alts, named = _resolve_alternatives(alternatives)
    utils = _check_utilities(_match_alternatives(alts, named, utilities, "utilities"))
    # See `_default_availability`: an omitted `availability` means "all available",
    # which has to be materialized because the probability loops index it per
    # alternative.
    avail = isempty(availability) ? _default_availability(alts, data) :
            _check_availability(_match_alternatives(alts, named, availability, "availability"), data)

    # Get id variable
    id = data[:,idvar]

    # 0. Ensure IDs are sorted
    @assert issorted(id) "The vector `id` must be sorted to ensure consistent draw assignment."

    # 1. Collect all Draw objects in the utility expressions
    draw_symbols = collect_draws(utils)

    # 2. : Identify unique individuals
    individuals = unique(id)
    N_individuals = length(individuals)
    
    # 3. Generate all draws per individual using external Draws.jl infrastructure
    draw_struct = generate_draws(draw_symbols, N_individuals, R; scheme=draw_scheme)
    raw_draws = Dict(s => draw_struct.values[s] for s in draw_symbols)

    # 4. Expand draws to observation level
    id_index_map = Dict(pid => idx for (idx, pid) in enumerate(individuals))
    N = nrow(data)
    expanded_draws = Dict{Symbol, Matrix{Float64}}()
    for (param, matrix) in raw_draws
        param_draws = zeros(N, R)
        for (i, pid) in enumerate(id)
            param_draws[i, :] .= matrix[id_index_map[pid], :]
        end
        expanded_draws[param] = param_draws
    end

    # 5. Build and return the model
    return MixedLogitModel(
        alts,
        utils,
        avail,
        data,
        (id_index_map, id),
        parameters,
        expanded_draws,
        R,
        draw_scheme
    )
end

"""
Rebuilds a Mixed Logit onto an externally supplied set of draws.

Used by `LatentClassModel`, which takes over draw generation for its classes so
that a draw dimension named in two classes really is the same draw — see
`_lc_share_draws`. Everything but the draws is carried over unchanged.
"""
_with_draws(m::MixedLogitModel, draws::Dict, R::Int) =
    MixedLogitModel(m.alternatives, m.utilities, m.availability, m.data,
                    m.id, m.parameters, draws, R, m.draw_scheme)

"""
Computes conditional choice probabilities for Mixed Logit using simulation draws.

Evaluates utility expressions for each alternative and draw, then applies the stable
softmax across the alternatives, incorporating availability constraints.

**The likelihood does not go through here.** It calls `chosen_logprob`, which needs
only the chosen alternative and so never materializes the `N × J × R` tensor — under
`ForwardDiff` an element of that tensor is a nested `Dual` occupying 72 bytes at the
chunk size `_hessian_config` sets, so the tensor is hundreds of megabytes on a
realistic model. This function serves `predict`, `evaluate(::MixedLogitModel, …)` and
the latent-class construction checks, which genuinely want every alternative.

# Arguments
- `utilities::Vector{<:DCMExpression}`: utility expressions, in the alternatives' order
- `data::DataFrame`: dataset with variables
- `parameters::Dict`: dictionary with values for all model parameters
- `availability::Vector{<:AbstractVector{Bool}}`: availability flags (length J, each
  vector of length N)
- `draws::Dict{Symbol, Matrix{Float64}}`: draws per draw name (each `N × R`)

Note the argument order — `parameters` before `availability` — which differs from the
`LogitModel` method of the same name.

# Returns
- Probability tensor of size `N × J × R` (observation, alternative, draw); `0` for
  unavailable alternatives. The element type follows `parameters`.
"""
function logit_prob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    parameters::Dict,
    availability::Vector{<:AbstractVector{Bool}},  # N × J
    draws::Dict{Symbol, Matrix{Float64}}, # N × R
)
    J = length(utilities)

    # Evaluated utilities: utils[j] is N × R
    raw = Vector{Any}(undef, J)

    Threads.@threads for j in 1:J
        raw[j] = evaluate(utilities[j], data, parameters, draws)
    end

    # Narrow to a concrete element type and hand off through a FUNCTION BARRIER.
    # The utilities' element type is only known at run time — it is whatever `Dual`
    # the caller happens to be differentiating through — so the softmax kernel has
    # to be reached via a separate call for Julia to specialize it. Held inline
    # (the container used to be `Vector{Matrix{<:Real}}`, abstract) every `utils[j]`
    # in the loop below went through dynamic dispatch.
    return _mxl_softmax(identity.(raw), availability)
end

"""
Stable softmax over the evaluated utilities: turns `J` matrices of size `N × R`
into the `N × J × R` probability tensor.

Split out of `logit_prob` as a function barrier (see the call site). Allocates two
arrays — the tensor itself and an `N × R` normalizer — and nothing per draw: the
column slices are views, the normalizer is accumulated in place, and the final
division writes back into `expU` rather than into a second tensor. Under
`ForwardDiff` every element is a nested `Dual`, so each avoided `N × J × R`
temporary is hundreds of megabytes on a realistic model.
"""
function _mxl_softmax(utils::AbstractVector, availability)
    J = length(utils)
    N, R = size(utils[1])
    T = eltype(first(utils))

    # Stack utils into a single tensor of size (N, J, R). `expU` holds exp(u - m)
    # first and is normalized into the probabilities in place at the end.
    expU = Array{T}(undef, N, J, R)
    s_expU = Array{T}(undef, N, R)

    Threads.@threads for r in 1:R
        @inbounds begin
            # Per-row max utility across *available* alternatives, for numerically
            # stable softmax. Without this, extreme utilities (e.g. lognormal
            # coefficients on large draws) make every exp(u) underflow to 0, the
            # normalisation collapses to 0/0, and the simulated likelihood is biased
            # sharply downward. Subtracting the max is mathematically identical (it
            # cancels in the ratio) but keeps the largest term at exp(0)=1.
            m = fill(T(-Inf), N)
            for j in 1:J
                m .= ifelse.(availability[j], max.(m, @view(utils[j][:, r])), m)
            end
            # Accumulate the normalizer as the columns are written, rather than
            # re-reading the slab with `sum(expU[:, :, r]; dims=2)` afterwards.
            # Sequential in `j`, which is the order `sum` reduces in, so the result
            # is bit-for-bit what it was.
            s = @view s_expU[:, r]
            fill!(s, zero(T))
            for j in 1:J
                u = @view utils[j][:, r]
                e = @view expU[:, j, r]
                e .= ifelse.(availability[j], exp.(u .- m), 0.0)
                s .+= e
            end
        end
    end

    # Normalize in place: `expU` becomes the probability tensor. Allocating a
    # separate `probs` here cost one full N × J × R tensor of Duals per call.
    @inbounds expU ./= max.(reshape(s_expU, N, 1, R), T(1e-30))

    return expU
end

"""
Log-probability of the CHOSEN alternative only, as an `N × R` matrix.

This is what the simulated likelihood actually consumes, and computing it directly
avoids the `N × J × R` probability tensor entirely: the likelihood never needs the
non-chosen alternatives' probabilities, only their contribution to the normalizer.

    log P(j_n | β_r) = (V_{j_n} - m) - log Σ_{k avail} exp(V_k - m)

`logit_prob` remains for `predict`, `evaluate(::MixedLogitModel, …)` and the
construction-time class checks, which genuinely want the whole tensor.

Not bit-for-bit what `log.(logit_prob(…))` produced — `a - log(s)` and
`log(exp(a)/s)` differ in the last ulp — but it is the same quantity, computed with
one fewer rounding step and without a `log` over every alternative.
"""
function chosen_logprob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    parameters::Dict,
    availability::Vector{<:AbstractVector{Bool}},
    draws::Dict{Symbol, Matrix{Float64}},
    choices::AbstractVector{Int},
)
    J = length(utilities)
    raw = Vector{Any}(undef, J)

    Threads.@threads for j in 1:J
        raw[j] = evaluate(utilities[j], data, parameters, draws)
    end

    # Function barrier, for the same reason as in `logit_prob`.
    return _mxl_chosen_logprob(identity.(raw), availability, choices)
end

function _mxl_chosen_logprob(utils::AbstractVector, availability, choices::AbstractVector{Int})
    J = length(utils)
    N, R = size(utils[1])
    T = eltype(first(utils))

    lp = Array{T}(undef, N, R)
    # What an unavailable-but-chosen alternative contributed under the old tensor
    # path: its probability was exactly 0, and `log(max(0, 1e-30))` floored it here.
    # Preserved deliberately — in log space the natural expression would instead
    # return a finite, wrong number.
    floorlp = log(T(1e-30))

    Threads.@threads for r in 1:R
        @inbounds begin
            # Same stable-softmax max-subtraction as `_mxl_softmax`; see the comment
            # there for why it is not optional.
            m = fill(T(-Inf), N)
            for j in 1:J
                m .= ifelse.(availability[j], max.(m, @view(utils[j][:, r])), m)
            end

            # Normalizer only — an N-vector per draw, not an N × J slab.
            s = zeros(T, N)
            for j in 1:J
                s .+= ifelse.(availability[j], exp.(@view(utils[j][:, r]) .- m), zero(T))
            end

            col = @view lp[:, r]
            for n in 1:N
                jc = choices[n]
                if jc == 0
                    # Defensive only: `_recode_choices` throws on a code no
                    # alternative claims, so `estimate` can never produce a 0 here.
                    # Matches what the old all-false one-hot row contributed.
                    col[n] = zero(T)
                elseif availability[jc][n]
                    # `m` is the max over AVAILABLE alternatives and the chosen one
                    # is available, so `s >= exp(0) = 1` and the log needs no guard.
                    #
                    # The `max(·, floorlp)` reproduces the old tensor path's
                    # `log(max(p, 1e-30))` EXACTLY, and is kept deliberately. It is
                    # not needed for numerical safety — this expression is finite
                    # however small the probability — but dropping it changes the
                    # objective in the tails, where an individual's every draw can
                    # sit below the floor, and that moves the optimizer onto
                    # different local optima. Measured on MXL_swissmetro at its
                    # optimum: 1060 of 3.38M cells bind, and while the resulting
                    # log-likelihood is unchanged to 10 digits (a floored cell is
                    # negligible inside the log-sum-exp over draws either way), the
                    # path through the tails is not. Removing it is a MODEL change,
                    # to be made on its own and validated on its own.
                    col[n] = max((utils[jc][n, r] - m[n]) - log(s[n]), floorlp)
                else
                    col[n] = floorlp
                end
            end
        end
    end

    return lp
end

"""
Predicts choice probabilities using a set of estimated parameters.

# Arguments
- `model::MixedLogitModel`: the model structure
- `results::NamedTuple`: estimation results with field `parameters`

# Returns
- `Array{Float64,3}`: simulated probabilities of shape `(N, J, R)`, i.e. **conditional**
  on each draw of the random coefficients. Average over the third dimension for the
  unconditional probabilities — which is what `evaluate(::MixedLogitModel, …)` returns
"""
function predict(model::MixedLogitModel, results)
    return logit_prob(
        model.utilities,
        model.data,
        results.parameters,
        model.availability,
        model.draws
    )
end

"""
Computes the simulated log-likelihood of a Mixed Logit model.

The contribution of individual `i` is `log (1/R) Σ_r Π_t P(j_t | β_r)`: the product
over that individual's observations happens **inside** the draw average, because the
random coefficients belong to the individual. Computing it the other way round
(averaging draws per observation) is a different and wrong model. The draw average is
taken by log-sum-exp rather than on probabilities — a sequence probability falls off
exponentially in the panel length, and the old `max(·, 1e-30)` failsafe silently
pinned every long-panel individual to the floor; see the comment at the loop.

# Arguments
- `model::MixedLogitModel`: the model object
- `choices::AbstractVector{Int}`: chosen alternative per observation, as a POSITION
  in `model.alternatives` (0 where no alternative claims the observed code). This
  used to be an `(N, J, R)` one-hot `Bool` tensor — R identical copies of one `N × J`
  mask, built only to be multiplied against a full tensor of log-probabilities. Both
  are gone; see `chosen_logprob`.
- `parameters::Dict = model.parameters`: values to evaluate at. `estimate` passes the
  dict its objective closures write into, so this is where the optimizer's current θ
  (and, during differentiation, ForwardDiff `Dual`s) enters

# Returns
- `Vector`: one contribution per **individual**, not per observation, so `G` is
  `I × K` and the resulting robust standard errors are clustered by individual. That
  is intended and structural — see item 3 of the TODO section in CLAUDE.md
"""
function loglikelihood(model::MixedLogitModel, choices::AbstractVector{Int};
                       parameters::Dict = model.parameters)
    # Only the chosen alternative's log-probability is ever needed, so the N × J × R
    # tensor is never built. See `chosen_logprob`.
    log_chosen = chosen_logprob(
        model.utilities,
        model.data,
        parameters,
        model.availability,
        model.draws,
        choices,
    )

    N, R = size(log_chosen)
    id_map, id = model.id
    I = length(id_map)

    T = eltype(log_chosen)

    log_indiv = zeros(T, I, R)
    loglik = zeros(T, I)
    Threads.@threads for r in 1:R
        @inbounds for n in 1:N
            log_indiv[id_map[id[n]], r] += log_chosen[n, r]
        end
    end

    # Average the sequence probability over draws IN LOGS:
    # log (1/R) Σ_r exp(Σ_t log P_t(r)).
    #
    # This used to materialize `exp.(log_indiv)` and average that, which is the
    # same quantity in exact arithmetic and silently wrong in Float64. An
    # individual's sequence probability is a product over their observations, so
    # it falls off exponentially in the panel length: once it drops below the
    # `1e-30` failsafe the contribution is CLAMPED, and every individual pins to
    # log(1e-30) = -69.08 regardless of the parameters. Measured on a 2-individual
    # fixture: correct at 10 observations each, but at 200/600/1200 the old code
    # returned -138.1551 every time (= 2 × the floor) while the true values are
    # -280.56 / -836.19 / -1668.50. The threshold is `T·|log p| > 69`, i.e. very
    # roughly 150-250 observations per individual — well inside real panel data,
    # and far below the ~745 where Float64 itself would underflow.
    #
    # Same failure mode as the softmax-underflow bug recorded in CLAUDE.md: a
    # `max(·, 1e-30)` failsafe converting an underflow into a plausible-looking
    # number instead of an error. The `LatentClassModel` panel path has always
    # done this in logs; this brings the standalone model into line.
    logR = log(T(R))
    Threads.@threads for i in 1:I
        @inbounds begin
            mx = -T(Inf)
            for r in 1:R
                mx = max(mx, log_indiv[i, r])
            end
            s = zero(T)
            for r in 1:R
                s += exp(log_indiv[i, r] - mx)
            end
            loglik[i] = mx + log(s) - logR
        end
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
function null_loglikelihood_mxl(availability::Vector{<:AbstractVector{Bool}})
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
Estimates the parameters of a Mixed Logit model using simulated maximum likelihood.

Uses optimization via `Optim.jl` to minimize the negative simulated log-likelihood.

# Arguments
- `model::MixedLogitModel`: model specification with draws and utility functions
- `choicevar::Symbol`: name of the column in `model.data` that contains observed choices
- `verbose::Bool = true`: whether to print optimizer output
- `hessian_method::Symbol=:ad`: how to compute the Hessian the covariance matrices are
  built from — `:ad` for exact ForwardDiff second derivatives, `:fd` for a finite-difference
  Jacobian of the exact gradient (Apollo's routine). See `model_hessian!`

# Returns
- `NamedTuple` with the same fields as `estimate(::LogitModel, …)`. Note that its
  robust standard errors are **clustered by individual**, since the likelihood
  returns one contribution per individual — see `loglikelihood`.
"""
function estimate(model::MixedLogitModel, choicevar::Symbol; verbose::Bool = true,
                  hessian_method::Symbol = :ad)
    # Validate before optimizing, not after: a typo'd method should not cost a full
    # estimation run before it is reported.
    _check_hessian_method(hessian_method)

    choice_data = model.data[:,choicevar]

    # Translate the analyst's alternative codes into positions in `model.alternatives`
    # once, here; `choices` and the utilities are both ordered by position.
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

    # Preallocate mutable parameter set (no deepcopy of full model)
    # `Dict{Symbol,Any}`, not `deepcopy`: the objective closures below write
    # ForwardDiff `Dual`s into this dict, which a user-supplied `parameters` dict
    # typed as `Dict{Symbol,Float64}` cannot hold (it raised a `MethodError` from
    # `convert`). The value type has to admit Duals regardless of what was passed.
    mutable_parameters = Dict{Symbol,Any}(model.parameters)

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
    cfg = _hessian_config(f_obj, θ0)
    H = model_hessian!(H, f_obj, θ0, cfg, hessian_method)
    
    ForwardDiff.gradient(f_obj, θ0)
    
    scores = zeros(length(f_obj_i(θ0)),length(θ0))
    ForwardDiff.jacobian!(scores,f_obj_i, θ0)

    ll0 = null_loglikelihood_mxl(model.availability)

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
            f_abstol=1e-6,
            g_abstol=1e-8);
            autodiff=:forward
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

    # Hessian
    if verbose
        println("Computing Standard Errors")
    end

    model_hessian!(H, f_obj, θ̂, cfg, hessian_method)

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
Evaluates derived expressions (e.g. WTP, elasticities) from a fitted `MixedLogitModel`,
with delta-method standard errors.

The expression must be a function of the **estimated parameters** — for a random
coefficient that means the parameters of its taste distribution, not the coefficient
itself. WTP at the mean of a normally-distributed time coefficient is `mu_time / β_cost`;
the median under a lognormal spec `-exp(mu + σ·ξ)` is `exp(mu_time) / exp(mu_cost)`.

An expression containing a `Draw` is an error, not an average over the draws: see
`delta_method`, which spells out the three distinct quantities that notation conflates
and why only this one has a delta-method standard error at all.
"""
evaluate(
    expressions::Dict{Symbol, <:DCMExpression},
    model::MixedLogitModel,
    results::NamedTuple
) = delta_method(expressions, collect_parameters(model.utilities), model.data, results)

# The symbolic children the `collect_*` walkers descend into when this model is
# nested inside another expression. See the traversal note in `Utils.jl`.
#
# `collect_draws` reaching in here is what lets a `LatentClassModel` discover the
# draw dimensions its Mixed Logit classes reference, which it needs in order to
# generate ONE shared draw set for all of them.
_children(m::MixedLogitModel) = m.utilities

"""
Cross-sectional `evaluate` for a nested `MixedLogitModel` term: the **unconditional**
choice probabilities, `(1/R) Σ_r P(j | β_r)`, as an `N × J` matrix.

WARNING — this is not the quantity a latent-class likelihood may be built from.
Averaging over draws here collapses the draws before the product over an individual's
observations has been taken, and the whole point of a panel Mixed Logit is that the
product happens *inside* the integral: `log (1/R) Σ_r Π_t P_t(r)`, not
`Σ_t log (1/R) Σ_r P_t(r)`. A Mixed Logit class inside a `LatentClassModel` therefore
goes through `_class_log_sequence`, which calls `chosen_logprob` for the per-draw
log-probabilities and never this method. What this is for is `predict`, and the
construction-time class checks (`_check_class_weights`, `_check_class_separation`),
which legitimately compare unconditional probabilities.
"""
function evaluate(e::MixedLogitModel, data::DataFrame, params::AbstractDict)
    P = logit_prob(e.utilities, data, params, e.availability, e.draws)  # N × J × R
    return dropdims(sum(P, dims=3), dims=3) ./ size(P, 3)
end