"""
Extracts all distinct parameters from a list of utility expressions.

Traverses the expression trees and returns all unique instances of `DCMParameter`.

# Arguments
- `utilities::Vector{<:DCMExpression}`: vector of symbolic utility expressions

# Returns
- `Vector{DCMParameter}`: unique parameters used in the utilities
"""
function collect_parameters(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, DCMParameter}()
    function visit(expr)
        if expr isa DCMParameter
            seen[expr.name] = expr
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        elseif expr isa LogitModel
            for u in expr.utilities
                visit(u)
            end
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(values(seen))
end

function collect_parameters(expr::DCMExpression)
    collect_parameters([expr])
end

"""
Extracts all variable names used in a list of utility expressions.

Traverses the expression trees to find all `DCMVariable` symbols.

# Arguments
- `utilities::Vector{<:DCMExpression}`: vector of symbolic utility expressions

# Returns
- `Vector{Symbol}`: names of variables appearing in the expressions
"""
function collect_variables(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, Bool}()
    function visit(expr)
        if expr isa DCMVariable
            seen[expr.name] = true
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(keys(seen))
end

"""
Recursively collects all unique draw names (`Symbol`) from a symbolic expression.

Used for identifying the random terms in Mixed Logit specifications.

# Arguments
- `expr::DCMExpression`: symbolic utility expression

# Returns
- `Vector{Symbol}`: names of draws used in the expression
"""
function collect_draws(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, Bool}()
    function visit(expr)
        if expr isa DCMDraw
            seen[expr.name] = true
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(keys(seen))
end

# ---------------------------------------------------------------------------
# Alternatives: names, codes, and the observed choice column
#
# Every model stores its alternative set as a NamedTuple mapping a name to the
# code that identifies that alternative in the data's choice column —
# `(car = 1, bus = 4, rail = 7)`. That NamedTuple is the single source of truth
# for (a) which alternatives exist, (b) their canonical order, and (c) how the
# observed choice column maps onto positions in the probability matrix.
#
# `utilities` and `availability` are matched to it BY NAME and stored as plain
# vectors in the NamedTuple's key order, so the likelihood keeps indexing by
# position and nothing in the hot loop ever touches the NamedTuple. Its keys
# live in its *type*, which is why the struct field is abstractly typed; that
# costs nothing as long as it stays construction/reporting metadata. Do not
# read it inside `logit_prob` or `loglikelihood`.
# ---------------------------------------------------------------------------

"""
Normalizes the user's `alternatives` argument into a canonical NamedTuple of
`name => code` pairs.

Two accepted forms:

- `NamedTuple` — `(car = 1, bus = 4, rail = 7)`: names chosen by the analyst,
  codes as they appear in the choice column. `utilities` and `availability`
  must then be NamedTuples over the same names.
- `AbstractVector{<:Integer}` — `[1, 4, 7]`: the codes alone, for alternative
  sets built programmatically where literal NamedTuple syntax is impractical.
  Names are generated as `:alt1, :alt2, …` (Apollo's convention for unnamed
  alternatives), and `utilities`/`availability` are then positional vectors.

# Returns
- `(alternatives::NamedTuple, named::Bool)` — `named` records which form was
  used, so the matching of `utilities`/`availability` can insist on the same one
  (mixing a NamedTuple with a positional vector is exactly the ambiguity this
  API exists to remove).
"""
function _resolve_alternatives(alternatives::NamedTuple)
    isempty(alternatives) && error("`alternatives` is empty: a model needs at least one alternative.")

    labels = collect(keys(alternatives))
    values_ = collect(values(alternatives))

    bad = findall(v -> !(v isa Integer), values_)
    if !isempty(bad)
        error("""
              `alternatives` must map each alternative name to the integer code identifying it \
              in the choice column, but $(join(string.(labels[bad]), ", ")) \
              $(length(bad) == 1 ? "is" : "are") not integer-valued \
              (got $(join(repr.(values_[bad]), ", "))).
              """)
    end

    codes = Int.(values_)
    _check_duplicate_codes(codes, labels)

    return NamedTuple{Tuple(labels)}(Tuple(codes)), true
end

function _resolve_alternatives(alternatives::AbstractVector{<:Integer})
    isempty(alternatives) && error("`alternatives` is empty: a model needs at least one alternative.")

    codes = Int.(alternatives)
    labels = [Symbol("alt", j) for j in 1:length(codes)]
    _check_duplicate_codes(codes, labels)

    return NamedTuple{Tuple(labels)}(Tuple(codes)), false
end

_resolve_alternatives(alternatives) = error("""
    `alternatives` must be a NamedTuple mapping alternative names to the codes used in the \
    choice column — e.g. `alternatives = (car = 1, bus = 4, rail = 7)` — or a vector of those \
    codes for unnamed alternatives. Got a $(typeof(alternatives)).
    """)

function _check_duplicate_codes(codes::Vector{Int}, labels::Vector{Symbol})
    for j in 1:length(codes), k in (j + 1):length(codes)
        if codes[j] == codes[k]
            error("""
                  Alternatives $(labels[j]) and $(labels[k]) share the code $(codes[j]) in \
                  `alternatives`. Each alternative needs its own code, since the code is what \
                  identifies it in the choice column.
                  """)
        end
    end
    return nothing
end

# "car => 1, bus => 4, rail => 7", for error messages.
_alternatives_str(alternatives::NamedTuple) =
    join(("$k => $v" for (k, v) in pairs(alternatives)), ", ")

"""
Matches a `utilities` or `availability` argument against the model's
`alternatives` and returns it as a plain `Vector` in the alternatives' order.

With named alternatives the argument must be a NamedTuple over exactly the same
names; its own order is irrelevant and is discarded here. With unnamed
alternatives it must be a vector of the right length. Anything else throws,
naming the alternatives responsible.

# Arguments
- `alternatives::NamedTuple`: canonical alternative set (from `_resolve_alternatives`)
- `named::Bool`: whether the user supplied named alternatives
- `x`: the `utilities` or `availability` argument
- `what::AbstractString`: which one, used in the messages
"""
function _match_alternatives(alternatives::NamedTuple, named::Bool, x::NamedTuple, what::AbstractString)
    if !named
        error("""
              `$what` was given as a NamedTuple, but `alternatives` was given as a plain vector \
              of codes, so the alternatives have no names to match it against. Either name the \
              alternatives — `alternatives = (car = 1, bus = 4, …)` — or pass `$what` as a \
              vector in the same order as `alternatives`.
              """)
    end

    want = collect(keys(alternatives))
    got = collect(keys(x))

    missing_ = setdiff(want, got)
    extra = setdiff(got, want)
    if !isempty(missing_) || !isempty(extra)
        problems = String[]
        isempty(missing_) || push!(problems, "`$what` is missing $(join(string.(missing_), ", "))")
        isempty(extra) || push!(problems,
            "`$what` has $(join(string.(extra), ", ")), which $(length(extra) == 1 ? "is" : "are") not in `alternatives`")
        error("""
              `$what` does not cover the same alternatives as `alternatives`: $(join(problems, "; ")).
              Declared alternatives: $(_alternatives_str(alternatives)).
              """)
    end

    return [x[k] for k in want]
end

function _match_alternatives(alternatives::NamedTuple, named::Bool, x::AbstractVector, what::AbstractString)
    if named
        error("""
              `alternatives` names its alternatives ($(join(string.(keys(alternatives)), ", "))), \
              so `$what` must be a NamedTuple over the same names — e.g. \
              `$what = ($(first(keys(alternatives))) = …, …)` — rather than a positional vector. \
              Matching by name is what makes the order irrelevant.
              """)
    end

    if length(x) != length(alternatives)
        error("""
              `$what` has $(length(x)) entries but there are $(length(alternatives)) alternatives \
              ($(_alternatives_str(alternatives))). With unnamed alternatives the two are matched \
              by position, so they must have the same length.
              """)
    end

    return collect(x)
end

_match_alternatives(alternatives::NamedTuple, named::Bool, x, what::AbstractString) = error("""
    `$what` must be $(named ? "a NamedTuple over the alternative names ($(join(string.(keys(alternatives)), ", ")))" :
                              "a vector with one entry per alternative"), \
    but got a $(typeof(x)).
    """)

"""
Checks that every entry matched out of `utilities` is a symbolic expression, so a
mistyped entry is reported by name rather than as a `MethodError` from the struct
constructor. Preserves a concrete element type when there is one — `logit_prob`
specializes on it.
"""
function _check_utilities(utils::AbstractVector)
    bad = findall(u -> !(u isa DCMExpression), utils)
    if !isempty(bad)
        error("""
              `utilities` entries $(join(string.(bad), ", ")) are not symbolic expressions \
              (got $(join(string.(typeof.(utils[bad])), ", "))). Build each utility from \
              `Parameter`, `Variable` and `Draw` terms.
              """)
    end
    return eltype(utils) <: DCMExpression ? utils : convert(Vector{DCMExpression}, utils)
end

"""
Checks the availability vectors: boolean, and one entry per row of `data`.
"""
function _check_availability(avail::AbstractVector, data::DataFrame)
    bad = findall(a -> !(a isa AbstractVector{Bool}), avail)
    if !isempty(bad)
        error("""
              `availability` entries $(join(string.(bad), ", ")) are not boolean vectors \
              (got $(join(string.(typeof.(avail[bad])), ", "))). Each entry must be a vector of \
              `Bool` with one element per observation — e.g. `df.av_car .== 1`.
              """)
    end

    N = nrow(data)
    wrong = findall(a -> length(a) != N, avail)
    if !isempty(wrong)
        error("""
              `availability` entries $(join(string.(wrong), ", ")) have \
              $(join(string.(length.(avail[wrong])), ", ")) elements, but `data` has $N rows. \
              Availability is recorded per observation.
              """)
    end

    return eltype(avail) <: AbstractVector{Bool} ? avail : convert(Vector{AbstractVector{Bool}}, avail)
end

"""
Recodes an observed choice column into **positions** in the model's alternative
ordering.

The utilities, the availability vectors and every probability matrix are ordered
by `alternatives`, and the likelihood indexes them directly (`probs[n, j]`), so
the choice column has to be translated from the analyst's codes to positions
exactly once — here, at the top of `estimate` — rather than being assumed to
already be `1:J` in the right order.

Errors on a missing value, and on any code no alternative claims (which would
otherwise be either a `BoundsError` deep in the likelihood or, worse, a silently
wrong model in which the code maps onto some other alternative).

# Arguments
- `choice_data`: the raw choice column
- `alternatives::NamedTuple`: the model's alternatives
- `choicevar::Symbol`: its column name, used in the messages

# Returns
- `Vector{Int}`: position in `1:J` of the alternative chosen in each row
"""
function _recode_choices(choice_data, alternatives::NamedTuple, choicevar::Symbol)
    if any(ismissing, choice_data)
        error("Choice vector contains missing values. Please clean your data.")
    end

    codes = collect(Int, values(alternatives))
    lookup = Dict(c => j for (j, c) in enumerate(codes))

    raw = try
        Int.(choice_data)
    catch
        error("""
              Choice column `$choicevar` is not integer-valued, so it cannot be matched against \
              the alternative codes ($(_alternatives_str(alternatives))).
              """)
    end

    positions = Vector{Int}(undef, length(raw))
    @inbounds for n in 1:length(raw)
        j = get(lookup, raw[n], 0)
        if j == 0
            error("""
                  Choice column `$choicevar` contains the value $(raw[n]) (row $n), which is not \
                  the code of any alternative. Declared alternatives: \
                  $(_alternatives_str(alternatives)).
                  Either add the missing alternative to `alternatives`, or correct the data.
                  """)
        end
        positions[n] = j
    end

    return positions
end

# ---------------------------------------------------------------------------
# Shared standard-error machinery
#
# All three `estimate` functions used to inline the same guarded `sqrt.(diag(V))`
# logic (6 near-identical sites), which is exactly the kind of duplication that
# lets a fix land in one model and not the other two. They now share these.
# ---------------------------------------------------------------------------

# `inv` throws on an exactly singular matrix; the pseudo-inverse at least gives
# something the diagonal checks below can reject.
_safe_inv(A::AbstractMatrix) = try inv(A) catch; pinv(A) end

# Create the directory an export path points at, so `file="output/x.xlsx"` works
# without the caller having to mkdir first. No-op for a bare filename.
function _ensure_dir(file::AbstractString)
    dir = dirname(file)
    isempty(dir) || mkpath(dir)
    return nothing
end

"""
Turns a variance-covariance matrix into standard errors, reporting `NaN` for any
negative diagonal entry rather than throwing a `DomainError` out of `sqrt`.

A negative variance means the matrix is not positive definite. That is worth
investigating (it usually points at the likelihood or the optimizer's stopping
point, not necessarily at identification) but it should not abort a run that has
already converged.

# Arguments
- `V::AbstractMatrix`: variance-covariance matrix
- `free_names::Vector{Symbol}`: free parameter names, in the same order as `V`
- `what::AbstractString`: label used in the warning (e.g. `"robust covariance"`)

# Returns
- `Vector{Float64}`: standard errors, `NaN` where the variance was negative
"""
function guarded_std_errors(V::AbstractMatrix, free_names::AbstractVector{Symbol}, what::AbstractString)
    d = diag(V)
    bad = free_names[findall(<(0), d)]
    if !isempty(bad)
        @warn "Non-positive-definite $what: negative variance for $(bad); reporting NaN standard error(s)."
    end
    return [dᵢ < 0 ? NaN : sqrt(dᵢ) for dᵢ in d]
end

"""
Classifies the Hessian at the optimum as `:posdef`, `:indefinite` or `:singular`,
and warns (naming the parameters responsible) when it is not positive definite.

**This is tested on the eigenvalues of `H` itself, not on the diagonal of
`inv(H)`.** The diagonal test only catches an *indefinite* Hessian; a merely
**singular** one — the signature of a parameter that is not identified — inverts
through `pinv` to a matrix with a perfectly non-negative diagonal, so it yields
confident-looking finite standard errors and no complaint at all. Two free ASCs
in a 2-alternative logit reproduce this: only their difference is identified, yet
each is reported with a tidy SE roughly half that of the identified single-ASC
version. Whatever else changes here, keep the check on `H`.

The two failure modes mean different things and are reported separately:

- `:indefinite` — `H` has a genuinely **negative** eigenvalue, so the optimizer
  stopped somewhere that is not a local maximum (a saddle, or a numerically
  broken likelihood — this is what the softmax-underflow bug produced). The
  parameters loading on that eigenvector define the direction of wrong curvature.
- `:singular` — `H` has a **zero** eigenvalue (to working precision), so the
  likelihood is flat in some direction: that combination of parameters is not
  identified by the data. The eigenvector names the combination.

# Arguments
- `H::AbstractMatrix`: Hessian of the **negative** log-likelihood at the optimum
- `free_names::Vector{Symbol}`: free parameter names, in the same order as `H`

# Returns
- `Symbol`: `:posdef`, `:indefinite`, or `:singular`
"""
function hessian_status(H::AbstractMatrix, free_names::AbstractVector{Symbol}; rtol::Real = 1e-10)
    k = size(H, 1)
    k == 0 && return :posdef

    # ForwardDiff Hessians come back with roundoff-level asymmetry; symmetrize so
    # `eigen` returns real eigenvalues and the classification is stable.
    M = Matrix(H)
    F = eigen(Symmetric((M .+ M') ./ 2))
    λ = F.values
    λmax = maximum(abs, λ)

    # An eigenvalue counts as zero when it is negligible RELATIVE to the largest.
    #
    # The textbook rank tolerance `k * eps * λmax` is far too tight here and was
    # measured to miss real cases: a Hessian accumulated by AD over N observations
    # carries much more roundoff than that idealized bound. On the two-free-ASC
    # example below (true zero eigenvalue, λmax ≈ 160) `eigen` returns 2.6e-13
    # while `eigvals` on the same matrix returns 2.1e-16 — both are numerically
    # zero, but they straddle a `k*eps*λmax` tolerance of 1.1e-13, so the check
    # silently passed. `rtol = 1e-10` sits above that noise floor while still
    # leaving eight orders of magnitude before a merely ill-conditioned (but
    # identified) model would trip it.
    tol = rtol * max(λmax, one(λmax))

    negative = findall(<(-tol), λ)
    flat     = findall(x -> abs(x) <= tol, λ)

    isempty(negative) && isempty(flat) && return :posdef

    # Parameters carrying the offending direction(s): those with a non-trivial
    # loading on the corresponding eigenvector.
    culprits(idxs) = unique(reduce(vcat, [
        free_names[abs.(F.vectors[:, i]) .> 0.1 * maximum(abs, F.vectors[:, i])]
        for i in idxs
    ]; init = Symbol[]))

    if !isempty(negative)
        @warn """
              The Hessian is NOT positive definite at the reported optimum: $(length(negative)) \
              of $(k) eigenvalue(s) are negative (smallest $(minimum(λ))).
              Parameters involved: $(culprits(negative)).
              This means the optimizer stopped at a point that is not a local maximum, so the \
              standard errors below are not trustworthy — the log-likelihood itself may be fine, \
              but check the convergence trace, the starting values, and the likelihood for \
              numerical problems before reporting these estimates.
              """
        return :indefinite
    end

    @warn """
          The Hessian is singular at the reported optimum: $(length(flat)) of $(k) eigenvalue(s) \
          are zero to working precision.
          Parameters involved: $(culprits(flat)).
          The likelihood is flat in that direction, i.e. that combination of parameters is NOT \
          identified by the data — a classic cause is two free alternative-specific constants \
          where only their difference is identified. Standard errors for those parameters are \
          computed from a pseudo-inverse and are meaningless; fix one of the parameters (or drop \
          it) and re-estimate.
          """
    return :singular
end

"""
Classifies the BHHH matrix `G = Σᵢ sᵢsᵢ'` as `:posdef` or `:singular`, warning
(and naming the parameters responsible) when it is rank-deficient.

This is the `G`-side counterpart of `hessian_status`, and it exists for exactly
the same reason: **a degenerate `G` cannot announce itself through the standard
error values.** `G` is positive semi-definite by construction, so `_safe_inv`
pseudo-inverts it to something with a perfectly non-negative diagonal, and
`guarded_std_errors` — which only rejects *negative* variances — stays silent.
The result is a standard error of exactly `0.0` for the unidentified direction,
i.e. `t = Inf` and `p = 0.0000`, reported as if it were the most precisely
estimated parameter in the model.

`G` singular is not a cosmetic problem, and it is **not** implied by a healthy
Hessian: measured on a fixture with a positive-definite `H` and `rank(G) = 1` of
2, `hessian_status` returned `:posdef` while the robust column reported
`0.0` — the sandwich `inv(H) G inv(H)` is built from `G`, so it degenerates even
though the arithmetic never fails. That is why this check is separate and why a
singular `G` suppresses *all three* estimators rather than just BHHH: robust is
built from `G` directly, and classical would be the lone survivor of a covariance
matrix we have just established cannot be computed.

Only the zero-eigenvalue case is classified. A genuinely negative eigenvalue is
impossible for an outer product of scores; anything below zero here is roundoff
and is treated as flat.

# Arguments
- `G::AbstractMatrix`: outer product of the per-observation score vectors
- `free_names::Vector{Symbol}`: free parameter names, in the same order as `G`

# Returns
- `Symbol`: `:posdef` or `:singular`
"""
function bhhh_matrix_status(G::AbstractMatrix, free_names::AbstractVector{Symbol}; rtol::Real = 1e-10)
    k = size(G, 1)
    k == 0 && return :posdef

    M = Matrix(G)
    F = eigen(Symmetric((M .+ M') ./ 2))
    λ = F.values
    λmax = maximum(abs, λ)

    # Same empirical tolerance as `hessian_status` — see the note there on why the
    # textbook `k*eps*λmax` bound is too tight for an AD-accumulated matrix.
    tol = rtol * max(λmax, one(λmax))
    flat = findall(x -> x <= tol, λ)

    isempty(flat) && return :posdef

    culprits = unique(reduce(vcat, [
        free_names[abs.(F.vectors[:, i]) .> 0.1 * maximum(abs, F.vectors[:, i])]
        for i in flat
    ]; init = Symbol[]))

    @warn """
          The BHHH matrix (the outer product of the scores) is singular at the reported \
          optimum: $(length(flat)) of $(k) eigenvalue(s) are zero to working precision.
          Parameters involved: $(culprits).
          That combination of parameters contributes no score variation in the sample, so it \
          is not identified. NO covariance matrix has been computed — classical, robust and \
          BHHH standard errors are all suppressed, because every one of them is built from \
          this matrix or would be the sole survivor of one that cannot be formed. Fix or drop \
          the parameter(s) above and re-estimate.
          """
    return :singular
end

"""
Computes all three maximum-likelihood covariance estimators at the optimum, plus
the `hessian_status` verdict.

The three are the standard trio, each a different estimator of the *same*
asymptotic covariance:

- **classical** `inv(H)` — inverse observed information.
- **robust / sandwich** `inv(H) G inv(H)` — White; valid under misspecification.
- **BHHH / OPG** `inv(G)` where `G = Σᵢ sᵢsᵢ'` — positive semi-definite **by
  construction**, so it survives a Hessian that isn't.

They are computed and reported **side by side, never substituted for one
another**. An earlier version swapped BHHH into the classical slot when `H` was
not positive definite and tagged the result with a `vcov_method` field; that was
a mistake. The package's whole reporting contract is that "Classic" and "Robust"
name two specific, comparable estimators — a column whose meaning silently
changes between runs breaks exactly the comparison it exists to support. Note
also that when `H` is bad the *robust* column is equally bad, since it is built
from the same `inv(H)`; a fallback in one column would have implied the other was
fine. `hessian_status` is what tells you whether to trust them.

**BHHH is computed and returned, but never presented.** `summarize_results` shows
only the classical and robust columns, following Apollo. BHHH's justification is
the information matrix equality `H = G`, which holds only at the true parameter
under correct specification — precisely the assumption in doubt whenever `H` is
not positive definite, which is the one situation an earlier version of this code
chose to display it in. Being PSD by construction makes BHHH *look* well-formed;
it does not make it a consistent estimator of a covariance whose defining
assumption has just failed. It stays in the returned `NamedTuple` for anyone who
explicitly wants it, and out of the report.

**A singular `G` suppresses everything.** When `bhhh_matrix_status` returns
`:singular` no covariance matrix is computed at all — every matrix and standard
error field comes back `nothing` (following Apollo: "if the BHHH matrix is
singular, no attempt will be made to calculate the full covariance matrix"). See
`bhhh_matrix_status` for why a degenerate `G` cannot be detected from the
standard error values themselves.

# Arguments
- `H::AbstractMatrix`: Hessian of the **negative** log-likelihood at the optimum
- `G::AbstractMatrix`: outer product of the per-observation score vectors
- `free_names::Vector{Symbol}`: free parameter names, in the same order as `H`

# Returns
- `NamedTuple` with `status` (the `hessian_status` verdict), `bhhh_matrix` (the
  `bhhh_matrix_status` verdict), and `vcov`/`std_errors`, `rob_vcov`/`rob_std_errors`,
  `bhhh_vcov`/`bhhh_std_errors`. The `*_std_errors` entries are `Dict`s keyed by
  parameter name, or `nothing` (as are the matrices) when `G` is singular.
"""
function covariance_estimates(H::AbstractMatrix, G::AbstractMatrix, free_names::AbstractVector{Symbol})
    status = hessian_status(H, free_names)
    bhhh_matrix = bhhh_matrix_status(G, free_names)

    # Apollo's rule: a singular BHHH matrix means no covariance matrix is
    # attempted at all. Not a partial result — the robust estimator is built from
    # `G` directly and would report 0.0 for the unidentified direction, and
    # reporting the classical column alone would present the one estimator that
    # happens to survive as if the covariance matrix were fine.
    if bhhh_matrix === :singular
        return (
            status          = status,
            bhhh_matrix     = bhhh_matrix,
            vcov            = nothing,
            std_errors      = nothing,
            rob_vcov        = nothing,
            rob_std_errors  = nothing,
            bhhh_vcov       = nothing,
            bhhh_std_errors = nothing,
        )
    end

    H_inv = _safe_inv(H)

    # Each estimator keeps its own identity. In particular the classical column is
    # ALWAYS inv(H) — the BHHH matrix is never substituted into it, because the
    # whole point of printing classical next to robust is that the reader knows
    # what each one is and can compare them.
    vcov      = H_inv
    rob_vcov  = H_inv * G * H_inv
    bhhh_vcov = _safe_inv(G)

    as_dict(v) = Dict{Symbol, Real}(name => v[i] for (i, name) in enumerate(free_names))

    return (
        status          = status,
        bhhh_matrix     = bhhh_matrix,
        vcov            = vcov,
        std_errors      = as_dict(guarded_std_errors(vcov,      free_names, "Hessian-based covariance")),
        rob_vcov        = rob_vcov,
        rob_std_errors  = as_dict(guarded_std_errors(rob_vcov,  free_names, "robust covariance")),
        bhhh_vcov       = bhhh_vcov,
        bhhh_std_errors = as_dict(guarded_std_errors(bhhh_vcov, free_names, "BHHH/OPG covariance")),
    )
end

"""
One-line human-readable rendering of a `hessian_status` verdict, for the model
summary block.

Kept to ≤10 characters so it right-aligns in the same column as the numeric
values in the Model Summary block; the diagnosis (which parameters, and why) is
in the `@warn` that `hessian_status` already emitted.
"""
_hessian_label(status::Symbol) =
    status === :posdef     ? "pos. def." :
    status === :indefinite ? "INDEFINITE" :
    status === :singular   ? "SINGULAR" :
                             string(status)

"""
Pretty-prints estimation results and optionally writes them to an Excel file.

Displays parameter estimates with classical and robust (sandwich) standard errors,
t-statistics and p-values. Optionally writes results to an Excel file (.xlsx) with
two sheets:
- "Estimates": full table of parameter results
- "Summary"  : log-likelihood, iterations, convergence status, and runtime.

# Arguments
- `results::NamedTuple`: Named tuple returned from model estimation, containing fields:
    - `parameters`: Dict of estimated parameter values
    - `std_errors`: Dict of classical standard errors
    - `rob_std_errors`: Dict of robust standard errors (optional)
    - `loglikelihood`: log-likelihood value
    - `iters`: number of iterations
    - `converged`: convergence flag
    - `estimation_time`: runtime in seconds
- `file::Union{String, Nothing}`: Optional path to export results as Excel file.

# Returns
- Nothing. Prints output to console and optionally writes Excel file.
"""
function summarize_results(results::NamedTuple; file::Union{String, Nothing}=nothing)
    params = results.parameters
    se_dict = results.std_errors
    se_robust = results.rob_std_errors
    ll = results.loglikelihood
    iters = results.iters
    converged = results.converged
    estimation_time = results.estimation_time

    hessian = get(results, :hessian, nothing)

    # A singular Hessian means a parameter (or combination) is not identified, so
    # the classical standard errors do not exist — `_safe_inv` only manufactures
    # plausible-looking numbers for them out of a pseudo-inverse. Following
    # Apollo, that is an estimation error and results are not summarized at all.
    # `estimate` still returns normally, so the caller keeps the converged
    # estimates and the verdict for debugging; what is refused is presenting them
    # as a result table with standard errors attached.
    if hessian === :singular
        error("""
              Refusing to summarize: the Hessian is singular at the reported optimum, so no \
              classical covariance matrix exists and the standard errors cannot be computed. \
              See the warning issued during estimation for the parameter(s) whose combination \
              is not identified by the data — fix one of them (or drop it) and re-estimate.
              The returned results object still holds the estimates and `hessian == :singular` \
              if you need to inspect the fit that produced this.
              """)
    end

    # `covariance_estimates` returns `nothing` for every covariance field when the
    # BHHH matrix is singular — no covariance matrix was attempted. The estimates
    # are still worth showing; the standard error columns simply do not exist.
    has_cov = !isnothing(se_dict)

    println("Estimation Results\n==================\n")

    # ---- one estimates table, used for BOTH the console blocks and the export --
    # Building the console output and the Excel sheet from the same `df` is what
    # keeps them from drifting apart; they used to be assembled independently.
    _p(t) = 2 * (1 - cdf(Normal(), abs(t)))

    rows = sort(collect(params); by=first)
    pname = String[]; est = Float64[]
    sc = Float64[]; tc = Float64[]; pc = Float64[]
    sr = Float64[]; tr = Float64[]; pr = Float64[]

    for (name, value) in rows
        push!(pname, string(name)); push!(est, value)
        has_cov || continue

        s = get(se_dict, name, NaN);   push!(sc, s); push!(tc, value / s); push!(pc, _p(value / s))
        s = get(se_robust, name, NaN); push!(sr, s); push!(tr, value / s); push!(pr, _p(value / s))
    end

    df = has_cov ?
        DataFrame(Parameter=pname, Estimate=est, StdError=sc, tStat=tc, PValue=pc) :
        DataFrame(Parameter=pname, Estimate=est)
    if has_cov
        df.RobustSE = sr; df.Robust_tStat = tr; df.Robust_PValue = pr
    end

    function print_block(title, se_col, t_col, p_col, se_header)
        println(title)
        println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", se_header, "t-Stat", "P-value"))
        println(repeat("-", 70))
        for row in eachrow(df)
            println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
                row.Parameter, row.Estimate, row[se_col], row[t_col], row[p_col]))
        end
    end

    if has_cov
        # These two headings always mean inv(H) and inv(H) G inv(H) respectively —
        # no estimator is ever substituted underneath either (see
        # `covariance_estimates`). BHHH is computed and returned but deliberately
        # never presented here.
        print_block("Classic Standard Errors", :StdError, :tStat, :PValue, "Std. Error")
        println()
        print_block("Robust Standard Errors (Sandwich)", :RobustSE, :Robust_tStat, :Robust_PValue, "Robust SE")
    else
        println("Parameter Estimates (no standard errors available)")
        println(@sprintf("%-20s %10s", "Parameter", "Estimate"))
        println(repeat("-", 32))
        for row in eachrow(df)
            println(@sprintf("%-20s %10.4f", row.Parameter, row.Estimate))
        end
        println("\n(The BHHH matrix is singular, so no covariance matrix was computed and no")
        println(" standard errors — classical, robust or BHHH — are available. See the warning")
        println(" issued during estimation for the parameter(s) involved.)")
    end

    # ---- one summary table, likewise shared -----------------------------------
    # `free_parameters` is carried by the results object; fall back to counting the
    # standard error dict for results produced before that field existed (it is
    # keyed by the free names). The fallback is unavailable when no covariance was
    # computed, which is exactly why the field was added.
    free_params = get(results, :free_parameters, has_cov ? length(se_dict) : 0)
    N = results.N
    ll0 = results.null_loglikelihood

    aic = -2 * ll + 2 * free_params
    bic = -2 * ll + log(N) * free_params
    rho2 = isfinite(ll0) ? 1 - ll / ll0 : NaN

    # (label, raw value for Excel, pre-formatted 10-wide text for the console).
    # Pre-formatting here rather than at the print site is what lets one list feed
    # both outputs — `@sprintf` needs a literal format string, so the alternative
    # would be duplicating the row list.
    summary = Tuple{String,Any,String}[
        ("Log-likelihood at optimum", ll,    @sprintf("%10.4f", ll)),
        ("Null Log-likelihood",       ll0,   @sprintf("%10.4f", ll0)),
        ("Iterations",                iters, @sprintf("%10d",   iters)),
        ("Converged",                 converged, @sprintf("%10s", converged)),
    ]
    # Sits next to `Converged` on purpose: "converged" alone is not enough to
    # trust the standard errors, and the `@warn` scrolls past above a long trace.
    if !isnothing(hessian)
        push!(summary, ("Hessian at optimum", _hessian_label(hessian),
                        @sprintf("%10s", _hessian_label(hessian))))
    end
    # Same rationale as the Hessian row: the reason there are no standard errors
    # belongs next to `Converged`, not only in a warning that scrolled past.
    if !has_cov
        push!(summary, ("BHHH matrix", "SINGULAR", @sprintf("%10s", "SINGULAR")))
        push!(summary, ("Covariance matrix", "not computed", @sprintf("%10s", "not comp.")))
    end
    append!(summary, Tuple{String,Any,String}[
        ("Estimation time (seconds)", estimation_time, @sprintf("%10.2f", estimation_time)),
        ("Number of free parameters", free_params,     @sprintf("%10d",   free_params)),
        ("Number of observations",    N,               @sprintf("%10d",   N)),
        ("AIC",                       aic,             @sprintf("%10.2f", aic)),
        ("BIC",                       bic,             @sprintf("%10.2f", bic)),
        ("Rho-squared (McFadden)",    rho2,            @sprintf("%10.4f", rho2)),
    ])

    println("\nModel Summary")
    for (label, _, text) in summary
        println(@sprintf("%-27s: %s", label, text))
    end

    if has_cov && !isnothing(hessian) && hessian !== :posdef
        println("\nNOTE: the Hessian is not positive definite, so the standard errors above are")
        println("      unreliable. See the warning issued during estimation for the parameters")
        println("      involved.")
    end

    # Exporta a Excel si se especifica
    if !isnothing(file)
        _ensure_dir(file)
        XLSX.openxlsx(file, mode="w") do xf
            # Hoja de resultados
            sheet = xf[1]
            XLSX.rename!(sheet,"Estimates")
            sheet1 = xf["Estimates"]

            # Escribir DataFrame celda por celda
            for (j, name) in enumerate(names(df))
                sheet1[1, j] = String(name)
            end
            for (i, row) in enumerate(eachrow(df))
                for (j, name) in enumerate(names(df))
                    sheet1[i+1, j] = row[name]
                end
            end

            # Hoja resumen — same rows, same labels, same order as the console.
            XLSX.addsheet!(xf, "Summary")
            sheet2 = xf["Summary"]
            for (i, (label, value, _)) in enumerate(summary)
                sheet2[i, 1:2] = [label, value]
            end
        end
    end
end

"""
Pretty-prints results from evaluating derived expressions (e.g., WTP, elasticities),
and optionally writes them to an Excel file.

Displays values and their classical and robust standard errors, t-statistics and p-values.

# Arguments
- `results::Dict{Symbol,<:NamedTuple}`: dictionary with results per expression, where each value has fields `value`, `std_error`, and `robust_std_error`
- `file::Union{String, Nothing}`: Optional path to export results as Excel file.

# Returns
- Nothing. Prints output to console and optionally writes Excel file.
"""
function summarize_expressions(results::Dict{Symbol,<:NamedTuple}; file::Union{String, Nothing}=nothing)
    println("Expression Evaluation\n======================\n")

    df = DataFrame(Expression=String[], Value=Float64[],
                   StdError=Float64[], tStat=Float64[], PValue=Float64[],
                   RobustSE=Float64[], Robust_tStat=Float64[], Robust_PValue=Float64[])

    for (name, output) in sort(collect(results); by=first)
        value = output.value

        # Clásico
        se_c = output.std_error
        t_c  = value / se_c
        p_c  = 2 * (1 - cdf(Normal(), abs(t_c)))

        # Robusto
        se_r = output.robust_std_error
        t_r  = value / se_r
        p_r  = 2 * (1 - cdf(Normal(), abs(t_r)))

        push!(df, (string(name), value, se_c, t_c, p_c, se_r, t_r, p_r))
    end

    # Imprime resultados clásicos
    println("Classic Standard Errors")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Expression, row.Value, row.StdError, row.tStat, row.PValue))
    end

    # Imprime resultados robustos
    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Expression, row.Value, row.RobustSE, row.Robust_tStat, row.Robust_PValue))
    end

    # Exportar a Excel si se especifica
    if !isnothing(file)
        _ensure_dir(file)
        XLSX.openxlsx(file, mode="w") do xf
            sheet = xf[1]
            XLSX.rename!(sheet,"Expressions")
            sheet = xf["Expressions"]

            # Escribir encabezados
            for (j, name) in enumerate(names(df))
                sheet[1, j] = String(name)
            end

            # Escribir filas
            for (i, row) in enumerate(eachrow(df))
                for (j, name) in enumerate(names(df))
                    sheet[i+1, j] = row[name]
                end
            end
        end
    end
end

# """
# Pretty-prints results from evaluating derived expressions (e.g., WTP, elasticities).

# Used to display values and their standard errors (e.g., computed via Delta method).

# # Arguments
# - `results::Dict{Symbol,<:NamedTuple}`: dictionary with results per expression, where each value has fields `value` and `std_error`

# # Returns
# - Nothing. Prints output to console.
# """
# function summarize_expressions(results::Dict{Symbol,<:NamedTuple})
#     println("Expression Evaluation\n======================\n")

#     println("Classic Standard Errors")
#     println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Std. Error", "t-Stat", "P-value"))
#     println(repeat("-", 70))

#     for (name, output) in sort(collect(results); by=first)
#         value = output.value
#         se = output.std_error
#         t = value / se
#         p = 2 * (1 - cdf(Normal(), abs(t)))
#         println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
#     end

#     println("\nRobust Standard Errors (Sandwich)")
#     println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Robust SE", "t-Stat", "P-value"))
#     println(repeat("-", 70))

#     for (name, output) in sort(collect(results); by=first)
#         value = output.value
#         se = output.robust_std_error
#         t = value / se
#         p = 2 * (1 - cdf(Normal(), abs(t)))
#         println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
#     end
# end



export summarize_results, summarize_expressions