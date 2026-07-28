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
# Shared standard-error machinery
#
# All three `estimate` functions used to inline the same guarded `sqrt.(diag(V))`
# logic (6 near-identical sites), which is exactly the kind of duplication that
# lets a fix land in one model and not the other two. They now share these.
# ---------------------------------------------------------------------------

# `inv` throws on an exactly singular matrix; the pseudo-inverse at least gives
# something the diagonal checks below can reject.
_safe_inv(A::AbstractMatrix) = try inv(A) catch; pinv(A) end

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
fine. `hessian_status` is what tells you whether to trust them, and BHHH is
offered as its own clearly-labelled third set.

# Arguments
- `H::AbstractMatrix`: Hessian of the **negative** log-likelihood at the optimum
- `G::AbstractMatrix`: outer product of the per-observation score vectors
- `free_names::Vector{Symbol}`: free parameter names, in the same order as `H`

# Returns
- `NamedTuple` with `status`, and `vcov`/`std_errors`, `rob_vcov`/`rob_std_errors`,
  `bhhh_vcov`/`bhhh_std_errors` (the `*_std_errors` entries are `Dict`s keyed by
  parameter name)
"""
function covariance_estimates(H::AbstractMatrix, G::AbstractMatrix, free_names::AbstractVector{Symbol})
    status = hessian_status(H, free_names)

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
"""
_hessian_label(status::Symbol) =
    status === :posdef     ? "positive definite" :
    status === :indefinite ? "NOT POS. DEFINITE (not a maximum)" :
    status === :singular   ? "SINGULAR (parameters unidentified)" :
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

    println("Estimation Results\n==================\n")

    # Prepara DataFrame con resultados
    df = DataFrame(Parameter=String[], Estimate=Float64[],
                   StdError=Float64[], tStat=Float64[], PValue=Float64[],
                   RobustSE=Float64[], Robust_tStat=Float64[], Robust_PValue=Float64[])

    for (name, value) in sort(collect(params); by=first)
        # Clásicos
        se_c = get(se_dict, name, NaN)
        t_c  = value / se_c
        p_c  = 2 * (1 - cdf(Normal(), abs(t_c)))

        # Robustos
        se_r = get(se_robust, name, NaN)
        t_r  = value / se_r
        p_r  = 2 * (1 - cdf(Normal(), abs(t_r)))

        push!(df, (string(name), value, se_c, t_c, p_c, se_r, t_r, p_r))
    end

    # Imprime resultados clásicos. This heading always means inv(H) — no estimator
    # is ever substituted underneath it (see `covariance_estimates`).
    println("Classic Standard Errors")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Parameter, row.Estimate, row.StdError, row.tStat, row.PValue))
    end

    # BHHH/OPG is printed only when the Hessian failed — and as its OWN block, not
    # in place of the classical column. When `H` is indefinite both blocks above
    # are built from `inv(H)` and are unusable, while `G = Σᵢ sᵢsᵢ'` stays PSD by
    # construction, so this is the only one of the three still worth reading.
    # Deliberately NOT printed for a `:singular` Hessian: `G` is rank-deficient in
    # the same direction there, so BHHH is degenerate too and showing it would
    # suggest a way out that does not exist.
    if get(results, :hessian, nothing) === :indefinite && haskey(results, :bhhh_std_errors)
        se_bhhh = results.bhhh_std_errors
        println("\nBHHH/OPG Standard Errors  <-- the Hessian is not positive definite, so the")
        println("                              classical and robust blocks (both built from")
        println("                              inv(H)) are unreliable; these are not.")
        println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "BHHH SE", "t-Stat", "P-value"))
        println(repeat("-", 70))
        for (name, value) in sort(collect(params); by=first)
            se_b = get(se_bhhh, name, NaN)
            t_b  = value / se_b
            p_b  = 2 * (1 - cdf(Normal(), abs(t_b)))
            println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se_b, t_b, p_b))
        end
    end

    # Imprime resultados robustos
    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Parameter, row.Estimate, row.RobustSE, row.Robust_tStat, row.Robust_PValue))
    end

    # Parámetros adicionales
    free_params = length(se_dict)
    N = results.N               # <- asegúrate de incluir esto en el NamedTuple
    ll0 = results.null_loglikelihood

    aic = -2 * ll + 2 * free_params
    bic = -2 * ll + log(N) * free_params
    rho2 = isfinite(ll0) ? 1 - ll / ll0 : NaN


    # The Hessian verdict sits next to `Converged` on purpose: "converged" alone
    # is not enough to trust the standard errors, and the `@warn` it also emits
    # scrolls past above a long optimizer trace.
    hessian = get(results, :hessian, nothing)

    # Imprime resumen
    println("\nModel Summary")
    println(@sprintf("Log-likelihood at optimum  : %10.4f", ll))
    println(@sprintf("Null Log-likelihood.       : %10.4f", ll0))
    println(@sprintf("Iterations                 : %10d", iters))
    println(@sprintf("Converged                  : %10s", converged))
    if !isnothing(hessian)
        println(@sprintf("Hessian at optimum         : %10s", _hessian_label(hessian)))
    end
    println(@sprintf("Estimation time (seconds)  : %10.2f", estimation_time))
    println(@sprintf("Number of free parameters  : %10d", free_params))
    println(@sprintf("Number of observations     : %10d", N))
    println(@sprintf("AIC                        : %10.2f", aic))
    println(@sprintf("BIC                        : %10.2f", bic))
    println(@sprintf("Rho-squared (McFadden)     : %10.4f", rho2))

    # Exporta a Excel si se especifica
    if !isnothing(file)
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

            # Hoja resumen
            XLSX.addsheet!(xf, "Summary")
            sheet2 = xf["Summary"]
            sheet2[1,1:2] = ["Log-likelihood at optimum", ll]
            sheet2[2,1:2] = ["Null Log-likelihood", ll0]
            sheet2[3,1:2] = ["Iterations", iters]
            sheet2[4,1:2] = ["Converged", converged]
            sheet2[5,1:2] = ["Estimation time (s)", estimation_time]
            sheet2[6,1:2] = ["Number of free parameters", free_params]
            sheet2[7,1:2] = ["Number of observations", N]
            sheet2[8,1:2] = ["AIC", aic]
            sheet2[9,1:2] = ["BIC", bic]
            sheet2[10,1:2] = ["Rho-squared", rho2]
            if !isnothing(hessian)
                sheet2[11,1:2] = ["Hessian at optimum", _hessian_label(hessian)]
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