### Expressions.jl — Symbolic Expressions for Discrete Choice Models

"""
Abstract type for all symbolic expressions in the DCM system.
All symbolic objects (parameters, variables, operators) must inherit from this type.
"""
abstract type DCMExpression end

abstract type DCMBinary <: DCMExpression end
abstract type DCMUnary <: DCMExpression end

"""
Represents a constant literal value in an expression tree.

# Fields
- `value::Float64`: the constant numeric value
"""
struct DCMLiteral <: DCMExpression
    value::Float64
end

"""
Represents a symbolic equality comparison between an expression and a numeric value.

# Fields
- `left::DCMExpression`: left-hand side symbolic expression
- `right::Real`: right-hand side numeric value
"""
struct DCMEqual{L<:DCMExpression, R<:Real} <: DCMBinary
    left::L
    right::R
end

"""
Symbolic addition of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMSum{L<:DCMExpression, R<:DCMExpression} <: DCMBinary
    left::L
    right::R
end

"""
Symbolic difference of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMDiff{L<:DCMExpression, R<:DCMExpression} <: DCMBinary
    left::L
    right::R
end

"""
Symbolic multiplication of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMMult{L<:DCMExpression, R<:DCMExpression} <: DCMBinary
    left::L
    right::R
end

"""
Symbolic division of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMDiv{L<:DCMExpression, R<:DCMExpression} <: DCMBinary
    left::L
    right::R
end

"""
Symbolic exponential of an expression.

# Fields
- `arg`: symbolic expression
"""
struct DCMExp{A<:DCMExpression} <: DCMUnary
    arg::A
end

"""
Symbolic logarithm of an expression.

# Fields
- `arg`: symbolic expression
"""
struct DCMLog{A<:DCMExpression} <: DCMUnary
    arg::A
end

"""
Symbolic negation of an expression (unary minus).

# Fields
- `arg`: symbolic expression
"""
struct DCMMinus{A<:DCMExpression} <: DCMUnary
    arg::A
end

"""
Symbolic power of an expression raised to a numeric exponent (`arg ^ exponent`).

Modeled as a `DCMUnary` node (its single symbolic child is `arg`; `exponent` is a
plain `Real`), so the `collect_*` tree-walkers traverse it automatically.

# Fields
- `arg`: symbolic base expression
- `exponent`: numeric exponent
"""
struct DCMPower{A<:DCMExpression, E<:Real} <: DCMUnary
    arg::A
    exponent::E
end

# Operator overloads

import Base: ==, +, *, /, ^, exp, log, -
==(a::DCMExpression, b::Real) = DCMEqual(a, b)
+(a::DCMExpression, b::DCMExpression) = DCMSum(a, b)
-(a::DCMExpression, b::DCMExpression) = DCMDiff(a, b)
*(a::DCMExpression, b::DCMExpression) = DCMMult(a, b)
/(a::DCMExpression, b::DCMExpression) = DCMDiv(a, b)
^(a::DCMExpression, b::Real) = DCMPower(a, b)
exp(a::DCMExpression) = DCMExp(a)
log(a::DCMExpression) = DCMLog(a)
-(a::DCMExpression) = DCMMinus(a)

+(a::Real, b::DCMExpression) = DCMLiteral(a) + b
+(a::DCMExpression, b::Real) = a + DCMLiteral(b)

-(a::Real, b::DCMExpression) = DCMLiteral(a) - b
-(a::DCMExpression, b::Real) = a - DCMLiteral(b)

*(a::Real, b::DCMExpression) = DCMLiteral(a) * b
*(a::DCMExpression, b::Real) = a * DCMLiteral(b)

/(a::Real, b::DCMExpression) = DCMLiteral(a) / b
/(a::DCMExpression, b::Real) = a / DCMLiteral(b)

"""
Represents a named parameter in a utility expression.

# Fields
- `name::Symbol`: parameter name
- `value::Float64`: initial value
- `fixed::Bool`: whether the parameter is fixed during estimation
"""
struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool
end

"""
Constructor for `DCMParameter`.

# Arguments
- `name::Symbol`: parameter name
- `value=0.0`: initial value (default: 0.0)
- `fixed::Bool=false`: fixed during estimation? (default: false)
"""
function Parameter(name::Symbol; value=0.0, fixed::Bool=false)
    return DCMParameter(name, value, fixed)
end

"""
Represents a data variable used in utility expressions.

# Fields
- `name::Symbol`: name of the variable, typically a column in the dataset
- `index::Union{Nothing, Int}`: optional index for panel data structure
"""
struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}  # For panel/individual data
end

"""
Constructor for `DCMVariable`.

# Arguments
- `name::Symbol`: variable name (must match column name in data)
- `index=nothing`: optional index for individual-specific variables

# Returns
- `DCMVariable` object
"""
function Variable(name::Symbol; index=nothing)
    return DCMVariable(name, index)
end


"""
Represents a symbolic placeholder for random draws in Mixed Logit models.

# Fields
- `name::Symbol`: name of the draw (e.g., `:draw_normal_time`)
"""
struct DCMDraw <: DCMExpression
    name::Symbol
end

"""
Constructor for `DCMDraw`.

# Arguments
- `name::Symbol`: name of the random draw

# Returns
- `DCMDraw` object
"""
function Draw(name::Symbol)
    return DCMDraw(name)
end

"""
Evaluates a symbolic utility expression for all observations in a dataset.

# Arguments
- `expr::DCMExpression`: symbolic expression to evaluate
- `data::DataFrame`: dataset with values for variables
- `params::AbstractDict`: dictionary with parameter names and values

# Returns
- `Vector{Float64}`: evaluated numeric result for each observation
"""
# One method per node type — dispatch replaces the former `if expr isa …` ladder,
# so each method is type-stable and specializable (the `LogitModel` case lives in
# models/LogitModel.jl, since that type is defined after this file is loaded).
evaluate(e::DCMParameter, data::DataFrame, params::AbstractDict) = fill(params[e.name], nrow(data))
evaluate(e::DCMVariable,  data::DataFrame, params::AbstractDict) = data[:, e.name]
evaluate(e::DCMLiteral,   data::DataFrame, params::AbstractDict) = fill(e.value, nrow(data))

evaluate(e::DCMSum,  data::DataFrame, params::AbstractDict) = evaluate(e.left, data, params) .+ evaluate(e.right, data, params)
evaluate(e::DCMDiff, data::DataFrame, params::AbstractDict) = evaluate(e.left, data, params) .- evaluate(e.right, data, params)
evaluate(e::DCMMult, data::DataFrame, params::AbstractDict) = evaluate(e.left, data, params) .* evaluate(e.right, data, params)
evaluate(e::DCMDiv,  data::DataFrame, params::AbstractDict) = evaluate(e.left, data, params) ./ evaluate(e.right, data, params)

evaluate(e::DCMExp,   data::DataFrame, params::AbstractDict) = exp.(evaluate(e.arg, data, params))
evaluate(e::DCMLog,   data::DataFrame, params::AbstractDict) = log.(evaluate(e.arg, data, params))
evaluate(e::DCMMinus, data::DataFrame, params::AbstractDict) = -evaluate(e.arg, data, params)
evaluate(e::DCMPower, data::DataFrame, params::AbstractDict) = evaluate(e.arg, data, params) .^ e.exponent

function evaluate(e::DCMEqual, data::DataFrame, params::AbstractDict)
    left_val = evaluate(e.left, data, params)
    return ifelse.(left_val .== e.right, one(eltype(left_val)), zero(eltype(left_val)))
end

"""
Evaluates a symbolic expression for all observations and draws in Mixed Logit models.

This version supports replication of data over simulation draws and handles parameter values,
random draws, and data variables.

# Arguments
- `expr::DCMExpression`: the symbolic expression to evaluate
- `data::DataFrame`: the dataset (N rows)
- `params::AbstractDict`: mapping from parameter names to values
- `draws::AbstractDict`: dictionary of random draws, each as an `N × R` array

# Returns
- `Array{Float64, 2}`: evaluated result of shape `N × R`
"""
# ---- N×R draws path --------------------------------------------------------
# Public entry recurses with `_evaluate_draws`, whose leaves keep their natural
# shape — parameters/literals stay scalar, variables stay length-N vectors,
# draws are the only genuinely N×R leaves — instead of being materialized to
# N×R up front with `fill`/`repeat`. Broadcasting reconciles the shapes, so a
# utility allocates a handful of N×R buffers rather than one per leaf (the old
# `fill(param, N, R)` cost was brutal in the ForwardDiff/Dual gradient path).
# The wrapper then guarantees an N×R result — broadcasting a draw-free subtree
# up if needed — so `logit_prob`'s `size(utils[j]) == (N, R)` contract holds.
function evaluate(e::DCMExpression, data::DataFrame, params::AbstractDict, draws::AbstractDict)
    N, R = size(first(values(draws)))
    # Because the operator nodes are parametric (DCMSum{L,R}, …), the full tree
    # type is concrete, so this call specializes on that type and the lazy
    # `Base.broadcasted` chain below fuses into a SINGLE materialized N×R kernel.
    return _as_nxr(Base.Broadcast.materialize(_evaluate_draws(e, data, params, draws)), N, R)
end

# Broadcast a scalar/vector result up to N×R; pass an already-N×R matrix through.
_as_nxr(v::AbstractMatrix, N::Int, R::Int) = v
function _as_nxr(v, N::Int, R::Int)
    out = Matrix{eltype(v)}(undef, N, R)
    out .= v
    return out
end

# Leaves — natural shape, no N×R materialization (params/literals stay scalar,
# variables stay length-N vectors, draws are the only genuinely N×R leaves).
_evaluate_draws(e::DCMParameter, data, params, draws) = params[e.name]
_evaluate_draws(e::DCMLiteral,   data, params, draws) = e.value
_evaluate_draws(e::DCMVariable,  data, params, draws) = data[:, e.name]
_evaluate_draws(e::DCMDraw,      data, params, draws) = draws[e.name]

# Operators — lazy: return an unmaterialized `Broadcasted`. Nesting these builds
# one fused broadcast tree for the whole utility, materialized once by the
# wrapper above. Concrete node types make the fused kernel fully specialized
# (incl. the ForwardDiff Dual path), so there is no per-node dispatch or buffer.
_evaluate_draws(e::DCMSum,  data, params, draws) = Base.broadcasted(+, _evaluate_draws(e.left, data, params, draws), _evaluate_draws(e.right, data, params, draws))
_evaluate_draws(e::DCMDiff, data, params, draws) = Base.broadcasted(-, _evaluate_draws(e.left, data, params, draws), _evaluate_draws(e.right, data, params, draws))
_evaluate_draws(e::DCMMult, data, params, draws) = Base.broadcasted(*, _evaluate_draws(e.left, data, params, draws), _evaluate_draws(e.right, data, params, draws))
_evaluate_draws(e::DCMDiv,  data, params, draws) = Base.broadcasted(/, _evaluate_draws(e.left, data, params, draws), _evaluate_draws(e.right, data, params, draws))

_evaluate_draws(e::DCMExp,   data, params, draws) = Base.broadcasted(exp, _evaluate_draws(e.arg, data, params, draws))
_evaluate_draws(e::DCMLog,   data, params, draws) = Base.broadcasted(log, _evaluate_draws(e.arg, data, params, draws))
_evaluate_draws(e::DCMMinus, data, params, draws) = Base.broadcasted(-,   _evaluate_draws(e.arg, data, params, draws))
_evaluate_draws(e::DCMPower, data, params, draws) = Base.broadcasted(^,   _evaluate_draws(e.arg, data, params, draws), e.exponent)

_evaluate_draws(e::DCMEqual, data, params, draws) =
    Base.broadcasted((l, r) -> ifelse(l == r, one(l), zero(l)), _evaluate_draws(e.left, data, params, draws), e.right)

export Parameter, Variable, Draw, evaluate