"""
ChoiceModels.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia.

## Features

* Symbolic utility specification using `Parameter`, `Variable` and `Draw`
* Four models, all with availability constraints: `LogitModel`, `MixedLogitModel`,
  `NestedLogitModel` and `LatentClassModel` (whose classes may be any of the other
  three, including Mixed Logit)
* Native compatibility with `DataFrames.jl`
* Maximum likelihood via `Optim.jl` (BFGS), with exact forward-mode `ForwardDiff`
  gradients and Hessians. `estimate(...; hessian_method = :fd)` switches the
  Hessian to a finite-difference Jacobian of that exact gradient, as Apollo does
* Classical and robust (sandwich) standard errors reported side by side, with the
  Hessian and BHHH matrices classified at the optimum rather than assumed sound
* Delta-method standard errors for derived quantities (WTP, elasticities) on all
  four models, via `evaluate(expressions, model, results)`

## Example

```julia
using ChoiceModels

asc = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)

V_car = asc + β_time * Variable(:time_car)
V_bus = β_time * Variable(:time_bus)

# `alternatives` maps each name to its code in the choice column; `utilities` and
# `availability` are matched to it by name.
model = LogitModel((car = 1, bus = 2);
                   utilities    = (car = V_car, bus = V_bus),
                   availability = (car = ..., bus = ...),
                   data         = df)
results = estimate(model, :choice)
P = predict(model, results)
```

## Files

Included in dependency order, which is why it is also the order they are listed in:

* `Expressions.jl`: the symbolic nodes (sum, product, exp, power, …) and the two
  `evaluate` paths — cross-sectional, and the `N × R` draws path
* `Utils.jl`: tree walkers, the shared alternatives/estimation-space/standard-error
  machinery, the delta method, and console + Excel reporting
* `Draws.jl`: draw generation for Mixed Logit (`:normal`, `:uniform`, `:halton`, `:mlhs`)
* `Models.jl`: the `DiscreteChoiceModel` base type, then the four model files

## Exports

* `Parameter`, `Variable`, `Draw`, `evaluate`
* `LogitModel`, `MixedLogitModel`, `NestedLogitModel`, `LatentClassModel`, `Nest`
* `estimate`, `predict`, `loglikelihood`
* `summarize_results`, `summarize_expressions`, `generate_draws`, `Draws`

## License

MIT License
"""

module ChoiceModels

__precompile__()

using Optim, LineSearches, DataFrames, ForwardDiff, FiniteDiff, LinearAlgebra, Distributions, Printf, XLSX, Base.Threads, StatsBase, Primes

include("Expressions.jl")
include("Utils.jl")
include("Draws.jl")
include("Models.jl")

end
