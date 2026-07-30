"""
ChoiceModels.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia.

## Features

* Symbolic utility specification using `Parameter` and `Variable`
* Support for Logit models with availability constraints
* Native compatibility with `DataFrames.jl`
* Estimation routines powered by `Optim.jl` (analytic or automatic differentiation)
* Modular design enabling future models like Mixed Logit, RRM, etc.

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

## Submodules

* `Expressions.jl`: symbolic expressions (sum, multiplication, exp, etc.)
* `Utils.jl`: utilities for model construction (parameter collection, updates)
* `Models.jl`: base model type and model-specific definitions

## Exports

User-facing types and functions, including:

* `Parameter`, `Variable`
* `LogitModel`, `estimate`, `predict`

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
