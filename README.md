# ChoiceModels.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia

**ChoiceModels.jl** is a Julia package designed for flexible and symbolic specification of Discrete Choice Models (DCM). Inspired by **Biogeme** and **Apollo**, it uses symbolic algebra, modular architecture, and Julia's type system to allow rapid prototyping and estimation of models like Logit, Nested Logit and Mixed Logit.

---

## 🚀 Key Features

- **Symbolic utility expressions** using `Parameter`, `Variable`, and algebraic operators.
- **Multinomial Logit, Nested Logit, Mixed Logit and Latent Class** estimation with availability conditions.
- **Integration with `DataFrames.jl`** for data handling.
- **Estimation via `Optim.jl`**, supporting automatic or analytic gradients.
- **Prediction tools** for probabilities and most likely alternatives.
- **Extensibility** for new models.

---

## ✨ Quick Example: Estimating a Logit Model

```julia
using ChoiceModels, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

asc_car = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)
β_cost = Parameter(:β_cost, value=0.0)

V_car = asc_car + β_time * Variable(:time_car) + β_cost * Variable(:cost_car)
V_bus = β_time * Variable(:time_bus) + β_cost * Variable(:cost_bus)

# `alternatives` maps each alternative's name to the code it carries in the choice
# column; `utilities` and `availability` are matched to it by name, so their order
# does not matter.
model = LogitModel((car = 1, bus = 2);
    utilities    = (car = V_car, bus = V_bus),
    availability = (car = trues(nrow(df)), bus = trues(nrow(df))),
    data         = df)
results = estimate(model, :choice)
probs = predict(model, results)
```

---

## 🌀 Example: Estimating a Mixed Logit Model

```julia
using ChoiceModels, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

asc_car = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)
σ_time = Parameter(:σ_time, value=1.0)

draw = Draw(:time_rnd)
V_car = asc_car + (β_time + σ_time * draw) * Variable(:time_car)
V_bus = (β_time + σ_time * draw) * Variable(:time_bus)

model = MixedLogitModel((car = 1, bus = 2);
    utilities    = (car = V_car, bus = V_bus),
    availability = (car = trues(nrow(df)), bus = trues(nrow(df))),
    data         = df,
    idvar        = :id,
    R            = 500,
    draw_scheme  = :mlhs)

results = estimate(model, :choice)
P = predict(model, results)
```

---

## 🌳 Example: Estimating a Nested Logit Model

```julia
using ChoiceModels, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

asc_train = Parameter(:asc_train, value=0.0)
β_time = Parameter(:β_time, value=0.0)
β_cost = Parameter(:β_cost, value=0.0)

# The nest scale parameter. Starting it at 1 is the neutral choice — estimation
# runs on θ = log λ, so this is θ₀ = 0, and λ = 1 is exactly the MNL special case.
λ_existing = Parameter(:λ_existing, value=1.0)

V_train = asc_train + β_time * Variable(:train_tt) + β_cost * Variable(:train_co)
V_sm    =             β_time * Variable(:sm_tt)    + β_cost * Variable(:sm_co)
V_car   =             β_time * Variable(:car_tt)   + β_cost * Variable(:car_co)

model = NestedLogitModel((train = 1, sm = 2, car = 3);
    utilities = (train = V_train, sm = V_sm, car = V_car),
    # The root is a plain Vector: its scale parameter is fixed at 1 by
    # normalization, so there is nothing to supply for it. An alternative that
    # belongs to no nest is simply listed at the root — no wrapper needed.
    tree      = [:sm, Nest(λ_existing, [:train, :car])],
    data      = df)

show(stdout, MIME"text/plain"(), model)   # prints the tree, to check the structure
results = estimate(model, :choice)
```

λ follows Apollo's convention, `V_nest = λ · log Σ exp(V_j / λ)`, with λ ≤ 1 the
range consistent with random utility maximization; λ̂ > 1 is estimated and warned
about rather than made unreachable. Biogeme parameterizes the reciprocal scale
μ = 1/λ ≥ 1, so the two packages fit the same model but report reciprocal nest
parameters.

---

## 📦 Installation

```julia
] dev https://github.com/ighdez/ChoiceModels.jl
```

---

## 🧱 Architecture Overview

- **Expressions**: `Parameter`, `Variable`, `Draw`, and algebraic combinators
- **Models**: `LogitModel`, `NestedLogitModel`, `MixedLogitModel`, `LatentClassModel`
- **Estimation**: `estimate(model, choices)` using `Optim.jl`
- **Prediction**: `predict(model, results)` returns probabilities

---

## 📚 Documentation

All public functions are documented via Julia docstrings. Use `?estimate` in the REPL to see inline help. Full documentation will be published using Documenter.jl soon.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

- [Biogeme](https://biogeme.epfl.ch/)
- [Apollo](https://www.apollochoicemodelling.com/)
- [Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html)

Built with ♥ and `ChatGPT` + `Claude Code` pair-programming.
