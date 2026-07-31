"""
Base type for every discrete choice model in ChoiceModels.jl.

A model is expected to provide `estimate(model, choicevar)`, `predict(model, results)`,
`loglikelihood(model, choices)`, an `evaluate(model, data, params)` giving its `N × J`
choice probabilities, and a `_children` method so the `collect_*` walkers can reach the
expressions it holds. There is no enforced interface — the methods are defined per model
in `models/`, and a missing one surfaces as a `MethodError`.

**It subtypes `DCMExpression` on purpose**: that is what lets a fitted model appear as a
term inside a symbolic expression, which is how a `LatentClassModel` is written
(`π_1 * model_1 + π_2 * model_2`). A model is therefore something the tree walkers must
descend into, not a leaf.
"""
abstract type DiscreteChoiceModel <: DCMExpression end

# function predict(model::DiscreteChoiceModel)
#     error("predict not implemented for $(typeof(model))")
# end

# function loglikelihood(model::DiscreteChoiceModel, choices)
#     error("loglikelihood not implemented for $(typeof(model))")
# end

# function estimate(model::DiscreteChoiceModel, choicevar; verbose = true)
#     error("estimate not implemented for $(typeof(model))")
# end

include("models/LogitModel.jl")
include("models/MixedLogit.jl")
# Nested Logit before Latent Class: it reuses `null_loglikelihood_mnl` from the
# Logit file, and `LatentClass.jl` needs `NestedLogitModel` to exist to give it a
# `_reorder_alternatives` method.
include("models/NestedLogit.jl")
include("models/LatentClass.jl")

export LogitModel, MixedLogitModel, NestedLogitModel, LatentClassModel, Nest,
       estimate, predict, loglikelihood