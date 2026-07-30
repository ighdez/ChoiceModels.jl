"""
Defines the abstract type DiscreteChoiceModel and the generic interface for models in ChoiceModels.jl.

This module establishes the base interface all discrete choice models must follow, including predict, loglikelihood, and estimate. Specific models (e.g., Logit, Mixed Logit) must subtype DiscreteChoiceModel and implement these methods.
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