using CSV, DataFrames, Statistics
using ChoiceModels

# Two-level nested logit with socio-demographics on mode choice SP data,
# mirroring Apollo's `NL_two_levels.r`: the three public-transport modes
# (bus, air, rail) share a nest, car sits outside it.
#
# Two differences from the Apollo script are worth knowing when comparing output:
#
#  1. Apollo calls `apollo_panelProd`, so its likelihood contributions are one per
#     *individual*. Here they are one per *observation*, as in every model in this
#     package. Point estimates and classical standard errors are unaffected; the
#     ROBUST standard errors will differ, because ours are not clustered by
#     individual. That is pre-existing and not specific to nested logit.
#  2. Apollo seeds its starting values from a previously estimated MNL
#     (`apollo_readBeta`). This starts everything at zero, and λ at 1.

df = CSV.read("../data/apollo_modeChoiceData.csv", DataFrame)
df = filter(:SP => x -> x == 1, df)

# Apollo builds `mean_income` as a column and then takes `income / mean_income`.
# Precomputing the ratio keeps the utility expression to one variable.
df.income_ratio = df.income ./ mean(df.income)

# Define variables and parameters
asc_car  = Parameter(:asc_car, value=0, fixed=true)
asc_bus  = Parameter(:asc_bus, value=0)
asc_air  = Parameter(:asc_air, value=0)
asc_rail = Parameter(:asc_rail, value=0)

asc_bus_interaction_female  = Parameter(:asc_bus_interaction_female, value=0)
asc_air_interaction_female  = Parameter(:asc_air_interaction_female, value=0)
asc_rail_interaction_female = Parameter(:asc_rail_interaction_female, value=0)

β_time_car  = Parameter(:β_time_car, value=0)
β_time_bus  = Parameter(:β_time_bus, value=0)
β_time_air  = Parameter(:β_time_air, value=0)
β_time_rail = Parameter(:β_time_rail, value=0)
β_time_interaction_business = Parameter(:β_time_interaction_business, value=0)

β_access = Parameter(:β_access, value=0)
β_cost   = Parameter(:β_cost, value=0)
β_cost_interaction_business = Parameter(:β_cost_interaction_business, value=0)
cost_income_elast = Parameter(:cost_income_elast, value=0)

β_no_frills = Parameter(:β_no_frills, value=0, fixed=true)
β_wifi = Parameter(:β_wifi, value=0)
β_food = Parameter(:β_food, value=0)

# The nest scale parameter for the public-transport nest. Started at 1, i.e. at
# θ₀ = log 1 = 0 in the space estimation actually searches, which is also exactly
# the MNL special case.
λ_PT = Parameter(:λ_PT, value=1.0)

# Interactions with socio-demographics
asc_bus_value  = asc_bus  + asc_bus_interaction_female  * Variable(:female)
asc_air_value  = asc_air  + asc_air_interaction_female  * Variable(:female)
asc_rail_value = asc_rail + asc_rail_interaction_female * Variable(:female)

β_time_car_value  = β_time_car  + β_time_interaction_business * Variable(:business)
β_time_bus_value  = β_time_bus  + β_time_interaction_business * Variable(:business)
β_time_air_value  = β_time_air  + β_time_interaction_business * Variable(:business)
β_time_rail_value = β_time_rail + β_time_interaction_business * Variable(:business)

# Written exactly as Apollo writes it: the exponent is an estimated parameter.
# `^` with a symbolic exponent is rewritten internally to exp(exponent · log(base)),
# which is exact for a positive base — see the `^` overload in Expressions.jl.
income_effect = Variable(:income_ratio) ^ cost_income_elast
β_cost_value = (β_cost + β_cost_interaction_business * Variable(:business)) * income_effect

# Define the alternatives: name => code used in the choice column
alternatives = (car = 1, bus = 2, air = 3, rail = 4)

utilities = (
    car  = asc_car + β_time_car_value * Variable(:time_car) + β_cost_value * Variable(:cost_car),
    bus  = asc_bus_value + β_time_bus_value * Variable(:time_bus) + β_access * Variable(:access_bus) +
           β_cost_value * Variable(:cost_bus),
    air  = asc_air_value + β_time_air_value * Variable(:time_air) + β_access * Variable(:access_air) +
           β_cost_value * Variable(:cost_air) + β_no_frills * (Variable(:service_air) == 1) +
           β_wifi * (Variable(:service_air) == 2) + β_food * (Variable(:service_air) == 3),
    rail = asc_rail_value + β_time_rail_value * Variable(:time_rail) + β_access * Variable(:access_rail) +
           β_cost_value * Variable(:cost_rail) + β_no_frills * (Variable(:service_rail) == 1) +
           β_wifi * (Variable(:service_rail) == 2) + β_food * (Variable(:service_rail) == 3)
)

# The nesting tree. Apollo needs two cross-referencing lists (nlNests + nlStructure);
# here the indentation is the structure, and the nest is identified by its own λ.
tree = [:car, Nest(λ_PT, [:bus, :air, :rail])]

availability = (
    car  = df.av_car .== 1,
    bus  = df.av_bus .== 1,
    air  = df.av_air .== 1,
    rail = df.av_rail .== 1
)

# Create model and estimate
model = NestedLogitModel(
    alternatives;
    utilities    = utilities,
    tree         = tree,
    availability = availability,
    data         = df
)

println()
show(stdout, MIME"text/plain"(), model)

results = estimate(model, :choice)

# Output results
summarize_results(results, file="output/NL_modeChoice_SP.xlsx")

# Predict
preds = predict(model, results)
println("\nAverage of Nested Logit predictions")
println(mean(preds, dims=1))
