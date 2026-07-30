using CSV, DataFrames, Statistics
using ChoiceModels

# Nested logit on the Swissmetro data, mirroring Biogeme's `b09_nested`
# (see plot_b09_nested.py): the two *existing* modes, train and car, share a nest;
# Swissmetro sits outside it.
#
# CONVENTION. This package follows Apollo: V_nest = λ · log Σ exp(V_j / λ), with
# λ ≤ 1 the RUM-consistent range. Biogeme parameterizes the reciprocal scale
# μ = 1/λ ≥ 1 (hence its `nest_parameter` bounded [1,3] and started at 1). The two
# fit the same model and report reciprocal nest parameters, so compare λ̂ against
# 1/μ̂, not against μ̂ itself.

df = CSV.read("../data/swissmetro.dat", DataFrame; delim='\t')
df = filter(row -> (row[:PURPOSE] == 1 || row[:PURPOSE] == 3) && row[:CHOICE] .!= 0, df)

# Scale
df.TRAIN_TT ./= 100
df.TRAIN_CO ./= 100
df.CAR_TT ./= 100
df.CAR_CO ./= 100
df.SM_TT ./= 100
df.SM_CO ./= 100

# If the person has a GA (season ticket) her incremental cost is actually 0
df.SM_CO    .= ifelse.(df.GA .== 0, df.SM_CO, 0.0)
df.TRAIN_CO .= ifelse.(df.GA .== 0, df.TRAIN_CO, 0.0)

# Define variables and parameters
asc_car = Parameter(:asc_car, value=0)
asc_train = Parameter(:asc_train, value=0)
asc_sm = Parameter(:asc_sm, value=0, fixed=true)

β_time = Parameter(:β_time, value=0)
β_cost = Parameter(:β_cost, value=0)

# The nest scale parameter. Starting it at 1 is the neutral choice: estimation is
# on θ = log λ, so this is θ₀ = 0, and λ = 1 is exactly the MNL special case.
λ_existing = Parameter(:λ_existing, value=1.0)

# Define the alternatives: name => code used in the CHOICE column
alternatives = (train = 1, sm = 2, car = 3)

# Define utility functions
utilities = (
    train = asc_train + β_time * Variable(:TRAIN_TT) + β_cost * Variable(:TRAIN_CO),
    sm    = asc_sm    + β_time * Variable(:SM_TT)    + β_cost * Variable(:SM_CO),
    car   = asc_car   + β_time * Variable(:CAR_TT)   + β_cost * Variable(:CAR_CO)
)

# The nesting tree. The root is a plain vector — its scale parameter is fixed at 1
# by normalization, so there is nothing to supply for it. Swissmetro hangs directly
# off the root, which is all a "degenerate nest" needs: no wrapper, no convention.
tree = [:sm, Nest(λ_existing, [:train, :car])]

# Load availability data
df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
df.CAR_AV_SP .= ifelse.(df.SP .!= 0, df.CAR_AV, 0)

availability = (
    train = df.TRAIN_AV_SP .== 1,
    sm    = df.SM_AV .== 1,
    car   = df.CAR_AV_SP .== 1
)

# Create model and estimate
model = NestedLogitModel(
    alternatives;
    utilities    = utilities,
    tree         = tree,
    availability = availability,
    data         = df
)

# The tree, printed, so the structure can be checked at a glance
println()
show(stdout, MIME"text/plain"(), model)

results = estimate(model, :CHOICE)

# Output results
summarize_results(results, file="output/NL_swissmetro.xlsx")
println('\n')

# Biogeme reports the reciprocal scale; print it so the two are directly comparable.
λ̂ = Float64(results.parameters[:λ_existing])
println("λ (Apollo/ChoiceModels convention) : ", round(λ̂, digits=4))
println("μ = 1/λ (Biogeme convention)       : ", round(1 / λ̂, digits=4))

# Evaluate WTP
expressions = Dict(
    :WTP => β_time / β_cost
)

wtp = evaluate(expressions, model, results)
summarize_expressions(wtp, file="output/NL_swissmetro_WTP.xlsx")

# Predict
preds = predict(model, results)
println("\nAverage of Nested Logit predictions")
println(mean(preds, dims=1))
