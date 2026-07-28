using CSV, DataFrames, Statistics
using ChoiceModels

# Load dataset and filter RP observations only
df = CSV.read("../data/swissmetro.dat", DataFrame; delim='\t')
df = filter(row -> (row[:PURPOSE] == 1 || row[:PURPOSE] == 3) && row[:CHOICE] .!= 0, df)

# Scale
df.TRAIN_TT ./= 100
df.TRAIN_CO ./= 100
df.CAR_TT ./= 100
df.CAR_CO ./= 100
df.SM_TT ./= 100
df.SM_CO ./= 100

#If the person has a GA (season ticket) her incremental cost is actually 0
df.SM_CO    .= ifelse.(df.GA .== 0, df.SM_CO, 0.0)
df.TRAIN_CO .= ifelse.(df.GA .== 0, df.TRAIN_CO, 0.0)

# Define variables and parameters
asc_car = Parameter(:asc_car, value=0)
asc_train = Parameter(:asc_train, value=0)
asc_sm = Parameter(:asc_sm, value=0, fixed=true)

β_time = Parameter(:β_time, value=0)
β_cost = Parameter(:β_cost, value=0)

# Define utility functions
V1 = asc_train + β_time * Variable(:TRAIN_TT) + β_cost * Variable(:TRAIN_CO)
V2 = asc_sm    + β_time * Variable(:SM_TT)    + β_cost * Variable(:SM_CO)
V3 = asc_car   + β_time * Variable(:CAR_TT)   + β_cost * Variable(:CAR_CO)

utilities = [V1,V2,V3]

# Load availability data
df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
df.CAR_AV_SP .= ifelse.(df.SP .!= 0, df.CAR_AV, 0)

availability = [
    df.TRAIN_AV_SP .== 1,
    df.SM_AV .== 1,
    df.CAR_AV_SP .== 1
]

# Create model and estimate
model = LogitModel(utilities; data=df, availability=availability)
results = estimate(model, :CHOICE)

# Output results
summarize_results(results, file="output/MNL_swissmetro.xlsx")
println('\n')

# Evaluate WTP
expressions = Dict(
    :WTP => β_time / β_cost,
    :ASC_ratio => asc_car / asc_train
)

wtp = evaluate(expressions, model, results)
summarize_expressions(wtp, file="output/MNL_swissmetro_WTP.xlsx")

# Predict
preds = predict(model,results)
println("\nAverage of Logit predictions")
println(mean(preds,dims=1))
