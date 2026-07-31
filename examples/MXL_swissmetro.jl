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
mu_asc_car = Parameter(:mu_asc_car, value=0)
s_asc_car = Parameter(:s_asc_car, value=1)
d_asc_car = Draw(:d_asc_car)
asc_car = (mu_asc_car + s_asc_car * d_asc_car)

mu_asc_train = Parameter(:mu_asc_train, value=0)
s_asc_train = Parameter(:s_asc_train, value=1)
d_asc_train = Draw(:d_asc_train)
asc_train = (mu_asc_train + s_asc_train * d_asc_train)

mu_asc_sm = Parameter(:mu_asc_sm, value=0, fixed=true)
s_asc_sm = Parameter(:s_asc_sm, value=1)
d_asc_sm = Draw(:d_asc_sm)
asc_sm = (mu_asc_sm + s_asc_sm * d_asc_sm)

mu_time = Parameter(:mu_time, value=-1)
s_time = Parameter(:s_time, value=1)
d_time = Draw(:d_time)
β_time = (mu_time + s_time * d_time)

β_cost = Parameter(:β_cost, value=-1)

# Define the alternatives: name => code used in the CHOICE column
alternatives = (train = 1, sm = 2, car = 3)

# Define utility functions
utilities = (
    train = asc_train + β_time * Variable(:TRAIN_TT) + β_cost * Variable(:TRAIN_CO),
    sm    = asc_sm    + β_time * Variable(:SM_TT)    + β_cost * Variable(:SM_CO),
    car   = asc_car   + β_time * Variable(:CAR_TT)   + β_cost * Variable(:CAR_CO)
)

# Load availability data
df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
df.CAR_AV_SP .= ifelse.(df.SP .!= 0, df.CAR_AV, 0)

availability = (
    train = df.TRAIN_AV_SP .== 1,
    sm    = df.SM_AV .== 1,
    car   = df.CAR_AV_SP .== 1
)

# Create model and estimate
using Random
Random.seed!(12345)
# 500 Halton draws, not 100 pseudo-random normals. With four random parameters,
# R=100 `:normal` gave a visibly noisy simulated likelihood: the optimizer landed
# on a different local optimum under any perturbation (three starting points gave
# −3612.3, −3641.0 and −3621.3), so the previously recorded log-likelihood was one
# draw from a lottery rather than a reproducible number.
model = MixedLogitModel(alternatives; utilities=utilities, availability=availability, data=df, idvar=:ID, R=500, draw_scheme=:halton)
results = estimate(model, :CHOICE)

# Output results
summarize_results(results, file="output/MXL_swissmetro.xlsx")

# Predict
# preds = predict(model,results)
# println("\nAverage of Logit predictions")
# println(mean(preds,dims=1))