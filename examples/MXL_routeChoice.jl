using CSV, DataFrames, Statistics
using ChoiceModels

# Load Swiss route choice dataset
df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)
df = sort(df, :ID)

# Scale
df.tt1 ./= 60
df.tt2 ./= 60
df.hw1 ./= 60
df.hw2 ./= 60
df.tc1 ./= 10
df.tc2 ./= 10

# Define parameters for lognormal random coefficients
mu_tt     = Parameter(:mu_tt, value=0)
sigma_tt  = Parameter(:sigma_tt, value=0.5,fixed=false)

mu_tc     = Parameter(:mu_tc, value=0)
sigma_tc  = Parameter(:sigma_tc, value=0.5,fixed=false)

mu_hw     = Parameter(:mu_hw, value=0)
sigma_hw  = Parameter(:sigma_hw, value=0.5,fixed=false)

mu_ch     = Parameter(:mu_ch, value=0)
sigma_ch  = Parameter(:sigma_ch, value=0.5,fixed=false)

# Define variables
tt = Draw(:tt)  # draw for travel time
tc = Draw(:tc)  # draw for travel cost
hw = Draw(:hw)  # draw for travel time
ch = Draw(:ch)  # draw for travel cost

# Define random parameters
b_tt = mu_tt + sigma_tt * tt
b_tc = mu_tc + sigma_tc * tc
b_hw = mu_hw + sigma_hw * hw
b_ch = mu_ch + sigma_ch * ch

# Define the alternatives: name => code used in the choice column
alternatives = (route1 = 1, route2 = 2)

# Define utility functions
utilities = (
    route1 = b_tt * Variable(:tt1) + b_tc * Variable(:tc1) + b_hw * Variable(:hw1) + b_ch * Variable(:ch1),
    route2 = b_tt * Variable(:tt2) + b_tc * Variable(:tc2) + b_hw * Variable(:hw2) + b_ch * Variable(:ch2)
)

# Availability (assumed all available for simplicity)
availability = (
    route1 = trues(nrow(df)),
    route2 = trues(nrow(df))
)

# Build and estimate the Mixed Logit model
using Random
Random.seed!(12345)
model = MixedLogitModel(alternatives; utilities=utilities, availability=availability, data=df, idvar=:ID, R=100, draw_scheme=:mlhs)
results = estimate(model, :choice)

# @show results

# Output results
summarize_results(results, file="output/MXL_routeChoice.xlsx")

# # Predict probabilities
# probs = predict(model)
# println("\nAverage predicted choice probabilities:")
# println(mean(probs, dims=1))
