using CSV, DataFrames, Statistics
using ChoiceModels

# Load Swiss route choice dataset
df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)
df = sort(df, :ID)

# Scale
# df.tt1 ./= 60
# df.tt2 ./= 60
# df.hw1 ./= 60
# df.hw2 ./= 60
# df.tc1 ./= 10
# df.tc2 ./= 10

# Define parameters
asc_1     = Parameter(:asc_1, value=-0.1)
asc_2     = Parameter(:asc_2, value=0, fixed=true)

b_tt_1     = Parameter(:b_tt_1, value=-0.1)
b_tt_2     = Parameter(:b_tt_2, value=-0.05)

b_tc_1     = Parameter(:b_tc_1, value=-0.5)
b_tc_2     = Parameter(:b_tc_2, value=-0.25)

b_hw_1     = Parameter(:b_hw_1, value=-0.1)
b_hw_2     = Parameter(:b_hw_2, value=-0.05)

b_ch_1     = Parameter(:b_ch_1, value=-1)
b_ch_2     = Parameter(:b_ch_2, value=-0.5)

# Define utility functions (alternatives 1, 2)
V1_1 = asc_1 + b_tt_1 * Variable(:tt1) + b_tc_1 * Variable(:tc1) + b_hw_1 * Variable(:hw1) + b_ch_1 * Variable(:ch1)
V2_1 = asc_2 + b_tt_1 * Variable(:tt2) + b_tc_1 * Variable(:tc2) + b_hw_1 * Variable(:hw2) + b_ch_1 * Variable(:ch2)

utilities_1 = (route1 = V1_1, route2 = V2_1)

V1_2 = asc_1 + b_tt_2 * Variable(:tt1) + b_tc_2 * Variable(:tc1) + b_hw_2 * Variable(:hw1) + b_ch_2 * Variable(:ch1)
V2_2 = asc_2 + b_tt_2 * Variable(:tt2) + b_tc_2 * Variable(:tc2) + b_hw_2 * Variable(:hw2) + b_ch_2 * Variable(:ch2)

utilities_2 = (route1 = V1_2, route2 = V2_2)

# Define the alternatives: name => code used in the choice column. Bound once and
# passed to both classes, so the latent class model inherits it from them.
alternatives = (route1 = 1, route2 = 2)

# Availability (assumed all available for simplicity)
availability = (
    route1 = trues(nrow(df)),
    route2 = trues(nrow(df))
)

# Define Class parameters
delta_1 = Parameter(:delta_1, value = 0)
delta_2 = Parameter(:delta_2, value = 0, fixed=true)

prob_1 = exp(delta_1) / (exp(delta_1) + exp(delta_2))
prob_2 = exp(delta_2) / (exp(delta_1) + exp(delta_2))

# Create conditional probabilities
model_1 = LogitModel(alternatives; utilities=utilities_1, availability=availability, data=df)
model_2 = LogitModel(alternatives; utilities=utilities_2, availability=availability, data=df)

# Create unconditional probability
prob_indiv = prob_1 * model_1 + prob_2 * model_2

# Create model
lc_model = LatentClassModel(prob_indiv;data=df,idvar=:ID)
results = estimate(lc_model, :choice)

# @show results

# Output results
summarize_results(results, file="output/LC2_routeChoice.xlsx")

# # Predict probabilities
# probs = predict(model)
# println("\nAverage predicted choice probabilities:")
# println(mean(probs, dims=1))
