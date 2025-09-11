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

# Define variables and parameters per class

# Class 1
asc_car_1 = Parameter(:asc_car_1, value=0)
asc_train_1 = Parameter(:asc_train_1, value=0)
asc_sm_1 = Parameter(:asc_sm_1, value=0, fixed=true)

β_time_1 = Parameter(:β_time_1, value=0)
β_cost_1 = Parameter(:β_cost_1, value=0)


# Define utility functions
V1_1 = asc_train_1 + β_time_1 * Variable(:TRAIN_TT) + β_cost_1 * Variable(:TRAIN_CO)
V2_1 = asc_sm_1    + β_time_1 * Variable(:SM_TT)    + β_cost_1 * Variable(:SM_CO)
V3_1 = asc_car_1   + β_time_1 * Variable(:CAR_TT)   + β_cost_1 * Variable(:CAR_CO)

utilities_1 = [V1_1,V2_1,V3_1]

# Class 2
asc_car_2 = Parameter(:asc_car_2, value=0)
asc_train_2 = Parameter(:asc_train_2, value=0)
asc_sm_2 = Parameter(:asc_sm_2, value=0, fixed=true)

β_time_2 = Parameter(:β_time_2, value=0)
β_cost_2 = Parameter(:β_cost_2, value=0)


# Define utility functions
V1_2 = asc_train_2 + β_time_2 * Variable(:TRAIN_TT) + β_cost_2 * Variable(:TRAIN_CO)
V2_2 = asc_sm_2    + β_time_2 * Variable(:SM_TT)    + β_cost_2 * Variable(:SM_CO)
V3_2 = asc_car_2   + β_time_2 * Variable(:CAR_TT)   + β_cost_2 * Variable(:CAR_CO)

utilities_2 = [V1_2,V2_2,V3_2]

# Load availability data
df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
df.CAR_AV_SP .= ifelse.(df.SP .!= 0, df.CAR_AV, 0)

availability = [
    df.TRAIN_AV_SP .== 1,
    df.SM_AV .== 1,
    df.CAR_AV_SP .== 1
]

# Define Class parameters
delta_1 = Parameter(:delta_1, value = 0)
delta_2 = Parameter(:delta_2, value = 0, fixed=true)

prob_1 = exp(delta_1) / (exp(delta_1) + exp(delta_2))
prob_2 = exp(delta_2) / (exp(delta_1) + exp(delta_2))

# Create conditional probabilities
model_1 = LogitModel(utilities_1; data=df, availability=availability)
model_2 = LogitModel(utilities_2; data=df, availability=availability)

# Create unconditional probability
prob_indiv = prob_1 * model_1 + prob_2 * model_2

# Create model
lc_model = LatentClassModel(prob_indiv;data=df,idvar=:ID)
results = estimate(lc_model, :CHOICE)

# Output results
summarize_results(results)
println('\n')