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
#
# The two classes MUST start at different values. If every class-specific
# parameter starts at 0 the classes are identical at θ₀, and with π_1 = π_2 the
# gradients w.r.t. β_time_1 and β_time_2 are then *bitwise* identical while the
# gradient w.r.t. delta_1 is exactly 0 — so BFGS moves both classes by the same
# amount at every step and {class 1 = class 2} is an invariant subspace the
# optimizer can only leave on floating-point noise. Seeding the slopes apart
# (as LC2_routeChoice.jl does) breaks the symmetry by construction.

# Class 1
asc_car_1 = Parameter(:asc_car_1, value=0)
asc_train_1 = Parameter(:asc_train_1, value=0)
asc_sm_1 = Parameter(:asc_sm_1, value=0, fixed=true)

β_time_1 = Parameter(:β_time_1, value=-0.1)
β_cost_1 = Parameter(:β_cost_1, value=-0.5)


# Define utility functions
V1_1 = asc_train_1 + β_time_1 * Variable(:TRAIN_TT) + β_cost_1 * Variable(:TRAIN_CO)
V2_1 = asc_sm_1    + β_time_1 * Variable(:SM_TT)    + β_cost_1 * Variable(:SM_CO)
V3_1 = asc_car_1   + β_time_1 * Variable(:CAR_TT)   + β_cost_1 * Variable(:CAR_CO)

utilities_1 = (train = V1_1, sm = V2_1, car = V3_1)

# Class 2
asc_car_2 = Parameter(:asc_car_2, value=0)
asc_train_2 = Parameter(:asc_train_2, value=0)
asc_sm_2 = Parameter(:asc_sm_2, value=0, fixed=true)

β_time_2 = Parameter(:β_time_2, value=-0.05)
β_cost_2 = Parameter(:β_cost_2, value=-0.25)


# Define utility functions
V1_2 = asc_train_2 + β_time_2 * Variable(:TRAIN_TT) + β_cost_2 * Variable(:TRAIN_CO)
V2_2 = asc_sm_2    + β_time_2 * Variable(:SM_TT)    + β_cost_2 * Variable(:SM_CO)
V3_2 = asc_car_2   + β_time_2 * Variable(:CAR_TT)   + β_cost_2 * Variable(:CAR_CO)

utilities_2 = (train = V1_2, sm = V2_2, car = V3_2)

# Define the alternatives: name => code used in the CHOICE column. Bound once and
# passed to both classes, so the latent class model inherits it from them.
alternatives = (train = 1, sm = 2, car = 3)

# Load availability data
df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
df.CAR_AV_SP .= ifelse.(df.SP .!= 0, df.CAR_AV, 0)

availability = (
    train = df.TRAIN_AV_SP .== 1,
    sm    = df.SM_AV .== 1,
    car   = df.CAR_AV_SP .== 1
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
results = estimate(lc_model, :CHOICE)

# Output results
summarize_results(results, file="output/LC2_swissmetro.xlsx")
println('\n')