using CSV, DataFrames, Statistics
using ChoiceModels

# Load dataset and filter SP observations only
df = CSV.read("../data/apollo_modeChoiceData.csv", DataFrame)
df = filter(:SP => x -> x == 1, df)

# Define variables and parameters
asc_car = Parameter(:asc_car, value=0, fixed=true)
asc_bus = Parameter(:asc_bus, value=0)
asc_air = Parameter(:asc_air, value=0)
asc_rail = Parameter(:asc_rail, value=0)

wtp_time_car = Parameter(:wtp_time_car, value=0)
wtp_time_bus = Parameter(:wtp_time_bus, value=0)
wtp_time_air = Parameter(:wtp_time_air, value=0)
wtp_time_rail = Parameter(:wtp_time_rail, value=0)

wtp_access = Parameter(:wtp_access, value=0)
wtp_no_frills = Parameter(:wtp_no_frills, value=0, fixed=true)
wtp_wifi = Parameter(:wtp_wifi, value=0)
wtp_food = Parameter(:wtp_food, value=0)

β_cost = Parameter(:β_cost, value=0)

# Define utility functions
V1 = asc_car                                         + β_cost * (wtp_time_car * Variable(:time_car)   + Variable(:cost_car))
V2 = asc_bus  +  β_cost * (wtp_time_bus * Variable(:time_bus)   + wtp_access * Variable(:access_bus)  + Variable(:cost_bus))      
V3 = asc_air  +  β_cost * (wtp_time_air * Variable(:time_air)   + wtp_access * Variable(:access_air)  + Variable(:cost_air)  + wtp_no_frills * (Variable(:service_air) == 1)  + wtp_wifi * (Variable(:service_air) == 2)   + wtp_food * (Variable(:service_air) == 3))    
V4 = asc_rail +  β_cost * (wtp_time_rail * Variable(:time_rail) + wtp_access * Variable(:access_rail) + Variable(:cost_rail) + wtp_no_frills * (Variable(:service_rail) == 1) + wtp_wifi * (Variable(:service_rail) == 2 ) + wtp_food * (Variable(:service_rail) == 3))    

utilities = [V1,V2,V3,V4]

# Load availability data
availability = [
    df.av_car .== 1,
    df.av_bus .== 1,
    df.av_air .== 1,
    df.av_rail .== 1
]

# Create model and estimate
model = LogitModel(utilities; data=df, availability=availability)
results = estimate(model, :choice)

# Output results
summarize_results(results, file="output/MNL_modeChoice_SP_WTP.xlsx")

# Predict
preds = predict(model,results)
println("\nAverage of Logit predictions")
println(mean(preds,dims=1))