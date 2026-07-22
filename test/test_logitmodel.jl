using Test
using ChoiceModels
using DataFrames

# Ground truth: for a 2-alternative logit the choice probability has the
# closed form  P1 = 1 / (1 + exp(V2 - V1)),  independent of how `logit_prob`
# builds its N×J matrix internally. We assert against that formula rather
# than snapshotting the implementation's output.
@testset "LogitModel.jl" begin

    # Shared tiny, hand-computable setup: 2 obs, 2 alternatives.
    asc = Parameter(:asc, value=0.0)
    β_time = Parameter(:β_time, value=-0.1)
    V1 = asc + β_time * Variable(:time1)
    V2 = β_time * Variable(:time2)

    df = DataFrame(time1 = [10.0, 20.0], time2 = [15.0, 5.0], CHOICE = [1, 2])
    params = Dict(:asc => 0.0, :β_time => -0.1)
    availability = [trues(2), trues(2)]

    @testset "logit_prob matches closed-form probabilities" begin
        probs = ChoiceModels.logit_prob([V1, V2], df, availability, params)

        # Return shape is an N×J matrix.
        @test size(probs) == (2, 2)

        # Closed-form expected values, derived independently of logit_prob.
        for n in 1:2
            v1 = params[:asc] + params[:β_time] * df.time1[n]
            v2 =                params[:β_time] * df.time2[n]
            p1 = 1 / (1 + exp(v2 - v1))
            @test probs[n, 1] ≈ p1
            @test probs[n, 2] ≈ 1 - p1
        end

        # Each row is a valid probability distribution.
        @test all(0.0 .<= probs .<= 1.0)
        @test all(abs.(sum(probs, dims=2) .- 1.0) .< 1e-12)
    end

    @testset "availability masks unavailable alternatives" begin
        # Make alternative 2 unavailable for observation 1.
        avail = [trues(2), [false, true]]
        probs = ChoiceModels.logit_prob([V1, V2], df, avail, params)

        @test probs[1, 2] == 0.0        # masked alternative gets zero probability
        @test probs[1, 1] ≈ 1.0         # all mass on the only available alternative
        @test abs(sum(probs[2, :]) - 1.0) < 1e-12  # obs 2 unaffected
    end

    @testset "estimate and predict" begin
        model = LogitModel([V1, V2]; data=df, availability=availability)
        results = estimate(model, :CHOICE; verbose=false)

        @test results.converged
        @test results.N == 2
        @test haskey(results.parameters, :asc)
        @test haskey(results.parameters, :β_time)

        preds = predict(model, results)
        @test size(preds) == (2, 2)
        @test all(0.0 .<= preds .<= 1.0)
        @test all(abs.(sum(preds, dims=2) .- 1.0) .< 1e-10)
    end

end
