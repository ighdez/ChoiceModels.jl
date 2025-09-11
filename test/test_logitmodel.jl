using Test
using ChoiceModels
using DataFrames

@testset "LogitModel.jl" begin

    @testset "Model creation and prediction" begin
        asc = Parameter(:asc_car, value=0.0)
        β_time = Parameter(:β_time, value=-0.1)
        V1 = asc + β_time * Variable(:time_car)
        V2 = asc + β_time * Variable(:time_bus)

        df = DataFrame(time_car = [10.0, 20.0], time_bus = [15.0, 5.0])
        availability = [trues(2), trues(2)]
        model = LogitModel([V1, V2]; data=df, availability=availability)

        results = estimate(model, [1, 2])
        probs = predict(model, results)

        @test size(probs) == (2, 2)
        @test all(x -> all(0.0 .<= x .<= 1.0), eachrow(probs))
        @test all(x -> abs(sum(x) - 1.0) < 1e-6, eachrow(probs))
    end

    # @testset "Log-likelihood evaluation" begin
    #     asc = Parameter(:asc_car, value=0.0)
    #     β_time = Parameter(:β_time, value=-0.1)
    #     V1 = asc + β_time * Variable(:time_car)
    #     V2 = asc + β_time * Variable(:time_bus)

    #     df = DataFrame(time_car = [10.0, 20.0], time_bus = [15.0, 5.0])
    #     availability = [trues(2), trues(2)]
    #     model = LogitModel([V1, V2]; data=df, availability=availability)

    #     ll = loglikelihood(model, [1, 2])
    #     @test isa(ll, Float64)
    #     @test ll < 0
    # end

end