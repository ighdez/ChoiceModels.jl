using Test
using ChoiceModels
using DataFrames

@testset "Expressions.jl" begin

    @testset "Parameter and Variable construction" begin
        β_cost = Parameter(:β_cost, value=-1.5)
        time = Variable(:time)

        @test β_cost.name == :β_cost
        @test β_cost.value == -1.5
        @test time.name == :time
    end

    @testset "Evaluation of expressions" begin
        asc = Parameter(:asc, value=0.5)
        β_time = Parameter(:β_time, value=-0.2)
        V = asc + β_time * Variable(:time)

        df = DataFrame(time = [10.0, 15.0, 20.0])
        params = Dict(:asc => 0.5, :β_time => -0.2)

        result = evaluate(V, df, params)
        @test result ≈ [0.5 - 2.0, 0.5 - 3.0, 0.5 - 4.0]
    end

    @testset "Logit probabilities" begin
        asc = Parameter(:asc, value=0.0)
        β_time = Parameter(:β_time, value=-0.1)
        V1 = asc + β_time * Variable(:time1)
        V2 = asc + β_time * Variable(:time2)

        df = DataFrame(time1 = [10.0, 20.0], time2 = [15.0, 5.0])
        params = Dict(:asc => 0.0, :β_time => -0.1)
        availability = [trues(2), trues(2)]

        probs = logit_prob([V1, V2], df, params, availability)
        @test length(probs) == 2
        @test all(length.(probs) .== 2)
        @test all(x -> all(0 .<= x .<= 1), probs)
    end

end
