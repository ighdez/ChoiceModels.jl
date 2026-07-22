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

    @testset "exp, log and division" begin
        a = Parameter(:a, value=0.0)
        b = Parameter(:b, value=0.0)
        df = DataFrame(x = [2.0, 4.0])
        params = Dict(:a => 0.5, :b => 2.0)

        # exp(a * x) with a = 0.5  →  exp(1.0), exp(2.0)
        @test evaluate(exp(a * Variable(:x)), df, params) ≈ exp.([1.0, 2.0])
        # log(x)  →  log(2), log(4)
        @test evaluate(log(Variable(:x)), df, params) ≈ log.([2.0, 4.0])
        # a / b  →  0.25 (broadcast over rows)
        @test evaluate(a / b, df, params) ≈ [0.25, 0.25]
    end

    @testset "power (^)" begin
        b = Parameter(:b, value=0.0)
        df = DataFrame(x = [2.0, 3.0])
        params = Dict(:b => 3.0)

        # x ^ 2  →  4, 9
        @test evaluate(Variable(:x)^2, df, params) ≈ [4.0, 9.0]
        # b ^ 2  →  9 (broadcast over rows)
        @test evaluate(b^2, df, params) ≈ [9.0, 9.0]
        # (b * x) ^ 2  with b = 3  →  36, 81
        @test evaluate((b * Variable(:x))^2, df, params) ≈ [36.0, 81.0]
        # collect_parameters must traverse the power's base
        @test :b in [p.name for p in ChoiceModels.collect_parameters(b^2)]
    end

end
