using Test
using ChoiceModels
using DataFrames
using ForwardDiff

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

        # A numeric exponent still builds the unary node — existing specs unchanged.
        @test Variable(:x)^2 isa ChoiceModels.DCMPower
    end

    # A symbolic exponent (an estimated Parameter) is rewritten to
    # exp(exponent * log(base)) rather than given a node of its own; see the `^`
    # overload in Expressions.jl. These pin the properties that rewrite has to have.
    @testset "power with a symbolic exponent" begin
        p = Parameter(:p, value=0.0)
        b = Parameter(:b, value=0.0)
        df = DataFrame(x = [2.0, 3.0, 0.5])
        params = Dict(:p => 1.7, :b => 4.0)

        # Values match the closed form for a variable base, a parameter base,
        # and a compound base.
        @test evaluate(Variable(:x)^p, df, params) ≈ df.x .^ 1.7
        @test evaluate(b^p, df, params) ≈ fill(4.0^1.7, 3)
        @test evaluate((b * Variable(:x))^p, df, params) ≈ (4.0 .* df.x) .^ 1.7
        # Numeric base, symbolic exponent.
        @test evaluate(2^p, df, params) ≈ fill(2.0^1.7, 3)
        # A symbolic exponent that happens to sit at an integer agrees with the
        # numeric-exponent node, i.e. the two paths do not disagree.
        @test evaluate(Variable(:x)^p, df, Dict(:p => 2.0)) ≈ evaluate(Variable(:x)^2, df, params)

        # THE thing a naive implementation breaks: the exponent's parameters must
        # be discoverable, or estimation silently never moves them.
        names = [q.name for q in ChoiceModels.collect_parameters((b * Variable(:x))^p)]
        @test :p in names
        @test :b in names
        @test :x in ChoiceModels.collect_variables([Variable(:x)^p])

        # The derivative with respect to the EXPONENT must be right, not just the
        # value: d/dp x^p = x^p · log x. Checked against ForwardDiff.
        for x0 in df.x
            g = ForwardDiff.derivative(t -> only(evaluate(Variable(:x)^p, DataFrame(x=[x0]), Dict(:p => t))), 1.7)
            @test g ≈ x0^1.7 * log(x0)
        end

        # Both evaluation paths must work — the draws path is separate code, and
        # a node covered in one and missed in the other is a recurring bug here.
        draws = Dict(:d => reshape(collect(0.1:0.1:0.6), 3, 2))
        got = evaluate((Variable(:x) + Draw(:d))^p, df, params, draws)
        @test size(got) == (3, 2)
        @test got ≈ (df.x .+ draws[:d]) .^ 1.7

        # A negative base is undefined in the reals and must fail loudly rather
        # than silently producing NaN.
        @test_throws DomainError evaluate(Variable(:x)^p, DataFrame(x=[-2.0]), params)
    end

    # The `collect_*` walkers share one `_children`/`_walk` recursion. `DCMEqual`
    # is the node that makes that non-trivial: it is a `DCMBinary`, but its
    # `right` is a plain `Real`, so the traversal is handed a bare number where
    # every other binary node hands it an expression. The original `isa`-ladder
    # ignored it by falling off the end of its `elseif` chain; a `_children`
    # recursion without an `Any` leaf case turns every `Variable(:x) == 3` in a
    # utility into a `MethodError` from deep inside `collect_parameters`.
    #
    # This is not hypothetical: it was introduced during the walker refactor and
    # the entire suite passed with it in place — only three examples caught it.
    @testset "walkers handle nodes with non-expression children" begin
        b = Parameter(:b, value=0.0)
        df = DataFrame(x = [1.0, 3.0], y = [2.0, 2.0])
        params = Dict(:b => 2.0)

        eq = b * (Variable(:x) == 3)
        @test eq isa ChoiceModels.DCMMult
        @test ChoiceModels.DCMEqual <: ChoiceModels.DCMBinary   # why it is a trap

        # The walk must survive the numeric child and still find what is past it.
        @test [p.name for p in ChoiceModels.collect_parameters(eq)] == [:b]
        @test ChoiceModels.collect_variables([eq]) == [:x]
        @test isempty(ChoiceModels.collect_draws([eq]))

        # ... and the node still evaluates on both paths, indicator-style.
        @test evaluate(eq, df, params) ≈ [0.0, 2.0]
        draws = Dict(:d => zeros(2, 2))
        @test evaluate(b * (Variable(:x) == 3) + Draw(:d), df, params, draws) ≈ [0.0 0.0; 2.0 2.0]

        # A comparison buried under other operators is reached too.
        deep = exp(b * (Variable(:y) == 2)) + Variable(:x)
        @test [p.name for p in ChoiceModels.collect_parameters(deep)] == [:b]
        @test sort(ChoiceModels.collect_variables([deep])) == [:x, :y]
    end

end
