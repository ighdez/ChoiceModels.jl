using Test
using ChoiceModels
using DataFrames
using Random

# The `alternatives` NamedTuple is what ties an alternative's name to the code it
# carries in the choice column. Before it existed, the choice column had to be
# consecutive integers `1:J` listed in exactly the order the utilities were given,
# and nothing checked it — a non-consecutive coding gave a `BoundsError` deep in
# the likelihood, and a differently-ordered one gave a silently wrong model.
#
# The two properties worth pinning are therefore: (1) codes are translated, so a
# `(car=1, bus=4, rail=7)` dataset gives exactly the same fit as the hand-recoded
# `1,2,3` one; and (2) names, not positions, decide what goes where.
@testset "alternatives.jl" begin

    # Shared fixture: 3 alternatives, deliberately non-consecutive codes, and a
    # choice column that never separates (so no singularity warnings).
    Random.seed!(20260730)
    N = 300
    codes = [1, 4, 7]

    df = DataFrame(x1=randn(N), x2=randn(N), x3=randn(N))
    b_true, a2_true, a3_true = -0.8, 0.4, -0.2
    chosen = Vector{Int}(undef, N)
    for n in 1:N
        v = [b_true * df.x1[n], b_true * df.x2[n] + a2_true, b_true * df.x3[n] + a3_true]
        p = exp.(v) ./ sum(exp.(v))
        r, j, acc = rand(), 1, p[1]
        while r > acc && j < 3
            j += 1
            acc += p[j]
        end
        chosen[n] = j
    end
    df.choice = codes[chosen]          # analyst's coding: 1, 4, 7
    df.position = chosen               # the same choices already recoded to 1:3
    av = trues(N)

    β  = Parameter(:b,  value=0.0)
    a2 = Parameter(:a2, value=0.0)
    a3 = Parameter(:a3, value=0.0)
    V = (car = β * Variable(:x1), bus = β * Variable(:x2) + a2, rail = β * Variable(:x3) + a3)

    named_model() = LogitModel((car=1, bus=4, rail=7);
                               utilities = V,
                               availability = (car=av, bus=av, rail=av),
                               data = df)

    @testset "codes are translated, not assumed" begin
        # Same model, two encodings of the same choices: the estimates must agree
        # exactly, not approximately — the likelihood sees identical numbers.
        named = estimate(named_model(), :choice; verbose=false)
        plain = estimate(LogitModel([1, 2, 3];
                                    utilities = [V.car, V.bus, V.rail],
                                    availability = [av, av, av],
                                    data = df), :position; verbose=false)

        @test named.loglikelihood == plain.loglikelihood
        for k in keys(named.parameters)
            @test named.parameters[k] == plain.parameters[k]
        end
    end

    @testset "utilities and availability are matched by name, not position" begin
        # Every argument written in a different order from `alternatives`, and from
        # each other. If anything were matched positionally this would fit a
        # different model.
        shuffled = LogitModel((rail=7, car=1, bus=4);
                              utilities = (bus=V.bus, rail=V.rail, car=V.car),
                              availability = (car=av, rail=av, bus=av),
                              data = df)
        reference = named_model()

        # Canonical order comes from `alternatives`, so this model's columns are
        # rail, car, bus — a permutation of the reference's car, bus, rail.
        @test collect(keys(shuffled.alternatives)) == [:rail, :car, :bus]
        @test [shuffled.alternatives[k] for k in keys(reference.alternatives)] ==
              collect(values(reference.alternatives))

        r_shuffled = estimate(shuffled, :choice; verbose=false)
        r_reference = estimate(reference, :choice; verbose=false)

        # Same model, so the same fit — the ordering only permutes the columns.
        @test r_shuffled.loglikelihood ≈ r_reference.loglikelihood atol=1e-9
        for k in keys(r_reference.parameters)
            @test r_shuffled.parameters[k] ≈ r_reference.parameters[k] atol=1e-6
        end

        preds_ref = predict(reference, r_reference)
        preds_shuf = predict(shuffled, r_shuffled)
        # Column j of one is the column of the other with the same *name*.
        for (j, name) in enumerate(keys(shuffled.alternatives))
            k = findfirst(==(name), collect(keys(reference.alternatives)))
            @test preds_shuf[:, j] ≈ preds_ref[:, k] atol=1e-6
        end
    end

    @testset "unnamed alternatives get alt1, alt2, … labels" begin
        model = LogitModel([1, 4, 7];
                           utilities = [V.car, V.bus, V.rail],
                           availability = [av, av, av],
                           data = df)
        @test model.alternatives == (alt1=1, alt2=4, alt3=7)

        # The vector form is an escape hatch for programmatic use, not a different
        # model: it recodes exactly as the named form does.
        @test estimate(model, :choice; verbose=false).loglikelihood ≈
              estimate(named_model(), :choice; verbose=false).loglikelihood atol=1e-9
    end

    @testset "a choice code no alternative claims is an error" begin
        # This is the case that used to be a BoundsError inside the likelihood
        # (code > J) or, worse, a silently wrong model (code within 1:J but
        # belonging to another alternative).
        model = LogitModel((car=1, bus=4, rail=99);
                           utilities = V,
                           availability = (car=av, bus=av, rail=av),
                           data = df)
        err = try
            estimate(model, :choice; verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("not the code of any alternative", err.msg)
        @test occursin("car => 1, bus => 4, rail => 99", err.msg)
    end

    @testset "alternatives are validated at construction" begin
        # Duplicated code: two alternatives could never be told apart in the data.
        @test_throws ErrorException LogitModel((car=1, bus=4, rail=4);
                                               utilities = V,
                                               availability = (car=av, bus=av, rail=av),
                                               data = df)

        # Key sets must match, in both directions.
        @test_throws ErrorException LogitModel((car=1, bus=4, rail=7);
                                               utilities = (car=V.car, bus=V.bus),
                                               availability = (car=av, bus=av, rail=av),
                                               data = df)
        @test_throws ErrorException LogitModel((car=1, bus=4, rail=7);
                                               utilities = V,
                                               availability = (car=av, bus=av, rail=av, plane=av),
                                               data = df)

        # Named alternatives with positional utilities (and vice versa) is the
        # ambiguity the API exists to remove.
        @test_throws ErrorException LogitModel((car=1, bus=4, rail=7);
                                               utilities = [V.car, V.bus, V.rail],
                                               availability = (car=av, bus=av, rail=av),
                                               data = df)
        @test_throws ErrorException LogitModel([1, 4, 7];
                                               utilities = V,
                                               availability = [av, av, av],
                                               data = df)

        # Wrong number of positional utilities.
        @test_throws ErrorException LogitModel([1, 4, 7];
                                               utilities = [V.car, V.bus],
                                               availability = [av, av, av],
                                               data = df)

        # Availability must be per-observation booleans.
        @test_throws ErrorException LogitModel((car=1, bus=4, rail=7);
                                               utilities = V,
                                               availability = (car=av, bus=av, rail=trues(N - 1)),
                                               data = df)

        # Codes must be integers: a float code cannot match a choice column entry.
        @test_throws ErrorException LogitModel((car=1.5, bus=4, rail=7);
                                               utilities = V,
                                               availability = (car=av, bus=av, rail=av),
                                               data = df)
    end

    @testset "MixedLogit carries the same contract" begin
        df_mxl = copy(df)
        df_mxl.ID = repeat(1:60, inner=5)

        mu    = Parameter(:mu,    value=-0.5)
        sigma = Parameter(:sigma, value=0.2)
        b = mu + sigma * Draw(:d)
        U = (car = b * Variable(:x1), bus = b * Variable(:x2), rail = b * Variable(:x3))

        Random.seed!(7)
        named = MixedLogitModel((car=1, bus=4, rail=7);
                                utilities = U,
                                availability = (rail=av, car=av, bus=av),
                                data = df_mxl, idvar=:ID, R=20, draw_scheme=:halton)
        Random.seed!(7)
        plain = MixedLogitModel([1, 2, 3];
                                utilities = [U.car, U.bus, U.rail],
                                availability = [av, av, av],
                                data = df_mxl, idvar=:ID, R=20, draw_scheme=:halton)

        @test named.alternatives == (car=1, bus=4, rail=7)
        @test estimate(named, :choice; verbose=false).loglikelihood ≈
              estimate(plain, :position; verbose=false).loglikelihood atol=1e-8
    end

    @testset "LatentClass inherits its alternatives from the class models" begin
        df_lc = copy(df)
        df_lc.ID = repeat(1:60, inner=5)
        alts = (car=1, bus=4, rail=7)
        avail = (car=av, bus=av, rail=av)

        b1 = Parameter(:b1, value=-1.0)
        b2 = Parameter(:b2, value=-0.4)
        d1 = Parameter(:d1, value=0.3)
        d2 = Parameter(:d2, value=0.0, fixed=true)
        π1 = exp(d1) / (exp(d1) + exp(d2))
        π2 = exp(d2) / (exp(d1) + exp(d2))

        utils(b) = (car = b * Variable(:x1), bus = b * Variable(:x2), rail = b * Variable(:x3))
        m1 = LogitModel(alts; utilities=utils(b1), availability=avail, data=df_lc)
        m2 = LogitModel(alts; utilities=utils(b2), availability=avail, data=df_lc)

        lc = LatentClassModel(π1 * m1 + π2 * m2; data=df_lc, idvar=:ID)
        @test lc.alternatives == alts

        # An explicitly supplied set is checked against the classes rather than
        # silently overriding them.
        @test LatentClassModel(π1 * m1 + π2 * m2; alternatives=alts, data=df_lc, idvar=:ID) isa LatentClassModel
        @test_throws ErrorException LatentClassModel(π1 * m1 + π2 * m2;
                                                     alternatives=(car=1, bus=2, rail=3),
                                                     data=df_lc, idvar=:ID)

        # Classes describing different alternatives cannot be mixed at all.
        m_other = LogitModel((car=1, bus=4, tram=9);
                             utilities = (car=utils(b2).car, bus=utils(b2).bus, tram=utils(b2).rail),
                             availability = (car=av, bus=av, tram=av), data=df_lc)
        @test_throws ErrorException LatentClassModel(π1 * m1 + π2 * m_other; data=df_lc, idvar=:ID)

        # Same alternatives, written in a different order: the mixture would add
        # mismatched columns, so the constructor reconciles the orders instead.
        m2_shuffled = LogitModel((rail=7, car=1, bus=4);
                                 utilities = (rail=utils(b2).rail, car=utils(b2).car, bus=utils(b2).bus),
                                 availability = avail, data=df_lc)
        lc_shuffled = LatentClassModel(π1 * m1 + π2 * m2_shuffled; data=df_lc, idvar=:ID)
        @test lc_shuffled.alternatives == alts

        pars = Dict(:b1 => -1.0, :b2 => -0.4, :d1 => 0.3, :d2 => 0.0)
        Y = zeros(Bool, nrow(df_lc), 3)
        for n in 1:nrow(df_lc)
            Y[n, df_lc.position[n]] = true
        end
        # Reconciled, not merely accepted: the reordered specification is the same
        # model, so it must give the same likelihood.
        @test loglikelihood(lc_shuffled, Y; parameters=pars) ≈
              loglikelihood(lc, Y; parameters=pars) rtol=1e-12

        # `estimate` recodes the choice column on the latent-class path too. The
        # check uses a single-class LC, which is exactly an MNL: that makes the
        # comparison against a hand-recoded positional model exact, and keeps the
        # fixture identified — fitting two classes to single-class data leaves the
        # class weight unidentified and would warn on every run.
        unit = Parameter(:unit, value=1.0, fixed=true)
        one_class = LatentClassModel(unit * m1; data=df_lc, idvar=:ID)
        r_lc = estimate(one_class, :choice; verbose=false)

        r_mnl = estimate(LogitModel([1, 2, 3];
                                    utilities = collect(utils(b1)),
                                    availability = [av, av, av],
                                    data = df_lc), :position; verbose=false)

        @test r_lc.N == nrow(df_lc)
        @test r_lc.loglikelihood ≈ r_mnl.loglikelihood atol=1e-6
        @test r_lc.parameters[:b1] ≈ r_mnl.parameters[:b1] atol=1e-6
    end

end
