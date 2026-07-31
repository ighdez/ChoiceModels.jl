using Test
using ChoiceModels
using DataFrames
using Random
using Statistics

# Ground truth for the simulated log-likelihood is recomputed here from scratch,
# always in the numerically stable logistic form  p_chosen = 1/(1+exp(V_other-V_chosen)).
# It must NOT be written as exp(V)/Σexp(V): the spec below deliberately produces
# utilities around -1e3 (lognormal coefficient × large variable), for which every
# exp(V) underflows to 0 — that underflow *is* the bug this test guards against,
# so a brute force written the same way would reproduce it and pass vacuously.
function stable_log_p_chosen(v_chosen, v_other)
    d = v_other - v_chosen
    return d > 0 ? -d - log1p(exp(-d)) : -log1p(exp(d))
end

function brute_force_simulated_ll(model, params, choices)
    dx = model.draws[:dx]
    dy = model.draws[:dy]
    N, R = size(dx)

    x1, x2 = model.data.x1, model.data.x2
    y1, y2 = model.data.y1, model.data.y2
    id_map, id = model.id
    I = length(id_map)

    log_indiv = zeros(I, R)
    for r in 1:R, n in 1:N
        bx = -exp(params[:mu_x] + params[:sigma_x] * dx[n, r])   # lognormal (negative)
        by = params[:mu_y] + params[:sigma_y] * dy[n, r]         # normal
        v1 = bx * x1[n] + by * y1[n]
        v2 = bx * x2[n] + by * y2[n]
        vc, vo = choices[n] == 1 ? (v1, v2) : (v2, v1)
        log_indiv[id_map[id[n]], r] += stable_log_p_chosen(vc, vo)
    end

    # Simulated LL: log of the draw-average of each individual's sequence probability.
    return sum(log(mean(exp.(log_indiv[i, :]))) for i in 1:I)
end

@testset "MixedLogit.jl" begin

    @testset "simulated log-likelihood matches an independent brute force" begin
        Random.seed!(20260727)

        I_ind, T_obs = 8, 3
        ids = repeat(1:I_ind, inner=T_obs)
        N = length(ids)

        # x1 ≈ x2 and both large: utilities are hugely negative (so exp(V)
        # underflows) while their *difference* stays moderate, keeping the true
        # choice probabilities well away from the 1e-30 floor in `loglikelihood`.
        x1 = 50.0 .+ 10.0 .* rand(N)
        x2 = x1 .+ 0.2 .* (rand(N) .- 0.5)
        y1 = rand(N)
        y2 = rand(N)
        choices = rand(1:2, N)

        df = DataFrame(ID=ids, x1=x1, x2=x2, y1=y1, y2=y2, choice=choices)

        mu_x    = Parameter(:mu_x,    value=2.0)
        sigma_x = Parameter(:sigma_x, value=0.5)
        mu_y    = Parameter(:mu_y,    value=-0.5)
        sigma_y = Parameter(:sigma_y, value=0.3)

        b_x = -exp(mu_x + sigma_x * Draw(:dx))
        b_y = mu_y + sigma_y * Draw(:dy)

        V1 = b_x * Variable(:x1) + b_y * Variable(:y1)
        V2 = b_x * Variable(:x2) + b_y * Variable(:y2)

        R = 40
        model = MixedLogitModel([1, 2]; utilities=[V1, V2],
                                availability=[trues(N), trues(N)],
                                data=df, idvar=:ID,
                                R=R, draw_scheme=:halton)

        params = Dict(:mu_x => 2.0, :sigma_x => 0.5, :mu_y => -0.5, :sigma_y => 0.3)

        # Sanity: the spec really does drive raw exp(V) to underflow, so a
        # non-stable softmax would be wrong here.
        utils1 = evaluate(V1, df, params, model.draws)
        @test minimum(utils1) < -700
        @test any(iszero, exp.(utils1))

        Y = zeros(Bool, N, 2, R)
        for n in 1:N
            Y[n, choices[n], :] .= true
        end

        ll = sum(loglikelihood(model, Y; parameters=params))
        expected = brute_force_simulated_ll(model, params, choices)

        @test isfinite(ll)
        @test ll ≈ expected rtol=1e-8
    end

    @testset "logit_prob draws path: shape, normalization, availability" begin
        Random.seed!(11)

        N = 6
        df = DataFrame(ID=1:N, x1=rand(N), x2=rand(N), y1=rand(N), y2=rand(N))

        b = Parameter(:mu, value=-1.0) + Parameter(:sigma, value=0.5) * Draw(:d)
        V1 = b * Variable(:x1)
        V2 = b * Variable(:x2)

        R = 5
        avail = [trues(N), [i != 1 for i in 1:N]]   # alt 2 unavailable for obs 1
        model = MixedLogitModel([1, 2]; utilities=[V1, V2], availability=avail,
                                data=df, idvar=:ID, R=R, draw_scheme=:normal)

        params = Dict(:mu => -1.0, :sigma => 0.5)
        probs = ChoiceModels.logit_prob(model.utilities, df, params, avail, model.draws)

        @test size(probs) == (N, 2, R)
        @test all(0.0 .<= probs .<= 1.0)
        @test all(abs.(sum(probs, dims=2) .- 1.0) .< 1e-10)
        @test all(probs[1, 2, :] .== 0.0)        # masked alternative gets zero mass
        @test all(probs[1, 1, :] .≈ 1.0)         # all mass on the only available alt
    end

    @testset "estimate runs and reports finite standard errors" begin
        Random.seed!(4242)

        # Simple spec on data simulated from known parameters: one normal random
        # coefficient plus a fixed ASC.
        I_ind, T_obs = 150, 6
        ids = repeat(1:I_ind, inner=T_obs)
        N = length(ids)

        x1 = 2.0 .* rand(N)
        x2 = 2.0 .* rand(N)

        true_asc, true_mu, true_sigma = 0.4, -1.0, 0.5
        b_ind = true_mu .+ true_sigma .* randn(I_ind)          # one taste per individual
        choices = Vector{Int}(undef, N)
        for n in 1:N
            v1 = true_asc + b_ind[ids[n]] * x1[n]
            v2 = b_ind[ids[n]] * x2[n]
            p1 = 1 / (1 + exp(v2 - v1))
            choices[n] = rand() < p1 ? 1 : 2
        end

        df = DataFrame(ID=ids, x1=x1, x2=x2, choice=choices)

        asc   = Parameter(:asc,   value=0.0)
        mu    = Parameter(:mu,    value=-0.5)
        sigma = Parameter(:sigma, value=0.2)

        b = mu + sigma * Draw(:d)
        V1 = asc + b * Variable(:x1)
        V2 = b * Variable(:x2)

        model = MixedLogitModel([1, 2]; utilities=[V1, V2],
                                availability=[trues(N), trues(N)],
                                data=df, idvar=:ID,
                                R=100, draw_scheme=:halton)

        results = estimate(model, :choice; verbose=false)

        @test results.converged
        @test results.N == N
        @test isfinite(results.loglikelihood)
        @test results.loglikelihood > results.null_loglikelihood

        for name in (:asc, :mu, :sigma)
            @test haskey(results.parameters, name)
            @test isfinite(results.parameters[name])
            @test isfinite(results.std_errors[name])       # PD Hessian ⇒ no NaN SEs
            @test isfinite(results.rob_std_errors[name])
            @test results.std_errors[name] > 0
        end

        # Loose recovery check on the well-identified parameters. `sigma` is NOT
        # asserted against its true value: the spread of a mixing distribution is
        # weakly identified at this sample size (σ̂ swings between ~0.2 and ~1.0
        # across seeds/sample sizes for a true 0.5), so a tight bound would be
        # flaky. We only require it to stay finite and in a sane range.
        @test results.parameters[:mu] < 0
        @test abs(results.parameters[:mu] - true_mu) < 0.4
        @test abs(results.parameters[:asc] - true_asc) < 0.4
        @test 0.0 < abs(results.parameters[:sigma]) < 2.0

        preds = predict(model, results)
        @test size(preds) == (N, 2, 100)
        @test all(abs.(sum(preds, dims=2) .- 1.0) .< 1e-10)
    end

    # An individual's contribution is the probability of their whole choice
    # SEQUENCE, a product over their observations, so it falls off exponentially
    # in the panel length. The simulated likelihood therefore has to average over
    # draws in LOGS. The original code materialized `exp.(log_indiv)` and averaged
    # that, so once the sequence probability dropped below the `max(·, 1e-30)`
    # failsafe every individual pinned to log(1e-30) = -69.08 regardless of the
    # parameters — a plausible-looking number, silently independent of the data.
    #
    # Measured on this fixture before the fix: correct at 10 observations per
    # individual, but -138.1551 (= 2 × the floor) at 200, 600 AND 1200, where the
    # true values are -280.56 / -836.19 / -1668.50. The threshold is
    # `T·|log p| > 69`, i.e. roughly 150-250 observations per individual — well
    # inside real panel data, and far below the ~745 at which Float64 itself
    # underflows. Same failure mode as the softmax-underflow bug: a failsafe
    # converting an underflow into a number rather than an error.
    #
    # The reference is the LatentClassModel panel path, which has always worked in
    # logs. It is an independent implementation, not a rearrangement of this one.
    @testset "long panels do not hit the probability floor" begin
        floor_ll = log(1e-30)   # what a clamped contribution would report

        for T_obs in (10, 200, 600)
            Random.seed!(3)
            I_ind = 2
            ids = repeat(1:I_ind, inner=T_obs)
            N_obs = length(ids)
            df_lp = DataFrame(ID=ids, x1=randn(N_obs), x2=randn(N_obs),
                              choice=rand(1:2, N_obs))
            Y2 = zeros(Bool, N_obs, 2)
            Y3 = zeros(Bool, N_obs, 2, 6)
            for n in 1:N_obs
                Y2[n, df_lp.choice[n]] = true
                Y3[n, df_lp.choice[n], :] .= true
            end

            alts_lp = (a=1, b=2)
            mu_lp = Parameter(:mu_lp, value=-0.3)
            sg_lp = Parameter(:sg_lp, value=0.4)
            utils = (a=(mu_lp + sg_lp*Draw(:z_lp)) * Variable(:x1),
                     b=(mu_lp + sg_lp*Draw(:z_lp)) * Variable(:x2))
            m_lp = MixedLogitModel(alts_lp; utilities=utils, data=df_lp,
                                   idvar=:ID, R=6, draw_scheme=:halton)

            # Compare against the latent-class panel path holding the SAME draws
            # (the latent class regenerates them), at a single class of weight 1.
            unit = Parameter(:unit_lp, value=1.0, fixed=true)
            lc_lp = LatentClassModel(unit * m_lp; data=df_lp, idvar=:ID)
            same_draws = ChoiceModels._lc_classes(lc_lp.expr)[1][2]

            p = Dict(:mu_lp=>-0.3, :sg_lp=>0.4, :unit_lp=>1.0)
            mxl_ll = loglikelihood(same_draws, Y3; parameters=p)
            lc_ll  = loglikelihood(lc_lp, Y2; parameters=p)

            @test sum(mxl_ll) ≈ sum(lc_ll)

            # No contribution may be sitting exactly on the floor.
            @test all(abs.(mxl_ll .- floor_ll) .> 1e-8)

            if T_obs >= 200
                # THE discriminating assertion. A long enough sequence has a
                # probability genuinely below 1e-30, so its correct log-likelihood
                # contribution is BELOW log(1e-30) — which clamping can never
                # produce, since the clamp is a floor on the probability and
                # therefore a floor on the contribution. Under the bug these came
                # out at exactly -69.08 each, flat in T; here they must keep
                # falling as the panel lengthens.
                @test all(mxl_ll .< floor_ll)
                @test sum(mxl_ll) < I_ind * floor_ll
            else
                @test all(mxl_ll .> floor_ll)   # short panel: nowhere near it
            end
        end
    end

end
