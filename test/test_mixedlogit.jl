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
        model = MixedLogitModel([V1, V2]; data=df, idvar=:ID,
                                availability=[trues(N), trues(N)],
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
        model = MixedLogitModel([V1, V2]; data=df, idvar=:ID,
                                availability=avail, R=R, draw_scheme=:normal)

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

        model = MixedLogitModel([V1, V2]; data=df, idvar=:ID,
                                availability=[trues(N), trues(N)],
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

end
