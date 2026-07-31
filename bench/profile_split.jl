# Where does an estimation actually spend itself?
#
#   cd bench && julia --project=@choicemodels -t 4 profile_split.jl [R]
#
# `R` defaults to 500 here — unlike the other two scripts — because this one runs
# a real `estimate` and is meant to be comparable with the committed
# `examples/MXL_swissmetro.jl`, not with the item 3/4 microbenchmarks.
#
# METHOD. No profiler: call COUNTS come from `Optim.f_calls`/`g_calls`, per-call
# COSTS from timing each site directly at θ̂, and the budget
#     f_calls*t_f + g_calls*t_g + t_H + t_J
# is compared against the reported `estimation_time`. Cheap, and accurate enough
# to select between "gradient-dominated" and "Hessian-dominated", which is the
# only decision it has to support.
#
# NOTE ONLY ONE HESSIAN AND ONE JACOBIAN ARE COUNTED. Each `estimate` evaluates
# both twice — a warm-up at θ₀ and the real one at θ̂ — but `t_start` sits AFTER
# the warm-up, so only the second falls inside `estimation_time`. The warm-up is
# therefore invisible to the reported metric while still costing wall time; see
# the warm-up note in CLAUDE.md item 5, Phase 2 before deciding that is a bug.
#
# Recorded result, MXL_swissmetro R=500 K=8 (CLAUDE.md item 5, Phase 2):
#     f_obj (line search)   55 calls   1.12s   17.7%
#     gradient (BFGS)       34 calls   2.92s   46.1%   <- dominant
#     hessian                1 call    1.49s   23.6%
#     score jacobian         1 call    0.10s    1.5%
# Verdict: gradient-dominated, which selects reverse-mode AD in the Phase 3 gate.

include("common.jl")

const RDRAW = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500

s   = bench_setup(R = RDRAW)
res = estimate(s.model, :CHOICE; verbose = false)

θ̂ = [res.parameters[n] for n in s.free_names]
K = s.K

# The library's own configurations, so the per-call costs are the ones an
# `estimate` actually pays: default chunk everywhere except the Hessian.
gcfg = ForwardDiff.GradientConfig(s.f_obj, θ̂)
hcfg = ChoiceModels._hessian_config(s.f_obj, θ̂)
jcfg = ForwardDiff.JacobianConfig(s.f_obj_i, θ̂)
H = zeros(K, K)
scores = zeros(length(s.f_obj_i(θ̂)), K)

s.f_obj(θ̂)                                        # compile every site first
ForwardDiff.gradient(s.f_obj, θ̂, gcfg)
ForwardDiff.hessian!(H, s.f_obj, θ̂, hcfg)
ForwardDiff.jacobian!(scores, s.f_obj_i, θ̂, jcfg)

med(f, n) = median([@elapsed f() for _ in 1:n])
t_f = med(() -> s.f_obj(θ̂), 10)
t_g = med(() -> ForwardDiff.gradient(s.f_obj, θ̂, gcfg), 10)
t_H = med(() -> ForwardDiff.hessian!(H, s.f_obj, θ̂, hcfg), 3)
t_J = med(() -> ForwardDiff.jacobian!(scores, s.f_obj_i, θ̂, jcfg), 3)

nf = Optim.f_calls(res.result)
ng = Optim.g_calls(res.result)
row(label, n, t) = println(
    rpad(label, 22), lpad(n, 6), lpad(round(n * t; digits = 2), 9), " s",
    lpad(round(100 * n * t / res.estimation_time; digits = 1), 8), " %")

println("\n=== estimation split: R=$(s.R)  K=$K  iters=", res.iters,
        "  estimation_time=", round(res.estimation_time; digits = 2), "s ===")
println(rpad("site", 22), lpad("calls", 6), lpad("total", 9), "  ", lpad("share", 8))
row("f_obj (line search)", nf, t_f)
row("gradient (BFGS)",     ng, t_g)
row("hessian",              1,  t_H)   # 1, not 2: see the header
row("score jacobian",       1,  t_J)
println("per-call: f=", round(t_f; digits = 4), "s  g=", round(t_g; digits = 4),
        "s  H=", round(t_H; digits = 3), "s  J=", round(t_J; digits = 3), "s")
println("LL = ", repr(res.loglikelihood), "  iters = ", res.iters)
