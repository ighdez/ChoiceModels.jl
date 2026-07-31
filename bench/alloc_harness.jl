# Per-AD-site allocation and timing, plus the VALUE PINS.
#
#   cd bench && julia --project=@choicemodels -t 4 alloc_harness.jl [chunk] [R]
#
# `chunk` defaults to 0 (ForwardDiff's default), `R` to 250 — see the note in
# common.jl for why 250 and not the example's committed 500.
#
# WHAT THIS MEASURES, AND WHAT IT DOES NOT. `@allocated` is CHURN, not the live
# set, and for the chunk question the two point in OPPOSITE directions: total
# allocation scales as ceil(K/C)*(1+C), which is minimized at the default chunk,
# while peak RSS falls as the chunk shrinks. Use `grad_rss.jl` for peak RSS.
# CLAUDE.md item 5, Phase 1 records the measurement that establishes this.
#
# The pins are the drift tripwire described in common.jl. Recorded values at θ₀,
# R=250 (CLAUDE.md item 5, Phase 0):
#     sum(P.^2)     = 945463.9473895603
#     sum(sqrt.(P)) = 2.5984283951159474e6
#     loglikelihood = -4280.208072020548
#     H[1,1]        = 173.2990104905588
#     H[end,end]    = 227.45759238880711
#
# Two pins that look natural and are USELESS — do not reinstate either:
#   * `sum(P)` is exactly N*R for any implementation, right or wrong, because
#     probabilities sum to 1 across alternatives.
#   * `sum(log P)` is -Inf for any implementation, because unavailable
#     alternatives have P exactly 0. `sum(sqrt.(P))` replaces it: finite
#     everywhere, and still weights the small probabilities heavily, which is
#     where an AD or softmax change would show up.

include("common.jl")

const CHUNK = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
const RDRAW = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 250

s = bench_setup(R = RDRAW)
K = s.K

# One tensor at the element size of the site being measured:
#   gradient / jacobian : Dual{Float64,C}         -> (1+C)*8   bytes
#   hessian             : Dual{Dual{Float64,C},C} -> (1+C)^2*8 bytes
const C = ForwardDiff.chunksize(chunk_of(s.θ0, CHUNK))
tensor_grad = s.N * s.J * s.R * (1 + C) * 8
tensor_hess = s.N * s.J * s.R * (1 + C)^2 * 8

fmt(x, d) = rpad(string(round(x; digits = d)), d + 6)
report(label, bytes, secs, unit) = println(
    rpad(label, 26), lpad(fmt(bytes / unit, 2), 10), " tensors ",
    lpad(fmt(bytes / 2^20, 1), 12), " MiB ", lpad(fmt(secs, 4), 11), " s")

println("MXL_swissmetro  N=$(s.N) J=$(s.J) R=$(s.R)  K=$K free params  ",
        "chunk=$(CHUNK == 0 ? "default($C)" : string(C))")
println("one tensor: grad-element ", round(tensor_grad / 2^20; digits = 1),
        " MiB   hess-element ", round(tensor_hess / 2^20; digits = 1), " MiB\n")

# --- value pins --------------------------------------------------------------
s.setθ!(s.θ0)
P = ChoiceModels.logit_prob(s.model.utilities, s.model.data, s.mutable_parameters,
                            s.model.availability, s.model.draws)   # N x J x R
println("PIN sum(P.^2)        = ", repr(sum(P .^ 2)))
println("PIN sum(sqrt.(P))    = ", repr(sum(sqrt.(P))))
println("PIN loglikelihood    = ", repr(-sum(s.f_obj_i(s.θ0))))

# --- the four AD sites -------------------------------------------------------
ch   = chunk_of(s.θ0, CHUNK)
gcfg = ForwardDiff.GradientConfig(s.f_obj, s.θ0, ch)
hcfg = ForwardDiff.HessianConfig(s.f_obj, s.θ0, ch)
jcfg = ForwardDiff.JacobianConfig(s.f_obj_i, s.θ0, ch)
scores = zeros(length(s.f_obj_i(s.θ0)), K)
H = zeros(K, K)

g = ForwardDiff.gradient(s.f_obj, s.θ0, gcfg)          # warm up (compile)
ForwardDiff.hessian!(H, s.f_obj, s.θ0, hcfg)
ForwardDiff.jacobian!(scores, s.f_obj_i, s.θ0, jcfg)
s.f_obj(s.θ0)

println("\nPIN gradient         = ", repr(g), "\n")

a = @allocated s.f_obj(s.θ0);                              t = @elapsed s.f_obj(s.θ0)
report("f_obj (value only)", a, t, tensor_grad)
a = @allocated ForwardDiff.gradient(s.f_obj, s.θ0, gcfg);  t = @elapsed ForwardDiff.gradient(s.f_obj, s.θ0, gcfg)
report("gradient", a, t, tensor_grad)
a = @allocated ForwardDiff.jacobian!(scores, s.f_obj_i, s.θ0, jcfg)
t = @elapsed ForwardDiff.jacobian!(scores, s.f_obj_i, s.θ0, jcfg)
report("score jacobian", a, t, tensor_grad)
a = @allocated ForwardDiff.hessian!(H, s.f_obj, s.θ0, hcfg)
t = @elapsed ForwardDiff.hessian!(H, s.f_obj, s.θ0, hcfg)
report("hessian", a, t, tensor_hess)

println("\nPIN H[1,1], H[end,end] = ", repr(H[1, 1]), "  ", repr(H[end, end]))
