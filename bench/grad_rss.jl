# Peak-RSS probe for ONE AD site at ONE chunk size, one site per process.
#
#   cd bench && /usr/bin/time -l julia --project=@choicemodels -t 4 \
#       grad_rss.jl <grad|jac|hess> <chunk, 0 = default> [reps] [R]
#
# Read the "maximum resident set size" line from `/usr/bin/time -l`.
#
# WHY A SEPARATE PROCESS PER SETTING. Peak RSS is a high-water mark, so two chunk
# settings measured in one process both report the larger. That also means the
# figure includes a fixed baseline (~0.95 GB at R=250: Julia, the data, the model
# and its draws); the baseline is recoverable by fitting across chunk sizes, since
# the marginal scales as (1+C) for a gradient and (1+C)^2 for a Hessian.
#
# WHY THIS EXISTS SEPARATELY FROM alloc_harness.jl. `@allocated` measures churn
# and points the OPPOSITE way from peak RSS on this question — see the header of
# alloc_harness.jl and CLAUDE.md item 5, Phase 1. Peak RSS is the constraint.
#
# Recorded results, MXL_swissmetro R=250 K=8 (CLAUDE.md item 5, Phase 1):
#     gradient default(8)  2.21 GB  0.059s     hessian chunk 2  2.32 GB  0.845s
#     gradient chunk 4     1.61 GB  0.099s     hessian default  8.75 GB  0.648s
#     gradient chunk 2     1.31 GB  0.112s
#     gradient chunk 1     1.23 GB  0.175s
# All sites return BITWISE IDENTICAL values at every chunk size; chunking is a
# memory/time knob and never a correctness one.

include("common.jl")

const SITE  = ARGS[1]
const CHUNK = parse(Int, ARGS[2])
const REPS  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const RDRAW = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 250

s  = bench_setup(R = RDRAW)
ch = chunk_of(s.θ0, CHUNK)

if SITE == "grad"
    cfg = ForwardDiff.GradientConfig(s.f_obj, s.θ0, ch)
    out = zeros(s.K)
    run!() = ForwardDiff.gradient!(out, s.f_obj, s.θ0, cfg)
elseif SITE == "jac"
    cfg = ForwardDiff.JacobianConfig(s.f_obj_i, s.θ0, ch)
    out = zeros(length(s.f_obj_i(s.θ0)), s.K)
    run!() = ForwardDiff.jacobian!(out, s.f_obj_i, s.θ0, cfg)
elseif SITE == "hess"
    cfg = ForwardDiff.HessianConfig(s.f_obj, s.θ0, ch)
    out = zeros(s.K, s.K)
    run!() = ForwardDiff.hessian!(out, s.f_obj, s.θ0, cfg)
else
    error("unknown site $SITE — expected grad, jac or hess")
end

run!()                     # compile, so the timings below are run time only
GC.gc(); GC.gc()
ts = [(@elapsed run!()) for _ in 1:REPS]

println("site=$SITE chunk=$(CHUNK == 0 ? "default" : CHUNK) R=$(s.R) K=$(s.K)  ",
        "median=", round(median(ts); digits = 4), "s  min=", round(minimum(ts); digits = 4), "s")
println("PIN out[1] = ", repr(out[1]), "   out[end] = ", repr(out[end]))
