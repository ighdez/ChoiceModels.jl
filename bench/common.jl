# Shared fixture for the benchmarks in this directory.
#
# ---------------------------------------------------------------------------
# DRIFT HAZARD, READ THIS FIRST.
#
# `bench_setup` DUPLICATES the specification in `examples/MXL_swissmetro.jl`
# (utilities, the /100 scaling, the GA cost rule, availability, the seed and the
# draw scheme). That is the copy-paste hazard CLAUDE.md opens with, and it is
# accepted here for one reason only: the benchmarks must keep measuring the same
# model as the numbers recorded in CLAUDE.md item 5, so they cannot follow the
# example if the example changes.
#
# The VALUE PINS printed by `alloc_harness.jl` are the tripwire. If they stop
# reproducing the values recorded in CLAUDE.md item 5, something moved — and you
# must work out WHICH of the two it was:
#   * the library changed  -> the pins are telling you about a real behaviour change
#   * this file drifted    -> re-sync it against examples/MXL_swissmetro.jl
# Do not "fix" a failing pin by editing the recorded value.
#
# `R` is a parameter and is deliberately NOT taken from the example. The example
# is committed at R=500; the benchmarks default to R=250 because every earlier
# measurement in CLAUDE.md items 3 and 4 was taken there and the point of these
# scripts is comparability with those.
# ---------------------------------------------------------------------------
#
# Run from INSIDE bench/ (the data path is relative, as in examples/):
#   cd bench && julia --project=@choicemodels -t 4 alloc_harness.jl
#
# The @choicemodels shared env is the one CLAUDE.md's Commands section sets up
# for the examples; these scripts need nothing added to it, reaching ForwardDiff
# and Optim through `ChoiceModels.` rather than importing them directly.

using CSV, DataFrames, Statistics, Random
using ChoiceModels

const ForwardDiff = ChoiceModels.ForwardDiff
const Optim       = ChoiceModels.Optim

"""
    bench_setup(; R = 250)

Build the `MXL_swissmetro` model and the two objective closures `estimate` uses,
without running an estimation. Returns a NamedTuple with `model`, `choices`,
`f_obj` (scalar, what BFGS and the Hessian see), `f_obj_i` (per-individual, what
the score Jacobian sees), `θ0`, `free_names`, and the problem dimensions.
"""
function bench_setup(; R::Int = 250)
    df = CSV.read("../data/swissmetro.dat", DataFrame; delim = '\t')
    df = filter(row -> (row[:PURPOSE] == 1 || row[:PURPOSE] == 3) && row[:CHOICE] .!= 0, df)
    for c in (:TRAIN_TT, :TRAIN_CO, :CAR_TT, :CAR_CO, :SM_TT, :SM_CO)
        df[!, c] ./= 100
    end
    # A season-ticket holder's incremental cost is zero.
    df.SM_CO    .= ifelse.(df.GA .== 0, df.SM_CO, 0.0)
    df.TRAIN_CO .= ifelse.(df.GA .== 0, df.TRAIN_CO, 0.0)

    mu_asc_car   = Parameter(:mu_asc_car,   value = 0)
    s_asc_car    = Parameter(:s_asc_car,    value = 1)
    mu_asc_train = Parameter(:mu_asc_train, value = 0)
    s_asc_train  = Parameter(:s_asc_train,  value = 1)
    mu_asc_sm    = Parameter(:mu_asc_sm,    value = 0, fixed = true)
    s_asc_sm     = Parameter(:s_asc_sm,     value = 1)
    mu_time      = Parameter(:mu_time,      value = -1)
    s_time       = Parameter(:s_time,       value = 1)
    β_cost       = Parameter(:β_cost,       value = -1)

    asc_car   = mu_asc_car   + s_asc_car   * Draw(:d_asc_car)
    asc_train = mu_asc_train + s_asc_train * Draw(:d_asc_train)
    asc_sm    = mu_asc_sm    + s_asc_sm    * Draw(:d_asc_sm)
    β_time    = mu_time      + s_time      * Draw(:d_time)

    alternatives = (train = 1, sm = 2, car = 3)
    utilities = (
        train = asc_train + β_time * Variable(:TRAIN_TT) + β_cost * Variable(:TRAIN_CO),
        sm    = asc_sm    + β_time * Variable(:SM_TT)    + β_cost * Variable(:SM_CO),
        car   = asc_car   + β_time * Variable(:CAR_TT)   + β_cost * Variable(:CAR_CO),
    )

    df.TRAIN_AV_SP .= ifelse.(df.SP .!= 0, df.TRAIN_AV, 0)
    df.CAR_AV_SP   .= ifelse.(df.SP .!= 0, df.CAR_AV,   0)
    availability = (train = df.TRAIN_AV_SP .== 1,
                    sm    = df.SM_AV       .== 1,
                    car   = df.CAR_AV_SP   .== 1)

    Random.seed!(12345)
    model = MixedLogitModel(alternatives; utilities = utilities,
                            availability = availability, data = df,
                            idvar = :ID, R = R, draw_scheme = :halton)

    # Mirrors what `estimate` builds, so the closures below are the same objects
    # the optimizer and the AD sites actually see.
    choices     = ChoiceModels._recode_choices(df[!, :CHOICE], model.alternatives, :CHOICE)
    params      = ChoiceModels.collect_parameters(model.utilities)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed    = [p.fixed for p in params]
    free_names  = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]
    θ0          = Float64[init_values[n] for n in free_names]

    # `Dict{Symbol,Any}`, not a typed dict: the closures write ForwardDiff Duals
    # into it. Same reasoning as the four `estimate` functions.
    mutable_parameters = Dict{Symbol,Any}(model.parameters)
    function setθ!(θ)
        for (i, n) in enumerate(free_names); mutable_parameters[n] = θ[i]; end
        for n in fixed_names;                mutable_parameters[n] = init_values[n]; end
    end

    f_obj_i(θ) = (setθ!(θ); -loglikelihood(model, choices; parameters = mutable_parameters))
    f_obj(θ)   = (setθ!(θ); -sum(loglikelihood(model, choices; parameters = mutable_parameters)))

    return (; model, df, choices, free_names, θ0, K = length(θ0),
              N = nrow(df), J = length(model.alternatives), R,
              f_obj, f_obj_i, setθ!, mutable_parameters)
end

"""
    chunk_of(θ0, chunk)

`ForwardDiff.Chunk` for a requested size, with `0` meaning "ForwardDiff's
default". Always goes through `Chunk(θ0, c)` and never `Chunk{c}()`: the latter
throws an `AssertionError` when the model has a single free parameter, since the
chunk may not exceed the input length. This form clamps. (CLAUDE.md item 3.)
"""
chunk_of(θ0, chunk::Int) = chunk == 0 ? ForwardDiff.Chunk(θ0) : ForwardDiff.Chunk(θ0, chunk)
