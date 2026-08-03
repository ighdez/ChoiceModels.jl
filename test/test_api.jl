using Test
using ChoiceModels

# The exported surface is what v1.0.0 promises under semver, so it is pinned
# here rather than left to whatever happens to carry an `export`. A symbol
# added to this list is a feature; one removed is a BREAKING change. Both
# should be a deliberate edit to this test, never a silent consequence of
# editing an `export` line.
#
# `Draws` and `generate_draws` are the two that were deliberately REMOVED from
# the surface before the tag (Pre-v1.0.0 item 3). They are internal machinery
# the model constructors drive: `MixedLogitModel` generates its own draws and
# `_lc_share_draws` takes that over for a latent class, so a user never calls
# `generate_draws` and never holds a `Draws`. Exporting them would have
# committed `generate_draws(names, N, R; scheme)` under semver — a signature
# whose Halton base assignment already changed shape once. The precedent they
# now follow is `logit_prob` / `chosen_logprob` / `covariance_estimates` /
# `hessian_status` / `bhhh_matrix_status`, all reachable as `ChoiceModels.foo`.
@testset "public API" begin

    @testset "exported surface is exactly what is promised" begin
        expected = Set([
            :ChoiceModels,                                          # the module itself
            :Parameter, :Variable, :Draw, :evaluate,                # symbolic layer
            :LogitModel, :MixedLogitModel, :NestedLogitModel,       # models
            :LatentClassModel, :Nest,
            :estimate, :predict, :loglikelihood,                    # estimation
            :summarize_results, :summarize_expressions,             # reporting
        ])
        @test Set(names(ChoiceModels)) == expected
    end

    @testset "draw machinery is reachable but unexported" begin
        for sym in (:Draws, :generate_draws)
            @test !(sym in names(ChoiceModels))
            @test isdefined(ChoiceModels, sym)
        end
        # Unexported does not mean unusable: the qualified call still works, and
        # this is the form the rest of the suite uses.
        d = ChoiceModels.generate_draws([:z], 5, 4; scheme = :halton)
        @test d isa ChoiceModels.Draws
        @test size(d.values[:z]) == (5, 4)
    end

    # Pre-v1.0.0 item 2: `DCMVariable.index` was accepted, stored, and read by
    # nothing — both `evaluate` paths take the column as `data[:, name]`. It was
    # removed before the tag rather than frozen into the API. Panel structure is
    # carried by the models' `idvar` instead.
    @testset "Variable takes a name and nothing else" begin
        @test fieldnames(ChoiceModels.DCMVariable) == (:name,)
        @test Variable(:x).name === :x
        @test_throws MethodError Variable(:x; index = 1)
    end
end
