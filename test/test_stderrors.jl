using Test
using ChoiceModels
using DataFrames
using LinearAlgebra
using XLSX
using Random

# `summarize_results` prints straight to stdout rather than taking an `io`, so
# `sprint` cannot capture it — redirect through a temp file instead.
function capture_stdout(f)
    mktemp() do path, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        return read(path, String)
    end
end

# Minimal results NamedTuple accepted by `summarize_results`, for exercising the
# reporting logic without running an estimation that happens to be ill-behaved.
function fake_results(; hessian, se = 0.5, bhhh_se = 0.25, has_cov = true,
                       bhhh_matrix = has_cov ? :posdef : :singular)
    names = [:a, :b]
    return (
        parameters      = Dict{Symbol,Real}(n => 1.0 for n in names),
        std_errors      = has_cov ? Dict{Symbol,Real}(n => se for n in names) : nothing,
        rob_std_errors  = has_cov ? Dict{Symbol,Real}(n => se for n in names) : nothing,
        bhhh_std_errors = has_cov ? Dict{Symbol,Real}(n => bhhh_se for n in names) : nothing,
        loglikelihood      = -100.0,
        null_loglikelihood = -200.0,
        iters = 5, converged = true, estimation_time = 1.0, N = 500,
        hessian = hessian,
        bhhh_matrix = bhhh_matrix,
        free_parameters = 2,
    )
end

@testset "Standard errors" begin

    names2 = [:a, :b]

    # The central invariant: the three estimators are computed side by side and
    # NEVER substituted for one another. "Classic" always means inv(H), whatever
    # state H is in — an earlier design swapped BHHH into that slot when H failed
    # and tagged it with a `vcov_method` field, which broke the classical-vs-robust
    # comparison the two columns exist to support.
    @testset "each estimator keeps its own identity" begin
        H = [4.0 1.0; 1.0 3.0]          # PD
        G = [9.0 0.0; 0.0 16.0]

        cov = ChoiceModels.covariance_estimates(H, G, names2)

        @test cov.status === :posdef
        @test cov.vcov ≈ inv(H)                          # classical == inv(H)
        @test cov.bhhh_vcov ≈ inv(G)                     # BHHH == inv(G)
        @test cov.rob_vcov ≈ inv(H) * G * inv(H)         # robust == sandwich
        @test cov.std_errors[:a] ≈ sqrt(inv(H)[1, 1])
        @test cov.bhhh_std_errors[:a] ≈ sqrt(inv(G)[1, 1])
        @test cov.rob_std_errors[:b] ≈ sqrt((inv(H) * G * inv(H))[2, 2])
    end

    @testset "a bad Hessian does not silently change the classical column" begin
        H = [1.0 0.0; 0.0 -1.0]         # indefinite
        G = [4.0 0.0; 0.0 16.0]

        cov = @test_logs (:warn,) match_mode=:any ChoiceModels.covariance_estimates(H, G, names2)

        @test cov.status === :indefinite
        # Classical is still inv(H) — negative variance surfaces as NaN rather
        # than being papered over with a different estimator.
        @test cov.vcov ≈ inv(H)
        @test cov.std_errors[:a] ≈ 1.0
        @test isnan(cov.std_errors[:b])
        # BHHH is computed and returned alongside it — but never presented; see
        # the reporting testset below.
        @test cov.bhhh_vcov ≈ inv(G)
        @test all(isfinite, values(cov.bhhh_std_errors))
        @test cov.bhhh_std_errors[:b] ≈ 0.25
    end

    # Apollo: "If the BHHH matrix is singular, no attempt will be made to
    # calculate the full covariance matrix." The case that makes this necessary is
    # a HEALTHY Hessian with a degenerate G — `hessian_status` reports :posdef, so
    # nothing else in the pipeline notices, and because G is PSD by construction
    # its pseudo-inverse has a non-negative diagonal: `guarded_std_errors` stays
    # silent and the ROBUST column (which is built from G) reports exactly 0.0,
    # i.e. t = Inf, p = 0.0000, for the unidentified direction.
    @testset "a singular BHHH matrix suppresses the whole covariance matrix" begin
        H = [2.0 0.0; 0.0 3.0]                    # perfectly fine, positive definite
        s = [1.0 0.0; 2.0 0.0; -1.5 0.0]          # scores carry nothing about `b`
        G = s' * s
        @test rank(G) == 1                        # singular BHHH matrix

        # Without the guard this is the number that would be printed.
        H_inv = inv(H)
        @test (H_inv * G * H_inv)[2, 2] == 0.0    # robust variance of `b`: exactly 0

        cov = @test_logs (:warn,) match_mode=:any ChoiceModels.covariance_estimates(H, G, names2)

        @test cov.status === :posdef              # H alone would have said "all fine"
        @test cov.bhhh_matrix === :singular
        # Nothing partial: every estimator is withheld, not just BHHH.
        @test cov.vcov === nothing
        @test cov.rob_vcov === nothing
        @test cov.bhhh_vcov === nothing
        @test cov.std_errors === nothing
        @test cov.rob_std_errors === nothing
        @test cov.bhhh_std_errors === nothing

        # A full-rank G leaves everything alone and says nothing.
        ok = ChoiceModels.covariance_estimates(H, [9.0 0.0; 0.0 16.0], names2)
        @test ok.bhhh_matrix === :posdef
        @test ok.vcov !== nothing
    end

    @testset "bhhh_matrix_status classifies and warns" begin
        @test ChoiceModels.bhhh_matrix_status([9.0 0.0; 0.0 16.0], names2) === :posdef
        @test_logs ChoiceModels.bhhh_matrix_status([9.0 0.0; 0.0 16.0], names2)

        G_sing = [4.0 2.0; 2.0 1.0]               # rank 1
        @test rank(G_sing) == 1
        # The trap this exists to catch: pinv gives it a clean non-negative
        # diagonal, so no downstream SE check can flag it.
        @test all(diag(pinv(G_sing)) .>= 0)
        st = @test_logs (:warn,) match_mode=:any ChoiceModels.bhhh_matrix_status(G_sing, names2)
        @test st === :singular
    end

    @testset "guarded_std_errors reports NaN instead of throwing" begin
        V = [4.0 0.0; 0.0 -1.0]

        se = @test_logs (:warn,) match_mode=:any ChoiceModels.guarded_std_errors(V, names2, "robust covariance")

        @test se[1] ≈ 2.0
        @test isnan(se[2])

        # No warning and no NaN when the matrix is fine.
        @test ChoiceModels.guarded_std_errors([4.0 0.0; 0.0 9.0], names2, "robust covariance") ≈ [2.0, 3.0]
    end

    @testset "hessian_status classifies and warns" begin
        # PD: no warning, no complaint.
        @test ChoiceModels.hessian_status([4.0 1.0; 1.0 3.0], names2) === :posdef
        @test_logs ChoiceModels.hessian_status([4.0 1.0; 1.0 3.0], names2)

        # Negative eigenvalue → not a maximum.
        indef = @test_logs (:warn,) match_mode=:any ChoiceModels.hessian_status([1.0 0.0; 0.0 -1.0], names2)
        @test indef === :indefinite

        # Zero eigenvalue → flat direction, i.e. unidentified combination. This is
        # the case a `diag(inv(H)) < 0` test cannot see, since pinv gives it a
        # perfectly non-negative diagonal.
        H_sing = [1.0 -1.0; -1.0 1.0]
        @test all(diag(pinv(H_sing)) .>= 0)                       # the old test would pass
        sing = @test_logs (:warn,) match_mode=:any ChoiceModels.hessian_status(H_sing, names2)
        @test sing === :singular
    end

    @testset "summarize_results presents only classical and robust" begin
        # Following Apollo: BHHH is computed and returned, but never presented.
        # Its justification is the information matrix equality H = G, which holds
        # only under correct specification at the true parameter — precisely what
        # is in doubt whenever H is not PD, which is the one case an earlier
        # version of this code chose to display it in.
        for status in (:posdef, :indefinite)
            out = capture_stdout(() -> summarize_results(fake_results(hessian=status)))
            @test occursin("Classic Standard Errors", out)
            @test occursin("Robust Standard Errors", out)
            @test !occursin("BHHH", out)
        end

        pd = capture_stdout(() -> summarize_results(fake_results(hessian=:posdef)))
        @test occursin("Hessian at optimum", pd)
        @test occursin("pos. def.", pd)

        indef = capture_stdout(() -> summarize_results(fake_results(hessian=:indefinite)))
        @test occursin("INDEFINITE", indef)
        # Indefinite still summarizes — the SEs exist, they are just not trustworthy.
        @test occursin("NOTE: the Hessian is not positive definite", indef)
    end

    @testset "a singular Hessian refuses to summarize at all" begin
        # Apollo treats a singular Hessian as an estimation error and does not
        # allow results to be summarized. `estimate` still returns normally, so
        # the caller keeps the estimates and the verdict — what is refused is
        # rendering them as a table with standard errors attached, which is what
        # `_safe_inv`'s pseudo-inverse would otherwise manufacture.
        @test_throws ErrorException summarize_results(fake_results(hessian=:singular))

        err = try
            summarize_results(fake_results(hessian=:singular)); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("singular", err)
        @test occursin("not identified", err)
    end

    @testset "no covariance matrix: estimates shown, standard errors withheld" begin
        out = capture_stdout(() -> summarize_results(fake_results(hessian=:posdef, has_cov=false)))

        # The estimates are still worth reporting...
        @test occursin("Parameter Estimates (no standard errors available)", out)
        @test occursin("Log-likelihood at optimum", out)
        # ...but no standard error column of any kind appears.
        @test !occursin("Classic Standard Errors", out)
        @test !occursin("Robust Standard Errors", out)
        @test !occursin("BHHH", out) || occursin("BHHH matrix", out)
        @test occursin("BHHH matrix", out)          # the reason, in the summary block
        @test occursin("not comp.", out)
        # AIC/BIC still need the free parameter count, which no longer comes from
        # the (absent) standard error dict.
        @test occursin("Number of free parameters", out)
        @test occursin("AIC", out)
    end

    @testset "console and Excel export stay in step" begin
        # These two used to be assembled from separate hand-written lists and had
        # already drifted (differing labels, different row order, the Hessian row
        # only in one). They now come from a single `summary` list — this pins it.
        mktempdir() do dir
            path = joinpath(dir, "nested", "out.xlsx")   # also exercises _ensure_dir
            printed = capture_stdout(() -> summarize_results(fake_results(hessian=:posdef); file=path))
            @test isfile(path)                            # nested dir created for us

            XLSX.openxlsx(path) do xf
                sheet = xf["Summary"]
                labels = String[]
                r = 1
                while !ismissing(sheet[r, 1])
                    push!(labels, sheet[r, 1])
                    r += 1
                end
                @test !isempty(labels)
                # Every exported summary label appears in the console output, in
                # the same order.
                pos = 0
                for label in labels
                    idx = findnext(label, printed, pos + 1)
                    @test idx !== nothing
                    idx === nothing || (pos = first(idx))
                end
                @test "Hessian at optimum" in labels
            end
        end

        # BHHH reaches neither the console nor the spreadsheet, in any Hessian state.
        for status in (:posdef, :indefinite)
            mktempdir() do dir
                path = joinpath(dir, "out.xlsx")
                printed = capture_stdout(() -> summarize_results(fake_results(hessian=status); file=path))
                @test !occursin("BHHH", printed)
                XLSX.openxlsx(path) do xf
                    header = [xf["Estimates"][1, j] for j in 1:8]
                    @test !("BHHH_SE" in header)
                    @test "RobustSE" in header
                    @test "StdError" in header
                end
            end
        end

        # With no covariance matrix the export carries the estimates alone — and
        # the summary sheet still explains why, in step with the console.
        mktempdir() do dir
            path = joinpath(dir, "out.xlsx")
            printed = capture_stdout(() -> summarize_results(
                fake_results(hessian=:posdef, has_cov=false); file=path))
            XLSX.openxlsx(path) do xf
                header = String[]
                j = 1
                while !ismissing(xf["Estimates"][1, j])
                    push!(header, xf["Estimates"][1, j]); j += 1
                end
                @test header == ["Parameter", "Estimate"]

                sheet = xf["Summary"]
                labels = String[]
                r = 1
                while !ismissing(sheet[r, 1])
                    push!(labels, sheet[r, 1]); r += 1
                end
                @test "BHHH matrix" in labels
                @test "Covariance matrix" in labels
                pos = 0
                for label in labels
                    idx = findnext(label, printed, pos + 1)
                    @test idx !== nothing
                    idx === nothing || (pos = first(idx))
                end
            end
        end
    end

    @testset "singular Hessian is detected on a real unidentified model" begin
        # Two free ASCs in a 2-alternative logit: only their difference is
        # identified, so the Hessian has an exact zero eigenvalue. Before the
        # eigenvalue check this estimated silently and reported tidy finite SEs.
        Random.seed!(99)
        N = 300
        x1, x2 = randn(N), randn(N)
        choices = [rand() < 1 / (1 + exp(-0.8 * (x2[n] - x1[n]) - 0.5)) ? 1 : 2 for n in 1:N]
        df = DataFrame(x1=x1, x2=x2, CHOICE=choices)

        asc1 = Parameter(:asc1, value=0.0)
        asc2 = Parameter(:asc2, value=0.0)
        b    = Parameter(:b,    value=0.0)
        model = LogitModel([asc1 + b * Variable(:x1), asc2 + b * Variable(:x2)];
                           data=df, availability=[trues(N), trues(N)])

        results = @test_logs (:warn,) match_mode=:any estimate(model, :CHOICE; verbose=false)
        @test results.converged
        # `converged == true` alongside `hessian == :singular` is exactly the pair
        # the summary line exists to show: convergence alone does not mean the
        # standard errors are usable.
        @test results.hessian === :singular
        # Both gates fire on this model, and that is not a coincidence: the score
        # w.r.t. asc1 is (y₁ − p₁) and w.r.t. asc2 is its exact negative, so the
        # score outer product is rank-deficient in the same direction the Hessian
        # is flat. No covariance matrix is computed.
        @test results.bhhh_matrix === :singular
        @test results.vcov === nothing
        @test results.std_errors === nothing

        # ...and summarizing is refused rather than rendering pseudo-inverse SEs.
        @test_throws ErrorException summarize_results(results)

        # The identified counterpart must NOT warn, so the check isn't just noisy.
        asc = Parameter(:asc, value=0.0)
        b2  = Parameter(:b,   value=0.0)
        ok = LogitModel([asc + b2 * Variable(:x1), b2 * Variable(:x2)];
                        data=df, availability=[trues(N), trues(N)])
        r_ok = estimate(ok, :CHOICE; verbose=false)
        @test r_ok.hessian === :posdef
        @test all(isfinite, values(r_ok.std_errors))
        @test all(>(0), values(r_ok.std_errors))
        @test occursin("pos. def.", capture_stdout(() -> summarize_results(r_ok)))

        # Same fit either way — the redundancy costs nothing in likelihood, which
        # is exactly why it is easy to miss without the Hessian check.
        @test results.loglikelihood ≈ r_ok.loglikelihood atol=1e-6
    end

    @testset "estimate returns all three estimators (BHHH returned, not reported)" begin
        Random.seed!(5)
        N = 200
        x1, x2 = randn(N), randn(N)
        choices = [rand() < 1 / (1 + exp(-0.7 * (x2[n] - x1[n]) - 0.3)) ? 1 : 2 for n in 1:N]
        df = DataFrame(x1=x1, x2=x2, CHOICE=choices)

        asc = Parameter(:asc, value=0.0)
        β = Parameter(:β, value=0.0)

        model = LogitModel([asc + β * Variable(:x1), β * Variable(:x2)];
                           data=df, availability=[trues(N), trues(N)])
        results = estimate(model, :CHOICE; verbose=false)

        @test results.hessian === :posdef
        @test size(results.vcov) == (2, 2)
        @test size(results.rob_vcov) == (2, 2)
        @test size(results.bhhh_vcov) == (2, 2)
        for d in (results.std_errors, results.rob_std_errors, results.bhhh_std_errors)
            @test all(isfinite, values(d))
            @test all(>(0), values(d))
        end
        # Three different estimators of the same quantity: same ballpark, not equal.
        @test results.std_errors[:β] ≈ results.bhhh_std_errors[:β] rtol=0.3
        @test results.std_errors[:β] != results.bhhh_std_errors[:β]
    end

end
