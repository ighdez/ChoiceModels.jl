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
function fake_results(; hessian, se = 0.5, bhhh_se = 0.25)
    names = [:a, :b]
    return (
        parameters      = Dict{Symbol,Real}(n => 1.0 for n in names),
        std_errors      = Dict{Symbol,Real}(n => se for n in names),
        rob_std_errors  = Dict{Symbol,Real}(n => se for n in names),
        bhhh_std_errors = Dict{Symbol,Real}(n => bhhh_se for n in names),
        loglikelihood      = -100.0,
        null_loglikelihood = -200.0,
        iters = 5, converged = true, estimation_time = 1.0, N = 500,
        hessian = hessian,
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
        # BHHH is available alongside it, finite, and is NOT the classical column.
        @test cov.bhhh_vcov ≈ inv(G)
        @test all(isfinite, values(cov.bhhh_std_errors))
        @test cov.bhhh_std_errors[:b] ≈ 0.25
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

    @testset "summarize_results shows the Hessian verdict and gates the BHHH block" begin
        # Headings must never lie: "Classic" is present in every case.
        for status in (:posdef, :indefinite, :singular)
            @test occursin("Classic Standard Errors",
                           capture_stdout(() -> summarize_results(fake_results(hessian=status))))
        end

        pd = capture_stdout(() -> summarize_results(fake_results(hessian=:posdef)))
        @test occursin("Hessian at optimum", pd)
        @test occursin("pos. def.", pd)
        @test !occursin("BHHH", pd)                 # no extra block when nothing is wrong

        # Indefinite: BHHH is the one estimator still worth reading, so show it.
        indef = capture_stdout(() -> summarize_results(fake_results(hessian=:indefinite)))
        @test occursin("INDEFINITE", indef)
        @test occursin("BHHH/OPG Standard Errors", indef)

        # Singular: G is rank-deficient in the same direction, so BHHH is degenerate
        # too — showing it would imply a way out that does not exist.
        sing = capture_stdout(() -> summarize_results(fake_results(hessian=:singular)))
        @test occursin("SINGULAR", sing)
        @test !occursin("BHHH/OPG Standard Errors", sing)
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

        # When BHHH is shown it must reach the spreadsheet too, not just the console.
        mktempdir() do dir
            path = joinpath(dir, "out.xlsx")
            printed = capture_stdout(() -> summarize_results(fake_results(hessian=:indefinite); file=path))
            @test occursin("BHHH/OPG Standard Errors", printed)
            XLSX.openxlsx(path) do xf
                header = [xf["Estimates"][1, j] for j in 1:11]
                @test "BHHH_SE" in header
                @test "RobustSE" in header
                @test "StdError" in header
            end
        end

        # ...and must be absent from both when it is not applicable.
        mktempdir() do dir
            path = joinpath(dir, "out.xlsx")
            printed = capture_stdout(() -> summarize_results(fake_results(hessian=:posdef); file=path))
            @test !occursin("BHHH", printed)
            XLSX.openxlsx(path) do xf
                header = [xf["Estimates"][1, j] for j in 1:8]
                @test !("BHHH_SE" in header)
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

        # The summary block must surface it, not just the (scrollable) @warn.
        printed = capture_stdout(() -> summarize_results(results))
        @test occursin("Hessian at optimum", printed)
        @test occursin("SINGULAR", printed)

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

    @testset "estimate returns all three estimators" begin
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
