# Benchmark: formula_pow vs NTT-based powering on CUDA
#
# Compares two approaches to computing (homogeneous polynomial)^pow:
#   - formula_pow: evaluates the multinomial expansion formula directly on GPU
#   - NTT:         Kronecker-substitution + multi-modular NTT + CRT
#
# Both approaches operate on the same polynomial.
# Plan construction is excluded; only kernel execution is measured.
# Requires a CUDA device — skips gracefully if none is present.
#
# Run standalone:
#   julia --project benchmarks/formula_pow_vs_ntt_bench.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GPUPolynomials
using Oscar
using CUDA
using KernelAbstractions
using Combinatorics
using BenchmarkTools
using Printf

if !CUDA.functional()
    println("No CUDA device found — skipping benchmark.")
    exit(0)
end

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Build a dense homogeneous Oscar polynomial of degree d in n_vars variables
# with coefficients coeffs[i] for the i-th monomial in
# with_replacement_combinations(1:n_vars, d) order.
function build_homog_poly(n_vars, d, coeffs)
    R, vars = polynomial_ring(ZZ, n_vars)
    f = zero(R)
    for (i, combo) in enumerate(with_replacement_combinations(1:n_vars, d))
        ev = zeros(Int, n_vars)
        for idx in combo; ev[idx] += 1; end
        mon = prod(vars[j]^ev[j] for j in 1:n_vars)
        f += coeffs[i] * mon
    end
    return f
end

function print_result(label, trial)
    t = median(trial)
    @printf("  %-44s  %8.3f ms\n", label, t.time / 1e6)
end

function print_speedup(label, ref_trial, cmp_trial)
    ratio = median(ref_trial).time / median(cmp_trial).time
    @printf("  %-44s  %.2fx  (vs %s)\n", "speedup", ratio, label)
end

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner for one (n_vars, d, pow) configuration
# ──────────────────────────────────────────────────────────────────────────────

function run_case(n_vars, d, pow)
    @printf("\n=== n_vars=%d  d=%d  pow=%d ===\n", n_vars, d, pow)

    n = binomial(n_vars + d - 1, d)
    coeffs = collect(Int64, 1:n)

    oscar_f = build_homog_poly(n_vars, d, coeffs)

    # ── formula_pow on CUDA ──────────────────────────────────────────────────
    cuda_backend = CUDABackend()
    fp_plan      = formula_pow_plan(n_vars, d, pow, cuda_backend)
    fp_original  = CUDA.CuArray{Int64}(coeffs)

    @printf("  formula_pow: short_rows=%d  long_rows=%d\n",
            length(fp_plan.short_rows), length(fp_plan.long_rows))

    # warm up
    formula_pow(fp_original, fp_plan, cuda_backend); CUDA.synchronize()

    fp_trial = @benchmark begin
        formula_pow($fp_original, $fp_plan, $cuda_backend)
        CUDA.synchronize()
    end samples=100 evals=3

    print_result("formula_pow (CUDA)", fp_trial)

    # ── NTT-based powering on CUDA ───────────────────────────────────────────
    cu_f    = cu(oscar_f)
    ntt_plan = MPowPlan(cu_f, pow)
    cu_f.opPlan = ntt_plan

    # warm up
    _ = cu_f ^ pow; CUDA.synchronize()

    ntt_trial = @benchmark begin
        result = $cu_f ^ $pow
        CUDA.synchronize()
    end samples=100 evals=3

    print_result("NTT powering (CUDA)", ntt_trial)

    print_speedup("NTT", ntt_trial, fp_trial)
end

# ──────────────────────────────────────────────────────────────────────────────
# Cases
# ──────────────────────────────────────────────────────────────────────────────

println("formula_pow vs NTT powering benchmark (CUDA)")
println("Plan construction excluded; kernel execution only.")

# Short-rows-only path for formula_pow
run_case(4, 4, 2)

# Hybrid short+long path for formula_pow
run_case(4, 4, 6)

println()
