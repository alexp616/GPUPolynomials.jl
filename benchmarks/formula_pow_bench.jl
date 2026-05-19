# Benchmark: formula_pow CPU vs Metal
#
# Measures formula_pow (kernel execution only; plan construction excluded)
# across two representative inputs:
#   - (n_vars=4, d=4, pow=2)  → short-rows-only path
#   - (n_vars=4, d=4, pow=6)  → hybrid short+long path
#
# Run standalone:
#   julia --project benchmarks/formula_pow_bench.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GPUPolynomials
using KernelAbstractions
using Metal
using BenchmarkTools
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

function make_coeffs(::Type{T}, n_vars, d) where T
    n = binomial(n_vars + d - 1, d)
    return collect(T, 1:n)
end

function print_result(label, trial)
    t = median(trial)
    @printf("  %-40s  %8.3f ms\n", label, t.time / 1e6)
end

function print_speedup(cpu_trial, gpu_trial)
    ratio = median(cpu_trial).time / median(gpu_trial).time
    @printf("  %-40s  %.2fx\n", "Metal speedup", ratio)
end

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner for one (n_vars, d, pow) configuration
# ──────────────────────────────────────────────────────────────────────────────

function run_case(n_vars, d, pow, ::Type{T}) where T
    @printf("\n=== n_vars=%d  d=%d  pow=%d  (%s) ===\n", n_vars, d, pow, T)

    cpu_backend = KernelAbstractions.CPU()
    cpu_coeffs  = make_coeffs(T, n_vars, d)
    cpu_plan    = formula_pow_plan(n_vars, d, pow, cpu_backend)
    @printf("  short rows: %d  long rows: %d\n",
            length(cpu_plan.short_rows), length(cpu_plan.long_rows))

    cpu_trial = @benchmark formula_pow($cpu_coeffs, $cpu_plan, $cpu_backend) samples=50 evals=3
    print_result("CPU", cpu_trial)

    if Metal.functional()
        metal_backend  = MetalBackend()
        metal_coeffs   = Metal.MtlArray{T}(cpu_coeffs)
        metal_plan     = formula_pow_plan(n_vars, d, pow, metal_backend)

        # Warm up
        formula_pow(metal_coeffs, metal_plan, metal_backend)
        Metal.synchronize()

        metal_trial = @benchmark begin
            formula_pow($metal_coeffs, $metal_plan, $metal_backend)
            Metal.synchronize()
        end samples=50 evals=3
        print_result("Metal", metal_trial)
        print_speedup(cpu_trial, metal_trial)
    else
        println("  Metal: not available (skipped)")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Cases
# ──────────────────────────────────────────────────────────────────────────────

println("formula_pow benchmark — CPU vs Metal")
println("Plan construction time is excluded; only kernel execution is measured.")

# Short-rows-only path (all rows below workgroup_size threshold)
run_case(4, 4, 2, Int64)

# Hybrid path (both short and long rows fire)
run_case(4, 4, 6, Int64)

# Matches the (5,5,2) and (4,8,3) cases covered by formula_pow_vs_ntt_bench.jl
# so cross-backend regressions get caught on all four tracked tuples.
run_case(5, 5, 2, Int64)
run_case(4, 8, 3, Int64)

println()
