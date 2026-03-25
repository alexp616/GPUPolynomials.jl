# GPUPolynomials.jl

GPU-accelerated polynomial multiplication and powering for Oscar.jl-compatible types (`ZZMPolyRingElem`, `fpMPolyRingElem`, `ZZPolyRingElem`).

## What it does

The core pipeline for multivariate powering:

1. **Kronecker substitution** — encode the multivariate polynomial as univariate via `x_i → t^{key^(i-1)}`
2. **Multi-modular NTT** — run GPU Number Theoretic Transforms over multiple NTT-friendly primes simultaneously
3. **Pointwise powering** — raise each frequency-domain value to the desired power (no repeated squaring needed)
4. **CRT reconstruction** — recover exact integer coefficients via Garner's algorithm

NTT primes are chosen so their product exceeds a tight coefficient bound derived from the input, minimizing the number of primes needed.

## Key types

- `CuZZPolyRingElem` — univariate polynomial over ZZ
- `CuZZMPolyRingElem` — multivariate polynomial over ZZ
- `CufpMPolyRingElem` — multivariate polynomial over a finite field GF(p)

All types store coefficients and exponents on the GPU as `CuVector`. Operation plans (`MulPlan`, `PowPlan`, `MPowPlan`) are lazily cached on first use and can be pre-built for hot paths.

## Dependencies

- `CUDA.jl` — GPU arrays and kernel launches
- `CudaNTTs.jl` — GPU NTT implementation
- `Oscar.jl` — computer algebra system integration
- `BitIntegers.jl` — wide integers (`Int256`/`Int512`/etc.) for CRT on GPU
- `FLINT_jll.jl` — direct C calls to construct Oscar polynomial objects

## Limitations

- Only homogeneous multivariate polynomials are supported; non-homogeneous inputs throw immediately.
- CUDA backend only (Metal path is broken).
