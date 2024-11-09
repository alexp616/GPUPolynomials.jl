# GPUPolynomials.jl

[![Build Status](https://github.com/alexp616/GPUPolynomials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexp616/GPUPolynomials.jl/actions/workflows/CI.yml?query=branch%3Amain)

The GPUPolynomials.jl package aims to allow users to perform fast polynomial arithmetic when used with [Oscar.jl](https://github.com/oscar-system/Oscar.jl).

Because GPUPolynomials.jl was developed for the sole purpose of raising multivariate homogeneous polynomials to powers, as used in [MMPSingularities.jl](https://github.com/jjgarzella/MMPSingularities.jl), this is the only functionality implemented for now.

## Disclaimer
This package is still in extremely early development, and needs much more work to be usable outside of the specific case I developed it for. Here is a checklist of things I plan to implement:

- More cutting-edge FFT using specific optimal primes, should remove need for pregeneration as well.
- Polynomial addition/subtraction
- Polynomial multiplication
- Polynomial evaluation

The GPU can make significant improvements to polynomial addition/subtraction/evaluation for both dense and sparse cases, but no sparse algorithms (that I know of) when parallelized on the GPU end up being faster than OSCAR.jl's FLINT backend. There are rare cases, (like 4-variate homogeneous polynomial powering), where using a dense algorithm on a sparse problem ends up being faster purely because of the GPU's throughput.

## Usage
To import GPUPolynomials, simply do:
```
pkg> add GPUPolynomials
```
As of now, GPUPolynomials requires Julia version 1.11 in order for division and modulo [BitIntegers.jl](https://github.com/rfourquet/BitIntegers.jl) to work with CUDA.jl.

```
julia> using GPUPolynomials
julia> R, (x, y, z, w) = polynomial_ring(GF(11), 4)
julia> f = x^4 + y^4 + z^4 + w^4 # Initialize an Oscar Polynomial
julia> f = HomogeneousPolynomial(f) # Wrapper struct that also includes a homogDegree field
julia> pregen = pregen_gpu_pow(f, 5) # Pregenerates constants needed for computation
julia> g_gpu = gpu_pow(f, 5, pregen)
julia> g_oscar = f ^ 5
julia> g_gpu = convert(FqmPolyRingelem, g_gpu)
julia> g_gpu == g_oscar
true
```
