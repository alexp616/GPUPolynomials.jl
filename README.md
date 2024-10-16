# GPUPolynomials.jl

[![Build Status](https://github.com/alexp616/GPUPolynomials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexp616/GPUPolynomials.jl/actions/workflows/CI.yml?query=branch%3Amain)

The GPUPolynomials.jl package aims to allow users to perform fast polynomial arithmetic when used with [Oscar.jl](https://github.com/oscar-system/Oscar.jl).

Because GPUPolynomials.jl was developed for the sole purpose of raising multivariate homogeneous polynomials to powers, as used in [MMPSingularities.jl](https://github.com/jjgarzella/MMPSingularities.jl), this is the only functionality implemented for now.

In the future, I (Alex Pan) aim to implement all of the methods of [AbstractAlgebra.jl](https://nemocas.github.io/AbstractAlgebra.jl/stable/mpolynomial/) for univariate and multivariate polynomials, but admittedly, GPU-parallelized algorithms are only suitable for a small subset of sparse multivariate problems, and for many use cases, GPUPolynomials.jl will simply call the implementations already defined in Oscar.jl.

## Usage
To import GPUPolynomials, simply do:
```
pkg> add GPUPolynomials
```
As of now, GPUPolynomials requires Julia version 1.11 in order for [BitIntegers.jl](https://github.com/rfourquet/BitIntegers.jl) to work with CUDA.jl.

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