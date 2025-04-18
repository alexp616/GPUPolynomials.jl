# GPUPolynomials.jl

GPUPolynomials.jl aims to implement fast polynomial arithmetic over finite fields, intended for use in pure-mathematical research at scale. Development so far has followed a paper-centric model, where features are developed as necessary for the papers of the authors. If you would like to use GPUPolynomials.jl in your paper, we welcome new contributors. 

As of now, the two main exported types are CuZZPolyRingElem, and CuZZMPolyRingElem. CuZZPolyRingElem implements univariate addition, subtraction, multiplication (FFT), and powering (FFT) on the GPU, and CuZZMPolyRingElem implements homogeneous polynomial powering (FFT) on the GPU. GPUPolynomials.jl was developed for this last operation, so it is the most optimized. Non-homogeneous polynomial powering is easier to implement than homogeneous, but we haven't needed it so badly as to justify completely optimizing it.

One goal of GPUPolynomials.jl is to interface with [Oscar.jl](https://github.com/oscar-system/Oscar.jl), which uses FLINT's GMP integers in its backend. This has not been fully achieved, and is indeed quite complicated - variable-width integers are possible on the GPU, but there isn't a Julia implementation (that we know of) yet. We handle large integers with BitIntegers, which works fine, but on-the-fly bound finding for operations other than powering, selecting primes, etc. isn't the easiest, and would require much more implementation. 

## Usage
To add GPUPolynomials, simply do:
```julia
pkg> add https://github.com/alexp616/GPUPolynomials.jl
```

To raise a homogeneous polynomial to a power:
```julia
using GPUPolynomials, Oscar

R, (x, y, z, w) = polynomial_ring(ZZ, 4)
f = x^4 + 2*y^4 + 3*z^4 + 4*w^4
cu_f = cu(f) # Send to GPU
cu_g = cu_f ^ 5 # Raise to 5th power
g_oscar = f ^ 5
g = ZZMPolyRingElem(cu_g) # Send back to CPU
@assert g == g_oscar
```

The algorithm powering uses is the FFT, which is a dense algorithm - so, ideally, we want to use GPUPolynomials.jl with dense inputs. The following benchmark compares the time taken to raise a 4-variate, 16-homogeneous polynomial with coefficients less than 5 to the 5th power:

```julia
using GPUPolynomials, Oscar, BenchmarkTools, CUDA

pow, p = 5, 5
R, vars = polynomial_ring(ZZ, 4)
f = GPUPolynomials.random_homog_poly_mod(p, vars, 16)
display(@benchmark $f ^ $pow) # 644 ms

cu_f = cu(f)
plan = MPowPlan(cu_f, pow)
cu_f.opPlan = plan

display(@benchmark CUDA.@sync $cu_f ^ $pow) # 4 ms
```

The full potential of GPUPolynomials.jl is obtained when performing many "similar" computations. This means same polynomial degree, same bound on coefficients, same number of variables. Then, a plan can be cached in the `opPlan` field of the polynomial, containing all needed information.
