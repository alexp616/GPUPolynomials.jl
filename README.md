# GPUPolynomials.jl
(Work in progress)

A library for fast polynomial arithmetic using CUDA.jl

## Usage

As of now, because the problem of interest I developed this for was for raising homogeneous, multivariate polynomials to powers, the only generally useful function is `gpu_pow()`. To use it, you can construct a `HomogeneousPolynomial` object,

```julia
coeffs = [1, 2, 3, 4]
degrees = [
    4 0 0 0
    0 4 0 0
    0 0 4 0
    0 0 0 4
]

hp = HomogeneousPolynomial(coeffs, degrees)
```

and pass it in as such:
```julia
hp2 = gpu_pow(hp, 5)
```

The HomogeneousPolynomial struct is also interfaceable with Oscar:

```julia
using Oscar

R, vars = polynomial_ring(GF(7), 4)
f = 2*x1^2*x2^2 + x4^4

hp = HomogeneousPolynomial(f)

hp2 = gpu_pow(hp, 2)

result = convert_to_oscar(hp2)
```

## Future features
Here is a list of features that I will be able to implement in the forseeable future:
- Univariate polynomials
- Non-homogeneous multivariate polynomials
- Polynomial addition / subtraction (gpu_merge, reduce_by_key)
- Polynomial multiplication (fast fourier transform)
- Polynomial evaluation

The grand idea for this library is for it to be able to seamlessly integrate with Oscar, and to use the GPU whenever the cases are big enough for the GPU to be faster than Oscar.