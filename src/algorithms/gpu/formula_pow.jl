
#TODO: 
#   * make the plan be on the GPU, i.e. change the vector type to some 
#   sort of a type parameter.
#   * make output be an array on the gpu (just need KernelAbstractions.zeros

using KernelAbstractions
using Adapt
using Oscar

struct FormulaPowPlanMonomial
    index::Int
    degree::Int
end

#struct FormulaPowPlanTerm#{T}
#    coefficient::Int
#    exp_vec::Vector{Int}
#end

#function vectortype(backend)
#    if backend == CPU()
#        Vector
#    elseif backend == MetalBackend()
#        MtlVector
#    end
#    #TODO add CUDA
#end

"""
A FormulaPowPlan is a struct with the necessary information for
a parallel kernel to evaluate a multiplication formula.

For the struct to be well formed, 
T should be a type of the form MyVector{MyVector{Int}},
and S should be MyVector{MyVector{MyVector{FormulaPowPlanMonomial}}}.
Here, MyVector should be the vector type of the backend you're using,
like CuVector or MtlVector or Vector (for the CPU).

The generic type parameters which enforce this doesn't really exist
in Julia... 

This should work:

{T1 <: AbstractArray{Int}, 
T <: AbstractArray{T1},
S1 <: AbstractArray{FormulaPowPlanMonomial},
S2 <: AbstractArray{S1},
S <: AbstractArray{S1}}

but then you have to put in all the types explicitly to construct it,
which is totally not what we want to do.

"""
struct FormulaPowPlan
    n::Int
    d::Int
    pow::Int
    coefficients::AbstractVector
    exponent_vectors::AbstractVector
end

"""
Using Oscar, compute a generic formula for the power of a polynomial
of fixed degree and number of variables
"""
function generic_power_formula(n,d,pow)

# using DeRham
# TODO: DeRham.jl really cannot be a dependency here. 


  # I think :lex is backwards compared to Oscar's default
  exps_list = DeRham.gen_exp_vec(n,d,:invlex)


  R, avars = polynomial_ring(ZZ,:a => 1:length(exps_list))
  S, xvars = polynomial_ring(R,n)

  mons = [prod(xvars .^ exps) for exps in exps_list]

  genericpoly = sum(mons .* avars)

  #@assert(all(MMPSingularities.DeRham.gen_exp_vec(5,8,:invlex) .== leading_exponent_vector.(terms(p))))

  genericpoly^pow
end

"""
Inputs a generic polynomial formula and outputs the 
"""
function expressions_from_poly(p)
    cs = coefficients(p)

    #expressions = Vector{Vector{FormulaPowPlanTerm}}()
    coeffs = Vector{Vector{Int}}()
    exp_vecs = Vector{Vector{Vector{FormulaPowPlanMonomial}}}()

    for c in cs
        #ts = Vector{FormulaPowPlanTerm}()
        term_coeffs = Vector{Int}()
        term_exp_vecs = Vector{Vector{FormulaPowPlanMonomial}}()

        for t in terms(c)
            vs = vars(t)
            coeff = leading_coefficient(t)
            exp_vec = Vector{FormulaPowPlanMonomial}()

            #println("vs: $vs")
            for v in vs
                ind = var_index(v)
                deg = degree(t,ind)
                #println("v: $v, ind: $ind")

                push!(exp_vec,FormulaPowPlanMonomial(ind,deg))
            end

            push!(term_coeffs,coeff)
            push!(term_exp_vecs,exp_vec)
        end

        push!(coeffs,term_coeffs)
        push!(exp_vecs,term_exp_vecs)
    end

    (coeffs,exp_vecs)
end

function formula_pow_plan(n,d,pow,backend::Backend)
    genericpow = generic_power_formula(n,d,pow)
    (coefs,exp_vecs) = expressions_from_poly(genericpow)

    to_b(x) = adapt(backend, x)
    vv = to_b.(coefs)
    println(typeof(vv))
    vecvec_to_b(x) = to_b(to_b.(x))
    b_coefs = to_b(vv)
    println("i am ok")
    b_exp_vecs = to_b(vecvec_to_b.(exp_vecs))

    FormulaPowPlan(n,d,pow,b_coefs,b_exp_vecs)
end

function formula_pow_plan(n,d,pow,sample_coefs_array::AbstractArray)
    backend = get_backend(sample_coefs_array)

    formula_pow_plan(n,d,pow,backend)
end

@kernel function formula_pow_kernel!(original,output,plan)
    i = @index(Global)
    coefs = plan.coefficients[i]
    exp_vecs = plan.exponent_vectors[i]

    result = 0

    for i in 1:length(coefs)
        termres = coefs[i]
        ev = exp_vecs[i]

        for m in ev
            val = original[m.index]
            deg = m.degree
            termres *= val^deg
        end
            
        result += termres
    end
        
    output[i] = result
end

function formula_pow(original,plan)
    backend = get_backend(plan.coefficients)
    output = zeros(Int,length(plan.coefficients))

    # make output be in the backend
    kernel = formula_pow_kernel!(backend)

    kernel(original,output,plan, ndrange = length(output))
    output
end

# TODO: Move the below to the tests folder

#using Oscar
using Test

function test_formula_pow()
    # tests on the cpu right now
    n = 5
    d = 4
    pow = 2

    evs = DeRham.gen_exp_vec(n,d,:invlex)
    S, xvars = polynomial_ring(ZZ,n)
    allones = sum([prod(xvars .^ vec) for vec in evs])
    square = allones^pow
    naive_powered = collect(coefficients(square))

    original = ones(Int,70)
    plan = formula_pow_plan(n,d,pow,original)

    powered = formula_pow(original,plan)

    @test all(powered .== naive_powered)
end

function test_formula_pow_metal()
    # this is currently broken because metal doesn't support
    # arrays of arrays or any sort of arrays of pointers.

    # Instead of re-architecturing this for now, I think we should
    # just try to test it on CUDA.
    
    n = 5
    d = 4
    pow = 2

    evs = DeRham.gen_exp_vec(n,d,:invlex)
    S, xvars = polynomial_ring(ZZ,n)
    allones = sum([prod(xvars .^ vec) for vec in evs])
    square = allones^pow
    naive_powered = collect(coefficients(square))

    original = Metal.ones(Int,70)
    plan = formula_pow_plan(n,d,pow,original)

    powered = formula_pow(original,plan)

    @test all(Array(powered) .== naive_powered)
end
