

struct FormulaPowPlanMonomial
    index::Int
    degree::Int
end

struct FormulaPowPlanTerm
    coefficient::Int
    monomials::Vector{FormulaPowPlanMonomial}
end

struct FormulaPowPlan
    n::Int
    d::Int
    pow::Int
    expressions::Vector{Vector{FormulaPowPlanTerm}}
end


function generic_power_formula(n,d,pow)

# using DeRham

  # I think :lex is backwards compared to Oscar's default
  exps_list = MMPSingularities.DeRham.gen_exp_vec(n,d,:invlex)


  R, avars = polynomial_ring(ZZ,:a => 1:length(exps_list))
  S, xvars = polynomial_ring(R,n)

  mons = [prod(xvars .^ exps) for exps in exps_list]

  genericpoly = sum(mons .* avars)

  @assert(all(MMPSingularities.DeRham.gen_exp_vec(5,8,:invlex) .== leading_exponent_vector.(terms(p))))

  genericpoly^pow
end

function expressions_from_poly(p)
    cs = coefficients(p)

    expressions = Vector{Vector{FormulaPowPlanTerm}}()

    for c in cs
        ts = Vector{FormulaPowPlanTerm}()

        for t in terms(c)
            vs = vars(t)
            coeff = leading_coefficient(t)

            mons = Vector{FormulaPowPlanMonomial}()

            for v in vs
                ind = var_index(v)
                deg = degree(t,ind)

                push!(mons,FormulaPowPlanMonomial(ind,deg))
            end

            push!(ts,FormulaPowPlanTerm(coeff,mons))
        end

        push!(expressions,ts)
    end

    expressions
end

function formula_pow_plan(n,d,pow)
    genericpow = generic_power_formula(n,d,pow)
    exps = expressions_from_poly(genericpow)

    FormulaPowPlan(n,d,pow,exps)
end

using KernelAbstractions

@kernel function formula_pow_kernel!(original,output,plan)
    i = @index(Global)
    terms = plan.expressions[i]

    result = 0

    for t in terms
        termres = t.coefficient

        for m in t.monomials
            val = original[m.index]
            deg = m.degree
            termres *= val^deg
        end
            
        result += termres
    end
        
    output[i] = result
end

function formula_pow(original,plan)
    backend = get_backend(original)
    output = zeros(Int,length(plan.expressions))

    # make output be in the backend
    
    kernel = formula_pow_kernel!(backend)

    kernel(original,output,plan, ndrange = length(output))
    output
end

# TODO: Move the below to the tests folder

using Oscar
using Test
    

function test_formula_pow()
    # tests on the cpu right now
    n = 5
    d = 4
    pow = 2

    evs = MMPSingularities.DeRham.gen_exp_vec(n,d,:invlex)
    allones = sum([prod(xvars .^ vec) for vec in evs])
    square = allones^pow
    naive_powered = collect(coefficients(square))

    plan = formula_pow_plan(n,d,pow)
    original = ones(Int,70)

    powered = formula_pow(original,plan)

    @test all(powered .== naive_powered)
end
