struct NTTPowPlan{T<:Integer}
    pow::Int
    forwardPlan::NTTPlan{T}
    invPlan::INTTPlan{T}

    function NTTPowPlan(len::Integer, pow::Integer, prime::T) where T<:Integer
        len = Int(len)
        pow = Int(pow)
        npru = primitive_nth_root_of_unity(len, prime)
        forwardPlan, invPlan = plan_ntt(len, prime, npru)

        return new{T}(pow, forwardPlan, invPlan)
    end
end

# Assumes vec is already properly padded
function ntt_pow(vec, plan::NTTPowPlan{T}) where T<:Integer
    @assert length(vec) == plan.forwardPlan.n
    @assert eltype(vec) == T

    p = plan.forwardPlan.p
    ntt!(vec, plan.forwardPlan)
    map!(x -> power_mod(x, p, plan.pow), vec, vec)
    intt!(vec, plan.invPlan)

    return nothing
end