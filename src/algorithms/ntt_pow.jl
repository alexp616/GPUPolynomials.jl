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

Base.eltype(::Type{NTTPowPlan{T}}) where T = T

# Assumes vec is already properly padded
function ntt_pow!(vect::CuVector{T}, plan::NTTPowPlan{T}) where T<:Integer
    @assert length(vect) == plan.forwardPlan.n
    @assert eltype(vect) == T

    reducer = plan.forwardPlan.reducer

    ntt!(vect, plan.forwardPlan, true)
    broadcast_pow!(vect, plan.pow, reducer)
    intt!(vect, plan.invPlan, true)

    return 
end

function broadcast_pow!(vec::CuVector{T}, pow::Int, m::CudaNTTs.Reducer{T}) where T<:Integer
    kernel = @cuda launch=false broadcast_pow_kernel!(vec, pow, m)
    config = launch_configuration(kernel.fun)
    threads = min(length(vec), Base._prevpow2(config.threads))
    blocks = div(length(vec), threads)

    kernel(vec, pow, m; threads = threads, blocks = blocks)

    return nothing
end

@inbounds function broadcast_pow_kernel!(vec::CuDeviceVector{T}, pow::Int, m::CudaNTTs.Reducer{T}) where T<:Integer
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    vec[idx] = CudaNTTs.power_mod(vec[idx], pow, m)

    return nothing
end

