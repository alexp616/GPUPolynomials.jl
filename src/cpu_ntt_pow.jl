include("ntt_utils.jl")

function cpu_ntt(p1, prime, npru, len, log2length; butterflied = false)
    if !butterflied
        perm = Array(generate_butterfly_permutations(len))
        p1 = p1[perm]
    end

    for i in 1:log2length
        m = 1 << i
        m2 = m >> 1
        alpha = 1
        alpha_m = power_mod(npru, len ÷ m, prime)
        for j in 0:m2-1
            for k in j:m:len-1
                t = alpha * p1[k + m2 + 1]
                u = p1[k + 1]

                p1[k + 1] = faster_mod((u + t), prime)
                p1[k + m2 + 1] = faster_mod((u - t), prime)
            end
            alpha *= alpha_m
            alpha = faster_mod(alpha, prime)
        end
    end

    return p1
end

function cpu_intt(vec, prime, npru, len, log2length, pregenButterfly = nothing)
    if pregenButterfly === nothing
        pregenButterfly = generate_butterfly_permutations(length)
    end

    arg1 = vec[pregenButterfly]
    result = cpu_ntt(arg1, prime, mod_inverse(npru, prime), len, log2length, butterflied = true)

    result = map(x -> x * mod_inverse(len, prime) % prime, result)
    return result
end

function cpu_pow(p1::Array{Int, 1}, pow; prime, npru, len = -1, pregenButterfly = nothing)
    if len == -1
        len = nextpow(2, (length(p1) - 1) * pow + 1)
    end
    log2length = Int(log2(len))
    finalLength = (length(p1) - 1) * pow + 1

    if pregenButterfly === nothing
        pregenButterfly = Array(generate_butterfly_permutations(len))
    end

    @assert length(pregenButterfly) == len "pregenerated butterfly doesn't have same length as input"

    append!(p1, zeros(Int, len - length(p1)))

    p1 = p1[pregenButterfly]

    cpu_ntt(p1, prime, npru, len, log2length, butterflied = true)

    p1 = map(x -> power_mod(x, pow, prime), p1)

    ans = cpu_intt(p1, prime, npru, len, log2length, pregenButterfly)

    return ans[1:finalLength]
end