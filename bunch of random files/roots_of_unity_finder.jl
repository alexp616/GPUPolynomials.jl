function nthPrincipalRootOfUnity(p, n)
    # numbers satisfying a^n = 1
    candidates = []
    results = []

    for i in 2:p-1
        println("i^n: $(i^n), i: $i")
        if i^n % p == 1
            push!(candidates, i)
        end
    end

    # println("candidates: $candidates")
    while length(candidates) > 0
        flag = true
        for k in 1:n-1
            if candidates[1] ^ k % p == 1
                flag = false
                break
            end
        end
        if flag 
            push!(results, candidates[1])
        end
        popfirst!(candidates)
    end

    return results
end