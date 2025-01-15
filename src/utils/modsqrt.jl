function legendre_symbol(a::T, p::T) where T<:Integer
    ls = powermod(a, (p - 1) ÷ 2, p)
    if ls == p - 1 
        return -1
    else
        return ls
    end
end

function modsqrt(a::T, p::T) where T<:Integer
    if legendre_symbol(a, p) != 1
        return 0
    elseif a == 0
        return 0
    elseif p == 2
        return p
    elseif p % 4 == 3
        return powermod(a, (p + 1) ÷ 4, p)
    end

    s = p - 1
    e = 0
    while s % 2 == 0
        s ÷= 2
        e += 1
    end

    n = T(2)

    while legendre_symbol(n, p) != -1
        n += T(1)
        if n > p
            throw("something messed up")
        end
    end

    x = powermod(a, (s + 1) ÷ 2, p)
    b = powermod(a, s, p)
    g = powermod(n, s, p)
    r = e

    while true
        t = b
        m = 0
        
        for i in 0:r-1
            if t == 1
                break
            end
            t = powermod(t, 2, p)
            m += 1
        end

        if m == 0
            return x
        end

        gs = powermod(g, 2 ^ (r - m - 1), p)
        g = mul_mod(gs, gs, p)
        x = mul_mod(x, gs, p)
        b = mul_mod(b, g, p)
        r = m
    end
end