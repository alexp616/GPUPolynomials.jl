function get_int_type(n)
    return eval(Symbol("Int", n))
end

function get_uint_type(n)
    return eval(Symbol("UInt", n))
end