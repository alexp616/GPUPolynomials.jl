function get_int_type(n::Integer)
    return eval(Symbol("Int", n))
end

function get_uint_type(n::Integer)
    return eval(Symbol("UInt", n))
end