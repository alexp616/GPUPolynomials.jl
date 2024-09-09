#module RandomPolynomials
#
using Combinatorics

"""
Returns a list of all monomials in varibles
vars of degree deg
"""
function allmonomialcombos(vars,deg)
  # for some reason, Combinatorics.multiset_partitions only does partitions which are subsets of the 
  #   array which is passed
  repeated_vars = repeat(vars,deg)
  multiset_combinations(repeated_vars,deg)

end#function

"""
coefficients are uniform random in 0:p-1

i.e. the polynomial is pretty dense, coef is nonzero with probability p-1/p
"""
function random_homog_poly_mod(p,vars,deg)
  nVars = length(vars)
  
  var_combos = collect(allmonomialcombos(vars,deg))
  nMons = length(var_combos)

  # the line where it all happens
  coefs = rand(0:p-1,nMons)

  res = zero(vars[1])
  for i in 1:nMons
    res = res + coefs[i] * prod(var_combos[i])
  end

  res
end#fucntion

function generate_all_monomials(vars, deg)
    var_combos = collect(allmonomialcombos(vars, deg))
    result = map(vec -> prod(vec), var_combos)
end



function random_homog_poly_mod_restricted(p, vars, mons)
    coefs = rand(0:p-1, length(mons))

    res = zero(vars[1])
    for i in eachindex(coefs)
        res += coefs[i] * mons[i]
    end

    return res
end

"""
Returns k random monomials in the variables
vars of degree deg.
"""
function random_monomials(vars,deg,k)
  var_combos = collect(allmonomialcombos(vars,deg))
  prod.(rand(var_combos,k))
end#function

"""
Returns a random polynomial of degree deg
in variables vars with about k 
nonzero coefficients

(there might be a few less if we randomly
pick the same monomial twice)
"""
function random_homog_poly_mod_k_coefs(p,vars,deg,k)

  monomials = random_monomials(vars,deg,k)
  coefs = rand(1:p-1,k)

  sum(coefs .* monomials)
end#function

#end#module