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