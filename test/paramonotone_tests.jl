using Distributions
using ProximalOperators

"""
    Problem 2 of   Yu. Malitsky, “Projected Reflected Gradient Methods for Monotone Variational Inequalities,” SIAM J. Optim., vol. 25, no. 1, pp. 502–520, Jan. 2015, doi: 10.1137/14097238X.

    We take F (x) = M x + q with the matrix M randomly generated as suggested in [6]:  M = AAT + B + D,  where every entry of the m×m matrix A and of the m×m skew-symmetric matrix B is uniformly generated from (−5, 5), and every diagonal entry of the m× m diagonal D is uniformly generated from (0, 0.3) (so M is positive definite), with every entry of q uniformly generated from (−500, 0).
"""

function func_prob2_Malitsky(dim_x::Int = 10; T::Type = Float64)
    # Generate random matrix A
    two = T(2)
    d = Uniform(T(-5), T(5))
    A = rand(d, dim_x, dim_x)
    # Generate random skew-symmetric matrix B
    B = rand(d, dim_x, dim_x)
    B = (B - B') / two
    # Generate random diagonal matrix D
    D = Diagonal(rand(Uniform(0,T(0.3)), dim_x))
    # Generate random vector q
    q = rand(Uniform(T(-500),T(0)), dim_x)
    
    # Compute M
    M = A * A' + B + D


    return x -> M*x + q

end


function set_prob2_Malitsky(dim_x::Int = 10, rhs::Int = 10; T::Type = Float64)
    # Define the feasible set C as the non-negative orthant in R^dim_matrix
    return IndAffine(ones(T, dim_x), T(rhs))
end