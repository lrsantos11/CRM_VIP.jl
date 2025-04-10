"""
    crm_vip(x₀, operator_F, projections; max_iteration=3_000, ε = 1e-6)

Implementation of the  Cirumcentered Reflections Method for solving variational inquality problems, i.e.,  to find ``x^*`` lying in a convex set ``C`` such that:
```math
    \\langle F(x^* ), x - x^* \\rangle \\leq 0, \\forall x \\in C
```
Usually, the set ``C`` is defined as the intersection of convex sets ``C_i``. 

Reference: Behling, et al (2025). On circumcentered direct methods for monotone variational inequality problem. 

# Arguments
- `x₀`: Initial point
- `operator_F`: Function that calculates the continous mapping ``F(x)``
- `projections`: Vector of functions that define the convex sets ``C_i``
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance

# Returns
- `xₖ`: Approximate solution
- `error`: Final error (distance between the last two iterations)
- `index_iteration`: Number of iterations performed
"""
function crm_vip(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    projections::Vector{Function},
    max_iteration::Int=3_000,
    β::Function=(i) -> 1 / (i^2),
    ε::Float64=1e-6) where {T}
    # Initializations
    xₖ = copy(x₀)
    index_iteration = 1
    x_temp = similar(x₀)
    solved = false
    tired = false
    error = zero(T)
    while !(solved || tired)
        copyto!(x_temp, xₖ)

        # Parameter calculations
        βₖ = β(index_iteration)
        F_xₖ = operator_F(xₖ)
        ηₖ = max(1, norm(F_xₖ))
        xₖ .-= (βₖ / ηₖ) * F_xₖ

        # Point update using Subgradients as Approximate Projections
        xₖ = crm_step!(xₖ, projections)


        index_iteration += 1


        # Convergence check
        error = norm(xₖ - x_temp)
        (error ≤ ε) && (solved = true)
        # Maximum iteration check
        (index_iteration ≥ max_iteration) && (tired = true)


    end


    return xₖ, error, index_iteration
end


""" 
    crm_step!(xₖ, projections)
Performs one step of the Circumcentered Reflections Method on the product space
# Arguments
- `xₖ`: Current point
- `projections`: Vector of functions that define the convex sets ``C_i``
# Returns
- `xₖ`: Updated point
"""
function crm_step!(xₖ::AbstractVecOrMat{T},
    projections::Vector{Function}) where {T}
    # Initialize variables
    num_sets = length(projections)
    # Transfer xₖ to the Product Space
    xCRM_prod = Vector{T}[copy(xₖ) for _ in 1:num_sets]
    xCRM_RA = copy(xCRM_prod)

    # Proceed with the Reflection on the Product Space
    for index in eachindex(projections)
        xCRM_RA[index] = projections[index](xCRM_prod[index])
    end
    xCRM_RA .*= 2*one(T)
    xCRM_RA .-= xCRM_prod
    # Reflection on the Diagonal space
    xCRM_RBRA = reflect_diagonal_prodspace(xCRM_RA)
    # Circumcenter of the reflected points
    find_circumcenter!(xCRM_prod, [xCRM_prod, xCRM_RA, xCRM_RBRA])

    return xCRM_prod[end]

   
end





### Circumcenter utils

"""
find_circumcenter!(CC, X)

Finds the Circumcenter of linearly independent vectors ``x_0,x_1,…,x_m``, belonging to ``X``,
as described in [^Behling2018a] and [^Behling2018b].

[^Behling2018a]: Behling, R., Bello Cruz, J.Y., Santos, L.-R.:
Circumcentering the Douglas–Rachford method. Numer. Algorithms. 78(3), 759–776 (2018).
[doi:10.1007/s11075-017-0399-5](https://doi.org/10.1007/s11075-017-0399-5)
[^Behling2018b]: Behling, R., Bello Cruz, J.Y., Santos, L.-R.:
On the linear convergence of the circumcentered-reflection method. Oper. Res. Lett. 46(2), 159-162 (2018).
[doi:10.1016/j.orl.2017.11.018](https://doi.org/10.1016/j.orl.2017.11.018)
"""
function find_circumcenter!(CC, X)
    # Finds the Circumcenter of  linearly independent points  X = [X1, X2, X3, ... Xn]
    # println(typeof(X))
    lengthX = length(X)
    if lengthX == 1
        return X[1]
    elseif lengthX == 2
        return 0.5 * (X[1] + X[2])
    end
    V = []
    b = Float64[]
    # Forms V = [X[2] - X[1] ... X[n]-X[1]]
    # and b = [dot(V[1],V[1]) ... dot(V[n-1],V[n-1])]
    for ind in 2:lengthX
        difXnX1 = X[ind] - X[1]
        push!(V, difXnX1)
        push!(b, dot(difXnX1, difXnX1))
    end

    # Forms Gram Matrix
    dimG = lengthX - 1
    G = diagm(b)

    for irow in 1:(dimG-1)
        for icol in (irow+1):dimG
            G[irow, icol] = dot(V[irow], V[icol])
            G[icol, irow] = G[irow, icol]
        end
    end
    try 
        L = cholesky(G)
    catch e
        if isa(e, ArgumentError)
            @warn "Circumcenter matrix is not positive definite. Circumcenter is not unique"
            y = G \ b
        else
            y = L \ b
        end 
    end
    copyto!(CC, X[1])
    for ind in 1:dimG
        CC += 0.5 * y[ind] * V[ind]
    end
    return CC
end

find_circumcenter!(X) = find_circumcenter!(similar(X[1]), X)


####################################
"""
project_diagonal_prodspace(X)


"""
function project_diagonal_prodspace(X::AbstractArray)
    inner_proj = mean(X)
    proj = similar(X)
    for index in eachindex(proj)
        proj[index] = inner_proj
    end
    return proj
end

""" 
    reflect_diagonal_prodspace(X)
"""

function reflect_diagonal_prodspace(X::AbstractArray)
    return 2 * project_diagonal_prodspace(X) - X
end


