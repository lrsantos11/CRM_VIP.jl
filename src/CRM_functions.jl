using ThreadsX

####################################
# CRM Functions
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




####################################
# PACA Functions
"""
    computevₖ(x₀, func_f, ∂f)

"""
function computevₖⁱ(x::Vector{T}, func_f, ∂f;
    ϵ::Real=0.0 # perturbation
) where {T}
    fx = func_f(x)
    ∂fx = ∂f(x)
    return (max(zero(T), fx + ϵ) / dot(∂fx, ∂fx)) .* ∂fx
end

"""
    computevₖ!(vₖ, x, Functions, Subgrads)

"""
function computevₖ!(vₖ::Vector{Vector{T}}, x::Vector{T}, Functions, Subgrads, m;
    ϵ::Number=0.0 # perturbation
) where {T}
    # numthreads = Threads.nthreads()
    for i in 1:m
        @views @inbounds copyto!(vₖ[i], computevₖⁱ(x, Functions[i], Subgrads[i], ϵ=ϵ))
    end
end


""" 
    paca_step!(yₖ, xₖ, approx_projections)
Performs one step of the Circumcentered Reflections Method on the product space using the same ideias of PACA method as in Behling et al. (2024)
# Arguments
- `xₖ`: Current point
- `functions_gi`: Vector of functions that define the convex sets ``C_i``
- `subgradients_gi`: Vector of subgradient functions that define the convex sets ``C_i``
# Returns
- `xₖ`: Updated point
"""

function paca_step!(yₖ::Vector{T}, xₖ::Vector{T},
    vₖ::Vector{Vector{T}},
    wₖ::Vector{T},
    functions_gi::Vector{Function},
    subgradients_gi::Vector{Function}) where {T}
    # Initialize variables
    m = length(functions_gi)
    invm = inv(m)
    ϵₖ = 0.0
    # Compute vₖ
    computevₖ!(vₖ, xₖ, functions_gi, subgradients_gi, m, ϵ=ϵₖ)
    copyto!(wₖ, mean(vₖ))
    αₖ = (mapreduce(v -> dot(v, v), +, vₖ) * invm) / dot(wₖ, wₖ)
    wₖ *= αₖ
    yₖ .-= wₖ
    return yₖ
end