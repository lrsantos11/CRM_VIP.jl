"""
    carm_vip(x₀, operator_F, functions_g, subgrads_g; max_iteration=3_000, ε = 1e-6)

Implementation of the  Cirumcentered-Approximated Reflections Method for solving variational inquality problems, i.e.,  to find ``x^*`` lying in a convex set ``C`` such that:
```math
    \\langle F(x^* ), x - x^* \\rangle \\leq 0, \\forall x \\in C
```
Usually, the set ``C`` is defined as the intersection of convex sets ``C_i``. 

Reference: Behling, et al (2025). On circumcentered direct methods for monotone variational inequality problem. 

# Arguments
- `x₀`: Initial point
- `operator_F`: Function that calculates the continous mapping ``F(x)``
- `functions_g`: Vector of functions that define the convex sets ``C_i``
- `subgrads_g`: Vector of subgradients of the functions ``g_i``
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance

# Returns
- `xₖ`: Approximate solution
- `error`: Final error (distance between the last two iterations)
- `index_iteration`: Number of iterations performed
"""
function carm_vip(x₀::AbstractVecOrMat{T},
                 operator_F::Function,
                 functions_g::Vector{Function},
                 subgrads_g::Vector{Function};
                 max_iteration::Int=3_000,
                 β::Function = (i) -> 1 / (i^2),
                 ε::Float64 = 1e-6) where {T}
    # Initializations
    xₖ = copy(x₀)
    num_sets = length(functions_g)
    vₖ = [similar(xₖ) for _ in 1:num_sets]
    wₖ = similar(xₖ)
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
        xₖ = carm_step!(xₖ, vₖ, wₖ, functions_g, subgrads_g)


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
    carm_step!(xₖ, vₖ, wₖ, functions_g, subgrads_g)
    Performs a single step of the CARM algorithm.
    The function updates the vector xₖ by subtracting the mean of the vectors vₖ and wₖ.
    The vectors vₖ and wₖ are computed using the provided functions and subgradients.
"""



function carm_step!(xₖ, vₖ, wₖ, functions_g, subgrads_g)
    m = length(functions_g)
    # Compute the mean of the vectors vₖ
    computevₖ!(xₖ, vₖ, functions_g, subgrads_g)
    copyto!(wₖ, mean(vₖ))
    αₖ = (mapreduce(v -> dot(v, v), +, vₖ)) / (m * dot(wₖ, wₖ))
    wₖ *= αₖ
    xₖ .-= wₖ
end







"""
    computevₖ(xₖ, func_g, ∂g)
    Computes the vector vₖ for the given function and its subgradient at xₖ.
    The vector is computed as vₖ = max(0, f(xₖ) + β) / ||∂g(xₖ)||² * ∂g(xₖ)
    where β is a perturbation parameter.
    The function func_g is a function that takes a vector x and returns a scalar value.
    The subgradient ∂g is a function that takes a vector x and returns a vector of the same size as x.
    The function returns the computed vector vₖ.
"""
function computevₖⁱ(xₖ::AbstractVector{T}, 
                   func_g::Function, 
                   ∂g::Function) where {T}
    gx = func_g(xₖ)
    ∂gx = ∂g(xₖ)
    return (max(zero(T), gx) / dot(∂gx, ∂gx)) .* ∂gx
end


"""
computevₖ!(xₖ, vₖ, functions_g, subgrads_g)

"""
function computevₖ!(
    xₖ,
    vₖ,
    functions_g,
    subgrads_g
)
    # numthreads = Threads.nthreads()
    for (vₖi, func_g, ∂g) in zip(vₖ, functions_g, subgrads_g)
        @views @inbounds copyto!(vₖi, computevₖⁱ(xₖ, func_g, ∂g))
    end
end
