"""
extragradient_vip(x₀, operator_F, projector_C; max_iteration=3_000, ε = 1e-6)

    This function implements the extragradient method for solving variational inequalities from Korpelevich and Antipin.
    It takes an initial point x₀, an operator F, and a set of projection functions.
    The function iteratively updates the point xₖ using the extragradient method until convergence or maximum iterations are reached.
    The function returns the final point, the error, and the number of iterations.

# Arguments
- `x₀`: Initial point
- `operator_F`: Function that calculates the continuous mapping ``F(x)``
- `projector_C`: Function that projects onto  closed convex set ``C``
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance
# Returns
- `xₖ`: Approximate solution
- `error`: Final error (distance between the last two iterations)

"""

function extragradient_vip(x₀::AbstractVecOrMat{T},
                            operator_F::Function,
                            projector_C::Function;
                            max_iteration::Int=3_000,
                            ε::Float64 = 1e-6, kwargs...) where {T}
    # Initializations
    xₖ = copy(x₀)
    yₖ = similar(x₀)
    solved = false
    tired = false
    error = one(T)
    index_iteration = 1
    γ = 0.1*one(T) # Step size for the Korpelevich method
    status = :Tired
    while !(solved || tired)
        
        copyto!(yₖ, projector_C(xₖ - γ * operator_F(xₖ)))
        # Point update using Subgradients as Approximate Projections
        copyto!(xₖ, projector_C(xₖ - γ * operator_F(yₖ)))
        index_iteration += 1

        # Convergence check
        error = norm(xₖ - yₖ)
        (error ≤ ε) && (solved = true)
        # Maximum iteration check
        (index_iteration ≥ max_iteration) && (tired = true)
    end
    solved ? status = :Solved : nothing
    return xₖ, error, index_iteration, status
end