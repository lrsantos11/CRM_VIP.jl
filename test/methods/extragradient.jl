"""
    Extragradient Method for Variational Inequality Problems

Implementation of the classical extragradient method (Korpelevich 1976, Antipin 1976)
for solving VI(F,C): find x* ∈ C such that ⟨F(x*), x - x*⟩ ≥ 0 ∀x ∈ C

References:
- Korpelevich, G. M. (1976). The extragradient method for finding saddle points 
  and other problems. Matecon, 12, 747-756.
- Antipin, A. S. (1976). On a method for convex programs using a symmetrical 
  modification of the Lagrange function. Ekonomika i Matematicheskie Metody, 12, 1164-1173.
"""

#==============================================================================#
#                    KORPELEVICH-ANTIPIN EXTRAGRADIENT                         #
#==============================================================================#

"""
    extragradient_vip(x₀, operator_F, projector_C; kwargs...)

Classical extragradient method (Korpelevich 1976, Antipin 1976) for VI(F,C).

# Algorithm
At each iteration k:
```
    yₖ = P_C(xₖ - γ F(xₖ))
    xₖ₊₁ = P_C(xₖ - γ F(yₖ))
```

Convergence requires γ < 1/L where L is the Lipschitz constant of F.

# Arguments
- `x₀::AbstractVecOrMat{T}`: Initial point
- `operator_F::Function`: Monotone, L-Lipschitz operator F: ℝⁿ → ℝⁿ
- `projector_C::Function`: Projection onto closed convex set C

# Keyword Arguments
- `max_iteration::Int=30_000`: Maximum iterations
- `γ::T=0.1`: Step size (should satisfy γ < 1/L)
- `ε::Float64=1e-6`: Convergence tolerance on ‖xₖ - yₖ‖
- `adaptive::Bool=true`: Enable adaptive step size reduction
- `verbose::Bool=false`: Print iteration info

# Returns
- `xₖ`: Approximate solution
- `error`: Final error ‖xₖ - yₖ‖
- `iterations`: Number of iterations
- `status`: `:Solved` or `:Tired`

# Complexity
- 2 projections per iteration
- 2 operator evaluations per iteration

# Notes
For problems where projection is expensive, consider using:
- `malitsky_2015` or `malitsky_2015_adaptive` (1 projection per iteration)
- `solodov_svaiter_vip` (2 projections, but no Lipschitz constant needed)
"""
function extragradient_vip(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    projector_C::Function;
    max_iteration::Int=30_000,
    γ::T=T(0.1),
    ε::Float64=1e-6,
    adaptive::Bool=true,
    verbose::Bool=false,
    kwargs...) where {T}
    # Pre-allocations
    xₖ = copy(x₀)
    yₖ = similar(x₀)
    Fxₖ = similar(x₀)
    Fyₖ = similar(x₀)
    temp = similar(x₀)

    # State variables
    solved = false
    tired = false
    error = one(T)
    error_old = one(T)
    iter = 0
    status = :Tired

    while !(solved || tired)
        iter += 1

        # Compute F(xₖ)
        copyto!(Fxₖ, operator_F(xₖ))

        # Half-step: yₖ = P_C(xₖ - γ F(xₖ))
        @. temp = xₖ - γ * Fxₖ
        copyto!(yₖ, projector_C(temp))

        # Compute F(yₖ)
        copyto!(Fyₖ, operator_F(yₖ))

        # Full step: xₖ₊₁ = P_C(xₖ - γ F(yₖ))
        @. temp = xₖ - γ * Fyₖ
        copyto!(xₖ, projector_C(temp))

        # Convergence check: ‖xₖ - yₖ‖
        error_old = error
        error = norm(xₖ - yₖ)

        if error ≤ ε
            solved = true
            status = :Solved
        end

        # Adaptive step size reduction when progress stalls or error increases
        if adaptive && (abs(error - error_old) ≤ ε || error > error_old)
            γ /= 2
            verbose && println("  Step size reduced to γ = $γ")
        end

        # Iteration limit
        iter ≥ max_iteration && (tired = true)

        verbose && println("Iter $iter: error = $error, γ = $γ")
    end

    return xₖ, error, iter, status
end

