"""
    malitsky_2015(x₀, operator_F, projector_C; max_iteration=30_000, ε=1e-8, L=nothing, λ=nothing, verbose=false)

Implements Algorithm 3.1 (Projected Reflected Gradient Method) from Malitsky (2015):
"Projected reflected gradient methods for monotone variational inequalities"
SIAM J. Optim. Vol. 25, No. 1, pp. 502–520

This method solves variational inequality problems (VIP): Find x* ∈ C such that
    ⟨F(x*), x - x*⟩ ≥ 0  ∀x ∈ C

The algorithm uses the iteration:
    x_{n+1} = P_C(x_n - λF(2x_n - x_{n-1}))

# Arguments
- `x₀`: Initial point (also used as y₀ = x₀)
- `operator_F`: Function that computes F(x)
- `projector_C`: Function that projects onto the closed convex set C
- `max_iteration`: Maximum number of iterations (default: 30,000)
- `ε`: Convergence tolerance (default: 1e-8)
- `L`: Lipschitz constant of F (required if λ is not provided)
- `λ`: Step size (default: computed as (√2-1)/L if not provided)
- `verbose`: Print iteration information (default: false)

# Returns
- `xₖ`: Approximate solution
- `error`: Final residual error r(x_n, y_n)
- `index_iteration`: Number of iterations performed
- `status`: :Solved or :Tired

# Notes
- Requires λ ∈ (0, (√2-1)/L) for convergence
- The default λ = (√2-1)/L ≈ 0.414/L is theoretically safe
- Convergence is guaranteed under monotonicity and Lipschitz continuity of F
- Has R-linear convergence rate under strong monotonicity
"""
function malitsky_2015(x₀::AbstractVecOrMat{T},
                       operator_F::Function,
                       projector_C::Function;
                       max_iteration::Int=30_000,
                       ε::Float64=1e-8,
                       L::Union{Nothing,Real}=nothing,
                       λ::Union{Nothing,Real}=nothing,
                       verbose::Bool=false,
                       kwargs...) where {T}
    
    # Determine step size
    if λ === nothing
        if L === nothing
            error("Either λ or L must be provided")
        end
        λ = (sqrt(2) - 1) / L  # Theoretical safe choice: λ ∈ (0, (√2-1)/L)
        verbose && println("Using λ = $(λ) = (√2-1)/L with L = $(L)")
    else
        if L !== nothing
            # Verify that λ is in the valid range
            λ_max = (sqrt(2) - 1) / L
            if λ >= λ_max
                @warn "λ = $(λ) is not < (√2-1)/L = $(λ_max). Convergence not guaranteed!"
            end
        end
    end
    
    # Initialization
    xₖ = copy(x₀)
    xₖ_prev = copy(x₀)  # x_{-1} = x_0
    yₖ = copy(x₀)        # y_0 = x_0 (used for reflected point)
    F_yₖ = similar(x₀)
    xₖ_new = similar(x₀)
    
    solved = false
    tired = false
    error = one(T)
    index_iteration = 0
    status = :Tired
    
    while !(solved || tired)
        # Compute reflected point: y_n = 2x_n - x_{n-1}
        @. yₖ = 2 * xₖ - xₖ_prev
        
        # Evaluate F at reflected point
        copyto!(F_yₖ, operator_F(yₖ))
        
        # Projection step: x_{n+1} = P_C(x_n - λF(y_n))
        @. xₖ_new = xₖ - λ * F_yₖ
        copyto!(xₖ_new, projector_C(xₖ_new))
        
        # Compute residual: r(x_n, y_n) = ||y_n - P_C(y_n - λF(y_n))|| + ||x_n - y_n||
        # Simplified check: ||x_{n+1} - y_n|| (since x_{n+1} = P_C(x_n - λF(y_n)))
        temp_residual = similar(x₀)
        @. temp_residual = yₖ - λ * F_yₖ
        copyto!(temp_residual, projector_C(temp_residual))
        @. temp_residual = yₖ - temp_residual  # y_n - P_C(y_n - λF(y_n))
        
        residual_1 = norm(temp_residual)
        residual_2 = norm(xₖ - yₖ)
        error = residual_1 + residual_2
        
        index_iteration += 1
        
        # Check convergence
        if error ≤ ε
            solved = true
            status = :Solved
        end
        
        # Check max iterations
        if index_iteration ≥ max_iteration
            tired = true
        end
        
        if verbose
            println("Iteration $index_iteration: error = $error, ||x_{n+1} - y_n|| = $(norm(xₖ_new - yₖ))")
        end
        
        # Update for next iteration
        copyto!(xₖ_prev, xₖ)
        copyto!(xₖ, xₖ_new)
    end
    
    return xₖ, error, index_iteration, status
end


"""
    malitsky_2015_adaptive(x₀, operator_F, projector_C; max_iteration=30_000, ε=1e-8, λ_init=0.01, α=0.4, λ_max=1e3, verbose=false)

Implements Algorithm 4.2 (Modified Projected Reflected Gradient Method with adaptive step size) 
from Malitsky (2015). This version does not require knowledge of the Lipschitz constant L.

The algorithm adaptively adjusts the step size λ_n to satisfy:
    λ_n ||F(y_n) - F(y_{n-1})|| ≤ α ||y_n - y_{n-1}||

where α ∈ (0, √2 - 1) ≈ (0, 0.414).

# Arguments
- `x₀`: Initial point in C
- `operator_F`: Function that computes F(x)
- `projector_C`: Function that projects onto the closed convex set C
- `max_iteration`: Maximum number of iterations (default: 30,000)
- `ε`: Convergence tolerance (default: 1e-8)
- `λ_init`: Initial step size (default: 0.01)
- `α`: Parameter for step size control, must be in (0, √2-1) (default: 0.4)
- `λ_max`: Maximum allowed step size (default: 1e3)
- `verbose`: Print iteration information (default: false)

# Returns
- `xₖ`: Approximate solution
- `error`: Final residual error
- `index_iteration`: Number of iterations performed
- `status`: :Solved or :Tired
- `λ_history`: History of step sizes used

# Notes
- Does not require knowledge of Lipschitz constant L
- Step size is adapted at each iteration
- Typically requires only one projection per iteration
- More practical than Algorithm 3.1 for real applications
"""
function malitsky_2015_adaptive(x₀::AbstractVecOrMat{T},
                                operator_F::Function,
                                projector_C::Function;
                                max_iteration::Int=30_000,
                                ε::Float64=1e-8,
                                λ_init::Float64=0.01,
                                α::Float64=0.4,
                                λ_max::Float64=1e3,
                                verbose::Bool=false,
                                kwargs...) where {T}
    
    # Verify parameter constraints
    sqrt2_minus_1 = sqrt(2) - 1
    if α <= 0 || α >= sqrt2_minus_1
        error("α must be in (0, √2-1) ≈ (0, 0.414), got α = $α")
    end
    
    # Initialization (following Algorithm 4.2 step 1)
    xₖ = copy(x₀)
    
    # Compute y₀ = P_C(x₀ - λ_{-1}F(x₀))
    λ_prev = λ_init
    F_xₖ = operator_F(x₀)
    temp = x₀ - λ_prev * F_xₖ
    y_prev = projector_C(temp)
    
    # Compute λ₀
    F_y_prev = operator_F(y_prev)
    dist_y = norm(x₀ - y_prev)
    dist_F = norm(F_xₖ - F_y_prev)
    
    if dist_F > 1e-14
        λₖ = min(α * dist_y / dist_F, λ_max)
    else
        λₖ = λ_max
    end
    
    # Compute x₁
    copyto!(xₖ, projector_C(x₀ - λₖ * F_y_prev))
    
    xₖ_prev = copy(x₀)
    yₖ = similar(x₀)
    τₖ = 1.0
    
    solved = false
    tired = false
    error = one(T)
    index_iteration = 1  # We already did one iteration
    status = :Tired
    
    λ_history = Float64[λₖ]
    
    while !(solved || tired)
        # Step 2: Compute y_n = 2x_n - x_{n-1}
        @. yₖ = 2 * xₖ - xₖ_prev
        
        # Evaluate F(y_n) and F(y_{n-1})
        F_yₖ = operator_F(yₖ)
        F_y_prev_current = operator_F(y_prev)
        
        # Compute adaptive step size
        dist_y = norm(yₖ - y_prev)
        dist_F = norm(F_yₖ - F_y_prev_current)
        
        if dist_F > 1e-14
            λ_proposed = α * dist_y / dist_F
        else
            λ_proposed = λ_max
        end
        
        # Include constraint from (1 + τ_{n-1})/τ * λ_{n-1}
        λ_bound = ((1 + τₖ) / 1.0) * λ_prev  # τ_n = 1 in step 2
        λₖ = min(λ_proposed, λ_bound, λ_max)
        
        # Compute x_{n+1} = P_C(x_n - λ_n F(y_n))
        xₖ_new = projector_C(xₖ - λₖ * F_yₖ)
        
        # Compute residual for convergence check
        temp_proj = projector_C(yₖ - λₖ * F_yₖ)
        residual_1 = norm(yₖ - temp_proj)
        residual_2 = norm(xₖ - yₖ)
        error = residual_1 + residual_2
        
        index_iteration += 1
        
        # Check convergence
        if error ≤ ε
            solved = true
            status = :Solved
        end
        
        # Check max iterations
        if index_iteration ≥ max_iteration
            tired = true
        end
        
        if verbose && (index_iteration % 100 == 0 || solved || tired)
            println("Iteration $index_iteration: error = $error, λ = $λₖ")
        end
        
        # Update for next iteration
        copyto!(xₖ_prev, xₖ)
        copyto!(xₖ, xₖ_new)
        copyto!(y_prev, yₖ)
        λ_prev = λₖ
        
        push!(λ_history, λₖ)
    end
    
    return xₖ, error, index_iteration, status, λ_history
end
