"""
    Solodov-Svaiter Projection Method for Variational Inequalities

Implementation of the projection algorithms from:
Solodov, M. V., & Svaiter, B. F. (1999). A New Projection Method for Variational 
Inequality Problems. SIAM J. Control Optim., 37(3), 765-776.

The method solves VI(F,C): find x* ∈ C such that ⟨F(x*), x - x*⟩ ≥ 0 ∀x ∈ C

Key features:
- Only requires continuity of F and pseudomonotonicity (weaker than Lipschitz)
- Uses Armijo-type linesearch (no need to know Lipschitz constant)
- Two projections per iteration: one onto C, one onto C ∩ Hₖ
- Geometric interpretation via separating hyperplanes
"""

using LinearAlgebra

#==============================================================================#
#                           ALGORITHM 2.1                                       #
#==============================================================================#

"""
    solodov_svaiter_vip(x₀, operator_F, projector_C; kwargs...)

Algorithm 2.1 from Solodov & Svaiter (1999).

# Algorithm Description
Given xⁱ, compute the residual r(xⁱ) = xⁱ - P_C[xⁱ - F(xⁱ)].
If r(xⁱ) = 0, stop (xⁱ is the solution).
Otherwise:
1. Find ηᵢ = γᵏⁱ where kᵢ is the smallest nonnegative integer k satisfying:
   ⟨F(xⁱ - γᵏr(xⁱ)), r(xⁱ)⟩ ≥ σ ‖r(xⁱ)‖²
2. Set zⁱ = xⁱ - ηᵢ r(xⁱ)
3. Define halfspace Hᵢ = {x : ⟨F(zⁱ), x - zⁱ⟩ ≤ 0}
4. Compute xⁱ⁺¹ = P_{C∩Hᵢ}[xⁱ]

# Arguments
- `x₀::AbstractVecOrMat{T}`: Initial point in C
- `operator_F::Function`: Continuous operator F: ℝⁿ → ℝⁿ (pseudomonotone)
- `projector_C::Function`: Projection onto closed convex set C

# Keyword Arguments
- `max_iteration::Int=30_000`: Maximum iterations
- `ε::Float64=1e-6`: Convergence tolerance on ‖r(xₖ)‖
- `σ::T=0.3`: Armijo parameter σ ∈ (0,1)
- `γ::T=0.5`: Backtracking parameter γ ∈ (0,1)
- `max_linesearch::Int=50`: Maximum linesearch iterations
- `verbose::Bool=false`: Print iteration info

# Returns
- `xₖ`: Approximate solution
- `error`: Final residual norm ‖r(xₖ)‖
- `iterations`: Number of iterations
- `status`: `:Solved`, `:Tired`, or `:LinesearchFailed`

# Complexity
- 2 projections per iteration (onto C and onto C ∩ Hₖ)
- Multiple F evaluations per iteration (due to linesearch)

# Reference
Solodov, M. V., & Svaiter, B. F. (1999). A New Projection Method for Variational 
Inequality Problems. SIAM J. Control Optim., 37(3), 765-776.
"""
function solodov_svaiter_vip(x₀::AbstractVecOrMat{T},
                             operator_F::Function,
                             projector_C::Function;
                             max_iteration::Int=30_000,
                             ε::Float64=1e-6,
                             σ::T=T(0.3),
                             γ::T=T(0.5),
                             max_linesearch::Int=50,
                             verbose::Bool=false,
                             kwargs...) where {T}
    
    @assert zero(T) < σ < one(T) "σ must be in (0,1)"
    @assert zero(T) < γ < one(T) "γ must be in (0,1)"
    
    # Pre-allocations
    xₖ = copy(x₀)
    rₖ = similar(x₀)      # Residual r(x) = x - P_C[x - F(x)]
    zₖ = similar(x₀)      # Linesearch point
    Fzₖ = similar(x₀)     # F(zₖ)
    temp = similar(x₀)
    
    solved = false
    tired = false
    iter = 0
    status = :Tired
    error = one(T)
    
    while !(solved || tired)
        iter += 1
        
        # Compute residual: r(xₖ) = xₖ - P_C[xₖ - F(xₖ)]
        Fxₖ = operator_F(xₖ)
        @. temp = xₖ - Fxₖ
        proj_temp = projector_C(temp)
        @. rₖ = xₖ - proj_temp
        
        # Check convergence
        error = norm(rₖ)
        if error ≤ ε
            solved = true
            status = :Solved
            break
        end
        
        # Armijo linesearch to find ηₖ = γ^kₖ
        # Find smallest k such that ⟨F(xₖ - γᵏ r(xₖ)), r(xₖ)⟩ ≥ σ ‖r(xₖ)‖²
        rₖ_norm_sq = error^2
        ηₖ = one(T)
        linesearch_success = false
        
        for k in 0:max_linesearch
            ηₖ = γ^k
            @. zₖ = xₖ - ηₖ * rₖ
            copyto!(Fzₖ, operator_F(zₖ))
            
            inner_prod = dot(Fzₖ, rₖ)
            
            if inner_prod ≥ σ * rₖ_norm_sq
                linesearch_success = true
                break
            end
        end
        
        if !linesearch_success
            verbose && @warn "Linesearch failed at iteration $iter"
            tired = true
            status = :LinesearchFailed
            continue
        end
        
        # Projection onto C ∩ Hₖ where Hₖ = {x : ⟨F(zₖ), x - zₖ⟩ ≤ 0}
        xₖ = project_onto_C_cap_H(xₖ, zₖ, Fzₖ, projector_C)
        
        iter ≥ max_iteration && (tired = true)
        
        verbose && println("Iter $iter: ‖r(x)‖ = $error, η = $ηₖ")
    end
    
    return xₖ, error, iter, status
end


#==============================================================================#
#                           ALGORITHM 2.2                                       #
#==============================================================================#

"""
    solodov_svaiter_vip_v2(x₀, operator_F, projector_C; kwargs...)

Algorithm 2.2 from Solodov & Svaiter (1999) - Practical variant with adaptive step sizes.

This variant uses r(x, μ) = x - P_C[x - μF(x)] with adaptive μₖ = min(θ·ηₖ₋₁, 1),
coordinating the first projection step with the linesearch for better practical performance.

# Arguments
Same as `solodov_svaiter_vip`, plus:
- `θ::T=4.0`: Step size growth parameter (θ > 1)
- `η₀::T=1.0`: Initial step size

# Reference
Solodov, M. V., & Svaiter, B. F. (1999). SIAM J. Control Optim., 37(3), 765-776.
"""
function solodov_svaiter_vip_v2(x₀::AbstractVecOrMat{T},
                                operator_F::Function,
                                projector_C::Function;
                                max_iteration::Int=30_000,
                                ε::Float64=1e-6,
                                σ::T=T(0.3),
                                γ::T=T(0.5),
                                θ::T=T(4.0),
                                η₀::T=one(T),
                                max_linesearch::Int=50,
                                verbose::Bool=false,
                                kwargs...) where {T}
    
    @assert zero(T) < σ < one(T) "σ must be in (0,1)"
    @assert zero(T) < γ < one(T) "γ must be in (0,1)"
    @assert θ > one(T) "θ must be > 1"
    
    # Pre-allocations
    xₖ = copy(x₀)
    rₖ_μ = similar(x₀)    # Residual r(x, μ)
    zₖ = similar(x₀)      # Linesearch point
    Fxₖ = similar(x₀)
    Fzₖ = similar(x₀)
    temp = similar(x₀)
    
    solved = false
    tired = false
    iter = 0
    status = :Tired
    error = one(T)
    
    # Adaptive step size
    ηₖ₋₁ = η₀
    
    while !(solved || tired)
        iter += 1
        
        # Adaptive μₖ = min(θ * ηₖ₋₁, 1)
        μₖ = min(θ * ηₖ₋₁, one(T))
        
        # Compute residual r(xₖ, μₖ) = xₖ - P_C[xₖ - μₖ F(xₖ)]
        copyto!(Fxₖ, operator_F(xₖ))
        @. temp = xₖ - μₖ * Fxₖ
        proj_temp = projector_C(temp)
        @. rₖ_μ = xₖ - proj_temp
        
        # Check convergence
        rₖ_μ_norm = norm(rₖ_μ)
        error = rₖ_μ_norm
        
        if rₖ_μ_norm ≤ ε
            solved = true
            status = :Solved
            break
        end
        
        # Armijo linesearch: find ηₖ = γ^kₖ * μₖ
        # satisfying ⟨F(xₖ - γᵏμₖ r(xₖ,μₖ)), r(xₖ,μₖ)⟩ ≥ (σ/μₖ) ‖r(xₖ,μₖ)‖²
        rₖ_μ_norm_sq = rₖ_μ_norm^2
        threshold = (σ / μₖ) * rₖ_μ_norm_sq
        
        ηₖ = μₖ
        linesearch_success = false
        
        for k in 0:max_linesearch
            ηₖ = (γ^k) * μₖ
            @. zₖ = xₖ - ηₖ * rₖ_μ
            copyto!(Fzₖ, operator_F(zₖ))
            
            inner_prod = dot(Fzₖ, rₖ_μ)
            
            if inner_prod ≥ threshold
                linesearch_success = true
                break
            end
        end
        
        if !linesearch_success
            verbose && @warn "Linesearch failed at iteration $iter"
            tired = true
            status = :LinesearchFailed
            continue
        end
        
        # Store ηₖ for next iteration
        ηₖ₋₁ = ηₖ
        
        # Projection onto C ∩ Hₖ
        xₖ = project_onto_C_cap_H(xₖ, zₖ, Fzₖ, projector_C)
        
        iter ≥ max_iteration && (tired = true)
        
        verbose && println("Iter $iter: ‖r(x,μ)‖ = $rₖ_μ_norm, η = $ηₖ, μ = $μₖ")
    end
    
    return xₖ, error, iter, status
end


#==============================================================================#
#                    PROJECTION ONTO C ∩ H                                      #
#==============================================================================#

"""
    project_onto_C_cap_H(x, z, Fz, projector_C; kwargs...)

Compute P_{C∩H}[x] where H = {w : ⟨Fz, w - z⟩ ≤ 0}.

Uses Dykstra's alternating projection algorithm.

# Arguments
- `x`: Point to project
- `z`: Point defining the halfspace
- `Fz`: F(z), normal direction of halfspace  
- `projector_C`: Projection onto C

# Returns
- Projection of x onto C ∩ H
"""
function project_onto_C_cap_H(x::AbstractVecOrMat{T},
                              z::AbstractVecOrMat{T},
                              Fz::AbstractVecOrMat{T},
                              projector_C::Function;
                              max_dykstra_iter::Int=100,
                              dykstra_tol::Float64=1e-10) where {T}
    
    Fz_norm_sq = dot(Fz, Fz)
    
    # Handle degenerate case
    if Fz_norm_sq < eps(T)
        return projector_C(x)
    end
    
    # Projection onto halfspace H = {w : ⟨Fz, w - z⟩ ≤ 0}
    function project_H(w)
        inner_prod = dot(Fz, w - z)
        if inner_prod ≤ zero(T)
            return copy(w)
        else
            coeff = inner_prod / Fz_norm_sq
            return w - coeff * Fz
        end
    end
    
    # Check if x is already in H and P_C[x] ∈ H
    if dot(Fz, x - z) ≤ zero(T)
        proj_c = projector_C(x)
        if dot(Fz, proj_c - z) ≤ zero(T)
            return proj_c
        end
    end
    
    # Dykstra's algorithm for projection onto C ∩ H
    y = copy(x)
    p = zeros(T, length(x))  # Increment for C
    q = zeros(T, length(x))  # Increment for H
    
    for _ in 1:max_dykstra_iter
        y_old = copy(y)
        
        # Project onto C
        y_temp = y + p
        y_new = projector_C(y_temp)
        p = y_temp - y_new
        y = y_new
        
        # Project onto H  
        y_temp = y + q
        y_proj_h = project_H(y_temp)
        q = y_temp - y_proj_h
        y = y_proj_h
        
        # Check convergence
        if norm(y - y_old) < dykstra_tol
            break
        end
    end
    
    return y
end


#==============================================================================#
#                         EXPORTS                                               #
#==============================================================================#

export solodov_svaiter_vip,
       solodov_svaiter_vip_v2,
       project_onto_C_cap_H
