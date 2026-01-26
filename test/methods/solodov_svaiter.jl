"""
    Solodov-Svaiter Projection Method for Variational Inequalities

Implementation of the projection algorithms from:
Solodov, M. V., & Svaiter, B. F. (1999). A New Projection Method for Variational 
Inequality Problems. SIAM J. Control Optim., 37(3), 765-776.
"""

using LinearAlgebra

#==============================================================================#
#                           ALGORITHM 2.1                                       #
#==============================================================================#

"""
    solodov_svaiter_vip(x₀, operator_F, projector_C; kwargs...)

Algorithm 2.1 from Solodov & Svaiter (1999).
"""
function solodov_svaiter_vip(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    projector_C::Function;
    max_iteration::Int=30_000,
    ε::Float64=1e-6,
    σ::T=T(0.3),
    γ::T=T(0.5),
    max_linesearch::Int=100,
    verbose::Bool=false,
    kwargs...) where {T}

    @assert zero(T) < σ < one(T) "σ must be in (0,1)"
    @assert zero(T) < γ < one(T) "γ must be in (0,1)"

    xₖ = copy(x₀)
    rₖ = similar(x₀)
    zₖ = similar(x₀)
    Fzₖ = similar(x₀)
    temp = similar(x₀)

    solved = false
    tired = false
    iter = 0
    status = :Tired
    error = one(T)

    while !(solved || tired)
        iter += 1

        # r(xₖ) = xₖ - P_C[xₖ - F(xₖ)]
        Fxₖ = operator_F(xₖ)
        @. temp = xₖ - Fxₖ
        proj_temp = projector_C(temp)
        @. rₖ = xₖ - proj_temp

        error = norm(rₖ)
        if error ≤ ε
            solved = true
            status = :Solved
            break
        end

        # Armijo linesearch
        rₖ_norm_sq = error^2
        ηₖ = one(T)
        linesearch_success = false

        for k in 0:max_linesearch
            ηₖ = γ^k
            @. zₖ = xₖ - ηₖ * rₖ
            copyto!(Fzₖ, operator_F(zₖ))

            if dot(Fzₖ, rₖ) ≥ σ * rₖ_norm_sq
                linesearch_success = true
                break
            end
        end

        if !linesearch_success
            verbose && @warn "Linesearch failed at iteration $iter"
            tired = true
            status = :LinesearchFailed
            break
        end

        # Projeção em C ∩ Hₖ: primeiro projeta em H, depois em C
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
    max_linesearch::Int=100,
    verbose::Bool=false,
    kwargs...) where {T}

    @assert zero(T) < σ < one(T) "σ must be in (0,1)"
    @assert zero(T) < γ < one(T) "γ must be in (0,1)"
    @assert θ > one(T) "θ must be > 1"

    xₖ = copy(x₀)
    rₖ_μ = similar(x₀)
    zₖ = similar(x₀)
    Fxₖ = similar(x₀)
    Fzₖ = similar(x₀)
    temp = similar(x₀)

    solved = false
    tired = false
    iter = 0
    status = :Tired
    error = one(T)

    ηₖ₋₁ = η₀

    while !(solved || tired)
        iter += 1

        # μₖ = min(θ * ηₖ₋₁, 1)
        μₖ = min(θ * ηₖ₋₁, one(T))

        # r(xₖ, μₖ) = xₖ - P_C[xₖ - μₖ F(xₖ)]
        copyto!(Fxₖ, operator_F(xₖ))
        @. temp = xₖ - μₖ * Fxₖ
        proj_temp = projector_C(temp)
        @. rₖ_μ = xₖ - proj_temp

        rₖ_μ_norm = norm(rₖ_μ)
        error = rₖ_μ_norm

        if rₖ_μ_norm ≤ ε
            solved = true
            status = :Solved
            break
        end

        # Armijo linesearch: ⟨F(xₖ - γᵏ r), r⟩ ≥ (σ/μₖ) ‖r‖²
        rₖ_μ_norm_sq = rₖ_μ_norm^2
        threshold = (σ / μₖ) * rₖ_μ_norm_sq

        ηₖ = one(T)
        linesearch_success = false

        for k in 0:max_linesearch
            ηₖ = γ^k
            @. zₖ = xₖ - ηₖ * rₖ_μ
            copyto!(Fzₖ, operator_F(zₖ))

            if dot(Fzₖ, rₖ_μ) ≥ threshold
                linesearch_success = true
                break
            end
        end

        if !linesearch_success
            verbose && @warn "Linesearch failed at iteration $iter"
            tired = true
            status = :LinesearchFailed
            break
        end

        ηₖ₋₁ = ηₖ

        # Projeção em C ∩ Hₖ
        xₖ = project_onto_C_cap_H(xₖ, zₖ, Fzₖ, projector_C)

        iter ≥ max_iteration && (tired = true)

        verbose && println("Iter $iter: ‖r(x,μ)‖ = $rₖ_μ_norm, η = $ηₖ, μ = $μₖ")
    end

    return xₖ, error, iter, status
end


#==============================================================================#
#                    PROJECTION ONTO C ∩ H (SIMPLIFIED)                         #
#==============================================================================#

"""
    project_onto_C_cap_H(x, z, Fz, projector_C)

Compute P_{C∩H}[x] where H = {w : ⟨Fz, w - z⟩ ≤ 0}.

Uses the simplified approach: first project x onto H, then project onto C.
This works because by construction x ∉ H but the solution x* ∈ C ∩ H.
"""
function project_onto_C_cap_H(x::AbstractVecOrMat{T},
    z::AbstractVecOrMat{T},
    Fz::AbstractVecOrMat{T},
    projector_C::Function) where {T}

    Fz_norm_sq = dot(Fz, Fz)

    if Fz_norm_sq < eps(T)
        return projector_C(x)
    end

    # Projeção em H: x̄ = x - max(0, ⟨Fz, x-z⟩/‖Fz‖²) * Fz
    inner_prod = dot(Fz, x - z)

    if inner_prod ≤ zero(T)
        # x já está em H
        x_bar = copy(x)
    else
        coeff = inner_prod / Fz_norm_sq
        x_bar = x - coeff * Fz
    end

    # Projeção em C
    return projector_C(x_bar)
end


#==============================================================================#
#                         EXPORTS                                               #
#==============================================================================#

export solodov_svaiter_vip,
    solodov_svaiter_vip_v2,
    project_onto_C_cap_H