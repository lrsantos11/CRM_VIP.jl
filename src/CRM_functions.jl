using ThreadsX
using Statistics: mean
using Base.Threads: nthreads

const PARALLEL_THRESHOLD = 8  # Número mínimo de funções para computação paralela

####################################
# CRM Functions


####################################
# PACA Functions (Optimized)

"""
    computevₖⁱ!(vᵢ, x, func_f, ∂f; ϵ=0.0)

Computes vᵢ in-place and returns ||vᵢ||².

The vector vᵢ is computed as:
    vᵢ = max{0, f(x) + ϵ} / ||∂f(x)||² * ∂f(x)
"""
function computevₖⁱ!(
    vᵢ::Vector{T},
    x::Vector{T},
    func_f,
    ∂f;
    ϵ::Real=0.0
) where {T}
    fx = func_f(x) + ϵ

    if fx > zero(T)
        ∂fx = ∂f(x)
        ∂fx_norm_sq = dot(∂fx, ∂fx)
        coef = fx / ∂fx_norm_sq
        @. vᵢ = coef * ∂fx
        # ||vᵢ||² = coef² * ||∂fx||²
        return coef^2 * ∂fx_norm_sq
    else
        fill!(vᵢ, zero(T))
        return zero(T)
    end
end



"""
    computevₖ!(vₖ, wₖ, x, Functions, Subgrads; ϵ=0.0)

Computes all vᵢ vectors and accumulates w = (1/m)Σvᵢ.
Returns Σ||vᵢ||² for use in αₖ calculation.
"""

function computevₖ!(
    vₖ::Vector{Vector{T}},
    wₖ::Vector{T},
    x::Vector{T},
    Functions,
    Subgrads;
    ϵ::Real=0.0,
    parallel::Bool=true
) where {T}
    m = length(Functions)

    # Usa versão paralela só se vale a pena
    if parallel && m > PARALLEL_THRESHOLD && nthreads() > 1
        return computevₖ_parallel!(vₖ, wₖ, x, Functions, Subgrads, ϵ=ϵ)
    else
        return computevₖ_sequential!(vₖ, wₖ, x, Functions, Subgrads, ϵ=ϵ)
    end
end


function computevₖ_sequential!(
    vₖ::Vector{Vector{T}},
    wₖ::Vector{T},
    x::Vector{T},
    Functions,
    Subgrads;
    ϵ::Real=0.0
) where {T}
    m = length(Functions)
    fill!(wₖ, zero(T))
    sum_vi_norm_sq = zero(T)

    @inbounds for i in 1:m
        vi_norm_sq = computevₖⁱ!(vₖ[i], x, Functions[i], Subgrads[i], ϵ=ϵ)
        sum_vi_norm_sq += vi_norm_sq
        @. wₖ += vₖ[i]
    end

    @. wₖ /= m
    return sum_vi_norm_sq
end


function computevₖ_parallel!(
    vₖ::Vector{Vector{T}},
    wₖ::Vector{T},
    x::Vector{T},
    Functions,
    Subgrads;
    ϵ::Real=0.0
) where {T}
    m = length(Functions)

    # Computa cada vᵢ em paralelo e coleta ||vᵢ||²
    vi_norm_sqs = ThreadsX.map(eachindex(Functions)) do i
        fx = Functions[i](x) + ϵ
        if fx > zero(T)
            ∂fx = Subgrads[i](x)
            ∂fx_norm_sq = dot(∂fx, ∂fx)
            coef = fx / ∂fx_norm_sq
            @. vₖ[i] = coef * ∂fx
            coef^2 * ∂fx_norm_sq
        else
            fill!(vₖ[i], zero(T))
            zero(T)
        end
    end

    # Acumula wₖ (sequencial - rápido)
    fill!(wₖ, zero(T))
    @inbounds for i in 1:m
        @. wₖ += vₖ[i]
    end
    @. wₖ /= m

    return sum(vi_norm_sqs)
end

""" 
    paca_step!(yₖ, xₖ, vₖ, wₖ, functions_gi, subgradients_gi; ϵ=0.0)

Performs one step of the PACA/ACRM method.

Implements the circumcentered approximate projection step:
    yₖ = xₖ - αₖ * wₖ

where:
    wₖ = (1/m) Σᵢ vᵢ
    αₖ = Σᵢ||vᵢ||² / (m ||wₖ||²)
    vᵢ = max{0, gᵢ(xₖ)} / ||∇gᵢ(xₖ)||² * ∇gᵢ(xₖ)

# Arguments
- `yₖ`: Output vector (will be modified)
- `xₖ`: Current point
- `vₖ`: Pre-allocated vectors for vᵢ
- `wₖ`: Pre-allocated buffer for w
- `functions_gi`: Vector of constraint functions gᵢ
- `subgradients_gi`: Vector of subgradient functions ∇gᵢ
- `ϵ`: Perturbation parameter (default: 0.0)

# Returns
- `yₖ`: Updated point
"""
function paca_step!(
    yₖ::Vector{T},
    xₖ::Vector{T},
    vₖ::Vector{Vector{T}},
    wₖ::Vector{T},
    functions_gi,
    subgradients_gi;
    ϵ::Real=0.0
) where {T}
    m = length(functions_gi)

    # Compute vₖ, wₖ and Σ||vᵢ||² in one pass
    sum_vi_norm_sq = computevₖ!(vₖ, wₖ, xₖ, functions_gi, subgradients_gi, ϵ=ϵ)

    # Compute αₖ = Σ||vᵢ||² / (m ||wₖ||²)
    wₖ_norm_sq = dot(wₖ, wₖ)

    if wₖ_norm_sq > eps(T)
        αₖ = sum_vi_norm_sq / (m * wₖ_norm_sq)
        @. yₖ = xₖ - αₖ * wₖ
    else
        # xₖ is already feasible (or very close)
        copyto!(yₖ, xₖ)
    end

    return yₖ
end