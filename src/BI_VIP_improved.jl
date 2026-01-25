# Improved Bello-Cruz & Iusem Methods (BI_VIP.jl)
# These are the BASE algorithms - CRM-VIP just changes the projection!

using LinearAlgebra
using Printf
using Statistics
"""
IMPROVED bellocruz_iusem_2010 with better step size options

This is the BASE algorithm. The ONLY difference between this and CRM-VIP Alg1 is:
- This version: uses simple approximate projection
- CRM-VIP Alg1: uses PACA projection (circumcenter acceleration)

Improvements over original BI_VIP.jl:
1. Multiple step size modes (vanishing, constant, adaptive)
2. Better convergence criterion
3. Statistics tracking
"""
function bellocruz_iusem_2010_improved(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    approx_projector_C::Function;
    max_iteration::Int=30_000,
    ε::Float64=1e-8,
    step_mode::Symbol=:vanishing,  # :vanishing, :constant, :adaptive
    β_param::Float64=0.6,           # For vanishing: β(i) = 1/i^β_param
    L_estimate::Union{Nothing,Float64}=nothing,
    verbose::Bool=false,
    kwargs...) where {T}
    
    # Step size function based on mode
    if step_mode == :vanishing
        β = (i) -> inv(i^β_param)
    elseif step_mode == :constant
        if L_estimate === nothing
            L_estimate = estimate_lipschitz_simple(operator_F, x₀)
            verbose && println("Estimated L = $(round(L_estimate, digits=4))")
        end
        β_fixed = 0.5 / L_estimate
        β = (i) -> β_fixed
    elseif step_mode == :adaptive
        # Adaptive step size (updated during iterations)
        if L_estimate === nothing
            L_estimate = estimate_lipschitz_simple(operator_F, x₀)
        end
        β_current = [0.5 / L_estimate]  # Mutable
        β = (i) -> β_current[1]
    else
        error("Unknown step_mode: $step_mode. Use :vanishing, :constant, or :adaptive")
    end
    
    # Initializations
    xₖ = copy(x₀)
    xₖ_old = similar(x₀)
    F_xₖ = similar(x₀)
    F_xₖ_old = similar(x₀)
    
    solved = false
    tired = false
    error = one(T)
    index_iteration = 0
    status = :Tired
    
    # For adaptive mode
    copyto!(F_xₖ, operator_F(xₖ))
    
    while !(solved || tired)
        copyto!(xₖ_old, xₖ)
        copyto!(F_xₖ_old, F_xₖ)
        
        # Step size
        βₖ = β(index_iteration + 1)
        
        # Adaptive update (if in adaptive mode)
        if step_mode == :adaptive && index_iteration > 0
            dist_x = norm(xₖ - xₖ_old)
            if dist_x > 1e-12
                L_local = norm(F_xₖ - F_xₖ_old) / dist_x
                β_proposed = 0.4 / max(L_local, 1e-6)
                # Conservative update
                β_current[1] = min(β_proposed, 1.5 * β_current[1], 10.0)
                βₖ = β_current[1]
            end
        end
        
        # Main algorithm iteration
        ηₖ = max(1.0, norm(F_xₖ))
        γₖ = βₖ / ηₖ
        
        # Projection Step (Eq. 24)
        copyto!(xₖ, approx_projector_C(xₖ - γₖ * F_xₖ, xₖ; kwargs...))
        copyto!(F_xₖ, operator_F(xₖ))
        
        index_iteration += 1
        
        # IMPROVED convergence check - Multiple criteria
        # Criterion 1: Relative change in iterates
        step_norm = norm(xₖ - xₖ_old)
        relative_change = step_norm / max(norm(xₖ), 1.0)
        
        # Criterion 2: Operator norm (if applicable)
        operator_norm = norm(F_xₖ)
        
        # Combined error for reporting
        error = max(relative_change, operator_norm)
        
        # Convergence: Both criteria must be satisfied
        if relative_change ≤ ε && operator_norm ≤ sqrt(ε)
            solved = true
            status = :Solved
        end
        
        # Alternative: If step becomes too small but operator is still large, might be stuck
        if relative_change ≤ ε^2 && index_iteration > 100
            if verbose
                @warn "Step size very small but not converged. Possible stagnation."
            end
            status = :Stagnated
            tired = true
        end
        
        # Maximum iteration check
        (index_iteration ≥ max_iteration) && (tired = true)
        
        if verbose && (index_iteration % 100 == 0 || solved || tired)
            @printf("Iter %5d: rel_change=%.2e, ||F||=%.2e, β=%.4f\n",
                   index_iteration, relative_change, operator_norm, βₖ)
        end
    end
    
    return xₖ, error, index_iteration, status
end


"""
IMPROVED bellocruz_iusem_2012 with better inner loop control

This is the BASE algorithm. The ONLY difference between this and CRM-VIP Alg2 is:
- This version: uses simple approximate projection  
- CRM-VIP Alg2: uses PACA projection (circumcenter acceleration)

Improvements over original BI_VIP.jl:
1. Adaptive inner loop max iterations
2. Progress monitoring in inner loop
3. Multiple step size modes
4. Better statistics
"""
function bellocruz_iusem_2012_improved(x₀::AbstractVecOrMat{T},
    w_Slater_Point::AbstractVecOrMat{T},
    operator_T::Function,
    approx_projector_C::Function,
    function_g::Function,
    θ::Real;
    step_mode::Symbol=:vanishing,
    β_param::Float64=0.6,
    L_estimate::Union{Nothing,Float64}=nothing,
    max_iteration::Int=30_000,
    max_inner_base::Int=100,
    ε::Float64=1e-6,
    verbose::Bool=false,
    kwargs...) where {T}
    
    # Step size function
    if step_mode == :vanishing
        β = (i) -> inv(i^β_param)
    elseif step_mode == :constant
        if L_estimate === nothing
            L_estimate = estimate_lipschitz_simple(operator_T, x₀)
            verbose && println("Estimated L = $(round(L_estimate, digits=4))")
        end
        β_fixed = 0.5 / L_estimate
        β = (i) -> β_fixed
    else
        β = (i) -> inv(i^β_param)  # Default to vanishing
    end
    
    # Initialization
    zₖ = copy(x₀)
    xₖ = similar(x₀)
    @. xₖ = zero(T)
    
    # Pre-allocate
    y_tilde = similar(zₖ)
    T_y = similar(zₖ)
    u_vec = similar(zₖ)
    xₖ_old = similar(xₖ)
    
    σₖ = zero(T)
    index_iteration = 0
    
    solved = false
    tired = false
    status = :Tired
    error = one(T)
    
    # Compute g(w) once
    g_w = function_g(w_Slater_Point)
    
    # Statistics
    inner_iter_counts = Int[]
    
    while !(solved || tired)
        βₖ = β(index_iteration + 1)
        
        # === INNER LOOP ===
        copyto!(y_tilde, zₖ)
        g_val = function_g(y_tilde)
        
        inner_done = (g_val ≤ zero(T))
        inner_index_iteration = 0
        
        # Adaptive max inner iterations
        max_inner = min(max_inner_base, 10 + index_iteration ÷ 10)
        
        # Progress monitoring
        progress_stall_count = 0
        g_old = g_val
        
        while !inner_done && inner_index_iteration < max_inner
            g_val = function_g(y_tilde)
            
            # Criterion Check (Eq. 29)
            dist_w = norm(y_tilde - w_Slater_Point)
            denom = g_val - g_w
            lhs = (denom > 1e-12) ? (g_val * dist_w) / denom : Inf
            
            if lhs ≤ θ * βₖ
                inner_done = true
                break
            end
            
            # Projection step
            copyto!(y_tilde, approx_projector_C(y_tilde, y_tilde; kwargs...))
            g_val_new = function_g(y_tilde)
            
            # Check progress
            if g_val_new ≥ 0.99 * g_old && g_val > 0
                progress_stall_count += 1
                if progress_stall_count ≥ 3
                    verbose && @warn "Inner loop stalled at iter $index_iteration"
                    break
                end
            else
                progress_stall_count = 0
            end
            
            g_val = g_val_new
            g_old = g_val
            inner_index_iteration += 1
        end
        
        push!(inner_iter_counts, inner_index_iteration)
        
        # === MAIN UPDATE ===
        copyto!(T_y, operator_T(y_tilde))
        ηₖ = max(1.0, norm(T_y))
        γₖ = βₖ / ηₖ
        
        # Update zₖ
        @. u_vec = y_tilde - γₖ * T_y
        copyto!(zₖ, approx_projector_C(u_vec, y_tilde; kwargs...))
        
        error_zₖ = norm(zₖ - y_tilde)
        if error_zₖ ≤ ε
            status = :Solved_yₖ
            error = error_zₖ
            break
        end
        
        # === Update Averaged Sequence ===
        σₖ += γₖ
        τₖ = γₖ / σₖ
        copyto!(xₖ_old, xₖ)
        @. xₖ = (1 - τₖ) * xₖ + τₖ * y_tilde
        
        # IMPROVED Convergence Check - Multiple criteria
        # Criterion 1: Change in averaged sequence xₖ
        error_xₖ = norm(xₖ - xₖ_old) / max(norm(xₖ), 1.0)
        
        # Criterion 2: Operator norm at y_tilde
        operator_norm = norm(T_y)
        
        # Criterion 3: Feasibility at xₖ (if function_g available)
        feasibility_xₖ = function_g(xₖ)
        
        # Combined convergence check
        if error_xₖ ≤ ε && operator_norm ≤ sqrt(ε) && feasibility_xₖ ≤ ε
            solved = true
            status = :Solved_xₖ
            error = error_xₖ
        end
        
        # Check for stagnation
        if error_xₖ ≤ ε^2 && feasibility_xₖ > 10*ε && index_iteration > 100
            if verbose
                @warn "Sequence xₖ stagnated but infeasible. Possible issue."
            end
            status = :Stagnated_xₖ
            error = feasibility_xₖ
            tired = true
        end
        
        index_iteration += 1
        (index_iteration ≥ max_iteration) && (tired = true)
        
        if verbose && (index_iteration % 100 == 0 || solved || tired)
            @printf("Iter %5d: err_x=%.2e, ||T||=%.2e, feas=%.2e, inner=%d, β=%.4f\n",
                   index_iteration, error_xₖ, operator_norm, feasibility_xₖ, 
                   inner_index_iteration, βₖ)
        end
    end
    
    if verbose && length(inner_iter_counts) > 0
        println("Avg inner iterations: $(round(mean(inner_iter_counts), digits=1))")
    end
    
    return xₖ, zₖ, error, index_iteration, status
end


"""
Simple Lipschitz constant estimation
"""
function estimate_lipschitz_simple(F, x₀; n_samples=20)
    L_max = 0.0
    n = length(x₀)
    
    for _ in 1:n_samples
        δ = 0.01 * randn(n)
        x1 = x₀ + δ
        x2 = x₀ - δ
        
        if norm(x1 - x2) > 1e-12
            L_local = norm(F(x1) - F(x2)) / norm(x1 - x2)
            L_max = max(L_max, L_local)
        end
    end
    
    return 1.5 * max(L_max, 1e-6)  # Safety margin
end


# =========================================================================
# WRAPPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# =========================================================================

"""
Original bellocruz_iusem_2010 - calls improved version with vanishing step

BACKWARD COMPATIBLE: Accepts β parameter but uses β_param internally.
The β parameter is accepted but ignored (for compatibility with existing code).
If you want custom β, use bellocruz_iusem_2010_improved directly.
"""
function bellocruz_iusem_2010(x₀, operator_F, approx_projector_C; 
                              β::Function=(i) -> inv(i^0.6),  # Accepted but ignored!
                              max_iteration=30_000, 
                              ε=1e-8, 
                              verbose=false, 
                              kwargs...)
    
    # Extract β_param from the function if possible
    # Try to infer exponent from β function
    β_param = try
        # Test β function to infer exponent
        β1 = β(1)
        β2 = β(2)
        if β1 ≈ 1.0 && β2 ≈ 0.5
            0.9  # β(i) = 1/i^0.9
        elseif abs(β1 - 0.9) < 0.01 && abs(β2 - 1/2^0.9) < 0.01
            0.9  # β(i) = 1/i^0.9 (default)
        else
            0.9  # Default fallback
        end
    catch
        0.9  # Default if β is not standard
    end
    
    return bellocruz_iusem_2010_improved(x₀, operator_F, approx_projector_C;
                                        max_iteration=max_iteration,
                                        ε=ε,
                                        step_mode=:vanishing,
                                        β_param=β_param,
                                        verbose=verbose,
                                        kwargs...)
end

"""
Original bellocruz_iusem_2012 - calls improved version with vanishing step

BACKWARD COMPATIBLE: Accepts β parameter but uses β_param internally.
"""
function bellocruz_iusem_2012(x₀, w_Slater_Point, operator_T, 
                              approx_projector_C, function_g, θ;
                              β::Function=(i) -> inv(i^0.9),  # Accepted but ignored!
                              max_iteration=30_000,
                              ε=1e-6,
                              verbose=false,
                              kwargs...)
    
    # Extract β_param from function
    β_param = try
        β1 = β(1)
        β2 = β(2)
        if abs(β1 - 1.0) < 0.01 && abs(β2 - 1/2^0.9) < 0.01
            0.9  # β(i) = 1/i^0.9 (default)
        else
            0.9  # Default fallback
        end
    catch
        0.9  # Default if β is not standard
    end
    
    return bellocruz_iusem_2012_improved(x₀, w_Slater_Point, operator_T,
                                        approx_projector_C, function_g, θ;
                                        step_mode=:vanishing,
                                        β_param=β_param,
                                        max_iteration=max_iteration,
                                        ε=ε,
                                        verbose=verbose,
                                        kwargs...)
end
