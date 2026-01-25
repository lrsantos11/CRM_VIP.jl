"""
bellocruz_iusem_2010(x₀, operator_F, approx_projector_C; max_iteration=3_000, ε = 1e-6)

    This function implements Algorithm 1  for solving variational inequalities problems 
    from Bello-Cruz and Iusem (2010).
    It takes an initial point x₀, an operator F, and a set of projection functions.
    The function returns the final point, the error, and the number of iterations.

# Arguments
- `x₀`: Initial point
- `operator_F`: Function that calculates the continuous mapping ``F(x)``
- `approx_projector_C`: Function that approximate projects onto  closed convex set ``C``
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance
# Returns
- `xₖ`: Approximate solution
- `error`: Final error (distance between the last two iterations)

"""



function bellocruz_iusem_2010(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    approx_projector_C::Function;
    max_iteration::Int=30_000,
    ε::Float64=1e-8, 
    β::Function=(i) -> inv(i^(2)), verbose = false,
    kwargs...) where {T}
    # Initializations
    xₖ = copy(x₀)
    xₖ_old = similar(x₀)
    F_xₖ = similar(x₀)
    solved = false
    tired = false
    error = one(T)
    index_iteration = 0
    status = :Tired
    while !(solved || tired)
        copyto!(xₖ_old, xₖ)
        #βₖ is square summable but not  summable, i.e., satisfies Eq. (6-7)
        βₖ = β(index_iteration+1)
        copyto!(F_xₖ, operator_F(xₖ))
        ηₖ = max(1.0, norm(F_xₖ))
        γₖ = βₖ / ηₖ
        # Projection Step (Eq. 24)
        copyto!(xₖ, approx_projector_C(xₖ - γₖ * F_xₖ, xₖ))
        index_iteration += 1

        # Convergence check
        error = norm(xₖ - xₖ_old) / max(norm(xₖ_old), one(T))
        if error ≤ ε
            solved = true
            status = :Solved
        end
        # Maximum iteration check
        (index_iteration ≥ max_iteration) && (tired = true)
        verbose && println("Iteration $index_iteration: error = $error")

    end
    return xₖ, error, index_iteration, status
end



"""
    bellocruz_iusem_2012(z₀, operator_T, approx_projector_C, function_g; max_iteration=3_000, ε=1e-6)

    Implements the "Algorithm A" structure  fby Bello-Cruz and Iusem (2012) for solving variational inequality problems
    using approximate projections onto the constraint set C. 

# Arguments
- `z₀`: Initial point in H
- `operator_T`: Function that calculates the operator T(x)
- `approx_projector_C`: Function P_C(x) that projects onto the constraint set C
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance

# Returns
- `xₖ`: The final averaged iterate (approximate solution)
- `zₖ`: The final primary iterate
- `error`: Norm of the step size (||z_new - z_old||)
- `status`: :Solved or :Tired
"""
function bellocruz_iusem_2012(x₀::AbstractVecOrMat{T},
    w_Slater_Point::AbstractVecOrMat{T},
    operator_T::Function,
    approx_projector_C::Function,
    function_g::Function,
    θ::Real;
    β::Function=(i) -> inv(i^(2)),
    max_iteration::Int=3_000,
    ε::Float64=1e-6, kwargs...) where {T}

    # --- Initialization ---
    # Algorithm A: x⁰ := 0, z⁰ ∈ H
    zₖ = copy(x₀)
    xₖ = similar(x₀)
    @. xₖ = zero(T)

    # Pre-allocate temporary vectors
    y_tilde = similar(zₖ)
    T_y = similar(zₖ)
    u_vec = similar(zₖ)
    z_old = similar(zₖ)
    xₖ_old = similar(xₖ)

    σₖ = zero(T) # Sigma accumulator (Eq. 33)
    index_iteration = 0    # Iteration counter (0-based to match paper indices)

    solved = false
    tired = false
    status = :Tired
    error = one(T)
    inner_done = false

    # Compute g(w) once
    g_w = function_g(w_Slater_Point)

    while !(solved || tired)
        # --- Step Size Calculation ---
        # βₖ sequence (example: 1/k^2)
        # Note: using index_iteration+1 to avoid division by zero if index_iteration starts at 0
        βₖ = β(index_iteration + 1)

        # --- Step 1: Compute y_tilde ---
      
        # If zₖ is feasible, y_tilde = zₖ.
        copyto!(y_tilde, zₖ)
        g_y_tilde = function_g(y_tilde)

        g_y_tilde ≤ zero(T) ? inner_done = true : inner_done = false

        inner_index_iteration = 0
        while !inner_done
            g_val = function_g(y_tilde)
            
            # --- Criterion Check (Eq. 29) ---
            # j(k) := min { j ≥ 0 : ... }
            dist_w = norm(y_tilde - w_Slater_Point)
            
            # Calculate LHS of (29)
            # Avoid division by zero: if g(y) ≈ g(w), we are likely feasible or w is bad.
            denom = g_val - g_w
            lhs = (denom > 1e-12) ? (g_val * dist_w) / denom : Inf

            if lhs ≤ θ * βₖ
                # Criterion met: Stop inner loop
                inner_done = true
            else
                # --- Step (28) ---
                # y^{k, j+1} := P_{C_{k,j}}(y^{k,j})
                # We use the provided projector function here.
                copyto!(y_tilde, approx_projector_C(y_tilde, y_tilde; kwargs...))
                # copyto!(y_tilde, y_next)
                
                inner_index_iteration += 1
                # Safety break
                if inner_index_iteration > max_iteration
                        @warn("Warning: Inner loop max iterations reached at k=$k")
                        inner_done = true
                end
            end
        end
        # --- Step 2: Operator Evaluation ---
        copyto!(T_y, operator_T(y_tilde))
        ηₖ = max(1.0, norm(T_y)) # Definition of ηₖ

        # --- Step 3: Compute z^{k+1} (Eq. 32) ---
        # Original: z^{k+1} := P_Ck ( y_tilde - (βₖ/ηₖ) * T(y_tilde) )
        # Adapted: We use approx_projector_C instead of P_Ck

        γₖ = βₖ / ηₖ
        @. u_vec = y_tilde - γₖ * T_y
        copyto!(zₖ, approx_projector_C(u_vec, y_tilde; kwargs...))
        error_zₖ = norm(zₖ - y_tilde)
        if error_zₖ ≤ ε
            status = :Solved_yₖ
            error = error_zₖ
            break
        end

        # --- Step 4: Update Averaged Sequence x^{k+1} (Eq. 33-34) ---
        # Update σₖ = σ_{k-1} + βₖ/ηₖ
        σₖ += γₖ

        # Coefficient for convex combination:
        # x^{k+1} := (1 - τₖ)x^k + τₖ * y_tilde
        # where τₖ = (βₖ/ηₖ) / σₖ
        τₖ = γₖ / σₖ
        copyto!(xₖ_old, xₖ)
        @. xₖ = (1 - τₖ) * xₖ + τₖ * y_tilde

        # --- Convergence Check ---
        error_xₖ = norm(xₖ - xₖ_old) / max(norm(xₖ_old), one(T))
        if error_xₖ ≤ ε
            solved = true
            status = :Solved_xₖ
            error = error_xₖ
        end



        index_iteration += 1
        if index_iteration ≥ max_iteration 
              tired = true
        end
    end

    return xₖ, zₖ, error, index_iteration, status
end