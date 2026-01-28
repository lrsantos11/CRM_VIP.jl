"""
    Example 5.3 - Test Suite for VIP Methods with Monotone (Non-Paramonotone) Operators
    
    This example tests operators that are monotone but NOT paramonotone.
    
    Operator: F(x) = Ax + c
    where A = [A₁  0 ]
              [0   A₂]
    with:
    - A₁ ∈ ℝ^{n₁×n₁} symmetric positive semidefinite
    - A₂ ∈ ℝ^{n₂×n₂} skew-symmetric (A₂ᵀ = -A₂)
    
    F is monotone because:
    - ⟨A₁x, x⟩ ≥ 0 (A₁ is PSD)
    - ⟨A₂x, x⟩ = 0 (A₂ is skew-symmetric)
    
    F is NOT paramonotone because:
    - rank(A + Aᵀ) < rank(A) when A₂ ≠ 0
    - For paramonotone operators, we need rank(J + Jᵀ) = rank(J)
    
    Test Scenarios follow the same structure as Examples 5.1 and 5.2:
    - 5.3A: Small size problems (n ∈ {5,10,20}, m ∈ {2,5,10})
    - 5.3B: Medium size problems (n ∈ {50,100}, m ∈ {5,8})
    - 5.3C: Large size problems (n ∈ {100,200,500}, m ∈ {20,30,50})
    
    Note: Since F is not paramonotone, Algorithm 2 (and BI2) may not converge
    to a solution. This example is useful to verify that:
    1. Algorithms designed for paramonotone operators may fail
    2. Extragradient-type methods should still work (only require monotonicity)
    
    Author: Luiz-Rafael Santos
"""

using LinearAlgebra
using Random
using DataFrames
using CSV

# Include common utilities
include("common_test_utils.jl")

#=============================================================================
    OPERATOR DEFINITION FOR EXAMPLE 5.3
=============================================================================#

"""
    operator_F_Ex_5_3(n::Int; T::Type=Float64, n1_ratio::Float64=0.5, seed::Union{Int, Nothing}=nothing)

Generates the operator F for Example 5.3 (monotone, NOT paramonotone).

F(x) = Ax + c

where A = [A₁  0 ]  with:
          [0   A₂]
- A₁ ∈ ℝ^{n₁×n₁}: symmetric positive semidefinite  
- A₂ ∈ ℝ^{n₂×n₂}: skew-symmetric (A₂ᵀ = -A₂)

The operator F is monotone:
- ⟨Ax, x⟩ = ⟨A₁x₁, x₁⟩ + ⟨A₂x₂, x₂⟩ = ⟨A₁x₁, x₁⟩ ≥ 0
  (since ⟨A₂x₂, x₂⟩ = 0 for skew-symmetric A₂)

F is NOT paramonotone because:
- rank(A + Aᵀ) = rank(2A₁) = rank(A₁) < rank(A) when A₂ ≠ 0
- For a differentiable paramonotone operator, rank(J + Jᵀ) = rank(J)

# Arguments
- `n`: Total dimension
- `T`: Numeric type (default: Float64)
- `n1_ratio`: Ratio of dimension for A₁ block (default: 0.5)
- `seed`: Random seed (default: nothing)

# Returns
- `operator_F`: The operator function
- `L_estimate`: Estimate of Lipschitz constant ||A||
"""
function operator_F_Ex_5_3(n::Int; T::Type=Float64, n1_ratio::Float64=0.5, 
                           seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Split dimension (ensure n2 ≥ 2 for non-trivial skew-symmetric matrix)
    n1 = max(1, round(Int, n * n1_ratio))
    n2 = n - n1
    
    # Ensure n2 ≥ 2 for meaningful skew-symmetric part
    if n2 < 2
        n1 = n - 2
        n2 = 2
    end
    
    # Generate A₁: symmetric positive semidefinite via M*M'
    M1 = randn(T, n1, n1)
    A1 = M1 * M1'
    A1 = (A1 + A1') / 2  # Ensure perfect symmetry
    
    # Generate A₂: skew-symmetric matrix (A₂ᵀ = -A₂)
    # Any matrix B can generate a skew-symmetric via (B - Bᵀ)/2
    B = randn(T, n2, n2)
    A2 = (B - B') / 2  # Skew-symmetric: ⟨A₂x, x⟩ = 0
    
    # Scale A₂ to have comparable magnitude to A₁
    if norm(A2) > 0
        A2 *= opnorm(A1) / opnorm(A2)
    end
    
    # Construct block diagonal A
    A = zeros(T, n, n)
    A[1:n1, 1:n1] = A1
    A[n1+1:end, n1+1:end] = A2
    
    # Generate c ≠ 0
    c = randn(T, n)
    if norm(c) == 0
        c[1] = one(T)
    end
    
    # Lipschitz constant is ||A||
    L_estimate = opnorm(A)
    
    function operator_F(x)
        return A * x + c
    end
    
    return operator_F, L_estimate
end

#=============================================================================
    VERIFICATION UTILITIES
=============================================================================#

"""
    verify_operator_properties_5_3(n::Int; seed::Int=42)

Debug function to verify that operator_F_Ex_5_3 has the expected properties:
1. A₁ is symmetric positive semidefinite
2. A₂ is skew-symmetric (A₂ᵀ = -A₂)
3. A is monotone (A + Aᵀ is PSD)
4. F is NOT paramonotone (rank(A + Aᵀ) < rank(A))
"""
function verify_operator_properties_5_3(n::Int; seed::Int=42)
    Random.seed!(seed)
    
    n1 = max(1, n ÷ 2)
    n2 = n - n1
    if n2 < 2
        n1 = n - 2
        n2 = 2
    end
    
    # Generate A₁: symmetric PSD
    M1 = randn(n1, n1)
    A1 = M1 * M1'
    A1 = (A1 + A1') / 2
    
    # Generate A₂: skew-symmetric
    B = randn(n2, n2)
    A2 = (B - B') / 2
    
    # Scale A₂
    if norm(A2) > 0
        A2 *= opnorm(A1) / opnorm(A2)
    end
    
    A = zeros(n, n)
    A[1:n1, 1:n1] = A1
    A[n1+1:end, n1+1:end] = A2
    
    println("Dimension: n = $n (n₁=$n1, n₂=$n2)")
    println("="^60)
    
    # Check A₁ symmetry
    A1_sym_error = norm(A1 - A1')
    println("A₁ symmetry error ||A₁ - A₁'|| = $(round(A1_sym_error, sigdigits=4))")
    
    # Check A₂ skew-symmetry
    A2_skew_error = norm(A2 + A2')
    is_skew = A2_skew_error < 1e-10
    println("A₂ skew-symmetry error ||A₂ + A₂'|| = $(round(A2_skew_error, sigdigits=4))")
    println("Is A₂ skew-symmetric? $is_skew")
    
    # Check ⟨A₂x, x⟩ = 0 for random x
    x_test = randn(n2)
    inner_prod = dot(A2 * x_test, x_test)
    println("⟨A₂x, x⟩ for random x = $(round(inner_prod, sigdigits=4)) (should be ≈ 0)")
    
    # Check monotonicity: A + Aᵀ should be PSD
    A_sym = A + A'
    eigs_sym = eigvals(Symmetric(A_sym))
    min_eig = minimum(eigs_sym)
    is_monotone = min_eig >= -1e-10
    println("\nMonotonicity check:")
    println("min eigenvalue of (A + A') = $(round(min_eig, sigdigits=4))")
    println("Is F monotone? $is_monotone")
    
    # Check paramonotonicity: rank(A + Aᵀ) vs rank(A)
    rank_A = rank(A)
    rank_A_sym = rank(A_sym)
    is_paramonotone = rank_A == rank_A_sym
    
    println("\nParamonotonicity check:")
    println("rank(A) = $rank_A")
    println("rank(A + A') = $rank_A_sym")
    println("Is F paramonotone? $is_paramonotone")
    
    if !is_paramonotone
        println("✓ Correctly NOT paramonotone (rank(A + A') < rank(A))")
    else
        @warn "Problem: F should NOT be paramonotone!"
    end
    
    return is_skew, is_monotone, !is_paramonotone
end

#=============================================================================
    SCENARIO FUNCTIONS
=============================================================================#

"""
    run_scenario_5_3A(; kwargs...)

Scenario 5.3A: Small size problems
- Dimensions: n ∈ {5, 10, 20}  
- Ellipsoids: m ∈ {2, 5, 10}
- All methods (to see which ones fail for non-paramonotone)
"""
function run_scenario_5_3A(; 
        num_repetitions::Int=20,
        max_iteration::Int=40_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_3;
        scenario_name="SCENARIO 5.3A: Small Size (Monotone, NOT Paramonotone)",
        example_name="Example 5.3",
        dimensions=[5, 10, 20],
        num_ellipsoids_list=[2, 5, 10],
        methods_list=METHODS_ALL,
        num_repetitions=num_repetitions,
        max_iteration=max_iteration,
        ε=ε,
        compute_time=compute_time,
        seed=seed)
end

"""
    run_scenario_5_3B(; kwargs...)

Scenario 5.3B: Medium size problems  
- Dimensions: n ∈ {50, 100}
- Ellipsoids: m ∈ {5, 8}
- All methods
"""
function run_scenario_5_3B(; 
        num_repetitions::Int=10,
        max_iteration::Int=300_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_3;
        scenario_name="SCENARIO 5.3B: Medium Size (Monotone, NOT Paramonotone)",
        example_name="Example 5.3",
        dimensions=[50, 100],
        num_ellipsoids_list=[5, 8],
        methods_list=METHODS_ALL,
        num_repetitions=num_repetitions,
        max_iteration=max_iteration,
        ε=ε,
        compute_time=compute_time,
        seed=seed)
end

"""
    run_scenario_5_3C(; kwargs...)

Scenario 5.3C: Large size / Many ellipsoids
- Dimensions: n ∈ {100, 200, 500}
- Ellipsoids: m ∈ {20, 30, 50}
- ONLY approximate projection methods
"""
function run_scenario_5_3C(; 
        num_repetitions::Int=5,
        max_iteration::Int=300_000,
        ε::Float64=1e-5,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_3;
        scenario_name="SCENARIO 5.3C: Large Size (Monotone NOT Paramonotone, Approx Only)",
        example_name="Example 5.3",
        dimensions=[100, 200, 500],
        num_ellipsoids_list=[20, 30, 50],
        methods_list=METHODS_APPROX_ONLY,
        num_repetitions=num_repetitions,
        max_iteration=max_iteration,
        ε=ε,
        compute_time=compute_time,
        seed=seed)
end

#=============================================================================
    QUICK TEST
=============================================================================#

"""
    quick_test_5_3(; n=10, m=3)

Quick test to verify all methods work correctly with Example 5.3 operator.
Note: We expect algorithms that require paramonotonicity (Alg2, BI2) to 
potentially not converge or behave differently.
"""
function quick_test_5_3(; n::Int=10, m::Int=3)
    Random.seed!(42)
    
    # First verify operator properties
    println("\n" * "="^60)
    println("Verifying operator properties for Example 5.3:")
    println("(Monotone but NOT Paramonotone)")
    println("="^60)
    is_skew, is_mono, is_not_para = verify_operator_properties_5_3(n)
    println()
    
    if !is_skew
        @warn "A₂ should be skew-symmetric!"
    end
    if !is_mono
        @warn "F should be monotone!"
    end
    if !is_not_para
        @warn "F should NOT be paramonotone!"
    end
    
    df = create_results_dataframe()
    
    println("\nQuick test Example 5.3: n=$n, m=$m")
    println("Note: Alg2/BI2 may not converge (designed for paramonotone operators)")
    test_example!(df, n, m, METHODS_ALL, operator_F_Ex_5_3;
        max_iteration=100_000, 
        ε=1e-4, 
        compute_time=true,
        verbose=true,
        example_name="Example 5.3")
    
    return df
end

#=============================================================================
    MAIN EXECUTION BLOCK
=============================================================================#

## Uncomment to run scenarios

# ## 5.3A - Small Size
# df_5_3A = run_scenario_5_3A(num_repetitions=20, compute_time=true)
# CSV.write(datadir("sims", "results_5_3A.csv"), df_5_3A)
# p_time_A, p_iter_A = generate_performance_profiles(df_5_3A; title_suffix=" (5.3A)")

# ## 5.3B - Medium Size  
# df_5_3B = run_scenario_5_3B(num_repetitions=10, compute_time=true)
# CSV.write(datadir("sims", "results_5_3B.csv"), df_5_3B)
# p_time_B, p_iter_B = generate_performance_profiles(df_5_3B; title_suffix=" (5.3B)")

# ## 5.3C - Large Size (approx methods only)
# df_5_3C = run_scenario_5_3C(num_repetitions=5, compute_time=true)
# CSV.write(datadir("sims", "results_5_3C.csv"), df_5_3C)
# p_time_C, p_iter_C = generate_performance_profiles(df_5_3C; title_suffix=" (5.3C)")

# Run quick test by default when file is executed
quick_test_5_3()
