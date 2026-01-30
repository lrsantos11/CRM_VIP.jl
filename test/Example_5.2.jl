"""
    Example 5.2 - Test Suite for VIP Methods with Paramonotone Operators
    
    This example tests operators that are paramonotone but NOT gradients of convex functions.
    
    Operator: F(x) = Ax + c
    where A = [A₁  0 ]
              [0   A₂]
    with:
    - A₁ ∈ ℝ^{n₁×n₁} symmetric positive semidefinite
    - A₂ ∈ ℝ^{n₂×n₂} nonsymmetric positive semidefinite
    
    F is not a gradient because A (the Jacobian) is not symmetric.
    F is paramonotone because rank(A + A') = rank(A).
    
    Test Scenarios follow the same structure as Example 5.1:
    - 5.2A: Small size problems (n ∈ {5,10,20}, m ∈ {2,5,10})
    - 5.2B: Medium size problems (n ∈ {50,100}, m ∈ {5,8})
    - 5.2C: Large size problems (n ∈ {100,200,500}, m ∈ {20,30,50})
    
    Author: Luiz-Rafael Santos
"""

using LinearAlgebra
using Random
using DataFrames
using CSV


#=============================================================================
    OPERATOR DEFINITION FOR EXAMPLE 5.2
=============================================================================#

"""
    operator_F_Ex_5_2(n::Int; T::Type=Float64, n1_ratio::Float64=0.5, seed::Union{Int, Nothing}=nothing)

Generates the operator F for Example 5.2 (paramonotone, NOT gradient of convex function).

F(x) = Ax + c

where A = [A₁  0 ]  with:
          [0   A₂]
- A₁ ∈ ℝ^{n₁×n₁}: symmetric positive semidefinite  
- A₂ ∈ ℝ^{n₂×n₂}: nonsymmetric positive semidefinite

The Jacobian of F is A, which is NOT symmetric since A₂ is not symmetric.
Hence F is NOT a gradient of a convex function.

For paramonotonicity, we need rank(A + A') = rank(A).
We construct A₂ = M*M' + B + D (Malitsky's method) where:
- M is random (gives M*M' symmetric PSD)
- B is skew-symmetric (B' = -B)  
- D is diagonal with positive entries (ensures positive definiteness)

This construction guarantees A₂ is positive definite (hence PSD) and nonsymmetric.

# Arguments
- `n`: Total dimension
- `T`: Numeric type (default: Float64)
- `n1_ratio`: Ratio of dimension for A₁ block (default: 0.5)
- `seed`: Random seed (default: nothing)

# Returns
- `operator_F`: The operator function
- `L_estimate`: Estimate of Lipschitz constant ||A||
"""
function operator_F_Ex_5_2(n::Int; T::Type=Float64, n1_ratio::Float64=0.5, 
                           seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Split dimension
    n1 = max(1, round(Int, n * n1_ratio))
    n2 = n - n1
    
    # Generate A₁: symmetric positive semidefinite via M*M'
    M1 = randn(T, n1, n1)
    A1 = M1 * M1'
    A1 = (A1 + A1') / 2  # Ensure perfect symmetry
    
    # Generate A₂: nonsymmetric positive definite using Malitsky's construction
    # A₂ = M*M' + B + D where B is skew-symmetric, D is diagonal positive
    M2 = randn(T, n2, n2)
    A2_base = M2 * M2'  # Symmetric PSD base
    
    # B: skew-symmetric matrix (B' = -B)
    B_rand = randn(T, n2, n2)
    B = (B_rand - B_rand') / 2  # Skew-symmetric: ⟨Bx,x⟩ = 0
    
    # D: diagonal with positive entries to ensure positive definiteness
    D = Diagonal(0.1 .+ 0.4 * rand(T, n2))
    
    # A₂ = M*M' + B + D is positive definite and nonsymmetric
    A2 = A2_base + B + D
    
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
    verify_operator_properties(n::Int; seed::Int=42)

Debug function to verify that operator_F_Ex_5_2 has the expected properties:
1. A is positive semidefinite (F is monotone)
2. A is NOT symmetric (F is NOT a gradient)
3. rank(A + A') = rank(A) (F is paramonotone)
"""
function verify_operator_properties(n::Int; seed::Int=42)
    Random.seed!(seed)
    
    n1 = n ÷ 2
    n2 = n - n1
    
    # Generate A₁: symmetric PSD
    M1 = randn(n1, n1)
    A1 = M1 * M1'
    A1 = (A1 + A1') / 2
    
    # Generate A₂: nonsymmetric PD using Malitsky's construction
    M2 = randn(n2, n2)
    A2_base = M2 * M2'
    B_rand = randn(n2, n2)
    B = (B_rand - B_rand') / 2  # Skew-symmetric
    D = Diagonal(0.1 .+ 0.4 * rand(n2))
    A2 = A2_base + B + D
    
    A = zeros(n, n)
    A[1:n1, 1:n1] = A1
    A[n1+1:end, n1+1:end] = A2
    
    println("Dimension: n = $n (n₁=$n1, n₂=$n2)")
    println("="^50)
    
    # Check symmetry
    sym_error = norm(A - A')
    is_nonsym = sym_error > 1e-10
    println("Symmetry error ||A - A'|| = $(round(sym_error, sigdigits=4))")
    println("Is A nonsymmetric? $is_nonsym")
    
    # Check positive semidefiniteness via eigenvalues of symmetric part
    A_sym = (A + A') / 2
    λ = eigvals(Symmetric(A_sym))
    min_λ = minimum(λ)
    is_psd = min_λ >= -1e-10
    println("Min eigenvalue of (A+A')/2: $(round(min_λ, sigdigits=4))")
    println("Is (A+A')/2 PSD? $is_psd")
    
    # Check monotonicity
    println("F is monotone? $is_psd")
    
    # Check paramonotonicity: rank(A + A') = rank(A)
    rank_A = rank(A)
    rank_A_sym = rank(A + A')
    is_paramonotone = rank_A == rank_A_sym
    println("rank(A) = $rank_A, rank(A+A') = $rank_A_sym")
    println("Is F paramonotone? $is_paramonotone")
    
    return is_nonsym, is_psd, is_paramonotone
end

#=============================================================================
    SCENARIO FUNCTIONS
=============================================================================#

"""
    run_scenario_5_2A(; kwargs...)

Scenario 5.2A: Small size problems 
- Dimensions: n ∈ {5, 10}  
- Ellipsoids: m ∈ {2, 5}
- All methods
"""
function run_scenario_5_2A(; 
        num_repetitions::Int=10,
        max_iteration::Int=40_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_2;
        scenario_name="SCENARIO 5.2A: Small Size (Paramonotone)",
        example_name="Example 5.2",
        dimensions=[5, 10],
        num_ellipsoids_list=[2, 5],
        methods_list=METHODS_ALL,
        num_repetitions=num_repetitions,
        max_iteration=max_iteration,
        ε=ε,
        compute_time=compute_time,
        seed=seed)
end

"""
    run_scenario_5_2B(; kwargs...)

Scenario 5.2B: Medium size problems  
- Dimensions: n ∈ {50, 100}
- Ellipsoids: m ∈ {5, 8}
- Only approximate projection methods
"""
function run_scenario_5_2B(; 
        num_repetitions::Int=10,
        max_iteration::Int=300_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_2;
        scenario_name="SCENARIO 5.2B: Medium Size (Paramonotone)",
        example_name="Example 5.2",
        dimensions=[50, 100],
        num_ellipsoids_list=[5, 8],
        methods_list=METHODS_APPROX_ONLY,
        num_repetitions=num_repetitions,
        max_iteration=max_iteration,
        ε=ε,
        compute_time=compute_time,
        seed=seed)
end

"""
    run_scenario_5_2C(; kwargs...)

Scenario 5.2C: Large size / Many ellipsoids
- Dimensions: n ∈ {100, 200, 500}
- Ellipsoids: m ∈ {20, 30, 50}
- ONLY approximate projection methods
"""
function run_scenario_5_2C(; 
        num_repetitions::Int=5,
        max_iteration::Int=300_000,
        ε::Float64=1e-5,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_2;
        scenario_name="SCENARIO 5.2C: Large Size (Paramonotone, Approx Only)",
        example_name="Example 5.2",
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
    quick_test_5_2(; n=10, m=3)

Quick test to verify all methods work correctly with Example 5.2 operator.
"""
function quick_test_5_2(; n::Int=10, m::Int=3)
    Random.seed!(42)
    
    # First verify operator properties
    println("\n" * "="^50)
    println("Verifying operator properties for Example 5.2:")
    is_nonsym, is_psd, is_para = verify_operator_properties(n)
    println()
    
    if !is_nonsym
        @warn "A should be nonsymmetric!"
    end
    if !is_psd
        @warn "A should be PSD!"
    end
    if !is_para
        @warn "F should be paramonotone!"
    end
    
    df = create_results_dataframe()
    
    println("\nQuick test Example 5.2: n=$n, m=$m")
    test_example!(df, n, m, METHODS_ALL, operator_F_Ex_5_2;
        max_iteration=10_000, 
        ε=1e-4, 
        compute_time=true,
        verbose=true,
        example_name="Example 5.2")
    
    return df
end
#=============================================================================
    MAIN EXECUTION BLOCK
=============================================================================#

## Uncomment to run scenarios

## 5.2A - Small Size (matches paper Table 4)
df_5_2A = run_scenario_5_2A(num_repetitions=10, compute_time=true)
CSV.write(datadir("sims", "results_5_2A.csv"), df_5_2A)
p_time_A, p_iter_A = generate_performance_profiles(df_5_2A; title_suffix=" (5.2A)")

## 5.2B - Medium Size  
df_5_2B = run_scenario_5_2B(num_repetitions=10, compute_time=true)
CSV.write(datadir("sims", "results_5_2B.csv"), df_5_2B)
p_time_B, p_iter_B = generate_performance_profiles(df_5_2B; title_suffix=" (5.2B)")

## 5.2C - Large Size (approx methods only)
df_5_2C = run_scenario_5_2C(num_repetitions=5, compute_time=true)
CSV.write(datadir("sims", "results_5_2C.csv"), df_5_2C)
p_time_C, p_iter_C = generate_performance_profiles(df_5_2C; title_suffix=" (5.2C)")


## Run quick test by default when file is executed
# quick_test_5_2()

