"""
    Example 5.1 - Test Suite for VIP Methods with Gradient Operators
    
    This example tests operators that are gradients of convex functions.
    
    Operator: F(x) = Ax + G(x) + c
    where G(x)_i = b_i * x_i^3 with b ≥ 0
    
    This is the gradient of: f(x) = ½⟨x,Ax⟩ + ⟨c,x⟩ + ¼∑ᵢbᵢxᵢ⁴
    
    Properties:
    - F is the gradient of a convex function (A symmetric PSD, G monotone)
    - F is paramonotone (gradients of convex functions are paramonotone)
    
    Test Scenarios:
    - 5.1A: Small size problems (n ∈ {5,10,20}, m ∈ {2,5,8})
    - 5.1B: Medium size problems (n ∈ {50,100}, m ∈ {5,8})  
    - 5.1C: Large size problems (n ∈ {100,200,500}, m ∈ {20,30,50})
    
    Author: Luiz-Rafael Santos
"""

using LinearAlgebra
using Random
using DataFrames
using CSV


#=============================================================================
    OPERATOR DEFINITION FOR EXAMPLE 5.1
=============================================================================#

"""
    operator_F_Ex_5_1(n::Int; T::Type=Float64, seed::Union{Int, Nothing}=nothing)

Generates the operator F for Example 5.1 (gradient of convex function).

F(x) = Ax + G(x) + c, where G(x)_i = b_i * x_i^3

This is the gradient of: f(x) = ½⟨x,Ax⟩ + ⟨c,x⟩ + ¼∑ᵢbᵢxᵢ⁴

# Arguments
- `n`: Problem dimension
- `T`: Numeric type (default: Float64)
- `seed`: Random seed (default: nothing)

# Returns
- `operator_F`: The operator function
- `L_estimate`: Estimate of Lipschitz constant for step size selection
"""
function operator_F_Ex_5_1(n::Int; T::Type=Float64, seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Generate A: symmetric positive semidefinite via A = M*M'
    M = randn(T, n, n)
    A = M * M'
    
    # Generate b ≥ 0
    b = abs.(randn(T, n))
    
    # Generate c ≠ 0
    c = randn(T, n)
    if norm(c) == 0
        c[1] = one(T)
    end
    
    # Estimate Lipschitz constant: L ≈ ||A|| + 3*max(b)*R² where R is domain bound
    L_estimate = opnorm(A) + 3 * maximum(b) * 10.0^2  # Assuming ||x|| ≤ 10

    function operator_F(x)
        Gx = b .* (x .^ 3)
        return A * x + Gx + c
    end

    return operator_F, L_estimate
end

#=============================================================================
    SCENARIO FUNCTIONS
=============================================================================#

"""
    run_scenario_5_1A(; kwargs...)

Scenario 5.1A: Small size problems
- Dimensions: n ∈ {5, 10}
- Ellipsoids: m ∈ {2, 5}
- All methods
"""
function run_scenario_5_1A(; 
        num_repetitions::Int=10,
        max_iteration::Int=100_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_1;
        scenario_name="SCENARIO 5.1A: Small Size (Gradient Operator)",
        example_name="Example 5.1",
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
    run_scenario_5_1B(; kwargs...)

Scenario 5.1B: Medium size problems  
- Dimensions: n ∈ {50, 100}
- Ellipsoids: m ∈ {5, 8}
- Only approximate projection methods
"""
function run_scenario_5_1B(; 
        num_repetitions::Int=10,
        max_iteration::Int=300_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_1;
        scenario_name="SCENARIO 5.1B: Medium Size (Gradient Operator)",
        example_name="Example 5.1",
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
    run_scenario_5_1C(; kwargs...)

Scenario 5.1C: Large size / Many ellipsoids
- Dimensions: n ∈ {100, 200, 500}
- Ellipsoids: m ∈ {20, 30, 50}
- ONLY approximate projection methods (exact projection too expensive)
"""
function run_scenario_5_1C(; 
        num_repetitions::Int=5,
        max_iteration::Int=300_000,
        ε::Float64=1e-5,
        compute_time::Bool=true,
        seed::Int=42)
    
    return run_scenario(operator_F_Ex_5_1;
        scenario_name="SCENARIO 5.1C: Large Size (Gradient, Approx Only)",
        example_name="Example 5.1",
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
    quick_test_5_1(; n=10, m=3)

Quick test to verify all methods work correctly with Example 5.1 operator.
"""
function quick_test_5_1(; n::Int=10, m::Int=3)
    Random.seed!(42)
    
    df = create_results_dataframe()
    
    println("\nQuick test Example 5.1: n=$n, m=$m")
    test_example!(df, n, m, METHODS_ALL, operator_F_Ex_5_1;
        max_iteration=10_000, 
        ε=1e-4, 
        compute_time=true,
        verbose=true,
        example_name="Example 5.1")
    
    return df
end

#=============================================================================
    MAIN EXECUTION BLOCK
=============================================================================#

## Uncomment to run scenarios

# ## 5.1A - Small Size
df_5_1A = run_scenario_5_1A(num_repetitions=10, compute_time=true)
CSV.write(datadir("sims", "results_5_1A.csv"), df_5_1A)
# p_time_A, p_iter_A = generate_performance_profiles(df_5_1A; title_suffix=" (5.1A)")

## 5.1B - Medium Size  
df_5_1B = run_scenario_5_1B(num_repetitions=10, compute_time=true)
CSV.write(datadir("sims", "results_5_1B.csv"), df_5_1B)
# p_time_B, p_iter_B = generate_performance_profiles(df_5_1B; title_suffix=" (5.1B)")

## 5.1C - Large Size (approx methods only)
df_5_1C = run_scenario_5_1C(num_repetitions=5, compute_time=true)
CSV.write(datadir("sims", "results_5_1C.csv"), df_5_1C)
# p_time_C, p_iter_C = generate_performance_profiles(df_5_1C; title_suffix=" (5.1C)")

# Run quick test by default when file is executed
# quick_test_5_1()

