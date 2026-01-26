"""
    Example 5.1 -  Test Suite for VIP Methods
    
    This file organizes the numerical experiments comparing:
    - Methods with APPROXIMATE projections: Alg1 (CRM-VIP), Alg2 (CRM-VIP), BI1, BI2
    - Methods with EXACT projections: Extragradient, Malitsky
    
    Test Scenarios:
    - 5.1A: Small size problems (n ∈ {10,20}, m ∈ {5,10}) - All methods
    - 5.1B: Medium size problems (n ∈ {50,100}, m ∈ {5,10}) - All methods  
    - 5.1C: Large size/many ellipsoids (n ∈ {100,200,300}, m ∈ {20,50,100}) 
            Only approximate projection methods (exact projection too expensive)
    
    Author: Luiz-Rafael Santos
"""

using LinearAlgebra
using Random
using BenchmarkTools
using Plots
using BenchmarkProfiles
using DataFrames
using CSV
using Printf

#=============================================================================
    OPERATOR DEFINITION
=============================================================================#

"""
    operator_F_Ex_5_1(n::Int; T::Type=Float64, seed::Union{Int, Nothing}=nothing)

Generates the operator F for Example 5.1 (gradient of convex function).

F(x) = Ax + G(x) + c, where G(x)_i = b_i * x_i^3

This is the gradient of: f(x) = ½⟨x,Ax⟩ + ⟨c,x⟩ + ¼∑ᵢbᵢxᵢ⁴

Returns also an estimate of the Lipschitz constant for Malitsky's method.
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
    METHOD REGISTRY - Defines available methods and their properties
=============================================================================#

"""
Method categories:
- :exact_projection - requires exact projection onto C (Extragradient, Malitsky)
- :approx_projection - uses approximate projections (Alg1, Alg2, BI1, BI2)

Note: Extragradient and Malitsky methods require computing P_C exactly, which 
for intersection of ellipsoids means running Dykstra's algorithm. This becomes
expensive when m (number of ellipsoids) is large.
"""
const METHOD_INFO = Dict(
    :extragradient_vip => (name="Extragradient", category=:exact_projection, requires_slater=false),
    :malitsky_2015 => (name="Malitsky (Fixed λ)", category=:exact_projection, requires_slater=false),
    :malitsky_2015_adaptive => (name="Malitsky (Adaptive)", category=:exact_projection, requires_slater=false),
    :bellocruz_iusem_2010 => (name="Bello-Cruz Iusem 2010 (BI1)", category=:approx_projection, requires_slater=false),
    :crm_vip_algorithm1 => (name="CRM-VIP Alg1", category=:approx_projection, requires_slater=false),
    :bellocruz_iusem_2012 => (name="Bello-Cruz Iusem 2012 (BI2)", category=:approx_projection, requires_slater=true),
    :crm_vip_algorithm2 => (name="CRM-VIP Alg2", category=:approx_projection, requires_slater=true),
    :solodov_svaiter_vip => (name="Solodov-Svaiter", category=:exact_projection, requires_slater=false),
    :solodov_svaiter_vip_v2 => (name="Solodov-Svaiter v2", category=:exact_projection, requires_slater=false),
)

#=============================================================================
    MAIN TEST FUNCTION
=============================================================================#

"""
    test_example_5_1!(df, n, num_ellipsoids, methods_list; kwargs...)

Runs Example 5.1 tests and appends results to DataFrame.

# Arguments
- `df`: DataFrame to store results
- `n`: Problem dimension
- `num_ellipsoids`: Number of ellipsoids (m)
- `methods_list`: Vector of method symbols to test

# Keyword Arguments
- `T`: Numeric type (default: Float64)
- `max_iteration`: Maximum iterations (default: 40_000)
- `ε`: Convergence tolerance (default: 1e-6)
- `verbose`: Print progress (default: true)
- `compute_time`: Benchmark timing (default: true)
- `seed`: Random seed for this instance (default: nothing)
"""
function test_example_5_1!(df::DataFrame, n::Int, num_ellipsoids::Int, 
                           methods_list::Vector{Symbol};
                           T::Type=Float64, 
                           max_iteration::Int=40_000, 
                           ε::Float64=1e-6,
                           verbose::Bool=true, 
                           compute_time::Bool=true,
                           seed::Union{Int, Nothing}=nothing)
    
    # Generate problem instance
    operator_F, L_estimate = operator_F_Ex_5_1(n; T=T, seed=seed)
    Ellipsoids_Sets, Slater_point = SampleEllipsoids(n, num_ellipsoids, 0.3; λ=1.1, γ=1.5)
    
    # Exact projector via Dykstra (expensive for large m)
    projections = [z -> proj_ellipsoid(z, ell) for ell in Ellipsoids_Sets]
    projector_C(x) = dykstra(x, projections; ε=1e-10, itmax=1_000)[1]
    
    # Approximate projector (separating halfspace)
    fi, ∇fi = ellipsoid_functions(Ellipsoids_Sets)
    function approx_proj_C(y, x)
        _, posmax = findmax([func(x) for func in fi])
        approx_proj_ellipsoid(y, Ellipsoids_Sets[posmax])
    end
    
    # g(x) = max_i g_i(x) for Alg2/BI2
    g(x) = maximum([func(x) for func in fi])
    
    # Starting point
    x₀ = starting_point(n; T=T)
    
    verbose && println("="^70)
    verbose && println("Example 5.1: n=$n, m=$num_ellipsoids")
    verbose && println("Estimated Lipschitz constant L ≈ $(round(L_estimate, digits=2))")
    
    # Run each method
    for method in methods_list
        info = get(METHOD_INFO, method, nothing)
        if info === nothing
            @warn "Method $method not recognized, skipping."
            continue
        end
        
        verbose && println("\n  Running $(info.name)...")
        
        result = run_method(method, x₀, operator_F, 
                           projector_C, approx_proj_C, 
                           fi, ∇fi, g, Slater_point,
                           L_estimate;
                           max_iteration=max_iteration, 
                           ε=ε,
                           verbose=false)
        
        x_sol, error, iterations, status = result
        
        # Compute time if converged
        if status in [:Solved, :Solved_xₖ, :Solved_yₖ] && compute_time
            t = benchmark_method(method, x₀, operator_F, 
                                projector_C, approx_proj_C,
                                fi, ∇fi, g, Slater_point, L_estimate;
                                max_iteration=max_iteration, ε=ε)
        elseif status in [:Solved, :Solved_xₖ, :Solved_yₖ]
            t = NaN
        else
            verbose && println("    Warning: Did not converge (status=$status)")
            t = Inf
        end
        
        verbose && println("    Status: $status, Iter: $iterations, Error: $(round(error, sigdigits=4)), Time: $(round(t, sigdigits=4))s")
        
        push!(df, (
            Method = info.name,
            Dimension = n,
            Num_Ellipsoids = num_ellipsoids,
            Iterations = iterations,
            Final_Error = error,
            Time = t,
            Status = status
        ))
    end
    
    verbose && println("="^70)
    return df
end

#=============================================================================
    METHOD DISPATCH
=============================================================================#

"""
Dispatch to appropriate method implementation.
"""
function run_method(method::Symbol, x₀, operator_F, 
                   projector_C, approx_proj_C, 
                   fi, ∇fi, g, Slater_point, L_estimate;
                   max_iteration, ε, verbose=false)
    
    T = eltype(x₀)
    
    if method == :extragradient_vip
        return extragradient_vip(x₀, operator_F, projector_C; 
                                max_iteration=max_iteration, ε=ε, verbose=verbose)
    
    elseif method == :malitsky_2015
        # Fixed step size version - requires L estimate
        result = malitsky_2015_adaptive(x₀, operator_F, projector_C;
                              malitsky_2015_adaptive=max_iteration, ε=ε,
                              L=L_estimate, verbose=verbose)
        return result  # Returns 4 values
        
    elseif method == :malitsky_2015_adaptive
        result = malitsky_2015_adaptive(x₀, operator_F, projector_C;
                                       max_iteration=max_iteration, ε=ε, 
                                       λ_init=0.5/L_estimate, verbose=verbose)
        # malitsky_2015_adaptive returns 5 values (includes λ_history)
        return result[1:4]
        
    elseif method == :bellocruz_iusem_2010
        return bellocruz_iusem_2010(x₀, operator_F, approx_proj_C; 
                                   max_iteration=max_iteration, ε=ε)
        
    elseif method == :crm_vip_algorithm1
        return crm_vip_algorithm1(x₀, operator_F, fi, ∇fi; 
                                 max_iteration=max_iteration, ε=ε)
        
    elseif method == :bellocruz_iusem_2012
        θ = 2.0 * one(T)
        x_sol, z_sol, error, iterations, status = bellocruz_iusem_2012(
            x₀, Slater_point, operator_F, approx_proj_C, g, θ;
            max_iteration=max_iteration, ε=ε)
        return x_sol, error, iterations, status
        
    elseif method == :crm_vip_algorithm2
        θ = 2.0 * one(T)
        x_sol, z_sol, error, iterations, status = crm_vip_algorithm2(
            x₀, Slater_point, operator_F, fi, ∇fi, θ;
            max_iteration=max_iteration, ε=ε)
        return x_sol, error, iterations, status

    elseif method == :solodov_svaiter_vip
        return solodov_svaiter_vip(x₀, operator_F, projector_C;
            max_iteration=max_iteration, ε=ε, verbose=verbose)

        elseif method == :solodov_svaiter_vip_v2
        return solodov_svaiter_vip_v2(x₀, operator_F, projector_C;
            max_iteration=max_iteration, ε=ε, verbose=verbose)
    else
        error("Method $method not implemented")
    end
end

"""
Benchmark a method using @belapsed.
"""
function benchmark_method(method::Symbol, x₀, operator_F,
                         projector_C, approx_proj_C,
                         fi, ∇fi, g, Slater_point, L_estimate;
                         max_iteration, ε)
    T = eltype(x₀)
    
    if method == :extragradient_vip
        return @belapsed extragradient_vip($x₀, $operator_F, $projector_C; 
                                          max_iteration=$max_iteration, ε=$ε, verbose=false)
    
    elseif method == :malitsky_2015
        return @belapsed malitsky_2015($x₀, $operator_F, $projector_C;
                                      max_iteration=$max_iteration, ε=$ε,
                                      L=$L_estimate, verbose=false)
        
    elseif method == :malitsky_2015_adaptive
        return @belapsed malitsky_2015_adaptive($x₀, $operator_F, $projector_C;
                                               max_iteration=$max_iteration, ε=$ε,
                                               λ_init=$(0.5/L_estimate), verbose=false)
        
    elseif method == :bellocruz_iusem_2010
        return @belapsed bellocruz_iusem_2010($x₀, $operator_F, $approx_proj_C;
                                             max_iteration=$max_iteration, ε=$ε)
        
    elseif method == :crm_vip_algorithm1
        return @belapsed crm_vip_algorithm1($x₀, $operator_F, $fi, $∇fi;
                                           max_iteration=$max_iteration, ε=$ε)
        
    elseif method == :bellocruz_iusem_2012
        θ = 2.0 * one(T)
        return @belapsed bellocruz_iusem_2012($x₀, $Slater_point, $operator_F, 
                                             $approx_proj_C, $g, $θ;
                                             max_iteration=$max_iteration, ε=$ε)
        
    elseif method == :crm_vip_algorithm2
        θ = 2.0 * one(T)
        return @belapsed crm_vip_algorithm2($x₀, $Slater_point, $operator_F,
                                           $fi, $∇fi, $θ;
                                           max_iteration=$max_iteration, ε=$ε)
    
    elseif method == :solodov_svaiter_vip
        return @belapsed solodov_svaiter_vip($x₀, $operator_F, $projector_C;
            max_iteration=$max_iteration, ε=$ε, verbose=false)

    elseif method == :solodov_svaiter_vip_v2
        return @belapsed solodov_svaiter_vip_v2($x₀, $operator_F, $projector_C;
            max_iteration=$max_iteration, ε=$ε, verbose=false)
    
    else
        return NaN
    end
end

#=============================================================================
    PERFORMANCE PROFILE UTILITIES
=============================================================================#

"""
    build_performance_matrices(df, metric::Symbol)

Build matrix for performance profile from DataFrame.
Returns (matrix, method_names).
"""
function build_performance_matrices(df::DataFrame, metric::Symbol)
    methods = unique(df.Method)
    n_methods = length(methods)
    
    # Group by problem instance (Dimension, Num_Ellipsoids) 
    # Each unique combination × repetition = one problem
    grouped = groupby(df, [:Method])
    
    n_problems = nrow(first(grouped))
    
    matrix = Matrix{Float64}(undef, n_problems, n_methods)
    
    for (j, method) in enumerate(methods)
        method_data = filter(row -> row.Method == method, df)
        matrix[:, j] = getproperty(method_data, metric)
    end
    
    return matrix, methods
end

"""
    generate_performance_profiles(df; output_dir=nothing)

Generate and optionally save performance profiles.
"""
function generate_performance_profiles(df::DataFrame; 
                                       output_dir::Union{String, Nothing}=nothing,
                                       title_suffix::String="")
    
    # Time profile
    Time_matrix, methods = build_performance_matrices(df, :Time)
    p_time = performance_profile(PlotsBackend(), Time_matrix, methods,
        title="Performance Profile - CPU Time $title_suffix",
        xlabel="τ", ylabel="Fraction of problems solved",
        legend=:bottomright)
    
    # Iteration profile
    Iter_matrix, _ = build_performance_matrices(df, :Iterations)
    Iter_matrix_float = Float64.(Iter_matrix)
    p_iter = performance_profile(PlotsBackend(), Iter_matrix_float, methods,
        title="Performance Profile - Iterations $title_suffix",
        xlabel="τ", ylabel="Fraction of problems solved",
        legend=:bottomright)
    
    if output_dir !== nothing
        savefig(p_time, joinpath(output_dir, "perf_profile_time$title_suffix.pdf"))
        savefig(p_iter, joinpath(output_dir, "perf_profile_iter$title_suffix.pdf"))
    end
    
    return p_time, p_iter
end

#=============================================================================
    TEST SCENARIOS
=============================================================================#

"""
    run_scenario_5_1A(; num_repetitions=5, max_iteration=300_000, ε=1e-6)

Scenario 5.1A: Small size problems
- Dimensions: n ∈ {5, 10, 20}
- Ellipsoids: m ∈ {2, 5, 8}
- All methods (exact projection still feasible)
"""
function run_scenario_5_1A(; 
        num_repetitions::Int=5,
        max_iteration::Int=300_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    Random.seed!(seed)
    
    df = DataFrame(
        Method=String[], 
        Dimension=Int[], 
        Num_Ellipsoids=Int[], 
        Iterations=Int[], 
        Final_Error=Float64[], 
        Time=Float64[], 
        Status=Symbol[]
    )
    
    # All methods available for small problems
    # Includes both Malitsky variants for comparison
    methods_all = [
        :extragradient_vip,
        :malitsky_2015,           # Fixed step size
        :malitsky_2015_adaptive,  # Adaptive step size
        :solodov_svaiter_vip,      # ← adicionar
        :solodov_svaiter_vip_v2,   # ← adicionar
        :bellocruz_iusem_2010,
        :crm_vip_algorithm1,
        :bellocruz_iusem_2012,  
        :crm_vip_algorithm2,    
    ]
    
    dimensions = [5, 10, 20]
    num_ellipsoids_list = [2, 5, 8]
    
    println("\n" * "="^70)
    println("SCENARIO 5.1A: Small Size Problems")
    println("Dimensions: $dimensions")
    println("Ellipsoids: $num_ellipsoids_list")
    println("Methods: $(length(methods_all))")
    println("Repetitions: $num_repetitions")
    println("="^70)
    
    for n in dimensions
        for m in num_ellipsoids_list
            for rep in 1:num_repetitions
                println("\n[n=$n, m=$m, rep=$rep]")
                test_example_5_1!(df, n, m, methods_all;
                    max_iteration=max_iteration, 
                    ε=ε, 
                    compute_time=compute_time,
                    verbose=true)
            end
        end
    end
    
    return df
end

"""
    run_scenario_5_1B(; num_repetitions=5, max_iteration=300_000, ε=1e-6)

Scenario 5.1B: Medium size problems  
- Dimensions: n ∈ {50, 100}
- Ellipsoids: m ∈ {5, 10}
- All methods (exact projection still reasonable)
"""
function run_scenario_5_1B(; 
        num_repetitions::Int=3,
        max_iteration::Int=300_000,
        ε::Float64=1e-6,
        compute_time::Bool=true,
        seed::Int=42)
    
    Random.seed!(seed)
    
    df = DataFrame(
        Method=String[], 
        Dimension=Int[], 
        Num_Ellipsoids=Int[], 
        Iterations=Int[], 
        Final_Error=Float64[], 
        Time=Float64[], 
        Status=Symbol[]
    )
    
    # All methods for medium problems
    methods_all = [
        :extragradient_vip,
        :malitsky_2015,           # Fixed step size
        :malitsky_2015_adaptive,  # Adaptive step size
        :solodov_svaiter_vip,      # ← adicionar
        :solodov_svaiter_vip_v2,   # ← adicionar
        :bellocruz_iusem_2010,
        :crm_vip_algorithm1,
    ]
    
    dimensions = [50, 100]
    num_ellipsoids_list = [5, 10]
    
    println("\n" * "="^70)
    println("SCENARIO 5.1B: Medium Size Problems")
    println("Dimensions: $dimensions")
    println("Ellipsoids: $num_ellipsoids_list")
    println("Methods: $(length(methods_all))")
    println("="^70)
    
    for n in dimensions
        for m in num_ellipsoids_list
            for rep in 1:num_repetitions
                println("\n[n=$n, m=$m, rep=$rep]")
                test_example_5_1!(df, n, m, methods_all;
                    max_iteration=max_iteration, 
                    ε=ε, 
                    compute_time=compute_time,
                    verbose=true)
            end
        end
    end
    
    return df
end

"""
    run_scenario_5_1C(; num_repetitions=3, max_iteration=300_000, ε=1e-5)

Scenario 5.1C: Large size / Many ellipsoids
- Dimensions: n ∈ {100, 200, 500}
- Ellipsoids: m ∈ {20, 30, 50}
- ONLY approximate projection methods (exact projection too expensive)

Note: Extragradient and Malitsky excluded because computing P_C via Dykstra
becomes prohibitively expensive when m is large.
"""
function run_scenario_5_1C(; 
        num_repetitions::Int=2,
        max_iteration::Int=300_000,
        ε::Float64=1e-5,  # Relaxed tolerance
        compute_time::Bool=true,
        seed::Int=42)
    
    Random.seed!(seed)
    
    df = DataFrame(
        Method=String[], 
        Dimension=Int[], 
        Num_Ellipsoids=Int[], 
        Iterations=Int[], 
        Final_Error=Float64[], 
        Time=Float64[], 
        Status=Symbol[]
    )
    
    # ONLY approximate projection methods
    methods_approx = [
        :bellocruz_iusem_2010,
        :crm_vip_algorithm1,
        :bellocruz_iusem_2012,  
        :crm_vip_algorithm2,    
    ]
    
    dimensions = [100, 200, 500]
    num_ellipsoids_list = [20, 30, 50]
    
    println("\n" * "="^70)
    println("SCENARIO 5.1C: Large Size / Many Ellipsoids")
    println("Dimensions: $dimensions")
    println("Ellipsoids: $num_ellipsoids_list")
    println("Methods (approx only): $(length(methods_approx))")
    println("Note: Extragradient/Malitsky excluded (exact proj too expensive)")
    println("="^70)
    
    for n in dimensions
        for m in num_ellipsoids_list
            for rep in 1:num_repetitions
                println("\n[n=$n, m=$m, rep=$rep]")
                test_example_5_1!(df, n, m, methods_approx;
                    max_iteration=max_iteration, 
                    ε=ε, 
                    compute_time=compute_time,
                    verbose=true)
            end
        end
    end
    
    return df
end

#=============================================================================
    MAIN EXECUTION BLOCK
=============================================================================#

## Uncomment to run scenarios

# ## 5.1A - Small Size
# df_5_1A = run_scenario_5_1A(num_repetitions=2, compute_time=true)
# CSV.write(datadir("sims", "results_5_1A.csv"), df_5_1A)
# p_time_A, p_iter_A = generate_performance_profiles(df_5_1A; title_suffix=" (5.1A)")

# ## 5.1B - Medium Size  
# df_5_1B = run_scenario_5_1B(num_repetitions=2, compute_time=true)
# CSV.write(datadir("sims", "results_5_1B.csv"), df_5_1B)
# p_time_B, p_iter_B = generate_performance_profiles(df_5_1B; title_suffix=" (5.1B)")

# ## 5.1C - Large Size (approx methods only)
# df_5_1C = run_scenario_5_1C(num_repetitions=2, compute_time=true)
# CSV.write(datadir("sims", "results_5_1C.csv"), df_5_1C)
# p_time_C, p_iter_C = generate_performance_profiles(df_5_1C; title_suffix=" (5.1C)")

#=============================================================================
    QUICK TEST FUNCTION
=============================================================================#

"""
    quick_test(; n=10, m=3)

Quick test to verify all methods work correctly.
"""
function quick_test(; n::Int=10, m::Int=3)
    Random.seed!(123)
    
    df = DataFrame(
        Method=String[], 
        Dimension=Int[], 
        Num_Ellipsoids=Int[], 
        Iterations=Int[], 
        Final_Error=Float64[], 
        Time=Float64[], 
        Status=Symbol[]
    )
    
    methods = [
        :extragradient_vip,
        :malitsky_2015,
        :malitsky_2015_adaptive,
        :solodov_svaiter_vip,      # ← adicionar
        :solodov_svaiter_vip_v2,   # ← adicionar
        :bellocruz_iusem_2010,
        :crm_vip_algorithm1,
        :bellocruz_iusem_2012,
        :crm_vip_algorithm2
    ]
    
    println("Quick test: n=$n, m=$m")
    test_example_5_1!(df, n, m, methods;
        max_iteration=10_000, 
        ε=1e-4, 
        compute_time=true,
        verbose=true)
    
    return df
end

quick_test()
