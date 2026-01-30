"""
    common_test_utils.jl - Shared utilities for VIP test examples
    
    This file contains common functions used across Example 5.1, 5.2, and 5.3:
    - Method registry (METHOD_INFO)
    - Method dispatch (run_method, benchmark_method)
    - Performance profile utilities
    - Generic test and scenario functions
    
    Author: Luiz-Rafael Santos
"""

using LinearAlgebra
using Random
using BenchmarkTools
using Plots
using BenchmarkProfiles
using DataFrames
using Statistics: median

#=============================================================================
    METHOD REGISTRY
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
    :malitsky_2015_adaptive => (name="Malitsky (Adaptive)", category=:exact_projection, requires_slater=false),
    :bellocruz_iusem_2010 => (name="Bello-Cruz Iusem 2010 (BI1)", category=:approx_projection, requires_slater=false),
    :crm_vip_algorithm1 => (name="CRM-VIP Alg1", category=:approx_projection, requires_slater=false),
    :bellocruz_iusem_2012 => (name="Bello-Cruz Iusem 2012 (BI2)", category=:approx_projection, requires_slater=true),
    :crm_vip_algorithm2 => (name="CRM-VIP Alg2", category=:approx_projection, requires_slater=true),
)

# Default method lists for different scenarios
const METHODS_ALL = [
    :extragradient_vip,
    :malitsky_2015_adaptive,
    :bellocruz_iusem_2010,
    :crm_vip_algorithm1,
    :bellocruz_iusem_2012,  
    :crm_vip_algorithm2,    
]

const METHODS_APPROX_ONLY = [
    :bellocruz_iusem_2010,
    :crm_vip_algorithm1,
    :bellocruz_iusem_2012,  
    :crm_vip_algorithm2,    
]

#=============================================================================
    METHOD DISPATCH
=============================================================================#

"""
    run_method(method, x₀, operator_F, projector_C, approx_proj_C, 
               fi, ∇fi, g, Slater_point, L_estimate; max_iteration, ε, verbose)

Dispatch to appropriate method implementation.
Returns (x_sol, error, iterations, status).
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
        result = malitsky_2015(x₀, operator_F, projector_C;
                              max_iteration=max_iteration, ε=ε,
                              L=L_estimate, verbose=verbose)
        return result
        
    elseif method == :malitsky_2015_adaptive
        result = malitsky_2015_adaptive(x₀, operator_F, projector_C;
                                       max_iteration=max_iteration, ε=ε, 
                                       λ_init=0.5/L_estimate, verbose=verbose)
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

    else
        error("Method $method not implemented")
    end
end

"""
    benchmark_method(method, x₀, operator_F, projector_C, approx_proj_C,
                     fi, ∇fi, g, Slater_point, L_estimate; max_iteration, ε)

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
    
    else
        return NaN
    end
end

#=============================================================================
    GENERIC TEST FUNCTION
=============================================================================#

"""
    test_example!(df, n, num_ellipsoids, methods_list, operator_generator; kwargs...)

Generic test function for VIP examples.

# Arguments
- `df`: DataFrame to store results
- `n`: Problem dimension
- `num_ellipsoids`: Number of ellipsoids (m)
- `methods_list`: Vector of method symbols to test
- `operator_generator`: Function (n; T, seed) -> (operator_F, L_estimate)

# Keyword Arguments
- `T`: Numeric type (default: Float64)
- `max_iteration`: Maximum iterations (default: 40_000)
- `ε`: Convergence tolerance (default: 1e-6)
- `verbose`: Print progress (default: true)
- `compute_time`: Benchmark timing (default: true)
- `seed`: Random seed for this instance (default: nothing)
- `example_name`: Name for printing (default: "Example")
"""
function test_example!(df::DataFrame, n::Int, num_ellipsoids::Int, 
                       methods_list::Vector{Symbol},
                       operator_generator::Function;
                       T::Type=Float64, 
                       max_iteration::Int=40_000, 
                       ε::Float64=1e-6,
                       verbose::Bool=true, 
                       compute_time::Bool=true,
                       seed::Union{Int, Nothing}=nothing,
                       example_name::String="Example")
    
    # Generate problem instance
    operator_F, L_estimate = operator_generator(n; T=T, seed=seed)
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
    verbose && println("$example_name: n=$n, m=$num_ellipsoids")
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
    GENERIC SCENARIO RUNNER
=============================================================================#

"""
    run_scenario(operator_generator; scenario_name, dimensions, num_ellipsoids_list,
                 methods_list, num_repetitions, max_iteration, ε, compute_time, seed)

Generic scenario runner for VIP tests.
"""
function run_scenario(operator_generator::Function;
                      scenario_name::String="Scenario",
                      example_name::String="Example",
                      dimensions::Vector{Int}=[5, 10, 20],
                      num_ellipsoids_list::Vector{Int}=[2, 5, 10],
                      methods_list::Vector{Symbol}=METHODS_ALL,
                      num_repetitions::Int=5,
                      max_iteration::Int=40_000,
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
    
    println("\n" * "="^70)
    println("$scenario_name")
    println("Dimensions: $dimensions")
    println("Ellipsoids: $num_ellipsoids_list")
    println("Methods: $(length(methods_list))")
    println("Repetitions: $num_repetitions")
    println("="^70)
    
    for n in dimensions
        for m in num_ellipsoids_list
            for rep in 1:num_repetitions
                println("\n[n=$n, m=$m, rep=$rep]")
                test_example!(df, n, m, methods_list, operator_generator;
                    max_iteration=max_iteration, 
                    ε=ε, 
                    compute_time=compute_time,
                    verbose=true,
                    example_name=example_name)
            end
        end
    end
    
    return df
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
    # Filter only solved instances
    df_solved = filter(row -> row.Status in [:Solved, :Solved_xₖ, :Solved_yₖ], df)
    
    if nrow(df_solved) == 0
        @warn "No solved instances found"
        return nothing, nothing
    end
    
    methods = unique(df_solved.Method)
    
    # Create problem identifiers
    df_solved.ProblemID = string.(df_solved.Dimension, "_", df_solved.Num_Ellipsoids)
    problems = unique(df_solved.ProblemID)
    
    n_problems = length(problems)
    n_methods = length(methods)
    
    # Build performance matrix
    perf_matrix = fill(Inf, n_problems, n_methods)
    
    for (i, prob) in enumerate(problems)
        for (j, meth) in enumerate(methods)
            rows = filter(row -> row.ProblemID == prob && row.Method == meth, df_solved)
            if nrow(rows) > 0
                # Use median for robustness
                perf_matrix[i, j] = median(rows[:, metric])
            end
        end
    end
    
    return perf_matrix, methods
end

"""
    generate_performance_profiles(df; title_suffix="")

Generate performance profiles for Time and Iterations.
"""
function generate_performance_profiles(df::DataFrame; title_suffix::String="")
    # Time profile
    T_matrix, methods = build_performance_matrices(df, :Time)
    if T_matrix !== nothing
        p_time = performance_profile(PlotsBackend(), T_matrix, 
            logscale=true,
            title="CPU Time Performance Profile" * title_suffix,
            legend=:bottomright)
        plot!(p_time, label=permutedims(methods))
    else
        p_time = nothing
    end
    
    # Iteration profile  
    I_matrix, _ = build_performance_matrices(df, :Iterations)
    if I_matrix !== nothing
        p_iter = performance_profile(PlotsBackend(), I_matrix,
            logscale=true, 
            title="Iterations Performance Profile" * title_suffix,
            legend=:bottomright)
        plot!(p_iter, label=permutedims(methods))
    else
        p_iter = nothing
    end
    
    return p_time, p_iter
end

#=============================================================================
    HELPER: Create empty results DataFrame
=============================================================================#

"""
    create_results_dataframe()

Create an empty DataFrame with the standard results schema.
"""
function create_results_dataframe()
    return DataFrame(
        Method=String[], 
        Dimension=Int[], 
        Num_Ellipsoids=Int[], 
        Iterations=Int[], 
        Final_Error=Float64[], 
        Time=Float64[], 
        Status=Symbol[]
    )
end
