using LinearAlgebra
using Random
using BenchmarkTools
using Plots
using BenchmarkProfiles
"""
    operator_F_Ex_5_1(n::Int; T::Type=Float64, seed::Union{Int, Nothing}=nothing)

Generates the operator F for Example 5.1.

# Arguments
- `n`: Dimension of the problem.

# Returns
- `F`: The operator function F(x).


# Description
First, we test examples with operators F which are gradients of convex functions. In these cases, VIP(F,C) is equivalent to the problem of minimizing the convex function with gradient F over C. Define
F(x) = Ax + G(x) + c
where G(x)_i = b_i * x_i^3 with b, c ∈ R^n, b ≥ 0, c ≠ 0. We generate randomly A ∈ R^{n×n} as a symmetric, positive semidefinite matrix. Note that F(x) is the gradient of the convex function
f(x) = 1/2 * <x, Ax> + <c, x> + 1/4 * sum(b_i * x_i^4).
"""
function operator_F_Ex_5_1(n::Int; T::Type=Float64, seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Generate A: symmetric positive semidefinite
    # Using A = M * M' ensures A is PSD
    M = randn(T, n, n)
    A = M * M'

    # Generate b >= 0
    # Using abs.(randn) ensures non-negative entries
    b = abs.(randn(T, n))

    # Generate c != 0
    c = randn(T, n)
    # Ensure c is not zero (unlikely with floats, but good to be safe)
    if norm(c) == 0
        c[1] = 1.0
    end

    # Define the operator F(x)
    function operator_F(x)
        # G(x)_i = b_i * x_i^3
        Gx = b .* (x .^ 3)
        return A * x + Gx + c
    end


    return operator_F
end

## Example 5.1 - Extragradient Method




function test_example_5_1!(df::DataFrame, n::Int, num_ellipsoids::Int, methods_list::Vector{Symbol};  T::Type=Float64, max_iteration::Int=40_000, ε::Float64=1e-6,verbose::Bool=true, compute_time::Bool=true)
    
    operator_F = operator_F_Ex_5_1(n; T=T)
    Ellipsoids_Sets, Slater_point = SampleEllipsoids(n, num_ellipsoids, 0.3; λ=1.1, γ=1.5)
    projections = [z -> proj_ellipsoid(z, ell) for ell in Ellipsoids_Sets]
    projector_C(x) = dykstra(x, projections; ε=1e-10, itmax=1_000)[1]
    x₀ = starting_point(n; T=T)

    # Approximate projector using f(x) = maximum(fi(x)) where fi(x) are the functions defining each ellipsoid
    fi, ∇fi =  ellipsoid_functions(Ellipsoids_Sets)
    function approx_proj_C(y, x)
        _, posmax = findmax([func(x) for func in fi])
        approx_proj_ellipsoid(y, Ellipsoids_Sets[posmax])
    end
    

    # dykstra_proj, _ = dykstra(x₀, projections; ε=1e-10, itmax=1_000, verbose=false, output_proj=false)
    verbose && println("="^70)
    verbose && println("Example 5.1 with n=$n and $num_ellipsoids ellipsoids.")
    for method in methods_list
        if method == :extragradient_vip
            println("Running Extragradient Method")
            x_sol, error, iterations, status = extragradient_vip(x₀, operator_F, projector_C; max_iteration=max_iteration, ε=ε, verbose=false)
            println("Status: $status, Iterations: $iterations, Final Error: $error")
            if status != :Solved
                verbose && println("Warning: Method did not converge to a solution.")
                t = Inf
            elseif compute_time 
                t = @belapsed extragradient_vip($x₀, $operator_F, $projector_C; max_iteration=$max_iteration, ε=$ε, verbose=false)
                verbose && println("Time taken by method: $(t) seconds")
            else
                t = NaN
            end
           
            push!(df, (Method="Extragradient", Dimension=n, Num_Ellipsoids=num_ellipsoids, Iterations=iterations, Final_Error=error, Time=t, Status=status))
        elseif method == :bellocruz_iusem_2010
            verbose && println("Running Bello-Cruz and Iusem 2010 Method")
            x_sol, error, iterations, status = bellocruz_iusem_2010(x₀, operator_F, approx_proj_C; max_iteration=max_iteration, ε=ε)
            verbose && println("Status: $status, Iterations: $iterations, Final Error: $error")
            if status != :Solved
                verbose && println("Warning: Method did not converge to a solution.")
                t = Inf
            elseif compute_time
                t = @belapsed bellocruz_iusem_2010($x₀, $operator_F, $approx_proj_C; max_iteration=$max_iteration, ε=$ε)
                verbose && println("Time taken by method: $(t) seconds")
            else
                t = NaN
            end
            push!(df, (Method="Bello-Cruz Iusem 2010", Dimension=n, Num_Ellipsoids=num_ellipsoids, Iterations=iterations, Final_Error=error, Time=t, Status=status))
        elseif method == :crm_vip_algorithm1
            println("Running CRM VIP Algorithm 1")
            x_sol, error, iterations, status = crm_vip_algorithm1(x₀, operator_F, fi, ∇fi; max_iteration=max_iteration, ε=ε)
            verbose && println("Status: $status, Iterations: $iterations, Final Error: $error")
            if status != :Solved
                verbose && println("Warning: Method did not converge to a solution.")
                t = Inf
            elseif compute_time
                t = @belapsed crm_vip_algorithm1($x₀, $operator_F, $fi, $∇fi; max_iteration=$max_iteration, ε=$ε)
                verbose && println("Time taken by method: $(t) seconds")
            else
                t = NaN 
            end
            push!(df, (Method="CRM-VIP Algorithm 1", Dimension=n, Num_Ellipsoids=num_ellipsoids, Iterations=iterations, Final_Error=error, Time=t, Status=status))
        elseif method == :bellocruz_iusem_2012
            verbose && println("Running Bello-Cruz and Iusem 2012 Method")
            g(x) = maximum([func(x) for func in fi])
            θ = 2.0*one(T)
            x_sol, z_sol, error, iterations, status = bellocruz_iusem_2012(x₀, Slater_point, operator_F, approx_proj_C, g, θ; max_iteration=max_iteration, ε=ε)
            verbose && println("Status: $status, Iterations: $iterations, Final Error: $error")
            if (status != :Solved_xₖ) && (status != :Solved_yₖ) 
                verbose && println("Warning: Method did not converge to a solution.")
                t = Inf
            elseif compute_time
                t = @belapsed bellocruz_iusem_2012($x₀, $Slater_point, $operator_F, $approx_proj_C, $g, $θ; max_iteration=$max_iteration, ε=$ε)
                verbose && println("Time taken by method: $(t) seconds")
            else
                t = NaN
            end
            push!(df, (Method="Bello-Cruz Iusem 2012", Dimension=n, Num_Ellipsoids=num_ellipsoids, Iterations=iterations, Final_Error=error, Time=t, Status=status))   
        elseif method == :crm_vip_algorithm2
            verbose && println("Running CRM VIP Algorithm 2")
            θ = 2.0*one(T)
            x_sol, z_sol, error, iterations, status = crm_vip_algorithm2(x₀, Slater_point, operator_F, fi, ∇fi, θ; max_iteration=max_iteration, ε=ε)
            verbose && println("Status: $status, Iterations: $iterations, Final Error: $error")
            if (status != :Solved_xₖ) && (status != :Solved_yₖ) 
                verbose && println("Warning: Method did not converge to a solution.")
                t = Inf
            elseif compute_time
                t = @belapsed crm_vip_algorithm2($x₀, $Slater_point, $operator_F, $fi, $∇fi, $θ; max_iteration=$max_iteration, ε=$ε)
                verbose && println("Time taken by method: $(t) seconds")
            else
                t = NaN
            end
            push!(df, (Method="CRM-VIP Algorithm 2", Dimension=n, Num_Ellipsoids=num_ellipsoids, Iterations=iterations, Final_Error=error, Time=t, Status=status))

        else 
            println("Method $(method) not recognized.")
            
        end
    end
    verbose && println("="^70)
    return df

end

## Run the test for Example 5.1
## 5.1.A - Small size problems 

Random.seed!(42) # For reproducibility

df_results = DataFrame(Method=String[], Dimension=Int[], Num_Ellipsoids=Int[], Iterations=Int[], Final_Error=Float64[], Time=Float64[], Status=Symbol[])

for n in [5, 10, 20]
    for num_ellipsoids in [2, 5, 8]
        for _ in 1:2  # Repeat each configuration 5 times
        test_example_5_1!(df_results, n, num_ellipsoids, [:extragradient_vip, 
                    :bellocruz_iusem_2010, :crm_vip_algorithm1, :bellocruz_iusem_2012, :crm_vip_algorithm2]; T=Float64, max_iteration=300_000, ε=1e-5)
        end
    end
end

CSV.write(datadir("sims", "results_example_5_1_small_size.csv"), df_results)

#Removing extragradient method for performance profile
# filter!(row -> row.Method != "Extragradient", df_results)
# Construir matrizes para performance profile
methods = unique(df_results.Method)
number_of_methods = length(methods)
n_problems = nrow(df_results) ÷ number_of_methods  # Assumindo igual número de problemas por método

# Inicializar matrizes com tamanho correto
Time = Matrix{Float64}(undef, n_problems, number_of_methods)
Iterations = Matrix{Int}(undef, n_problems, number_of_methods)

# Preencher as matrizes
for (j, method) in enumerate(methods)
    method_data = filter(row -> row.Method == method, df_results)
    Time[:, j] = method_data.Time
    Iterations[:, j] = method_data.Iterations
end

# Criar performance profiles
p1 = performance_profile(PlotsBackend(), Time, methods,
    title="Performance Profile - Time",
    xlabel="τ", ylabel="Fraction of problems solved")

p2 = performance_profile(PlotsBackend(), Iterations, methods,
    title="Performance Profile - Iterations",
    xlabel="τ", ylabel="Fraction of problems solved")

display(p1)
display(p2)
## 5.1.B - Large size problems

Random.seed!(42) # For reproducibility

df_results = DataFrame(Method=String[], Dimension=Int[], Num_Ellipsoids=Int[], Iterations=Int[], Final_Error=Float64[], Time=Float64[], Status=Symbol[])
for n in [100, 200, 300]
    for num_ellipsoids in [20, 50, 100]
        for _ in 1:2  # Repeat each configuration 5 times
        test_example_5_1!(df_results, n, num_ellipsoids, [:bellocruz_iusem_2010, :crm_vip_algorithm1]; T=Float64, max_iteration=300_000,ε=1e-5)
        end
    end
end

CSV.write(datadir("sims", "results_example_5_1_large_size.csv"), df_results) 


##


df_results = DataFrame(Method=String[], Dimension=Int[], Num_Ellipsoids=Int[], Iterations=Int[], Final_Error=Float64[], Time=Float64[], Status=Symbol[])
##
Random.seed!(42) # For reproducibility
test_example_5_1!(df_results, 500, 20, [#:bellocruz_iusem_2010,
 :crm_vip_algorithm1, :bellocruz_iusem_2012, :crm_vip_algorithm2]; T=Float64, max_iteration=100_000, compute_time=true)