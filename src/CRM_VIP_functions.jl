"""
    crm_vip_algorithm1(x₀, operator_F, projections; max_iteration=3_000, ε = 1e-6)

Implementation of the  Cirumcentered Reflections Method for solving variational inquality problems, i.e.,  to find ``x^*`` lying in a convex set ``C`` such that:
```math
    \\langle F(x^* ), x - x^* \\rangle \\leq 0, \\forall x \\in C
```
Usually, the set ``C`` is defined as the intersection of convex sets ``C_i``. 

Reference: Behling, et al (2025). On circumcentered direct methods for monotone variational inequality problem. 

# Arguments
- `x₀`: Initial point
- `operator_F`: Function that calculates the continous mapping ``F(x)``
- `projections`: Vector of functions that define the convex sets ``C_i``
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance

# Returns
- `xₖ`: Approximate solution
- `error`: Final error (distance between the last two iterations)
- `index_iteration`: Number of iterations performed
"""
function crm_vip_algorithm1(x₀::AbstractVecOrMat{T},
    operator_F::Function,
    functions_gi::Vector{Function},
    subgradients_gi::Vector{Function};
    max_iteration::Int=3_000,
    # β::Function=(i) -> inv(i^(.8)),
    ε::Float64=1e-8, kwargs...) where {T}
    m = length(functions_gi)
    vₖ = [similar(x₀) for _ in 1:m]
    wₖ = similar(x₀)

    # Compute Circumcenter Operator
    function circumcenter_operator(y, x) 
        # Define approx projections onto each set C_i       
    
        return paca_step!(y, x, vₖ, wₖ, functions_gi, subgradients_gi)
    end 

    return bellocruz_iusem_2010(x₀,
        operator_F,
        circumcenter_operator;
        max_iteration=max_iteration,
        ε=ε,
        kwargs...)

end





"""
    crm_vip_algorithm2(x₀, w_Slater_Point, operator_F, functions_gi, subgradients_gi; θ, β=(i) -> inv(i^(0.9)), max_iteration=3_000, ε=1e-6)
    
    Implements the "Algorithm A" structure  fby Bello-Cruz and Iusem (2012) for solving variational inequality problems using the Circumcentered Reflections Method on the product space with the same ideias of PACA method as in Behling et al. (2024)    
# Arguments
- `x₀`: Initial point
- `w_Slater_Point`: A Slater point for the feasible set
- `operator_F`: Function that calculates the continuous mapping ``F(x)``
- `functions_gi`: Vector of functions that define the convex sets ``C_i``
- `subgradients_gi`: Vector of subgradient functions that define the convex sets ``C_i``
- `θ`: Positive parameter used in the algorithm
- `β`: Function that defines the step size sequence
- `max_iteration`: Maximum number of iterations
- `ε`: Convergence tolerance
# Returns
- `xₖ`: Approximate solution
- `wₖ`: Approximate solution in the product space
- `error`: Final error (distance between the last two iterations)
- `index_iteration`: Number of iterations performed


"""
function crm_vip_algorithm2(x₀::AbstractVecOrMat{T},
    w_Slater_Point::AbstractVecOrMat{T},
    operator_F::Function,
    functions_gi::Vector{Function},
    subgradients_gi::Vector{Function},
    θ::Real;
    β::Function=(i) -> inv(i^(0.9)),
    max_iteration::Int=3_000,
    ε::Float64=1e-6, kwargs...) where {T}

    m = length(functions_gi)
    vₖ = [similar(x₀) for _ in 1:m]
    wₖ = similar(x₀)

    # Compute Circumcenter Operator
     circumcenter_operator(y, x) = paca_step!(y, x, vₖ, wₖ, functions_gi, subgradients_gi)
    function_g(x) = maximum([func(x) for func in functions_gi])

    return bellocruz_iusem_2012(x₀,
        w_Slater_Point,
        operator_F,
        circumcenter_operator,
        function_g,
        θ,
        β=β,
        max_iteration=max_iteration,
        ε=ε,
        kwargs...)

end

