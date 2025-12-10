using LinearAlgebra
using ProximalOperators


"""
    proj = proj_indicator(indicator,x)
    Projection using Indicator Function from `ProximalOperators.jl`
    """
function proj_indicator(indicator, x)
    proj, _ = prox(indicator, x)
    return proj
end

####################################
"""
reflec = reflec_indicator(indicator,x)
Reflection using Indicator Function from `ProximalOperators.jl`
"""
function reflec_indicator(indicator, x)
    proj = proj_indicator(indicator, x)
    return 2 * proj - x
end
####################################




"""
    dykstra(x₀, projections)

        This function implements the Dykstra's algorithm for projecting a point onto the intersection of convex sets.  The Algorithm is in the form of the algorithm in Birgin, E. G., & Raydan, M. (2005). Robust stopping criteria for Dykstra's algorithm. SIAM Journal on Scientific Computing, 26(4), 1405-1414.
    
        input: 
            x₀::AbstractVector
            projections::Vector{Function}
        output:
            prohj::AbstractVector
"""
function dykstra(x₀::AbstractArray,
            projections::Vector{Function};
            ε::Float64 = 1e-10, 
            itmax::Int = 1_000,
            verbose::Bool = false,
            output_proj::Bool = false)
    # Initialize variables
    T = eltype(x₀)
    tired = false
    solved = false
    iterate = 0
    error_ck_I = zero(T)
    num_sets = length(projections)
    var_size = length(x₀)
    # Initialize the projection variables
    vtemp = zeros(T, var_size)
    xDykstra = Vector{T}[copy(x₀) for _ in 1:num_sets]

    yDykstra = Vector{T}[zeros(T,var_size) for _ in 1:num_sets]
    xproj =  copy(xDykstra[end])
    while !(tired || solved)
        error_ck_I = zero(T)
        # Robust criteria for stopping according to Birgin and Raydan 2006
        for (index, proj) in enumerate(projections)
            if index == 1
                copyto!(xDykstra[index],  proj(xproj - yDykstra[index]))
                # y[index]_{k+1} = y[index]_{k+1} + xDykstra[index] - xproj
                copyto!(vtemp, xDykstra[index] - xproj)
                yDykstra[index] .+= vtemp
                # Robust criteria for stopping according to Birgin and Raydan 2006
                error_ck_I += dot(vtemp, vtemp)
            else
                copyto!(xDykstra[index],  proj(xDykstra[index-1] - yDykstra[index]))
                # y[index]_{k+1} = y[inxdex]_{k+1} + xDykstra[index] - xDykstra[index-1]
                copyto!(vtemp, xDykstra[index] - xDykstra[index-1])
                yDykstra[index] .+= vtemp
                # Robust criteria for stopping according to Birgin and Raydan 2006
                error_ck_I += dot(vtemp, vtemp)

            end 
           
        end 
        copyto!(xproj, xDykstra[end])


        iterate += 1
        verbose && @info "Iteration: $iterate, Error: $error_ck_I"

        (error_ck_I < ε) && (solved = true) 
 
          (iterate ≥ itmax) && (tired = true)
        xproj =  xDykstra[end]

    end
    verbose && @info "Dykstra's Iterates: $iterate, Error: $error_ck_I"
    if output_proj
        return xproj
    else
        return xproj, iterate, error_ck_I
    end

end


# """
# Example of Dykstra's Algorithm as described in Birgin and Raydan 2006
# """
# Ω_1 = IndHalfspace(-ones(2), -10.)
# Ω_2 = IndBox([3, 0.], [10., 4])
# Projections = [x -> proj_indicator(Ω_1, x), x -> proj_indicator(Ω_2, x)]
# x₀ = [-49.,50]
# dykstra(x₀, Projections; verbose=true)