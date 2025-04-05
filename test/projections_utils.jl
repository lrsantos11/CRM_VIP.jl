"""
    Dykstra(x₀, Projections)

        This function implements the Dykstra algorithm for projecting a point onto the intersection of convex sets.  The Algorithm is in the form of the algorithm in Birgin, E. G., & Raydan, M. (2005). Robust stopping criteria for Dykstra's algorithm. SIAM Journal on Scientific Computing, 26(4), 1405-1414.
    
        input: 
            x₀::AbstractVector
            Projections::Vector{Function}
        output:
            prohj::AbstractVector
"""
function Dykstra(x₀::AbstractArray,
            Projections::Vector{Function};
            EPSVAL::Float64=1e-8, 
            itmax::Int=1_000,
            verbose::Bool=false)
    # Initialize variables
    T = eltype(x₀)
    tired = false
    solved = false
    iterate = 0
    error_ck_I = zero(T)
    num_sets = length(Projections)
    var_size = length(x₀)
    # Initialize the projection variables
    vtemp = zeros(T, var_size)
    xDykstra = Vector{T}[copy(x₀) for _ in 1:num_sets]

    yDykstra = Vector{T}[zeros(T,var_size) for _ in 1:num_sets]
    xproj =  copy(xDykstra[end])
    while !(tired || solved)
        error_ck_I = zero(T)
        # Robust criteria for stopping according to Birgin and Raydan 2006
        for (index, proj) in enumerate(Projections)
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

        (error_ck_I ≤ EPSVAL) && (solved = true)

        (iterate ≥ itmax) && (tired = true)
        xproj =  xDykstra[end]

    end
    return xproj, iterate, error_ck_I

end

