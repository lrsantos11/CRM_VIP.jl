## Creates a random point in R^n with type T and norm between min_value and max_value
function starting_point(n::Int64;T::Type=Float64, max_value::Number = 15, min_value::Number = 5)
    x = zeros(T, n)
    while norm(x) < 2
        x = randn(T, n)
    end
    # norm between min_value and max_value
    foonorm = (max_value - min_value) * rand(T) + min_value
    return foonorm * x / norm(x)

end

####################################
## Creates a random point in R^{m\times n}

function starting_point(m::Int64, n::Int64; T::Type=Float64)
    X = zeros(T, m, n)
    for j in 1:n
        X[:, j] = starting_point(m, T=T)
    end
    return X
end