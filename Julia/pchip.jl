# pchip.jl

"""
    pchip(x, y)

Return an interpolation function `interp(xq)` that computes the pchip interpolant
for the data points in vectors `x` and `y`. The vectors must be sorted by `x`.
The returned function supports scalar and vector inputs (via broadcasting).
"""
function pchip(x::AbstractVector, y::AbstractVector)
    n = length(x)
    if length(y) != n
        error("x and y must be of the same length")
    end

    # Compute intervals and secant slopes
    h = diff(x)
    δ = diff(y) ./ h

    # Allocate array for derivatives
    d = zeros(n)

    # Compute derivatives at interior points
    for i in 2:n-1
        if δ[i-1] * δ[i] > 0   # same sign
            w1 = 2h[i] + h[i-1]
            w2 = h[i] + 2h[i-1]
            d[i] = (w1 + w2) / (w1/δ[i-1] + w2/δ[i])
        else
            d[i] = 0.0
        end
    end

    # Endpoint derivative at x[1]
    d[1] = begin
        if n == 1
            0.0
        elseif n == 2
            δ[1]
        else
            # Use one-sided formula
            d1 = ((2h[1] + h[2]) * δ[1] - h[1] * δ[2]) / (h[1] + h[2])
            # If sign mismatch or overshoot, set to zero
            if sign(d1) != sign(δ[1])
                0.0
            elseif (sign(δ[1]) != sign(δ[2])) && (abs(d1) > 3abs(δ[1]))
                3 * δ[1]
            else
                d1
            end
        end
    end

    # Endpoint derivative at x[n]
    d[n] = begin
        if n == 1
            0.0
        elseif n == 2
            δ[end]
        else
            # Use one-sided formula at the right endpoint
            d_end = ((2h[end] + h[end-1]) * δ[end] - h[end] * δ[end-1]) / (h[end-1] + h[end])
            if sign(d_end) != sign(δ[end])
                0.0
            elseif (sign(δ[end]) != sign(δ[end-1])) && (abs(d_end) > 3abs(δ[end]))
                3 * δ[end]
            else
                d_end
            end
        end
    end

    # The interpolation function
    interp = function(xq::Real)
        # Handle extrapolation by clamping to the endpoints
        if xq <= x[1]
            return y[1]
        elseif xq >= x[end]
            return y[end]
        end

        # Find the interval index such that x[i] <= xq < x[i+1]
        i = searchsortedlast(x, xq)
        h_i = x[i+1] - x[i]
        t = (xq - x[i]) / h_i

        # Hermite basis functions
        h00 = 2t^3 - 3t^2 + 1
        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 = t^3 - t^2

        return h00*y[i] + h10*h_i*d[i] + h01*y[i+1] + h11*h_i*d[i+1]
    end

    return interp
end

# using Plots

# # Sample data
# x = 0:0.1:20
# y = sin.(x).*exp.(-0.1.*x)

# # Create the interpolant
# f = pchip(collect(x), collect(y))

# # Create a fine grid for plotting
# xq = range(first(x), stop=last(x), length=200)
# yq = f.(xq)   # broadcast evaluation

# plot(x, y, seriestype=:scatter, label="Data Points")
# plot!(xq, yq, label="PCHIP Interpolation", lw=2)
# savefig("pchip_interpolation.png")
# println("beep")
