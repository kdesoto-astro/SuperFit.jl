struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    LogUniform{T}(a::T, b::T) where {T <: Real} = new{T}(a, b)
end

"""
LogUniform(a::Real, b::Real) = LogUniform(promote(a, b)...)
LogUniform(a::Integer, b::Integer) = LogUniform(float(a), float(b))
LogUniform() = LogUniform(0.0, 1.0, check_args=false)


#### Conversions
convert(::Type{LogUniform{T}}, a::Real, b::Real) where {T<:Real} = LogUniform(T(a), T(b))
convert(::Type{LogUniform{T}}, d::LogUniform{S}) where {T<:Real, S<:Real} = LogUniform(T(d.a), T(d.b), check_args=false)
"""

function LogUniform(a::T, b::T; check_args=true) where {T <: Real}
    #check_args && @check_args(LogUniform, a < b) && @check_args(LogUniform, a > 0) && @check_args(LogUniform, b > 0)
    if b <= a || a <=0 || b <= 0
        error("Invalid limits")
    end
    return LogUniform{T}(a, b)
end

params(d::LogUniform) = (d.a, d.b)
partype(::LogUniform{T}) where {T<:Real} = T

Distributions.maximum(d::LogUniform) = d.b
Distributions.minimum(d::LogUniform) = d.a
Distributions.mean(d::LogUniform) = (d.b - d.a) / (log(d.b) - log(d.a))
Distributions.logpdf(d::LogUniform{T}, x::Real) where {T<:Real} = insupport(d, x) ? 1. / (log(d.b) - log(d.a)) : -Inf
Distributions.median(d::LogUniform) = exp(Distributions.median(Uniform(log(d.a), log(d.b))))
Distributions.rand(rng::AbstractRNG, d::LogUniform) = exp(Distributions.rand(rng, Uniform(log(d.a), log(d.b))))

