const PHASE_MIN::Float64 = 58058.
const PHASE_MAX::Float64 = 59528.
const ITERATIONS::Int64 = 5000
const WALKERS::Int64 = 6
const FILTERS::Vector{String} = ["r", "g"]
const ZEROPOINT_MAG::Float64 = 22.

@assert PHASE_MIN < PHASE_MAX "Phase minimum must be less than phase maximum."
@assert ITERATIONS > 0 "ITERATIONS constant should be positive"
@assert WALKERS > 0 "WALKERS should be positive (NOTE: to use included convergence metrics, WALKERS should be > 1)"
