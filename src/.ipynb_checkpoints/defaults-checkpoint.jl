const PHASE_MIN = 58058.
const PHASE_MAX = 59528.
const ITERATIONS = 5000
const WALKERS = 6
const FILTERS = ["r", "g"]
const ZEROPOINT_MAG = 22.

@assert PHASE_MIN < PHASE_MAX "Phase minimum must be less than phase maximum."
@assert ITERATIONS > 0 "ITERATIONS constant should be positive"
@assert WALKERS > 0 "WALKERS should be positive (NOTE: to use included convergence metrics, WALKERS should be > 1)"
