__precompile__()

module SuperFit

using Distributed
using CSV, DataFrames
using Turing, MCMCChains
using Plots, DataFrames
using LinearAlgebra, PDMats
using Distributions
import ForwardDiff
import ReverseDiff
using Random
using Formatting, ArgParse
using Optim
using Suppressor
import AbstractMCMC
import StatsBase

include("defaults.jl")
include("loguniform.jl")
include("priors.jl")
include("format_inputs.jl")
include("model.jl")
include("convergence_metrics.jl")
include("sample_multithread.jl")
include("diagnostics.jl")

end
