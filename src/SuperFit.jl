__precompile__()

module SuperFit

using Distributed
using CSV, DataFrames
using Turing, MCMCChains
using Plots, StatsPlots
using LinearAlgebra, PDMats
using Distributions
import ForwardDiff
import ReverseDiff
using Random
using Formatting, ArgParse
using Optim
using Suppressor
using AbstractMCMC: AbstractModel
import StatsBase
using Logging
using Dates

include("defaults.jl")
include("loguniform.jl")
include("priors.jl")
include("format_inputs.jl")
include("model.jl")
include("convergence_metrics.jl")
include("sample_multithread.jl")
include("diagnostics.jl")

#precompile(sample_or_load_trace, (AbstractModel, String))
end
