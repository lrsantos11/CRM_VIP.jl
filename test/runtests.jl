using DrWatson
@quickactivate :CRM_VIP


using CRM_VIP
using CSV
using DataFrames
using Random

include("test_utils.jl")
include("ellipsoids_utils.jl")
include("projections_utils.jl")
include("methods/extragradient.jl")
include("methods/malitsky_2015.jl")
##
include("Example_5.1.jl")
include("Example_5.2.jl")
include("Example_5.3.jl")
