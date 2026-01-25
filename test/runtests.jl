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
##
# include("Example_5.1.jl")
# include("paramonotone_tests.jl")