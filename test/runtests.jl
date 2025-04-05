using DrWatson, Test
@quickactivate "CRM_VIP.jl"


include(srcdir("CRM_VIP.jl"))
using Main.CRM_VIP

include("ellipsoids_utils.jl")
include("projections_utils.jl")