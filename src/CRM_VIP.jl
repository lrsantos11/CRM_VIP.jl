
module CRM_VIP

using Reexport
@reexport using LinearAlgebra, SparseArrays, ProximalOperators, Statistics
const LA = LinearAlgebra
export carm_vip, LA

include("CARM_functions.jl")
include("VIP_utils.jl")



end