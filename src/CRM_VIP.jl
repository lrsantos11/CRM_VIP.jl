
module CRM_VIP

using Reexport
@reexport using LinearAlgebra, SparseArrays, ProximalOperators, Statistics
const LA = LinearAlgebra
export carm_vip, bellocruz_iusem_2010, bellocruz_iusem_2012

include("CRM_functions.jl")
include("BI_VIP.jl")
include("CRM_VIP_functions.jl")



end