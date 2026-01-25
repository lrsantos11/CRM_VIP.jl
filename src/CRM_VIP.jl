
module CRM_VIP

using Reexport
@reexport using LinearAlgebra, SparseArrays, ProximalOperators, Statistics, Random
const LA = LinearAlgebra
export starting_point, crm_vip_algorithm1, crm_vip_algorithm2, bellocruz_iusem_2010, bellocruz_iusem_2012

include("CRM_functions.jl")
include("BI_VIP_improved.jl")
include("CRM_VIP_functions.jl")



end