
module CRM_VIP

using Reexport
@reexport using LinearAlgebra, SparseArrays, ProximalOperators

include("CRM_utils.jl")
include("VIP_utils.jl")


export CRM_VIP, CRM_VIP!, find_circumcenter!, find_circumcenter, ProjectIndicator, ReflectIndicator

end