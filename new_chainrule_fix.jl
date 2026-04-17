include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using LaTeXStrings
using Zygote
using Plots
using ChainRulesCore

"Returns bond dimension of link connecting sites j and j+1"
function bonddim(N::Int, χ::Int, j::Int)
    return min(2^j, χ, 2^(N-j))
end

function genPoint(N::Int, χ::Int, b::Int)
    # generate random point on M
    bond = j -> bonddim(N, χ, j)

    local U
    if b == 1
        U1 = randn(ComplexF64, 2, bond(1))
        U1 /= norm(U1)
        UR = [random_right_isometry(bond(j-1), 2*bond(j)) for j in 2:N]
        U = vcat([U1], UR)
    elseif b == N
        UL = [random_left_isometry(bond(j-1)*2, bond(j)) for j in 1:N-1]
        UN = randn(ComplexF64, bond(N-1), 2)
        UN /= norm(UN)
        U = vcat(UL, [UN])
    else
        UL = [random_left_isometry(bond(j-1)*2, bond(j)) for j in 1:b-1]
        UC = randn(ComplexF64, bond(b-1), 2, bond(b))
        UC /= norm(UC)
        UR = [random_right_isometry(bond(j-1), 2*bond(j)) for j in b+1:N]
        U = vcat(UL, [UC], UR)
    end
    return U
end

function genTanVec(arrU, b::Int)
    arrV = [randn(ComplexF64, size(U)) for U in arrU]    
    arrV = projectMixed(arrU, arrV, b)
    arrV /= sqrt(innerMixed(arrV, arrV))
end

function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
    U0 = genPoint()
    V = genTanVec(U0)
    func, grad = computeCostGrad(U0)
    gradV = inner(grad, V) 
    E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end



# CHECK GRADIENT
N = 4; χ = 2; ogc = 4
V_array = genPoint(N, χ, ogc)

function cost(arrV::Vector{<:AbstractArray}, b::Int)
    psi = vecToITensor(arrV, b)
    psi = orthogonalize(psi, b)
    return norm(psi, b)
end

tpsi = vecToITensor(V_array, 4)
orthogonalize(tpsi, 4)

fRgrad = (func, arrV) -> begin
    N = length(arrV)
    fU, gU = withgradient(func, arrV)
    gU = gU[1]
    riemG = projectMixed(arrV, gU, 4) 
    return fU, riemG
end

cost_ogcfix = arrV -> cost(arrV, 4)
cost(V_array, 4)
gradient(cost_ogcfix, V_array)
fRgrad(cost_ogcfix, V_array)


function retractMixedFixed(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, 4)
end


testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), arrV -> fRgrad(cost_ogcfix, arrV), innerMixed, retractMixedFixed)



# CHECK SCALING