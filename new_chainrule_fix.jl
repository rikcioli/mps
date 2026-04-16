include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using LaTeXStrings
using Zygote
using Plots
using ChainRulesCore


function genPoint(N::Int)
    # generate random point on M
    U0 = [[random_left_isometry(2, 2)]; 
            [random_left_isometry(4, 2) for _ in 2:N-1]; 
            [randn(ComplexF64, 2, 2)]]
    return U0
end

function genTanVec(arrU)
    N = length(arrU)
    arrV = [[randn(ComplexF64, 2, 2)]; 
            [randn(ComplexF64, 4, 2) for _ in 2:N-1]; 
            [randn(ComplexF64, 2, 2)]]
    arrV = projectMixed(arrU, arrV, N)
    arrV /= sqrt(inner(arrV, arrV))
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




V_array = genPoint(4)
psi = vecToITensor(V_array, 4)

function cost(arrV::Vector{<:Matrix})
    psi = vecToITensor(arrV, 4)
    psi = orthogonalize(psi, 4)
    return norm(psi, 4)
end

function cost_simple(arrV::Vector{<:Matrix})
    return real(tr(arrV[1]' * arrV[1] + arrV[2]' * arrV[2] + arrV[3]' * arrV[3] + arrV[4]' * arrV[4]))
end
cost_simple(V_array)
gradient(cost_simple, V_array)[1]
fRgrad(cost_simple, V_array)

fRgrad = (func, arrV) -> begin
    N = length(arrV)
    fU, gU = withgradient(func, arrV)
    gU = gU[1]
    riemG = projectMixed(arrV, gU, N) 
    return fU, riemG
end

fRgrad(cost, V_array)

function retractMixed4(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, 4)
end

testGrad(() -> genPoint(4), genTanVec, arrV -> fRgrad(cost, arrV), inner, retractMixed4)