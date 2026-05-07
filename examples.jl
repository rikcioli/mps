using MKL
include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS

function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
    U0 = genPoint()
    func, grad = computeCostGrad(U0)

    V = genTanVec(U0)
    gradV = inner(grad, V) 
    E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end



# Checking gradient of sre2 with isometries

N = 4; χ = 4; ogc = 2
V_array = genPoint(N, χ, ogc)

withgrad_Riemannian = (func, arrV::Vector{<:AbstractArray}, ogc::Int, args...) -> begin
    val, grad = withgradient(func, arrV, args...)
    Rgrad = projectMixed(arrV, grad[1], ogc)
    return val, Rgrad
end

function retractMixed_ogc(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, ogc)
end

sre2(V_array, ogc, :direct)
sre2(V_array, ogc, :zipup)
sre2(V_array, ogc, :direct; trunc_pauli=(atol=1e-1,))
sre2(V_array, ogc, :zipup; trunc_pauli=(atol=1e-1,), trunc_product=(atol=1e-1,))

gradient(sre2, V_array, ogc, :zipup)

sre2_red = arrV -> sre2(arrV, ogc, :zipup; trunc_pauli=(atol=1e-1,), trunc_product=(atol=1e-1,))
fg_sre2 = arrV::Vector{<:AbstractArray} -> withgrad_Riemannian(sre2_red, arrV, ogc)
fg_sre2(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2, innerMixed, retractMixed_ogc)




# checking gradient of sre2 with unitaries

function genUnitary(nU)
    U0 = [random_unitary(4) for _ in 1:nU]
    return U0
end

function genTanVec(Uvec)
    V = [randn(ComplexF64, 4, 4) for _ in eachindex(Uvec)]
    V = skew.(V)
    V = Uvec .* V
    V /= sqrt(inner(V, V))
end

withgrad_Riemannian = (func, arrU, args...) -> begin
    fU, gU = withgradient(func, arrU, args...)
    riemG = project(arrU, gU[1]) 
    return fU, riemG
end


N = 4; χ = 2; nU = 3;
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims = χ)
U_array = [random_unitary(4) for _ in 1:nU]

sre2(U_array, psi, :direct)
sre2(U_array, psi, :zipup)
sre2(U_array, psi, :direct; trunc_bw = (maxrank=χ,), trunc_pauli = (atol=1e-1,))
sre2(U_array, psi, :zipup; trunc_bw = (maxrank=χ,), trunc_pauli=(atol=1e-1,), trunc_product = (atol=1e-2,))

gradient(sre2, U_array, psi, :zipup)

sre2_red = arrU -> sre2(arrU, psi, :zipup; trunc_bw = (maxrank=χ,), trunc_pauli=(atol=1e-1,), trunc_product = (atol=1e-2,))
fg_sre2 = arrU::Vector{<:AbstractMatrix} -> withgrad_Riemannian(sre2_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_sre2, inner, retract)