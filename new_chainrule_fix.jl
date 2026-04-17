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

# WATCH OUT: THIS IS STILL NOT GAUGE FIXED, YOU NEED TO FIX A PERMUTATION OF THE BASIS STATES VIA QR
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
N = 4; χ = 4; ogc = 1
V_array = genPoint(N, χ, ogc)

fRgrad = (func, arrV) -> begin
    N = length(arrV)
    fU, gU = withgradient(func, arrV)
    gU = gU[1]
    riemG = projectMixed(arrV, gU, ogc) 
    return fU, riemG
end

function cost(arrV::Vector{<:AbstractArray}, b::Int)
    psi = vecToITensor(arrV, b)
    psi = orthogonalize(psi, b)
    return norm(psi, b)
end

tpsi = vecToITensor(V_array, ogc)
tpsi_og = orthogonalize(tpsi, ogc)
tpsi_og2 = orthogonalize(tpsi_og, ogc)

cost(V_array, ogc)
gradient(cost, V_array, ogc)
cost_ogc = arrV -> cost(arrV, ogc)
G_array = gradient(cost_ogc, V_array)[1]

@code_warntype projectMixed(V_array, G_array, ogc) 
fRgrad(cost_ogc, V_array)


function retractMixed_ogc(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, ogc)
end

testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), arrV -> fRgrad(cost_ogc, arrV), innerMixed, retractMixed_ogc)


# CHECK SCALING WITH N
Nrange = 4:2:50
results = let cost = cost, Nrange=Nrange
    χ = 2; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for N in Nrange
        ftime_N = Float64[]
        gtime_N = Float64[]
        for _ in 1:100
            V_array = genPoint(N, χ, ogc)
            ftime = @elapsed cost(V_array, ogc)
            gtime = @elapsed gradient(cost, V_array, ogc)
            push!(ftime_N, ftime)
            push!(gtime_N, gtime)
        end
        push!(ftimes, sum(ftime_N)/100)
        push!(gtimes, sum(gtime_N)/100)
    end
    ftimes, gtimes    
end

Plots.plot(xlabel="N", ylabel="t (s)")
Plots.plot(Nrange, results[1], label="tf")
Plots.plot!(Nrange, results[2], label="tg")


# CHECK SCALING WITH Chi
chirange = 2 .^(1:8)
results = let cost = cost, chirange=chirange
    N = 50; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        for _ in 1:100
            V_array = genPoint(N, χ, ogc)
            ftime = @elapsed cost(V_array, ogc)
            gtime = @elapsed gradient(cost, V_array, ogc)
            push!(ftime_χ, ftime)
            push!(gtime_χ, gtime)
        end
        push!(ftimes, sum(ftime_χ)/100)
        push!(gtimes, sum(gtime_χ)/100)
    end
    ftimes, gtimes    
end

Plots.plot(xlabel="chi", ylabel="t (s)")
Plots.plot!(chirange, results[1], label="tf")
Plots.plot!(chirange, results[2], label="tg")
Plots.plot!(chirange, 3e-5*chirange, yscale=:log10, xscale=:log10, label="O(chi)")
Plots.plot!(chirange, 1e-5*chirange .^2, yscale=:log10, xscale=:log10, label="O(chi^2)")
Plots.plot!(chirange, 1e-7*chirange .^3, yscale=:log10, xscale=:log10, label="O(chi^3)", legend=:bottomright)