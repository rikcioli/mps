include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using LaTeXStrings
using Zygote
using Plots
using ChainRulesCore
using Test


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


function test_genPoint()
    N = 4; χ = 4;
    for ogc in 1:4
        V_array = genPoint(N, χ, ogc)
        for j in 1:ogc-1
            U = V_array[j]
            if norm(U'*U - I) > 1e-12
                return false
            end
        end
        for j in ogc+1:N
            V = V_array[j]
            if norm(V*V' - I) > 1e-12
                return false
            end
        end
    end
    return true
end


function ordered_inds(psi::Vector{ITensor})
    N = length(psi)
    sites = siteinds(MPS(psi))
    links = linkinds(MPS(psi))
    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]

    inds_all = [inds1; indsbulk; indsN]
    return inds_all
end


function test_vecToITensor()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = vecToITensor(V_arr, ogc)
        !is_orthogonal(psi, ogc) && return false
    end
    return true
end

"Check orthogonality"
function test_move_center()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = vecToITensor(V_arr, ogc)
        psi = move_center(psi, ogc, N; check_og=true)
        for ogc_final in 1:N
            psi_final = move_center(psi, N, ogc_final; check_og=true)
            !is_orthogonal(psi_final, ogc_final) && return false
        end
    end
    return true
end

"Check directly against ITensor orthogonalize, tensor by tensor, modulo phases"
function test_move2()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = vecToITensor(V_arr, ogc)
        psi = move_center(psi, ogc, N)
        psi = move_center(psi, N, 1)

        psi_mps = MPS(psi)
        orthogonalize!(psi_mps, N)
        orthogonalize!(psi_mps, 1)

        for ogc_final in 1:N
            psi_final = move_center(psi, 1, ogc_final)
            inds_final = ordered_inds(psi_final)
            psi_tensors = [Array{ComplexF64}(psi_final[j], inds_final[j]) for j in 1:N]

            psi_mps_final = orthogonalize(psi_mps, ogc_final)
            inds_mps_final = ordered_inds(psi_mps_final[1:N])
            psi_mps_tensors = [Array{ComplexF64}(psi_mps_final[j], inds_mps_final[j]) for j in 1:N]

            # tensors should be the same up to overall phase set by chosen decomposition
            # for move_center it's svd, for orthogonalize it's probably qr
            sum(norm.([abs.(mat) for mat in psi_tensors] .- [abs.(mat) for mat in psi_mps_tensors])) > 1e-12 && return false
        end
    end
    return true
end

function applyBW(U_array::Vector{<:AbstractMatrix}, psi::MPS)
    N = length(psi)
    sites = siteinds(psi)
    n_unitaries = length(U_array)

    # we prepare the values of j to use in the next section here
    pattern = vcat([1:2:N-1; N-2:-2:2])
    # repeat pattern as needed to match n_unitaries, then truncate to exact length
    # if n_unitaries < length(pattern), only the first k elements are used
    jvals = repeat(pattern, ceil(Int, n_unitaries/length(pattern)))[1:n_unitaries]

    gates = [ITensor(U_array[unit_no], sites[j]', sites[j+1]', sites[j], sites[j+1]) for (unit_no, j) in enumerate(jvals)]
    
    n_twolayers = div(n_unitaries, N-1)
    if mod(n_unitaries, N-1) > 0
        n_twolayers += 1
    end

    for j in 1:n_twolayers
        psi = ITensorMPS.apply(gates[1+(j-1)*(N-1) : min(j*(N-1), n_unitaries)], psi)
    end

    return psi
end

function applyED(Uvec, ψ, N, depth)
    j=1
    for i in 1:depth
        if isodd(i)
            lastj = div(N,2)
            ψ = reduce(kron, reverse(Uvec[j:j+lastj-1]))*ψ
        else
            lastj = div(N,2)-1
            ψ = reduce(kron, [[Id]; Uvec[j:j+lastj-1]; [Id]])*ψ
        end
        j += lastj
    end
    return ψ
end

function test_applyED()
    N=6
    Uarr = [random_unitary(4) for _ in 1:8]

    sites =siteinds("Qubit", N)
    zeromps = random_mps(ComplexF64, sites, linkdims=2)
    psi = applyBW(Uarr, zeromps)
    psivec = reshape(Array{ComplexF64}(prod(psi), sites), 2^N)

    zerostate = reshape(Array{ComplexF64}(prod(zeromps), sites), 2^N)
    ψED = applyED(Uarr, zerostate, N, 3)

    return isapprox(real(psivec'*ψED), 1)
end

function test_apply()
    N = 6
    sites = siteinds("Qubit", N)
    zeromps = random_mps(ComplexF64, sites, linkdims=2)
    ψ = zeromps[1:N]
    Uarr = [random_unitary(4) for _ in 1:8]
    ψfinal = apply(Uarr, ψ)
    @show inner(ψfinal,ψfinal)
    ψfinal_vec = reshape(Array{ComplexF64}(prod(ψfinal), sites), 2^N)

    zerostate = reshape(Array{ComplexF64}(prod(zeromps), sites), 2^N)
    ψED = applyED(Uarr, zerostate, N, 3)

    return isapprox(real(ψfinal_vec'*ψED), 1)
end



@test test_genPoint()
@test test_vecToITensor()
@test test_move_center()
@test test_move2()

@test test_applyED()
@test test_apply()




# CHECK GRADIENT

N = 4; χ = 4; ogc = 2
V_array = genPoint(N, χ, ogc)


fRgrad = (func, arrV) -> begin
    N = length(arrV)
    fU, gU = withgradient(func, arrV)
    gU = gU[1]
    riemG = projectMixed(arrV, gU, ogc) 
    return fU, riemG
end

function cost(arrV::Vector{<:AbstractArray}, ogc::Int)
    N = length(arrV)
    psi = vecToITensor(arrV, ogc)
    psi = move_center(psi, ogc, N-1; check_og=false, trunc=(maxrank=2,))
    return inner(psi, psi)
end


cost(V_array, ogc)
gradient(cost, V_array, ogc)
cost_ogc = arrV -> cost(arrV, ogc)
G_array = gradient(cost_ogc, V_array)[1]
fRgrad(cost_ogc, V_array)


function retractMixed_ogc(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, ogc)
end

testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), arrV -> fRgrad(cost_ogc, arrV), innerMixed, retractMixed_ogc)


# CHECK SCALING WITH N
Nrange = 4:2:50
results = let cost = cost, Nrange=Nrange
    χ = 8; ogc = 1 
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


# GRADIENT OF ISOMETRIES
N = 4; χ = 2; ogc = 3
V_array = genPoint(N, χ, ogc)

withgrad_Riemannian = (func, arrV, ogc, args...) -> begin
    val, grad = withgradient(func, arrV, args...)
    Rgrad = projectMixed(arrV, grad[1], ogc)
    return val, Rgrad
end

function retractMixed_ogc(arrA, arrD, t)
    return retractMixed(arrA, arrD, t, ogc)
end


# GRADIENT OF INNER
W_array = genPoint(N, χ, ogc)
psiW = vecToITensor(W_array, ogc)
function cost_inner(arrV::Vector{<:AbstractArray})
    ogc = 3
    psiV = vecToITensor(arrV, ogc; sites = siteinds(psiW))
    return real(inner(psiV, psiW))
end

cost_inner(V_array)
gradient(cost_inner, V_array)[1]
fg_inner = arrV -> withgrad_Riemannian(cost_inner, arrV, ogc)
fg_inner(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_inner, innerMixed, retractMixed_ogc)


# GRADIENT OF PAULIMPS
function cost_pauli(arrV::Vector{<:AbstractArray})
    ogc = 3
    psi = vecToITensor(arrV, ogc)
    Ppsi = pauliMPS(psi, ogc)
    return real(inner(Ppsi, Ppsi))
end

cost_pauli(V_array)
gradient(cost_pauli, V_array)[1]
fg_pauli = arrV -> withgrad_Riemannian(cost_pauli, arrV, ogc)
fg_pauli(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_pauli, innerMixed, retractMixed_ogc)


# GRADIENT OF APPLY
function cost_apply(arrV::Vector{<:AbstractArray})
    ogc = 3
    psi = vecToITensor(arrV, ogc)
    psimpo = getMPO(psi)
    res = apply(psimpo, psi)
    return real(inner(res, res))
end

cost_apply(V_array)
gradient(cost_apply, V_array)[1]
fg_apply = arrV -> withgrad_Riemannian(cost_apply, arrV, ogc)
fg_apply(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_apply, innerMixed, retractMixed_ogc)




function cost(arrU::Vector{<:AbstractMatrix}, arrV::Vector{<:AbstractArray})
    psi0 = vecToITensor(arrV, 1)
    psi = apply(arrU, psi0)
    return real(inner(psi, psi0))
end
N = 10; χ = 2
V_array = genPoint(N, χ, 1)
U_array = [random_unitary(4) for _ in 1:10]
costfn = arrU -> cost(arrU, V_array)
costfn(U_array)
gradient(costfn, U_array)
G_array = gradient(costfn, U_array)[1]

fRgrad = (func, arrU) -> begin
    N = length(arrU)
    fU, gU = withgradient(func, arrU)
    gU = gU[1]
    riemG = project(arrU, gU) 
    return fU, riemG
end
fRgrad(costfn, U_array)


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

testGrad(() -> genUnitary(3), genTanVec, arrU -> fRgrad(costfn, arrU), inner, retract)

