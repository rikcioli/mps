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


function ordered_inds(psi::Union{Vector{ITensor}, MPS})
    N = length(psi)
    sites = siteinds(psi)
    links = linkinds(psi)
    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]

    inds_all = [inds1; indsbulk; indsN]
    return inds_all
end


function test_MPS()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = MPS(V_arr, ogc)
        !is_orthogonal(psi, ogc) && return false
    end
    return true
end

"Check orthogonality"
function test_move_center()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = MPS(V_arr, ogc)
        psi = move_center(psi, N)
        for ogc_final in 1:N
            psi_final = move_center(psi, ogc_final)
            !is_orthogonal(psi_final[:], ogc_final) && return false     #this explicitly checks orthogonalization
        end
    end
    return true
end

"Check directly against ITensor orthogonalize, tensor by tensor, modulo phases"
function test_move2()
    N = 8; χ = 8
    for ogc in 1:N
        V_arr = genPoint(N, χ, ogc)
        psi = MPS(V_arr, ogc)
        psi = move_center(psi, N)
        psi = move_center(psi, 1)

        psi_mps = copy(psi)
        orthogonalize!(psi_mps, N)
        orthogonalize!(psi_mps, 1)

        for ogc_final in 1:N
            psi_final = move_center(psi, ogc_final)
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
    sites = siteinds("Qubit", N)

    for trial in 1:100
        Uarr = [random_unitary(4) for _ in 1:8]
        mps = random_mps(ComplexF64, sites, linkdims=4)

        psi = applyBW(Uarr, mps)
        psivec = reshape(Array{ComplexF64}(prod(psi), sites), 2^N)

        mps_state = reshape(Array{ComplexF64}(prod(mps), sites), 2^N)
        ψED = applyED(Uarr, mps_state, N, 3)
        !isapprox(real(psivec'*ψED), 1) && return false
    end

    return true
end

function test_apply_brickwork()
    N = 6
    sites = siteinds("Qubit", N)

    for trial in 1:1000
        ψ = random_mps(ComplexF64, sites, linkdims=4)
        Uarr = [random_unitary(4) for _ in 1:8]

        ψfinal = apply_brickwork(Uarr, ψ)
        ψfinal_statevec = reshape(Array{ComplexF64}(prod(ψfinal), sites), 2^N)

        ψ_statevec = reshape(Array{ComplexF64}(prod(ψ), sites), 2^N)
        ψED = applyED(Uarr, ψ_statevec, N, 3)

        !isapprox(real(ψfinal_statevec'*ψED), 1) && return false
    end
    return true
end


function sre2(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    Uψ = apply_brickwork(arrU, ψ)     # assume odd number of layers
    PMPS = get_pauli_mps(Uψ)
    PMPO = MPO(PMPS)
    P2 = product(PMPO, PMPS)
    return -log2(real(sproduct(P2,P2))) - length(Uψ)
end

function test_sre2()
    N = 4; χ = 2
    sites = siteinds("Qubit", N)

    for trial in 1:100
        psi = random_mps(ComplexF64, sites; linkdims = χ)
        U_array = [random_unitary(4) for _ in 1:5]

        sre2_mps = sre2(U_array, psi)

        psi_statevec = reshape(Array{ComplexF64}(prod(psi), sites), 2^N)
        Upsi = applyED(U_array, psi_statevec, N, 3)

        sre2_ED = fastEDMagic(Upsi)

        !(abs(sre2_mps - sre2_ED) < 1e-12) && return false
    end

    return true
end


@test test_genPoint()
@test test_MPS()
@test test_move_center()
@test test_move2()

@test test_applyED()
@test test_apply_brickwork()

@test test_sre2()



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


# GRADIENT OF SPRODUCT WORKS
W_array = genPoint(N, χ, ogc)
psiW = MPS(W_array, ogc)
function cost_sproduct(arrV::Vector{<:AbstractArray})
    ogc = 3
    psiV = MPS(arrV, ogc; sites = siteinds(psiW))
    return real(sproduct(psiW, psiV))
end

cost_sproduct(V_array)
gradient(cost_sproduct, V_array)[1]
fg_sproduct = arrV -> withgrad_Riemannian(cost_sproduct, arrV, ogc)
fg_sproduct(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sproduct, innerMixed, retractMixed_ogc)


# GRADIENT OF SVDcontract WORKS
function cost_SVDcontract(arrV::Vector{<:AbstractArray})
    psi = vec(MPS(arrV, ogc))
    sites = siteinds(psi)
    linds = (sites[1],)
    U, R = SVDcontract(psi, linds)
    newpsi = [U, R]
    return real(sproduct(newpsi, newpsi))
end
cost_SVDcontract(V_array)
gradient(cost_SVDcontract, V_array)[1]
fg_SVDcontract = arrV -> withgrad_Riemannian(cost_SVDcontract, arrV, ogc)
fg_SVDcontract(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_SVDcontract, innerMixed, retractMixed_ogc)


# GRADIENT OF PAULIMPS WORKS
W_array = genPoint(N, χ, ogc)
psiW = MPS(W_array, ogc)
function cost_pauli(arrV::Vector{<:AbstractArray})
    N = length(arrV)
    ogc = 3
    psi = MPS(arrV, ogc)
    sites = siteinds(4, N)
    Ppsi = get_pauli_mps(psi; sites=sites)
    Ppsi2 = get_pauli_mps(psiW; sites=sites)
    return real(sproduct(Ppsi, Ppsi2))
end

cost_pauli(V_array)
gradient(cost_pauli, V_array)[1]
fg_pauli = arrV -> withgrad_Riemannian(cost_pauli, arrV, ogc)
fg_pauli(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_pauli, innerMixed, retractMixed_ogc)


# GRADIENT OF PRODUCT 
function cost_product(arrV::Vector{<:AbstractArray})
    ogc = 3
    psi = MPS(arrV, ogc)
    psimpo = MPO(psi)
    contr = product(psimpo, psi)
    res = real(sproduct(contr, contr))
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_product), arrV::Vector{<:AbstractArray})
    ogc = 3
    psi, MPS_back = pullback(MPS, arrV, ogc)
    psimpo, MPO_back = pullback(MPO, psi)
    contr, product_back = pullback(product, psimpo, psi)
    res, sproduct_back = pullback(sproduct, contr, contr)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_product_pullback(Δresreal)
        _, Δreal = real_back(Δresreal)

        Δcontr1, Δcontr2 = sproduct_back(Δreal)
        @assert isa(Δcontr1, MPS)
        @assert isa(Δcontr2, MPS)
        Δcontr_vec = Δcontr1[:] .+ Δcontr2[:]
        Δcontr = MPS(Δcontr_vec)
        reset_ortho_lims!(Δcontr)

        Δpsimpo, Δpsi = product_back(Δcontr)
        @assert isa(Δpsimpo, MPO)
        @assert isa(Δpsi, MPS)

        Δpsi2 = MPO_back(Δpsimpo)[1]
        @assert isa(Δpsi2, MPS)

        Δpsi_vec = Δpsi[:] .+ Δpsi2[:]
        Δpsi = MPS(Δpsi_vec)
        set_ortho_lims!(Δpsi, ortho_lims(psi))
        ΔarrV, _ = MPS_back(Δpsi)

        return (NoTangent(), ΔarrV)
    end
    return resreal, cost_product_pullback
end
cost_product(V_array)
gradient(cost_product, V_array)[1]
fg_product = arrV -> withgrad_Riemannian(cost_product, arrV, ogc)
fg_product(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_product, innerMixed, retractMixed_ogc)


# GRADIENT OF SRE2 
function cost_sre2(arrV::Vector{<:AbstractArray})
    ogc = 3
    psi = MPS(arrV, ogc)
    Ppsi = get_pauli_mps(psi)
    Pmpo = MPO(Ppsi)
    contr = product(Pmpo, Ppsi)
    res = real(sproduct(contr, contr))
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_sre2), arrV::Vector{<:AbstractArray})
    ogc = 3
    psi, MPS_back = pullback(MPS, arrV, ogc)
    Ppsi, get_pauli_mps_pullback = pullback(get_pauli_mps, psi)
    Pmpo, MPO_back = pullback(MPO, Ppsi)
    contr, product_back = pullback(product, Pmpo, Ppsi)
    res, sproduct_back = pullback(sproduct, contr, contr)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_sre2_pullback(Δresreal)
        _, Δreal = real_back(Δresreal)

        Δcontr1, Δcontr2 = sproduct_back(Δreal)
        @assert isa(Δcontr1, MPS)
        @assert isa(Δcontr2, MPS)
        Δcontr_vec = Δcontr1[:] .+ Δcontr2[:]
        Δcontr = MPS(Δcontr_vec)
        reset_ortho_lims!(Δcontr)

        ΔPmpo, ΔPpsi = product_back(Δcontr)
        @assert isa(ΔPmpo, MPO)
        @assert isa(ΔPpsi, MPS)

        ΔPpsi2 = MPO_back(ΔPmpo)[1]
        @assert isa(ΔPpsi2, MPS)

        ΔPpsi_vec = ΔPpsi[:] .+ ΔPpsi2[:]
        ΔPpsi = MPS(ΔPpsi_vec)
        set_ortho_lims!(ΔPpsi, ortho_lims(Ppsi))

        Δpsi = get_pauli_mps_pullback(ΔPpsi)[1]
        @assert isa(Δpsi, MPS)

        ΔarrV, _ = MPS_back(Δpsi)

        return (NoTangent(), ΔarrV)
    end
    return resreal, cost_sre2_pullback
end
cost_sre2(V_array)
gradient(cost_sre2, V_array)[1]
fg_sre2 = arrV -> withgrad_Riemannian(cost_sre2, arrV, ogc)
fg_sre2(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2, innerMixed, retractMixed_ogc)






###### COST FUNCTIONS WITH UNITARIES ######

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

N = 4; χ = 2
nU = 2;
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims = χ)
orthogonalize!(psi, 1)
psivec = psi[:]
U_array = [random_unitary(4) for _ in 1:nU]


# WORKS
function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::Vector{ITensor})
    ψ2 = apply(arrU, ψ)
    return real(inner(ψ, ψ2))
end
cost_applyU(U_array, psivec)
cost_applyU_red = arrU -> cost_applyU(arrU, psivec)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)


# WORKS
function cost_move_center(arrU::Vector{<:AbstractMatrix}, ψ::Vector{ITensor})
    N = length(ψ)
    ψ2 = apply(arrU, ψ)
    ψ3 = move_center(ψ2, N, 1)
    return real(inner(ψ, ψ3))
end
cost_move_center(U_array, psivec)
cost_move_center_red = arrU -> cost_move_center(arrU, psivec)
gradient(cost_move_center_red, U_array)
fg_cost_move_center = arrU -> withgrad_Riemannian(cost_move_center_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_move_center, inner, retract)



# WORKS
function cost_pauli(arrU::Vector{<:AbstractMatrix}, ψ::Vector{ITensor})
    N = length(ψ)
    Uψ = apply(arrU, ψ)
    sites_pauli = siteinds(4, N)
    Pψ = pauliMPS(ψ, 1; sites=sites_pauli)
    PUψ = pauliMPS(Uψ, N; sites=sites_pauli, trunc=(atol=1e-12,))
    return real(inner(Pψ, PUψ))
end
cost_pauli(U_array, psivec)
cost_pauli_red = arrU -> cost_pauli(arrU, psivec)
gradient(cost_pauli_red, U_array)
fg_cost_pauli = arrU -> withgrad_Riemannian(cost_pauli_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_pauli, inner, retract)


# WORKS
function cost_mpo(arrU::Vector{<:AbstractMatrix}, ψ::Vector{ITensor})
    ψ = apply(arrU, ψ)
    ψ2 = move_center(ψ, 4, 1)
    ψ2 = apply(arrU, ψ2)
    W = getMPO(ψ)
    W2 = getMPO(ψ2)
    return real(inner(W, W2))
end
cost_mpo(U_array, psivec)
cost_mpo_red = arrU -> cost_mpo(arrU, psivec)
gradient(cost_mpo_red, U_array)
fg_cost_mpo = arrU -> withgrad_Riemannian(cost_mpo_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_mpo, inner, retract)


# 
function cost_apply(arrU::Vector{<:AbstractMatrix}, ψ::Vector{ITensor})
    N = length(ψ)
    ψ2 = apply(arrU, ψ)
    Pψ = pauliMPS(ψ2, N)
    W = getMPO(Pψ)
    P2 = apply(W, Pψ)
    return real(inner(P2, P2))
end
cost_apply(U_array, psivec)
cost_apply_red = arrU -> cost_apply(arrU, psivec)
gradient(cost_apply_red, U_array)
fg_cost_apply = arrU -> withgrad_Riemannian(cost_apply_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_apply, inner, retract)



# CHECK SCALING WITH Chi
chirange = 2 .^(1:8)
results = let cost = cost, chirange=chirange
    N = 30; ogc = 1; nU = 29
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        for _ in 1:100
            U_array = genPoint(N, χ, ogc)
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