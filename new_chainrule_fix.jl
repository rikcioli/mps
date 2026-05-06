using MKL
include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using LaTeXStrings
using Zygote
using Plots
using ChainRulesCore
using Test
using Logging

#Logging.disable_logging(Logging.Warn)

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

function applyED(Uvec::Vector{<:AbstractMatrix}, ψ::AbstractVector, N::Int, depth::Int)
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

function sre2zip(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    Uψ = apply_brickwork(arrU, ψ)     # assume odd number of layers
    PMPS = get_pauli_mps(Uψ)
    PMPO = MPO(PMPS)
    P2 = zipup(PMPO, PMPS)
    return -log2(real(sproduct(P2,P2))) - length(Uψ)
end

function test_sre2(sre2_func)
    N = 4; χ = 2
    sites = siteinds("Qubit", N)

    for trial in 1:100
        psi = random_mps(ComplexF64, sites; linkdims = χ)
        U_array = [random_unitary(4) for _ in 1:5]

        sre2_mps = sre2_func(U_array, psi)

        psi_statevec = reshape(Array{ComplexF64}(prod(psi), sites), 2^N)
        Upsi = applyED(U_array, psi_statevec, N, 3)

        sre2_ED = fastEDMagic(Upsi)

        !(abs(sre2_mps - sre2_ED) < 1e-12) && return false
    end

    return true
end


@test test_genPoint()
@test test_MPS()
#@code_warntype test_MPS()
@test test_move_center()
#@code_warntype test_move_center()
@test test_move2()
#@code_warntype test_move2()
@test test_applyED()
@test test_apply_brickwork()
#@code_warntype test_apply_brickwork()

@test test_sre2(sre2)
#@code_warntype test_sre2(sre2)
@test test_sre2(sre2zip)
@code_warntype test_sre2(sre2zip)



# GRADIENT OF ISOMETRIES
N = 4; χ = 4; ogc = 3
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
    psiV = MPS(arrV, ogc; sites = siteinds(psiW))
    return real(sproduct(psiW, psiV))
end

cost_sproduct(V_array)
gradient(cost_sproduct, V_array)[1]
fg_sproduct = arrV -> withgrad_Riemannian(cost_sproduct, arrV, ogc)
fg_sproduct(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sproduct, innerMixed, retractMixed_ogc)


# GRADIENT OF SVDcontract DOESNT SEEM TO WORK WHEN BRA IS INVOLVED
function cost_SVDcontract(arrV::Vector{<:AbstractArray})
    N = length(arrV)
    n = div(N,2)
    psi = toITensors(arrV, ogc)
    psibra, unbra = bra(psi)

    sites = siteinds(psi)
    links = linkinds(psi)
    brasites = siteinds(psibra)
    bralinks = linkinds(psibra)


    Tn = delta(sites[n], brasites[n])*psi[n]*psibra[n]
    linds = [links[n-1], bralinks[n-1]]
    tensors = [Tn, psi[n+1], psibra[n+1], delta(sites[n+1], brasites[n+1])]
    ((U, R), eps), _ = SVDcontract(tensors, linds)
    
    state = [U, R]
    return real(sproduct(state, dag.(state)))
end

function ChainRulesCore.rrule(::typeof(cost_SVDcontract), arrV::Vector{<:AbstractArray})
    N = length(arrV)
    n = div(N,2)
    psi, backten = pullback(toITensors, arrV, ogc)
    psibra, unbra = bra(psi)

    sites = siteinds(psi)
    links = linkinds(psi)
    bralinks = linkinds(psibra)
    brasites = siteinds(psibra)

    Tn = psi[n]*delta(sites[n], brasites[n])*psibra[n]
    linds = [links[n-1], bralinks[n-1]]
    tensors = [Tn, psi[n+1], psibra[n+1], delta(sites[n+1], brasites[n+1])]
    ((U, R), eps), tape = SVDcontract(tensors, linds)
    
    state = [U, R]
    res, backsprod = pullback(sproduct, state, dag.(state))

    function cost_SVDcontract_pullback(Δres)
        Δres*=1.0+0.0im
        Δstate, Δstate_dg = backsprod(Δres)
        Δstate = 2*Δstate   # CHECKED, THIS IS CORRECT
        (ΔTn, Δpsi_np1, Δpsibra_np1, _) = SVDcontract_pullback(Δstate, tape)

        Δpsi_np1 = 2*Δpsi_np1   # CHECKED, THIS IS CORRECT
        Δpsi_n = 2*ΔTn*delta(sites[n], brasites[n])*dag(psibra[n])  # CHECKED, THIS IS CORRECT

        Δpsi = [ITensor(inds(T)) for T in psi]
        Δpsi[n] = Δpsi_n
        Δpsi[n+1] = Δpsi_np1

        (ΔarrV,) = backten(Δpsi)
        return (NoTangent(), ΔarrV)
    end

    return real(res), cost_SVDcontract_pullback
end
cost_SVDcontract(V_array)
gradient(cost_SVDcontract, V_array)[1];
fg_SVDcontract = arrV -> withgrad_Riemannian(cost_SVDcontract, arrV, ogc)
fg_SVDcontract(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_SVDcontract, innerMixed, retractMixed_ogc)


# THIS WORKS
W_array = genPoint(N, χ, ogc)
function cost_SVDcontract2(arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray})
    psi = toITensors(arrV, ogc)
    sites = siteinds(psi)
    linds = [sites[1];]
    ((U, R), eps), _ = SVDcontract(psi, linds)
    state = [U, R]

    psiW = toITensors(arrW, ogc; sites)
    ((UW, RW), eps), _ = SVDcontract(psiW, linds)
    stateW = [UW, RW]
    return real(sproduct(state, stateW))
end

function ChainRulesCore.rrule(::typeof(cost_SVDcontract2), arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray})
    psi, backten = pullback(toITensors, arrV, ogc)
    sites = siteinds(psi)
    linds = [sites[1];]
    ((U, R), eps), tape = SVDcontract(psi, linds)
    state = [U, R]

    psiW, backtenW = pullback((arr, oc) -> toITensors(arr, oc; sites=sites), arrW, ogc)
    ((UW, RW), eps), tapeW = SVDcontract(psiW, linds)
    stateW = [UW, RW]

    res, backsprod = pullback(sproduct, state, stateW)

    function cost_SVDcontract2_pullback(Δres)
        Δres*=1.0+0.0im
        Δstate, ΔstateW = backsprod(Δres)
        Δpsi = SVDcontract_pullback(Δstate, tape)
        ΔpsiW = SVDcontract_pullback(ΔstateW, tapeW)

        (ΔarrV,) = backten(Δpsi)
        (ΔarrW,) = backtenW(ΔpsiW)

        return (NoTangent(), ΔarrV, ΔarrW)
    end

    return real(res), cost_SVDcontract2_pullback
end
cost_SVDcontract2(V_array, W_array)
svdred = arrV -> cost_SVDcontract2(arrV, W_array)
gradient(svdred, V_array)[1]
fg_SVDcontract2 = arrV -> withgrad_Riemannian(svdred, arrV, ogc)
fg_SVDcontract2(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_SVDcontract2, innerMixed, retractMixed_ogc)




# GRADIENT OF PAULIMPS WORKS, SOMETIMES RETURNS NAN FOR DEGENERATE SINGULAR VALUES
W_array = genPoint(N, χ, ogc)
psiW = MPS(W_array, ogc)
function cost_pauli(arrV::Vector{<:AbstractArray})
    N = length(arrV)
    psi = MPS(arrV, ogc)
    sites = siteinds(4, N)
    Ppsi = get_pauli_mps(psi; sites=sites)
    Ppsi2 = get_pauli_mps(psiW; sites=sites)
    return real(sproduct(Ppsi, Ppsi2))
end
@time cost_pauli(V_array)
@time gradient(cost_pauli, V_array)[1];
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
function cost_sre2_product(arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple())
    N = length(arrV)
    psi = MPS(arrV, ogc)
    Ppsi = get_pauli_mps(psi; trunc=trunc_pauli)
    Pmpo = MPO(Ppsi)
    contr = product(Pmpo, Ppsi)
    res = -log2(real(sproduct(contr, contr)))-N
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_sre2_product), arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple())
    N = length(arrV)
    psi, MPS_back = pullback(MPS, arrV, ogc)
    Ppsi, get_pauli_mps_pullback = pullback(mps -> get_pauli_mps(mps; trunc=trunc_pauli), psi)
    Pmpo, MPO_back = pullback(MPO, Ppsi)
    contr, product_back = pullback(product, Pmpo, Ppsi)
    res, sproduct_back = pullback(sproduct, contr, contr)

    m2, m2_back = -log2(real(res))-N, Δm2 -> (NoTangent(), -Δm2/(log(2)*real(res)))

    function cost_sre2_pullback(Δm2)
        _, Δres = m2_back(Δm2)

        Δcontr1, Δcontr2 = sproduct_back(Δres)
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

        return (NoTangent(), ΔarrV, NoTangent())
    end
    return m2, cost_sre2_pullback
end
cost_sre2_product(V_array, ogc)
gradient(cost_sre2_product, V_array, ogc)[1]
fg_sre2 = arrV -> withgrad_Riemannian(cost_sre2_product, arrV, ogc, ogc)
fg_sre2(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2, innerMixed, retractMixed_ogc)


# GRADIENT OF SRE2 WITH zipup
function cost_sre2_zip(arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple(), trunc_zip = NamedTuple())
    N = length(arrV)
    ψ = MPS(arrV, ogc)
    Pψ = get_pauli_mps(ψ; trunc = trunc_pauli)
    W = MPO(Pψ)
    WP = zipup(W, Pψ; trunc = trunc_zip)
    res = -log2(real(sproduct(WP, WP))) - N
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_sre2_zip), arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple(), trunc_zip = NamedTuple())
    N = length(arrV)
    ψ, MPS_back = pullback(MPS, arrV, ogc)
    Pψ, get_pauli_mps_pullback = pullback(psi -> get_pauli_mps(psi; trunc=trunc_pauli), ψ)
    W, MPO_back = pullback(MPO, Pψ)    # at this point Pψ and W have same ortho lims
    WP, zipup_back = pullback((mpo, mps) -> zipup(mpo, mps; trunc=trunc_zip), W, Pψ)
    res, sproduct_back = pullback(sproduct, WP, WP)
    
    m2, m2_back = -log2(real(res))-N, Δm2 -> (NoTangent(), -Δm2/(log(2)*real(res)))

    function cost_sre2_zip_pullback(Δm2)
        _, Δres = m2_back(Δm2)

        ΔWP_1, ΔWP_2 = sproduct_back(Δres)
        @assert isa(ΔWP_1, MPS)
        @assert isa(ΔWP_2, MPS)
        ΔWP_vec = ΔWP_1[:] .+ ΔWP_2[:]
        ΔWP = MPS(ΔWP_vec)
        set_ortho_lims!(ΔWP, ortho_lims(WP))

        ΔW, ΔPψ_1 = zipup_back(ΔWP)
        @assert isa(ΔW, MPO)
        @assert isa(ΔPψ_1, MPS)

        ΔPψ_2 = MPO_back(ΔW)[1]
        @assert isa(ΔPψ_2, MPS)

        ΔPψ_vec = ΔPψ_1[:] .+ ΔPψ_2[:]
        ΔPψ = MPS(ΔPψ_vec)
        set_ortho_lims!(ΔPψ, ortho_lims(Pψ))
        Δψ = get_pauli_mps_pullback(ΔPψ)[1]
        @assert isa(Δψ, MPS)

        ΔarrV, _ = MPS_back(Δψ)

        return (NoTangent(), ΔarrV, NoTangent())
    end
    return m2, cost_sre2_zip_pullback
end
function cost_sre2_zip(arrV)
    return cost_sre2_zip(arrV, ogc; trunc_pauli=(atol=1e-6,), trunc_zip=(atol=1e-6,))
end
cost_sre2_zip(V_array)
gradient(cost_sre2_zip, V_array)[1]
fg_sre2_zip = arrV -> withgrad_Riemannian(cost_sre2_zip, arrV, ogc)
fg_sre2_zip(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2_zip, innerMixed, retractMixed_ogc)




V_array = genPoint(30, 8, 1);
function cost_sre2_zip(arrV)
    return cost_sre2_zip(arrV, 1; trunc_pauli=(maxrank=8,), trunc_zip=(maxrank=8,))
end
@time cost_sre2_zip(V_array);
@time gradient(cost_sre2_zip, V_array)[1];
@profview gradient(cost_sre2_zip, V_array)[1];


### SCALINGS
V_array = genPoint(10, 10, 1);
W_array = genPoint(10, 10, 1);
function cost_pauli(arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray}, ogc; trunc=(maxrank=10,))
    N = length(arrV)
    sites = siteinds(4, N)

    psi = MPS(arrV, ogc)
    Ppsi = get_pauli_mps(psi; sites=sites, trunc)
    psiW = MPS(arrW, ogc)
    Ppsi2 = get_pauli_mps(psiW; sites=sites, trunc)
    return real(sproduct(Ppsi, Ppsi2))
end
@time cost_pauli(V_array, W_array, 1);
@time gradient(cost_pauli, V_array, W_array, 1)[1];
@profview gradient(cost_pauli, V_array, W_array, 1)[1];



chirange = 2 .^(1:4)
results = let cost = cost_pauli, chirange=chirange
    N = 30; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        cost_red = (arrV, arrW, ogc) -> cost(arrV, arrW, ogc; trunc=(maxrank=χ,))
        for iter in 1:10
            @show iter
            V_array = genPoint(N, χ, ogc)
            W_array = genPoint(N, χ, ogc)
            ftime = @elapsed cost_red(V_array, W_array, ogc)
            gtime = @elapsed gradient(cost_red, V_array, W_array, ogc)
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
Plots.plot!(chirange, 1e-5*chirange .^4, yscale=:log10, xscale=:log10, label="O(chi^4)", legend=:bottomright)
Plots.plot!(chirange, 1e-5*chirange .^5, yscale=:log10, xscale=:log10, label="O(chi^5)")



# CHECK SCALING WITH N
Nrange = 4:2:30
results = let cost = cost_sre2_zip, Nrange=Nrange
    χ = 2; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for N in Nrange
        @show N
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
chirange = 2 .^(1:4)
results = let cost = cost_sre2_zip, chirange=chirange
    N = 30; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        cost_red = (arr, ogc) -> cost(arr, ogc; trunc_pauli=(maxrank=χ,), trunc_zip=(maxrank=χ,))
        for iter in 1:10
            @show iter
            V_array = genPoint(N, χ, ogc)
            ftime = @elapsed cost_red(V_array, ogc)
            gtime = @elapsed gradient(cost_red, V_array, ogc)
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
Plots.plot!(chirange, 1e-5*chirange .^4, yscale=:log10, xscale=:log10, label="O(chi^4)", legend=:bottomright)
Plots.plot!(chirange, 1e-5*chirange .^5, yscale=:log10, xscale=:log10, label="O(chi^5)")





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
U_array = [random_unitary(4) for _ in 1:nU]


# WORKS
function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2 = apply_brickwork(arrU, ψ)
    return real(sproduct(ψ, ψ2))
end
cost_applyU(U_array, psi)
cost_applyU_red = arrU -> cost_applyU(arrU, psi)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)


# WORKS
function cost_move_center(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2 = apply_brickwork(arrU, ψ)
    ψ3 = move_center(ψ2, 1)
    return real(sproduct(ψ, ψ3))
end
cost_move_center(U_array, psi)
cost_move_center_red = arrU -> cost_move_center(arrU, psi)
gradient(cost_move_center_red, U_array)
fg_cost_move_center = arrU -> withgrad_Riemannian(cost_move_center_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_move_center, inner, retract)



# WORKS
function cost_pauli(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    N = length(ψ)
    Uψ = apply_brickwork(arrU, ψ)
    sites_pauli = siteinds(4, N)
    Pψ = get_pauli_mps(ψ; sites=sites_pauli)
    PUψ = get_pauli_mps(Uψ; sites=sites_pauli, trunc=(atol=1e-12,))
    return real(sproduct(Pψ, PUψ))
end
cost_pauli(U_array, psi)
cost_pauli_red = arrU -> cost_pauli(arrU, psi)
gradient(cost_pauli_red, U_array)
fg_cost_pauli = arrU -> withgrad_Riemannian(cost_pauli_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_pauli, inner, retract)



# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function cost_product(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2 = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2 = product(W, Pψ)
    return real(sproduct(P2, P2))
end

function ChainRulesCore.rrule(::typeof(cost_product), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    P2, product_back = pullback(product, W, Pψ)
    res, sproduct_back = pullback(sproduct, P2, P2)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_product_pullback(Δresreal)
        _, Δres = real_back(Δresreal)

        ΔP2_1, ΔP2_2 = sproduct_back(Δres)
        @assert isa(ΔP2_1, MPS)
        @assert isa(ΔP2_2, MPS)
        ΔP2_vec = ΔP2_1[:] .+ ΔP2_2[:]
        ΔP2 = MPS(ΔP2_vec)
        reset_ortho_lims!(ΔP2)

        ΔW, ΔPψ_1 = product_back(ΔP2)
        @assert isa(ΔW, MPO)
        @assert isa(ΔPψ_1, MPS)

        ΔPψ_2 = MPO_back(ΔW)[1]
        @assert isa(ΔPψ_2, MPS)

        ΔPψ_vec = ΔPψ_1[:] .+ ΔPψ_2[:]
        ΔPψ = MPS(ΔPψ_vec)
        set_ortho_lims!(ΔPψ, ortho_lims(Pψ))

        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]
        @assert isa(Δψ2, MPS)

        ΔarrU, Δψ = apply_brickwork_back(Δψ2)
        @show typeof(ΔarrU)
        @show typeof(Δψ)

        return (NoTangent(), ΔarrU, NoTangent())
    end
    return resreal, cost_product_pullback
end
cost_product(U_array, psi)
cost_product_red = arrU -> cost_product(arrU, psi)
gradient(cost_product_red, U_array)
fg_cost_product = arrU -> withgrad_Riemannian(cost_product_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_product, inner, retract)



# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function cost_zipup(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2 = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2 = zipup(W, Pψ)
    return real(sproduct(P2, P2))
end

function ChainRulesCore.rrule(::typeof(cost_zipup), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    P2, zipup_back = pullback(zipup, W, Pψ)
    res, sproduct_back = pullback(sproduct, P2, P2)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_zipup_pullback(Δresreal)
        _, Δres = real_back(Δresreal)

        ΔP2_1, ΔP2_2 = sproduct_back(Δres)
        @assert isa(ΔP2_1, MPS)
        @assert isa(ΔP2_2, MPS)
        ΔP2_vec = ΔP2_1[:] .+ ΔP2_2[:]
        ΔP2 = MPS(ΔP2_vec)
        set_ortho_lims!(ΔP2, ortho_lims(P2))

        ΔW, ΔPψ_1 = zipup_back(ΔP2)
        @assert isa(ΔW, MPO)
        @assert isa(ΔPψ_1, MPS)

        ΔPψ_2 = MPO_back(ΔW)[1]
        @assert isa(ΔPψ_2, MPS)

        ΔPψ_vec = ΔPψ_1[:] .+ ΔPψ_2[:]
        ΔPψ = MPS(ΔPψ_vec)
        set_ortho_lims!(ΔPψ, ortho_lims(Pψ))

        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]
        @assert isa(Δψ2, MPS)

        ΔarrU, Δψ = apply_brickwork_back(Δψ2)
        @show typeof(ΔarrU)
        @show typeof(Δψ)

        return (NoTangent(), ΔarrU, NoTangent())
    end
    return resreal, cost_zipup_pullback
end
cost_zipup(U_array, psi)
cost_zipup_red = arrU -> cost_zipup(arrU, psi)
gradient(cost_zipup_red, U_array)
fg_cost_zipup = arrU -> withgrad_Riemannian(cost_zipup_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_zipup, inner, retract)



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