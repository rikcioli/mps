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
using HDF5, JLD2

#Logging.disable_logging(Logging.Warn)


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


function genUnitary(nU)
    U0 = [random_unitary(4) for _ in 1:nU]
    return U0
end

function genUnitaryProduct(nU)
    U0 = [kron(random_unitary(2), random_unitary(2)) for _ in 1:nU]
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

            psi_mps_final = ITensorMPS.orthogonalize(psi_mps, ogc_final)
            inds_mps_final = ordered_inds(psi_mps_final[1:N])
            psi_mps_tensors = [Array{ComplexF64}(psi_mps_final[j], inds_mps_final[j]) for j in 1:N]

            # tensors should be the same up to overall phase set by chosen decomposition
            # for move_center it's svd, for orthogonalize it's probably qr
            sum(norm.([abs.(mat) for mat in psi_tensors] .- [abs.(mat) for mat in psi_mps_tensors])) > 1e-12 && return false
        end
    end
    return true
end


"Check orthogonality"
function test_move_center_MPO()
    N = 8; χ = 8
    for ogc in 1:N
        sites = siteinds("Qubit", N)
        psi = random_mps(ComplexF64, sites; linkdims=χ)
        orthogonalize!(psi, ogc)
        rho = density_matrix(psi)
        rho = move_center(rho, N)
        for ogc_final in 1:N
            rho_final = move_center(rho, ogc_final)
            !is_orthogonal(rho_final[:], ogc_final) && return false     #this explicitly checks orthogonalization
        end
    end
    return true
end


"Uses ITensor apply, can be used to check later functions"
function applyBW(U_array::Vector{<:AbstractMatrix}, psi::MPS; shift=0)
    N = length(psi)
    sites = siteinds(psi)
    nU = length(U_array)

    # we prepare the values of j to use in the next section here
    pattern = iseven(N) ? vcat([1+shift:2:N-1; N-2+shift:-2:2-shift]) : vcat([1+shift:2:N-1; N-1-shift:-2:2-shift])
    # repeat pattern as needed to match nU, then truncate to exact length
    # if nU < length(pattern), only the first k elements are used
    jvals = repeat(pattern, ceil(Int, nU/length(pattern)))[1:nU]

    gates = [ITensor(U_array[unit_no], sites[j]', sites[j+1]', sites[j], sites[j+1]) for (unit_no, j) in enumerate(jvals)]
    
    ngates_2layer = N-1
    n_twolayers = div(nU, ngates_2layer)
    if mod(nU, ngates_2layer) > 0
        n_twolayers += 1
    end
 
    for j in 1:n_twolayers
        psi = ITensorMPS.apply(gates[1+(j-1)*ngates_2layer : min(j*ngates_2layer, nU)], psi)
    end

    return psi
end

function applyED(Uvec::Vector{<:AbstractMatrix}, ψ::AbstractVector, N::Int, depth::Int, shift::Int=0)
    @assert shift == 0 || shift == 1
    j = 1
    Id = Matrix{ComplexF64}(I, 2, 2)
    for i in 1:depth
        start_site = isodd(i+shift) ? 1 : 2
        ngates = start_site == 1 ? div(N, 2) : div(N-1, 2)
        layer = Uvec[j:j+ngates-1]
        if isodd(i)
            layer = reverse(layer)
        end

        if isodd(N)
            if start_site==1
                pushfirst!(layer, Id)
            else
                push!(layer, Id)
            end
        else
            if start_site==2
                layer = [[Id]; layer; [Id]]
            end
        end

        ψ = reduce(kron, layer)*ψ
        j += ngates
    end
    return ψ
end

function test_applyED()
    for trial in 1:200
        N = rand([6,7])
        sites = siteinds("Qubit", N)
        mps = random_mps(ComplexF64, sites, linkdims=4)

        tau = rand([1,2,3])
        shift = rand([0,1])
        nU = n_unitaries(N, tau, shift)
        Uarr = [random_unitary(4) for _ in 1:nU]

        psi = applyBW(Uarr, mps; shift)
        psivec = reshape(Array{ComplexF64}(prod(psi), sites), 2^N)

        mps_state = reshape(Array{ComplexF64}(prod(mps), sites), 2^N)
        ψED = applyED(Uarr, mps_state, N, tau, shift)
        !isapprox(real(psivec'*ψED), 1) && return false
    end

    return true
end

function test_apply_brickwork()
    for trial in 1:1000
        N = rand([6,7])
        sites = siteinds("Qubit", N)
        ψ = random_mps(ComplexF64, sites, linkdims=4)

        tau = rand([1,2,3])
        shift = rand([0,1])
        nU = n_unitaries(N, tau; shift)
        Uarr = [random_unitary(4) for _ in 1:nU]

        ψfinal, = apply_brickwork(Uarr, ψ; shift=shift)
        ψfinal_statevec = reshape(Array{ComplexF64}(prod(ψfinal), sites), 2^N)

        ψ_statevec = reshape(Array{ComplexF64}(prod(ψ), sites), 2^N)
        ψED = applyED(Uarr, ψ_statevec, N, tau, shift)

        !isapprox(real(ψfinal_statevec'*ψED), 1) && return false
    end
    return true
end

function test_apply_brickwork_toleft()

    for trial in 1:1000
        N = rand([6,7])
        sites = siteinds("Qubit", N)
        ψ = random_mps(ComplexF64, sites, linkdims=4)

        tau = 1
        shift = rand([0,1])
        nU = n_unitaries(N, tau, shift)
        Uarr = [random_unitary(4) for _ in 1:nU]

        ψfinal, = apply_brickwork(Uarr, ψ; shift=shift, to_right=false)
        ψfinal_statevec = reshape(Array{ComplexF64}(prod(ψfinal), sites), 2^N)

        ψ_statevec = reshape(Array{ComplexF64}(prod(ψ), sites), 2^N)
        ψED = applyED(reverse(Uarr), ψ_statevec, N, tau, shift)

        !isapprox(real(ψfinal_statevec'*ψED), 1) && return false
    end
    return true
end

function sre2direct(arrU::Vector{<:AbstractMatrix}, ψ::MPS; truncbw=NamedTuple())
    Uψ, = apply_brickwork(arrU, ψ; trunc=truncbw)     # assume odd number of layers
    PMPS = get_pauli_mps(Uψ)
    PMPO = MPO(PMPS)
    P2 = direct(PMPO, PMPS)
    return -log2(real(sproduct(P2,P2))) - length(Uψ)
end

function sre2zip(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    Uψ, = apply_brickwork(arrU, ψ)     # assume odd number of layers
    PMPS = get_pauli_mps(Uψ)
    PMPO = MPO(PMPS)
    P2, = zipup(PMPO, PMPS)
    return -log2(real(sproduct(P2,P2))) - length(Uψ)
end

function sre2zip2(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    N = length(ψ)
    Uψ, = apply_brickwork(arrU, ψ)     # assume odd number of layers
    PMPS = get_pauli_mps(Uψ)
    PMPO = MPO(PMPS)
    P2, Snorms = zipup(PMPO, PMPS; normalize=true)
    return -2*sum(log2.(Snorms)) - N
end

### SRE2 OF A DENSITY MATRIX ψ
function sre2dm(arrU::Vector{<:AbstractMatrix}, ψ::MPS; truncbw=NamedTuple())
    Uψ, _ = apply_brickwork(arrU, ψ; trunc=truncbw)
    rho = density_matrix(Uψ)
    Pψ = get_pauli_mps(rho)
    W = MPO(Pψ)
    P2, = product(W, Pψ, :direct)
    m2 = -log2(real(sproduct(P2, P2))) - length(Uψ)
    return m2
end

function sre2_normalized(arrU::Vector{<:AbstractMatrix}, ψ::MPS; trunc_bw = NamedTuple(), trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    d = space(siteind(ψ,1))
    b = log2(d)
    N = length(ψ)*b
    ψf, = apply_brickwork(arrU, ψ; trunc=trunc_bw)
    Pψ = get_pauli_mps(ψf; trunc=trunc_pauli)
    W = MPO(Pψ)
    P2, logF = zipup_normalize(W, Pψ; trunc=trunc_product)
    # F = abs(ITensorMPS.inner(P2, P2))
    # @show "fidelityF: ", F
    m2 = -2*logF/log(2) - N
    return m2
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
@test test_apply_brickwork_toleft()
#@code_warntype test_apply_brickwork()

@test test_sre2(sre2direct)
#@code_warntype test_sre2(sre2)
@test test_sre2(sre2zip)
@code_warntype test_sre2(sre2zip)

test_sre2(sre2_normalized)
test_sre2(sre2zip2)
test_sre2(sre2dm)



# GRADIENT OF ISOMETRIES
N = 4; χ = 4; ogc = 2
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


# GRADIENT OF SNORM
function cost_snorm(arrV::Vector{<:AbstractArray})
    psiV = MPS(arrV, ogc; sites = siteinds(psiW))
    return snorm(psiV)
end

cost_snorm(V_array)
gradient(cost_snorm, V_array)[1]
fg_snorm = arrV -> withgrad_Riemannian(cost_snorm, arrV, ogc)
fg_snorm(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_snorm, innerMixed, retractMixed_ogc)


# GRADIENT OF SVDCONTRACT WORKS
W_array = genPoint(N, χ, ogc)
function cost_SVDcontract2(arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray})
    psi = toITensors(arrV, ogc)
    sites = siteinds(psi)
    linds = [sites[1];]
    ((U, R), eps), _ = SVDcontract(psi, linds)

    psiW = toITensors(arrW, ogc; sites)
    ((UW, RW), eps), _ = SVDcontract(psiW, linds)
    
    res = only(Array{ComplexF64}((U*R)*(UW*RW)))
    return real(res)
end

function ChainRulesCore.rrule(::typeof(cost_SVDcontract2), arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray})
    psi, backten = pullback(toITensors, arrV, ogc)
    sites = siteinds(psi)
    linds = [sites[1];]
    ((U, R), eps), tape = SVDcontract(psi, linds)

    psiW, backtenW = pullback((arr, oc) -> toITensors(arr, oc; sites=sites), arrW, ogc)
    ((UW, RW), eps), tapeW = SVDcontract(psiW, linds)

    res = only(Array{ComplexF64}((U*R)*(UW*RW)))

    function cost_SVDcontract2_pullback(Δres)
        Δres*=1.0+0.0im
        ΔU = (Δres*dag(R))*dag(UW*RW)
        ΔR = (dag(U)*Δres)*dag(UW*RW)
        ΔUW = dag(U*R)*(Δres*dag(RW))
        ΔRW = dag(U*R)*(dag(UW)*Δres)

        Δpsi = SVDcontract_pullback((ΔU, ΔR), tape)
        ΔpsiW = SVDcontract_pullback((ΔUW, ΔRW), tapeW)

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




# GRADIENT OF PAULIMPS WORKS, BUT FUNCTION IS NOT DIFFERENTIABLE IF WE SPLIT DEGENERATE EIGENVALUES APART
W_array = genPoint(N, χ, ogc)
psiW = MPS(W_array, ogc)
function cost_pauli(arrV::Vector{<:AbstractArray})
    N = length(arrV)
    psi = MPS(arrV, ogc)
    sites = siteinds(4, N)
    Ppsi = get_pauli_mps(psi; sites=sites, trunc=(maxrank=2,))
    Ppsi2 = get_pauli_mps(psiW; sites=sites, trunc=(maxrank=2,))
    return real(sproduct(Ppsi, Ppsi2))
end
@time cost_pauli(V_array)
@time gradient(cost_pauli, V_array)[1];
fg_pauli = arrV -> withgrad_Riemannian(cost_pauli, arrV, ogc)
fg_pauli(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_pauli, innerMixed, retractMixed_ogc)


# GRADIENT OF DIRECT 
function cost_direct(arrV::Vector{<:AbstractArray})
    psi = MPS(arrV, ogc)
    psimpo = MPO(psi)
    contr = direct(psimpo, psi)
    res = real(sproduct(contr, contr))
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_direct), arrV::Vector{<:AbstractArray})
    psi, MPS_back = pullback(MPS, arrV, ogc)
    psimpo, MPO_back = pullback(MPO, psi)
    contr, direct_back = pullback(direct, psimpo, psi)
    res, sproduct_back = pullback(sproduct, contr, contr)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_direct_pullback(Δresreal)
        _, Δreal = real_back(Δresreal)

        Δcontr1, Δcontr2 = sproduct_back(Δreal)
        Δcontr = Δcontr1 .+ Δcontr2

        Δpsimpo, Δpsi1 = direct_back(Δcontr)

        Δpsi2 = MPO_back(Δpsimpo)[1]

        Δpsi = Δpsi1 .+ Δpsi2
        ΔarrV, _ = MPS_back(Δpsi)

        return (NoTangent(), ΔarrV)
    end
    return resreal, cost_direct_pullback
end
cost_direct(V_array)
gradient(cost_direct, V_array)[1]
fg_direct = arrV -> withgrad_Riemannian(cost_direct, arrV, ogc)
fg_direct(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_direct, innerMixed, retractMixed_ogc)


# GRADIENT OF SRE2 WITH DIRECT MULTIPLICATION OF MPO AND MPS
function cost_sre2_direct(arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple())
    N = length(arrV)
    psi = MPS(arrV, ogc)
    Ppsi = get_pauli_mps(psi; trunc=trunc_pauli)
    Pmpo = MPO(Ppsi)
    contr = direct(Pmpo, Ppsi)
    res = -log2(real(sproduct(contr, contr)))-N
    return res
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(cost_sre2_direct), arrV::Vector{<:AbstractArray}, ogc::Int; trunc_pauli = NamedTuple())
    N = length(arrV)
    psi, MPS_back = pullback(MPS, arrV, ogc)
    Ppsi, get_pauli_mps_pullback = pullback(mps -> get_pauli_mps(mps; trunc=trunc_pauli), psi)
    Pmpo, MPO_back = pullback(MPO, Ppsi)
    contr, direct_back = pullback(direct, Pmpo, Ppsi)
    res, sproduct_back = pullback(sproduct, contr, contr)

    m2, m2_back = -log2(real(res))-N, Δm2 -> (NoTangent(), -Δm2/(log(2)*real(res)))

    function cost_sre2_pullback(Δm2)
        _, Δres = m2_back(Δm2)

        Δcontr1, Δcontr2 = sproduct_back(Δres)

        Δcontr = Δcontr1 .+ Δcontr2

        ΔPmpo, ΔPpsi1 = direct_back(Δcontr)

        ΔPpsi2 = MPO_back(ΔPmpo)[1]

        ΔPpsi = ΔPpsi1 .+ ΔPpsi2

        Δpsi = get_pauli_mps_pullback(ΔPpsi)[1]

        ΔarrV, _ = MPS_back(Δpsi)

        return (NoTangent(), ΔarrV, NoTangent())
    end
    return m2, cost_sre2_pullback
end
cost_sre2_direct(V_array, ogc)
gradient(cost_sre2_direct, V_array, ogc)[1]
fg_sre2_direct = arrV -> withgrad_Riemannian(cost_sre2_direct, arrV, ogc, ogc)
fg_sre2_direct(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2_direct, innerMixed, retractMixed_ogc)


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
        ΔWP = ΔWP_1 .+ ΔWP_2

        ΔW, ΔPψ_1 = zipup_back(ΔWP)

        ΔPψ_2 = MPO_back(ΔW)[1]

        ΔPψ = ΔPψ_1 .+ ΔPψ_2

        Δψ = get_pauli_mps_pullback(ΔPψ)[1]

        ΔarrV, _ = MPS_back(Δψ)

        return (NoTangent(), ΔarrV, NoTangent())
    end
    return m2, cost_sre2_zip_pullback
end
function cost_sre2_zip(arrV)
    return cost_sre2_zip(arrV, ogc; trunc_pauli=(atol=1e-3,), trunc_zip=(atol=1e-3,))
end
cost_sre2_zip(V_array)
gradient(cost_sre2_zip, V_array)[1]
fg_sre2_zip = arrV -> withgrad_Riemannian(cost_sre2_zip, arrV, ogc)
fg_sre2_zip(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2_zip, innerMixed, retractMixed_ogc)


sre2(V_array, ogc, :direct)
sre2(V_array, ogc, :zipup; trunc_product = (maxrank=χ,))
gradient(sre2, V_array, ogc, :direct)

sre2_red = arrV -> sre2(arrV, ogc, :direct; trunc_pauli=(atol=1e-4,), trunc_product=(atol=1e-4,))
fg_sre2 = arrV -> withgrad_Riemannian(sre2_red, arrV, ogc)
fg_sre2(V_array)
testGrad(() -> genPoint(N, χ, ogc), arrV -> genTanVec(arrV, ogc), fg_sre2, innerMixed, retractMixed_ogc)




### ADDITIONAL CHECK ON THE SLOWEST PART OF THE CODE

X_array = genPoint(10, 10, 1);
Y_array = genPoint(10, 10, 1);
function cost_pauli(arrV::Vector{<:AbstractArray}, arrW::Vector{<:AbstractArray}, ogc; trunc=(maxrank=8,))
    N = length(arrV)
    sites = siteinds(4, N)

    psi = MPS(arrV, ogc; check_og=false)
    Ppsi = get_pauli_mps(psi; sites=sites, trunc)
    psiW = MPS(arrW, ogc; check_og=false)
    Ppsi2 = get_pauli_mps(psiW; sites=sites, trunc)
    return real(sproduct(Ppsi, Ppsi2))
end
red_pauli = arr -> cost_pauli(arr, Y_array, 1)
@time red_pauli(X_array);
@time gradient(red_pauli, X_array)[1];
@profview gradient(red_pauli, X_array)[1];


chirange = 2 .^(1:5)
results = let cost = cost_pauli, chirange=chirange
    N = 12; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        cost_red = (arrV, arrW, ogc) -> cost(arrV, arrW, ogc; trunc=(atol=1e-1,))
        for iter in 1:3
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
Plots.plot!(chirange, 2e-6*chirange .^4, yscale=:log10, xscale=:log10, label="O(chi^4)", legend=:bottomright)
Plots.plot!(chirange, 1e-7*chirange .^5, yscale=:log10, xscale=:log10, label="O(chi^5)")



### SCALINGS OF SRE2

# SCALING WITH N
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


# SCALING WITH χ
chirange = 2 .^(1:6)
results = let cost = cost_sre2_zip, chirange=chirange
    N = 30; ogc = 1 
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        cost_red = (arr, ogc) -> cost(arr, ogc; trunc_pauli=(atol=1e-1,), trunc_zip=(atol=1e-1,))
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

N = 4; χ = 2
nU = n_unitaries(N, 2)
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims = χ)
orthogonalize!(psi, 1)
U_array = [random_unitary(4) for _ in 1:nU]


# WORKS
function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ; trunc=(maxrank=2,))
    return real(sproduct(ψ, ψ2))
end
cost_applyU(U_array, psi)
cost_applyU_red = arrU -> cost_applyU(arrU, psi)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)

function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, Snorms = apply_brickwork(arrU, ψ; trunc=(maxrank=2,))
    return real(sproduct(ψ, ψ2)) - sum(log.(Snorms))
end
cost_applyU(U_array, psi)
cost_applyU_red = arrU -> cost_applyU(arrU, psi)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)


chirange = 20:20:100
trials = 50
results = let chirange=chirange, trials=trials
    N = 30
    sites = siteinds("Qubit", N)
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        for _ in 1:trials
            ψ = random_mps(ComplexF64, sites; linkdims = χ)
            Uarr = genUnitary(n_unitaries(N, 2))
            ftime = @elapsed cost_applyU(Uarr, ψ)
            gtime = @elapsed gradient(cost_applyU, Uarr, ψ)
            push!(ftime_χ, ftime)
            push!(gtime_χ, gtime)
        end
        push!(ftimes, sum(ftime_χ)/trials)
        push!(gtimes, sum(gtime_χ)/trials)
    end
    ftimes, gtimes    
end


Plots.plot(xlabel="chi", ylabel="t (s)")
Plots.plot!(chirange, results[1], label="tf")
Plots.plot!(chirange, results[2], label="tg")

Plots.plot(xlabel="chi", ylabel="tf/tg")
Plots.plot!(chirange, results[2] ./ results[1], label="tg/tfù")

Plots.plot!(chirange, 3e-2*chirange, yscale=:log10, xscale=:log10, label="O(chi)")
Plots.plot!(chirange, 3e-3*chirange .^2, yscale=:log10, xscale=:log10, label="O(chi^2)")
Plots.plot!(chirange, 1e-7*chirange .^3, yscale=:log10, xscale=:log10, label="O(chi^3)", legend=:bottomright)


# WORKS
function cost_move_center(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ)
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
    Uψ, = apply_brickwork(arrU, ψ)
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
function cost_direct(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2 = direct(W, Pψ)
    return real(sproduct(P2, P2))
end

function ChainRulesCore.rrule(::typeof(cost_direct), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    (ψ2,), apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    P2, direct_back = pullback(direct, W, Pψ)
    res, sproduct_back = pullback(sproduct, P2, P2)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_direct_pullback(Δresreal)
        _, Δres = real_back(Δresreal)
        ΔP2_1, ΔP2_2 = sproduct_back(Δres)
        ΔP2 = ΔP2_1 .+ ΔP2_2

        ΔW, ΔPψ_1 = direct_back(ΔP2)
        ΔPψ_2 = MPO_back(ΔW)[1]

        ΔPψ = ΔPψ_1 .+ ΔPψ_2
        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]
        ΔarrU, Δψ = apply_brickwork_back((Δψ2, ZeroTangent()))

        return (NoTangent(), ΔarrU, NoTangent())
    end
    return resreal, cost_direct_pullback
end
cost_direct(U_array, psi)
cost_direct_red = arrU -> cost_direct(arrU, psi)
gradient(cost_direct_red, U_array)
fg_cost_direct = arrU -> withgrad_Riemannian(cost_direct_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_direct, inner, retract)



# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function cost_zipup(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2, = zipup(W, Pψ)
    return real(sproduct(P2, P2))
end

function ChainRulesCore.rrule(::typeof(cost_zipup), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    (ψ2,), apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    (P2,), zipup_back = pullback(zipup, W, Pψ)
    res, sproduct_back = pullback(sproduct, P2, P2)
    resreal, real_back = real(res), Δresreal -> (NoTangent(), Δresreal*(1.0+0.0im))

    function cost_zipup_pullback(Δresreal)
        _, Δres = real_back(Δresreal)

        ΔP2_1, ΔP2_2 = sproduct_back(Δres)
        ΔP2 = ΔP2_1 .+ ΔP2_2

        ΔW, ΔPψ_1 = zipup_back((ΔP2, ZeroTangent()))

        ΔPψ_2 = MPO_back(ΔW)[1]

        ΔPψ = ΔPψ_1 .+ ΔPψ_2

        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]

        ΔarrU, Δψ = apply_brickwork_back((Δψ2,ZeroTangent()))

        return (NoTangent(), ΔarrU, NoTangent())
    end
    return resreal, cost_zipup_pullback
end
cost_zipup(U_array, psi)
cost_zipup_red = arrU -> cost_zipup(arrU, psi)
gradient(cost_zipup_red, U_array)
fg_cost_zipup = arrU -> withgrad_Riemannian(cost_zipup_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_zipup, inner, retract)


# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function cost_zipup2(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2, Snorms = zipup(W, Pψ; normalize=true)
    return sum(Snorms)
end

function ChainRulesCore.rrule(::typeof(cost_zipup2), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    (ψ2,), apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    (P2, Snorms), zipup_back = pullback((a, b) -> zipup(a, b; normalize=true), W, Pψ)
    res, res_back = pullback(sum, Snorms)

    function cost_zipup_pullback(Δres)
        ΔSnorms, = res_back(Δres)
        ΔP2 = [ITensor(ComplexF64, inds(P2[j])) for j in eachindex(P2)]
        ΔW, ΔPψ_1 = zipup_back((ΔP2, ΔSnorms))

        ΔPψ_2 = MPO_back(ΔW)[1]

        ΔPψ = ΔPψ_1 .+ ΔPψ_2

        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]

        ΔarrU, Δψ = apply_brickwork_back((Δψ2,ZeroTangent()))

        return (NoTangent(), ΔarrU, Δψ)
    end
    return res, cost_zipup_pullback
end
cost_zipup2(U_array, psi)
cost_zipup_red = arrU -> cost_zipup2(arrU, psi)
gradient(cost_zipup_red, U_array)
fg_cost_zipup = arrU -> withgrad_Riemannian(cost_zipup_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_zipup, inner, retract)



function cost_zipup3(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, = apply_brickwork(arrU, ψ)
    Pψ = get_pauli_mps(ψ2)
    W = MPO(Pψ)
    P2, Snorms = zipup(W, Pψ; normalize=true)
    return sum(Snorms)
end

function ChainRulesCore.rrule(::typeof(cost_zipup3), arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    (ψ2,), apply_brickwork_back = pullback(apply_brickwork, arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(get_pauli_mps, ψ2)
    W, MPO_back = pullback(MPO, Pψ)
    (P2, Snorms), zipup_back = pullback((a, b) -> zipup(a, b; normalize=true), W, Pψ)
    res, res_back = pullback(sum, Snorms)

    function cost_zipup_pullback(Δres)
        ΔSnorms, = res_back(Δres)
        ΔP2 = ZeroTangent()
        ΔW, ΔPψ_1 = zipup_back((ΔP2, ΔSnorms))

        ΔPψ_2 = MPO_back(ΔW)[1]

        ΔPψ = ΔPψ_1 .+ ΔPψ_2

        Δψ2 = get_pauli_mps_pullback(ΔPψ)[1]

        ΔarrU, Δψ = apply_brickwork_back((Δψ2,ZeroTangent()))

        return (NoTangent(), ΔarrU, Δψ)
    end
    return res, cost_zipup_pullback
end
cost_zipup3(U_array, psi)
cost_zipup_red = arrU -> cost_zipup3(arrU, psi)
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


### COST OF SRE2
function cost_sre2(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    return sre2(arrU, ψ)
end
cost_sre2(U_array, psi)
cost_red = arrU -> cost_sre2(arrU, psi)
gradient(cost_red, U_array)
fg_cost = arrU -> withgrad_Riemannian(cost_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost, inner, retract)





"Extends m x n isometry to M x M unitary, where M is the power of 2 which bounds max(m, n) from above"
function iso_to_unitary(V::AbstractArray)
    nrows, ncols = size(V, 1), size(V, 2)
    dagger = false
    if ncols > nrows
        V = V'
        nrows, ncols = ncols, nrows
        dagger = true
    end
    V = V[:, vec(sum(abs.(V), dims=1) .> 1E-10)]
    nrows, ncols = size(V, 1), size(V, 2)

    bitlength = length(digits(nrows-1, base=2))     # find number of sites of dim 2
    D = 2^bitlength

    U = zeros(ComplexF64, (D, D))
    U[1:nrows, 1:ncols] = V
    kerU = nullspace(U')
    U[:, ncols+1:D] = kerU

    if dagger
        U = copy(U')
    end

    return U
end

"Extract isometries from bond dimension-1 MPS and convert them to depth-1 brickwork circuit"
function to_layer(ψ::MPS)
    N = length(ψ)
    @assert maxlinkdim(ψ)==1
    orthogonalize!(ψ, N)
    sites = siteinds(ψ)
    links = linkinds(ψ)

    combiners1 = [combiner((sites[1], links[1]))]
    combiners = [combiner([sites[j]; links[j-1:j]]) for j in 2:N-1]
    combinersN = [combiner((sites[N], links[N-1]))]
    combiners = [combiners1; combiners; combinersN]

    combinedinds = [combinedind(comb) for comb in combiners]
    tensor_list = [combiners[j]*ψ[j] for j in 1:N]

    Vlist = [Array{ComplexF64}(tensor_list[j], combinedinds[j]) for j in 1:N]

    Ulist = [iso_to_unitary(V) for V in Vlist]
    layer = [kron(Ulist[2*j], Ulist[2*j-1]) for j in 1:div(N,2)]
    return layer
end


sites = siteinds("Qubit", 20)
psi = random_mps(ComplexF64, sites; linkdims = 2)
N = length(psi)
orthogonalize!(psi, 1)

psi_cut = move_center(psi, N; trunc=(maxrank=1,), normalize=true)
@show maxlinkdim(psi)
@show norm(psi_cut)
@show real(dot(psi_cut, psi))
@show maxlinkdim(psi_cut)

# Extract the U_start from the truncated mps
U_start = to_layer(psi_cut)

# We add some random noise to help escaping the saddle point
Vs = skew([randn(ComplexF64, 4, 4) for _ in eachindex(U_start)])
newU = [retract(Matrix{ComplexF64}(I, (4,4)), V, 0.01)[1] for V in Vs]
U_start = newU .* U_start

zeromps = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zeromps, 1)
trunc = (maxerror=1e-2,)

# Captures ψ, zeromps and kwargs
cost_function = (arrU) -> begin
    ϕ, = apply_brickwork(arrU, zeromps; trunc=trunc, normalize=true)
    return -real(sproduct(psi, ϕ))
end

# Combines function and projected gradient
fg = arrU -> begin
    func, grad = withgradient(cost_function, arrU)
    grad = project(arrU, grad[1])
    return func, grad
end

nU = n_unitaries(N, 5)

testGrad(() -> U_start, genTanVec, fg, inner, retract)

testGrad(() -> genUnitary(nU), genTanVec, fg, inner, retract)

testGrad(() -> genUnitaryProduct(nU), genTanVec, fg, inner, retract)




Base.@kwdef mutable struct InversionInstructions
    maxrank::Union{Nothing, Int} = nothing
    maxerror::Union{Nothing, Float64} = nothing
    atol::Float64 = 1e-8
    maxiter::Int = 1000000
    gradtol::Float64 = 1e-8
    n_checkpoint::Int = 5000
    skip_outer::Bool = false
    m::Int = 5                                      
    Σε_max::Float64 = 1e-2
    c_ref::Float64 = Inf                             
    ρ0::Float64 = 1.0
    ρ::Float64 = 1.0
    λ0::Float64 = 1.0
    λ::Float64 = 1.0                                
    outer_iters::Int = 25
    inner_maxiter::Int = 10000
    inner_gradtol::Float64 = 1e-5
    ρ_growth::Float64 = 2.0
end

# copy for a mutable struct (field-by-field)
Base.copy(x::InversionInstructions) = InversionInstructions(
    (getfield(x, f) for f in fieldnames(InversionInstructions))...)

folder = "testdata\\ising\\g1.5\\maxiter\\"
f = h5open(folder*"60_mps.h5","r")
psi_og = read(f,"psi",MPS)
close(f)

psi = dense(psi_og)
sites = siteinds(psi)
N = length(psi)

result = load_object(folder*"N60_T7.jld2");
plot(result.normgradhistory, yscale=:log)
U_array = result[:arrU]

zeromps = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zeromps, 1)

instr = load_object(folder*"N60_T12_instructions.jld2")
trunc = (maxrank=instr.maxrank,)

# Captures ψ, zeromps
cost_function = (arrU) -> begin
    phi, Snorms = apply_brickwork(arrU, zeromps; trunc=trunc)
    return -log(abs(sproduct(psi, phi))) - sum(log.(Snorms))
end

# Combines function and projected gradient
fg = arrU -> begin
    func, grad = withgradient(cost_function, arrU)
    grad = project(arrU, grad[1])
    return func, grad
end

nU = n_unitaries(N, 12)

plt = testGrad(() -> U_array, genTanVec, fg, inner, retract)

testGrad(() -> genUnitary(nU), genTanVec, fg, inner, retract)

testGrad(() -> genUnitaryProduct(nU), genTanVec, fg, inner, retract)


### SCALING WITH TAU

folder = "testdata\\xxz\\Jz2.5\\m50\\"
f = h5open(folder*"300_mps.h5","r")
psi_og = read(f,"psi",MPS)
close(f)
psi = dense(psi_og)
sites = siteinds(psi)
N = length(psi)

zeromps = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zeromps, 1)
instr = load_object(folder*"N$(N)_T1_instructions.jld2")
maxrank = isnothing(instr.maxrank) ? maxlinkdim(psi) : instr.maxrank
trunc = (maxrank=maxrank, rtol=1e-14)


taurange = 1:7
results = let N=N, zeromps=zeromps, trunc=trunc, taurange = taurange
    # Captures ψ, zeromps
    cost_function = (arrU) -> begin
        phi, Snorms = apply_brickwork(arrU, zeromps; trunc=trunc)
        return -log(abs(sproduct(psi, phi))) - sum(log.(Snorms))
    end

    # Combines function and projected gradient
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    ftimes = Float64[]
    gtimes = Float64[]
    gctimes = Float64[]
    bytes = Float64[]
    for tau in taurange
        @show tau
        ftime_tau = Float64[]
        gtime_tau = Float64[]
        gctime_tau = Float64[]
        bytes_tau = Int64[]
        for _ in 1:10
            result = load_object(folder*"N$(N)_T$(tau).jld2")
            U_array = result[:arrU]
            ftime = @elapsed cost_function(U_array)
            res = @timed gradient(cost_function, U_array)
            gtime = res.time
            gctime = res.gctime
            byte = res.bytes
            @show gtime, gctime, bytes
            push!(ftime_tau, ftime)
            push!(gtime_tau, gtime)
            push!(gctime_tau, gctime)
            push!(bytes_tau, byte)
        end
        push!(ftimes, sum(ftime_tau)/length(ftime_tau))
        push!(gtimes, sum(gtime_tau)/length(ftime_tau))
        push!(gctimes, sum(gctime_tau)/length(gctime_tau))
        push!(bytes, sum(bytes_tau)/length(bytes_tau))
    end
    ftimes, gtimes, gctimes, bytes   
end

Plots.plot(xlabel="tau", ylabel="t (s)")
Plots.plot!(taurange, results[1], label="tf")
Plots.plot!(taurange, results[2], label="tg")
Plots.plot!(taurange , 3e-5*taurange , yscale=:log10, xscale=:log10, label="O(chi)")
Plots.plot!(taurange , 1e-5*taurange  .^2, yscale=:log10, xscale=:log10, label="O(chi^2)")
Plots.plot!(taurange , 1e-7*taurange  .^3, yscale=:log10, xscale=:log10, label="O(chi^3)", legend=:bottomright)


taurange = 1:10
results = let N=N, zeromps=zeromps, trunc=trunc, taurange = taurange
    # Captures ψ, zeromps
    cost_function = (arrU) -> begin
        phi, Snorms = apply_brickwork(arrU, zeromps; trunc=trunc)
        return -log(abs(sproduct(psi, phi))) - sum(log.(Snorms))
    end

    # Combines function and projected gradient
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    ftimes = Float64[]
    gtimes = Float64[]
    for tau in taurange
        @show tau
        ftime_tau = Float64[]
        gtime_tau = Float64[]
        for _ in 1:10
            U_array = random_circuit(N, tau)
            ftime = @elapsed cost_function(U_array)
            gtime = @elapsed gradient(cost_function, U_array)
            push!(ftime_tau, ftime)
            push!(gtime_tau, gtime)
        end
        push!(ftimes, sum(ftime_tau)/length(ftime_tau))
        push!(gtimes, sum(gtime_tau)/length(ftime_tau))
    end
    ftimes, gtimes    
end

Plots.plot(xlabel="tau", ylabel="t (s)")
Plots.plot!(taurange, results[1], label="tf")
Plots.plot!(taurange, results[2], label="tg")

Plots.plot!(taurange , 3e-5*taurange , yscale=:log10, xscale=:log10, label="O(chi)")
Plots.plot!(taurange , 1e-5*taurange  .^2, yscale=:log10, xscale=:log10, label="O(chi^2)")
Plots.plot!(taurange , 1e-7*taurange  .^3, yscale=:log10, xscale=:log10, label="O(chi^3)", legend=:bottomright)


### TESTING COMPRESSION AND PULLBACK OF COMPRESSION

# Real Riemannian metric (must match what pcg_solve/warm_start use)
tinner(x, y) = innerLC(x, y)
tnorm(x)     = sqrt(innerLC(x, x))

# Random tangent at arrA (Grassmann sites projected, center site free)
function random_tangent(arrA)
    D = [randn(ComplexF64, size(a)) for a in arrA]
    return projectLC(arrA, D)
end

# Standalone Riemannian gradient, mirroring `fg` inside `compress`
function riem_grad(arrA, ψ, sites, N)
    cost = a -> begin
        ψA = MPS(a, N; sites=sites)
        return snorm(ψA)^2 - 2*real(sproduct(ψ, ψA))
    end
    _, grad = withgradient(cost, arrA)
    return projectLC(arrA, grad[1])
end

# ---- Test 1: self-adjointness of the implemented HVP ----
# ⟨ξ, Hη⟩ == ⟨Hξ, η⟩  for the real metric. No gradients/retraction needed.
function test_self_adjoint(hvp, arrA; ntrials=5)
    println("== self-adjointness ==")
    for _ in 1:ntrials
        ξ = random_tangent(arrA)
        η = random_tangent(arrA)
        a = tinner(ξ, hvp(η))
        b = tinner(hvp(ξ), η)
        println("  rel.asym = ", abs(a - b) / max(abs(a), abs(b), 1e-30))
    end
end

# ---- Test 2: HVP vs central-difference of the re-projected gradient ----
# At the converged point, (P_A grad(R_A(tξ)) - P_A grad(R_A(-tξ)))/2t  ->  Hess[ξ].
function test_fd_hvp(hvp, arrA, ψ, sites, N; ts=[1e-3, 1e-4, 1e-5, 1e-6])
    println("== FD vs HVP (central difference) ==")
    ξ  = random_tangent(arrA)
    Hξ = hvp(ξ)
    for t in ts
        Ap, _ = retractLC(arrA, ξ,  t)
        Am, _ = retractLC(arrA, ξ, -t)
        gp = projectLC(arrA, riem_grad(Ap, ψ, sites, N))   # transport back to T_A
        gm = projectLC(arrA, riem_grad(Am, ψ, sites, N))
        fd = [(p - m) / (2t) for (p, m) in zip(gp, gm)]
        relerr = tnorm(fd .- Hξ) / max(tnorm(Hξ), 1e-30)
        println("  t = $t   rel.err = $relerr")
    end
    # Scalar (Rayleigh) cross-check — most robust to retraction higher-order terms
    println("  scalar check (⟨ξ,Hξ⟩):")
    for t in ts
        Ap, _ = retractLC(arrA, ξ,  t)
        Am, _ = retractLC(arrA, ξ, -t)
        hp = tinner(ξ, projectLC(arrA, riem_grad(Ap, ψ, sites, N)))
        hm = tinner(ξ, projectLC(arrA, riem_grad(Am, ψ, sites, N)))
        fd = (hp - hm) / (2t)
        ex = tinner(ξ, Hξ)
        println("    t = $t   rel.err = $(abs(fd - ex)/max(abs(ex),1e-30))")
    end
end

# ---- Driver ----
function run_hvp_tests(ψ::MPS, χ::Int)
    ψA, _ = compress(ψ, χ; gradtol=1e-12)
    N     = length(ψ)
    sites = siteinds(ψ)
    arrψA = toMatricesLC(ψA)

    L0, R0 = build_environments(ψ, ψA)
    Σ      = compute_sigma(ψ, ψA, arrψA, L0, R0)
    hvp(ξ) = hessian_vector_product(ψ, ψA, arrψA, ξ, L0, R0, Σ)

    test_self_adjoint(hvp, arrψA)
    test_fd_hvp(hvp, arrψA, ψ, sites, N)
end


N = 10
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims=4)
orthogonalize!(psi, N)
run_hvp_tests(psi, 2)


using Printf

# Instrumented PCG: returns solution, iteration count, and residual history.
function pcg_solve_instrumented(hvp, precond, b, x0; tol=1e-10, maxiter=500)
    x = deepcopy(x0)
    r = b .- hvp(x)
    z = precond(r)
    p = deepcopy(z)
    rz_old = tinner(r, z)
    b_norm = max(tnorm(b), 1e-30)
    reshist = Float64[]
    for iter in 1:maxiter
        res = tnorm(r) / b_norm
        push!(reshist, res)
        res < tol && return x, iter, reshist
        Hp = hvp(p)
        α  = rz_old / tinner(p, Hp)
        x  = x .+ α .* p
        r  = r .- α .* Hp
        z  = precond(r)
        rz_new = tinner(r, z)
        β = rz_new / rz_old
        p = z .+ β .* p
        rz_old = rz_new
    end
    return x, maxiter, reshist
end

# Largest/smallest eigenvalue of an operator on the tangent space, via power
# iteration (largest) and inverse-shift-free smallest through CG-solve power
# iteration. Cheap, approximate — just to *see* the conditioning.
function estimate_condition(op, arrA; iters=60)
    x = let D = [randn(ComplexF64, size(a)) for a in arrA]; projectLC(arrA, D); end
    x = x ./ tnorm(x)
    λmax = 0.0
    for _ in 1:iters
        y = op(x); λmax = tinner(x, y); x = y ./ max(tnorm(y), 1e-30)
    end
    # smallest: power-iterate (λmax·I - op) to get the eigenvalue furthest from λmax
    x = let D = [randn(ComplexF64, size(a)) for a in arrA]; projectLC(arrA, D); end
    x = x ./ tnorm(x)
    μ = 0.0
    for _ in 1:iters
        y = λmax .* x .- op(x); μ = tinner(x, y); x = y ./ max(tnorm(y), 1e-30)
    end
    λmin = λmax - μ
    return λmax, λmin, abs(λmax / λmin)
end

function benchmark_preconditioner(ψ::MPS, χ::Int; tol=1e-10)
    ψA, _ = compress(ψ, χ)
    arrψA = toMatricesLC(ψA)
    L0, R0 = build_environments(ψ, ψA)
    Σ      = compute_sigma(ψ, ψA, arrψA, L0, R0)
    hvp(ξ) = hessian_vector_product(ψ, ψA, arrψA, ξ, L0, R0, Σ)

    # A representative right-hand side: a random tangent (stands in for -ΔψA)
    b = let D = [randn(ComplexF64, size(a)) for a in arrψA]; projectLC(arrψA, D); end
    x0 = [zero(a) for a in arrψA]

    # ----- conditioning of H and of M^{-1}H -----
    blocks = build_preconditioner_blocks(arrψA, Σ)
    Minv(r) = apply_preconditioner(blocks, r)
    MinvH(ξ) = Minv(hvp(ξ))

    λmax, λmin, κ      = estimate_condition(hvp,   arrψA)
    λmaxP, λminP, κP   = estimate_condition(MinvH, arrψA)
    @printf("κ(H)        ≈ %.3e   (λmax=%.3e, λmin=%.3e)\n", κ,  λmax,  λmin)
    @printf("κ(M⁻¹H)     ≈ %.3e   (λmax=%.3e, λmin=%.3e)\n", κP, λmaxP, λminP)

    # ----- iteration counts -----
    identity_precond(r) = r
    _, it_none, h_none = pcg_solve_instrumented(hvp, identity_precond, b, x0; tol=tol)
    xP, it_prec, h_prec = pcg_solve_instrumented(hvp, Minv,            b, x0; tol=tol)
    xN, _,       _      = pcg_solve_instrumented(hvp, identity_precond, b, x0; tol=tol)

    @printf("CG iters: no-precond = %d,  precond = %d\n", it_none, it_prec)
    @printf("solutions agree (rel): %.3e\n", tnorm(xP .- xN) / max(tnorm(xN), 1e-30))

    println("residual history (no-precond):")
    for (k, r) in enumerate(h_none); @printf("  %3d  %.3e\n", k, r); end
    println("residual history (precond):")
    for (k, r) in enumerate(h_prec); @printf("  %3d  %.3e\n", k, r); end

    return (it_none=it_none, it_prec=it_prec, κ=κ, κP=κP)
end


N = 20
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims=16)
orthogonalize!(psi, N)
benchmark_preconditioner(psi, 2)



### TEST MIXED PARTIAL ADJOINT

# Perturb the bra-MPS ψ by a matrix-space tangent η (step t), staying in ψ's
# own site/link gauge so the environments remain index-compatible.
function perturb_mps(ψ::MPS, η::Vector{<:AbstractMatrix}, t::Real)
    arr   = toMatricesLC(ψ)
    arrp  = [arr[j] .+ t .* η[j] for j in eachindex(arr)]
    N     = length(ψ)
    return toITensors(arrp, N; check_og=false, sites=siteinds(ψ), links=linkinds(ψ))
end

# Ambient (Wirtinger) gradient of the OVERLAP part  -2 Re⟨ψ_pert | ψA⟩  w.r.t. A,
# as a function of the (perturbed) bra ψ_pert. Mirrors exactly the overlap piece
# used in compute_sigma / fg: E_j = L0[j]·dag(ψ_pert[j])·R0[j+1], then -2·conj.
# NOTE: environments must be rebuilt for the perturbed bra each time.
function ambient_overlap_grad_A(ψ_pert::MPS, ψA::MPS)
    N = length(ψA)
    L0, R0 = build_environments(ψ_pert, ψA)
    Ej_tensors = [L0[j]*dag(ψ_pert[j])*R0[j+1] for j in 1:N]
    Ej = toMatricesLC(Ej_tensors; check_og=false)
    return [-2 .* conj(Ej[j]) for j in 1:N]   # ambient ∂/∂A* of -2Re⟨ψ_pert|ψA⟩
end

function test_mixed_adjoint(ψ::MPS, ψA::MPS, arrψ, arrψA, L0, R0; ntrials=4, t=1e-6)
    N = length(ψ)
    println("== mixed_partial_adjoint pairing test ==")
    for _ in 1:ntrials
        λ = projectLC(arrψA, [randn(ComplexF64, size(a)) for a in arrψA])
        η = projectLC(arrψ,  [randn(ComplexF64, size(b)) for b in arrψ])

        # LHS: ⟨ mixed_partial_adjoint(λ), η ⟩_B
        MB  = mixed_partial_adjoint(ψ, arrψ, ψA, λ, L0, R0)
        lhs = innerLC(MB, η)

        # RHS: ⟨ λ, D_B g[η] ⟩_A  via central difference of the ambient A-gradient
        gp  = ambient_overlap_grad_A(MPS(perturb_mps(ψ, η,  t)), ψA)
        gm  = ambient_overlap_grad_A(MPS(perturb_mps(ψ, η, -t)), ψA)
        DBg = [(gp[j] .- gm[j]) ./ (2t) for j in 1:N]
        rhs = innerLC(λ, projectLC(arrψA, DBg))

        relerr = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        @show lhs rhs relerr
    end
end

# Driver
function run_mixed_adjoint_test(ψ::MPS, χ::Int)
    ψA, _  = compress(ψ, χ)
    arrψ   = toMatricesLC(ψ)
    arrψA  = toMatricesLC(ψA)
    L0, R0 = build_environments(ψ, ψA)
    Σ      = compute_sigma(ψ, ψA, arrψA, L0, R0)
    test_mixed_adjoint(ψ, ψA, arrψ, arrψA, L0, R0)
end

N = 10
sites = siteinds("Qubit", N)
psi = random_mps(sites; linkdims=4)
orthogonalize!(psi, N)
run_mixed_adjoint_test(psi, 2)






# GRADIENT OF compress 



function genPointLC(N, χ)
    ψ = random_mps(ComplexF64, siteinds("Qubit", N); linkdims=χ)
    orthogonalize!(ψ, N)
    arrV = toMatricesLC(ψ)
    return arrV
end

function genTanLC(arrV)
    D = [randn(ComplexF64, size(a)) for a in arrV]
    return projectLC(arrV, D)
end

function genPoint(N::Int, χ::Int, b::Int)
    ψ = random_mps(ComplexF64, siteinds("Qubit", N); linkdims=χ)
    orthogonalize!(ψ, N)
    orthogonalize!(ψ, 1)
    orthogonalize!(ψ, b)
    arrV = toMatrices(ψ, b)
    return arrV
end

function genPointRC(N, χ)
    ψ = random_mps(ComplexF64, siteinds("Qubit", N); linkdims=χ)
    orthogonalize!(ψ, 1)
    arrV = toMatricesRC(ψ)
    return arrV
end

function genTanRC(arrV)
    D = [randn(ComplexF64, size(a)) for a in arrV]
    return projectRC(arrV, D)
end


N = 4; χ = 4

W_array = Vector{Matrix{ComplexF64}}(genPoint(N, χ, 1))
withgrad_Riemannian = (func, arrV, args...) -> begin
    val, grad = withgradient(func, arrV, args...)
    Rgrad = projectRC(arrV, grad[1])
    return val, Rgrad
end

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

function cost_compress(arrV::Vector{<:AbstractMatrix})
    N = length(arrV)
    psiV = MPS(arrV, 1)
    psiVcompr, _ = compress(psiV, 2)
    return real(sproduct(psiV, psiVcompr))
end

function ChainRulesCore.rrule(::typeof(cost_compress), arrV::Vector{<:AbstractMatrix})
    N = length(arrV)
    psiV, MPS_back = pullback(MPS, arrV, 1)
    (psiVcompr, conv_info), compr_back = pullback(compress, psiV, 2)   # your custom rrule

    # The scalar overlap, with BOTH MPS arguments held open so we can pull back
    # through each leg separately. Use Zygote for this cheap, multilinear part.
    func = (A, B) -> real(sproduct(A, B))
    cost, ovlp_back = pullback(func, psiV, psiVcompr)

    function cost_compress_pullback(c̄)
        # 1. Pull c̄ back through the overlap to BOTH legs.
        ΔpsiV_direct, ΔpsiVcompr = ovlp_back(c̄)        # adjoints w.r.t. bra and ket

        # 2. Pull ΔpsiVcompr back through compress to its input psiV.
        #    compr_back returns (NoTangent(), Δ_into_psiV, NoTangent()).
        (ΔpsiV_viacompr,) = compr_back((ΔpsiVcompr, nothing))

        # 3. Sum the two contributions to psiV's adjoint.
        ΔpsiV = ΔpsiV_direct .+ ΔpsiV_viacompr

        # 4. Pull ΔpsiV_total back through MPS(arrV, N) to arrV.
        (ΔarrV,) = MPS_back(ΔpsiV)

        return NoTangent(), ΔarrV
    end

    return cost, cost_compress_pullback
end

cost_compress(W_array)
gradient(cost_compress, W_array)[1]
fg_compress = arrV -> withgrad_Riemannian(cost_compress, arrV)
res, tes = fg_compress(W_array)
testGrad(() -> Vector{Matrix{ComplexF64}}(genPoint(N, χ, 1)), arrV -> genTanRC(arrV), fg_compress, innerRC, retractRC)





### Test scaling with bond dimension

chirange = 2 .^(6:7)
trials = 5
results2 = let chirange=chirange, trials=trials
    N = 30
    sites = siteinds("Qubit", N)
    ftimes = Float64[]
    gtimes = Float64[]
    for χ in chirange
        @show χ
        ftime_χ = Float64[]
        gtime_χ = Float64[]
        for _ in 1:trials
            Varr = genPointRC(N, χ)
            ftime = @elapsed cost_compress(Varr)
            gtime = @elapsed gradient(cost_compress, Varr)
            push!(ftime_χ, ftime)
            push!(gtime_χ, gtime)
        end
        push!(ftimes, sum(ftime_χ)/trials)
        push!(gtimes, sum(gtime_χ)/trials)
    end
    ftimes, gtimes    
end

chirange = 2 .^(2:5)
Plots.plot(xlabel="chi", ylabel="t (s)")
Plots.plot!(chirange, results[1], label="tf")
Plots.plot!(chirange, results[2], label="tg")
Plots.plot!(chirange, 3e-2*chirange, yscale=:log10, xscale=:log10, label="O(chi)")
Plots.plot!(chirange, 3e-3*chirange .^2, yscale=:log10, xscale=:log10, label="O(chi^2)")
Plots.plot!(chirange, 1e-7*chirange .^3, yscale=:log10, xscale=:log10, label="O(chi^3)", legend=:bottomright)



### GRADIENT OF APPLY BRICKWORK VARIATIONAL

N = 4; χ = 2
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims = χ)
orthogonalize!(psi, 1)
nU = n_unitaries(N, 1)
U_array = [random_unitary(4) for _ in 1:nU]


function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPS)
    ψ2, lognorm_factors = apply_brickwork_variational(arrU, ψ, 2)
    return -log(abs(sproduct(ψ, ψ2))) - sum(lognorm_factors)
end
cost_applyU(U_array, psi)
cost_applyU_red = arrU -> cost_applyU(arrU, psi)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)



### GRADIENT OF APPLY BRICKWORK NORMALIZE ON A DENSITY MATRIX REPRESENTED AS MPO

N = 4; χ = 2
sites = siteinds("Qubit", N)
psi = random_mps(ComplexF64, sites; linkdims = χ)
orthogonalize!(psi, 1)
rho = density_matrix(psi)

rho_compr = move_center(rho, N; trunc=(maxrank=3, atol=1e-15))

nU = n_unitaries(N, 3)
U_array = [random_unitary(4) for _ in 1:nU]

function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPO)
    ψ2, Snorms = apply_brickwork(arrU, ψ; trunc=(maxrank=2,))
    return -log(abs(sproduct(ψ, ψ2))) - sum(log.(Snorms))
end
cost_applyU(U_array, rho_compr)
cost_applyU_red = arrU -> cost_applyU(arrU, rho_compr)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)

function cost_applyU(arrU::Vector{<:AbstractMatrix}, ψ::MPO)
    ψ2, _ = apply_brickwork(arrU, ψ; normalize=false, trunc=(maxrank=2,))
    return -log(abs(sproduct(ψ, ψ2)))
end
cost_applyU(U_array, rho_compr)
cost_applyU_red = arrU -> cost_applyU(arrU, rho_compr)
gradient(cost_applyU_red, U_array)
fg_cost_applyU = arrU -> withgrad_Riemannian(cost_applyU_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost_applyU, inner, retract)


function cost_sre2(arrU::Vector{<:AbstractMatrix}, ψ::MPO)
    Uψ, _ = apply_brickwork(arrU, ψ; normalize=false, trunc=(maxrank=2,))
    return sre2(Uψ, :direct)
end
cost_sre2(U_array, rho_compr)
cost_red = arrU -> cost_sre2(arrU, rho_compr)
gradient(cost_red, U_array)
fg_cost = arrU -> withgrad_Riemannian(cost_red, arrU)
testGrad(() -> genUnitary(nU), genTanVec, fg_cost, inner, retract)

