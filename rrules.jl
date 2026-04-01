using ITensors, ITensorMPS
using LinearAlgebra
using MatrixAlgebraKit
using ChainRulesCore

const Id = (1.0+0.0im)*[1.0 0; 0 1.0]
const X = (1.0+0.0im)*[0 1.0 ; 1.0 0]
const Z = (1.0+0.0im)*[1.0 0; 0 -1.0]
const Y = [0.0 -1.0im; 1.0im 0.0]

Psm = zeros(ComplexF64,4,2,2)
Psm[1,:,:] = Id
Psm[2,:,:] = X
Psm[3,:,:] = Z
Psm[4,:,:] = Y
const Ps = deepcopy(Psm)


function truncDM(ψ::MPS; trunc=NamedTuple())
    N = length(ψ)
    orthogonalize!(ψ, 1)
    maxranks = linkdims(ψ)
    if maximum(maxranks) == 1
        return ψ
    end
    if haskey(trunc, :maxrank)
        # remove maxrank from trunc tuple
        kwarg_maxrank = trunc[:maxrank]
        if maximum(maxranks) < kwarg_maxrank    # if it's equal it's a truncated orthogonalization, which could be useful
            return ψ
        end
        trunc = (; filter(p -> first(p) != :maxrank, collect(pairs(trunc)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks = [min(kwarg_maxrank, maxranks[j]) for j in 1:N-1]
    end
    if !haskey(trunc, :atol)
        trunc = (trunc..., atol=eps())
    end

    sites = siteinds(ψ)
    links = linkinds(ψ)

    #envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    #for j in N-1:-1:2
    #    push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    #end
    envR = [delta(ComplexF64, links[j], links[j]') for j in N-1:-1:1]

    vect = ITensor[]    # store tensors that make the final mps
    errs = Float64[]    # store truncation errors
    ilinks = Index[]    # store intermediate links
    isites = [sites[1]]     # store intermediate site1
    iψ12 = [ψ[1:2]]

    icombiners = ITensor[]      # store intermediate combiners

    for j in 1:N-1
        ψ1, ψ2 = iψ12[j]
        site1 = isites[j]
        rhoj_ten = ψ1*envR[N-j]*dag(ψ1)'

        rhoj = Matrix{ComplexF64}(rhoj_ten, site1, site1')
        _, U, ϵ = eigh_trunc(rhoj, trunc=(trunc..., maxrank=maxranks[j]))
        push!(errs, ϵ)

        Ulinkind = Index(size(U)[2], "Link, l=$(j)")
        push!(ilinks, Ulinkind)
        Uten = ITensor(U, isites[j], Ulinkind)
        if j>1
            push!(vect, dag(icombiners[end])*Uten)
        else
            push!(vect, Uten)
        end    

        if j == N-1
            push!(vect, dag(Uten)*ψ1*ψ2)
        else
            c12 = combiner(Ulinkind, sites[j+1])
            push!(icombiners, c12)
            ψ1new = c12*(dag(Uten)*ψ1*ψ2)
            push!(iψ12, [ψ1new; ψ[j+2]])
            site1new = combinedind(c12)
            push!(isites, site1new)
        end
    end

    ψt = MPS(vect)
    ψt.llim = N-1
    ψt.rlim = N+1

    return ψt
end


function ChainRulesCore.rrule(::typeof(truncDM), ψ::MPS; trunc=NamedTuple())
    N = length(ψ)
    orthogonalize!(ψ, 1)
    maxranks = linkdims(ψ)
    if maximum(maxranks) == 1
        return ψ, Δψt -> (NoTangent(), Δψt)
    end
    if haskey(trunc, :maxrank)
        # remove maxrank from trunc tuple
        kwarg_maxrank = trunc[:maxrank]
        @show kwarg_maxrank
        if maximum(maxranks) < kwarg_maxrank    # if it's equal it's a truncated orthogonalization, which could be useful
            return ψ, Δψt -> (NoTangent(), Δψt)
        end
        trunc = (; filter(p -> first(p) != :maxrank, collect(pairs(trunc)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks = [min(kwarg_maxrank, maxranks[j]) for j in 1:N-1]
    end
    if !haskey(trunc, :atol)
        trunc = (trunc..., atol=2*eps())
    end
    sites = siteinds(ψ)
    links = linkinds(ψ)

    ## old one, needed only if mps is not orthonormal
    # envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    # for j in N-1:-1:2
    #   push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    # end
    ## instead we orthogonalize first and then insert deltas for the environment
    envR = [delta(links[j], links[j]') for j in N-1:-1:1]


    vect = ITensor[]    # store tensors that make the final mps
    errs = Float64[]    # store truncation errors
    ilinks = Index[]    # store intermediate (truncated) links
    isites = [sites[1]]     # store intermediate site1
    icombiners = ITensor[]      # store intermediate combiners
    ieigh_trunc = Tuple{Matrix{ComplexF64}, Diagonal{Float64, Vector{Float64}}, Matrix{ComplexF64}}[]    # store intermediate rdm and eigh_trunc results
    iψ12 = [ψ[1:2]]    # store ψ[1] and ψ[2]

    for j in 1:N-1
        ψj1, ψj2 = iψ12[j]
        site1 = isites[j]
        rhoj_ten = ψj1*envR[N-j]*dag(ψj1)'

        rhoj = Matrix{ComplexF64}(rhoj_ten, site1, site1')
        D, U, ϵ = eigh_trunc(rhoj, trunc=(trunc..., maxrank=maxranks[j]))
        push!(ieigh_trunc, (rhoj, D, U))
        push!(errs, ϵ)

        Ulinkind = Index(size(U)[2], "Link_trunc, l=$(j)")
        push!(ilinks, Ulinkind)
        Uten = ITensor(U, isites[j], Ulinkind)
        if j>1
            push!(vect, dag(icombiners[end])*Uten)
        else
            push!(vect, Uten)
        end    

        if j == N-1
            push!(vect, dag(Uten)*ψj1*ψj2)
        else
            c12 = combiner(Ulinkind, sites[j+1])
            push!(icombiners, c12)
            ψ1new = c12*(dag(Uten)*ψj1*ψj2)
            push!(iψ12, [ψ1new; ψ[j+2]])
            site1new = combinedind(c12)
            push!(isites, site1new)
        end
    end

    ψt, MPS_pullback = ChainRulesCore.rrule(MPS, vect)
    ψt.llim = N-1
    ψt.rlim = N+1

    function truncDM_pullback(Δψt)
        _, Δψt_vec = MPS_pullback(Δψt)  #vector to MPS
        # we call its elements [ΔB1, ΔB2, ..., ΔBN-1, ΔψN] as the last element is just ΔψN
        # every B is actually just U with split indices, that must be combined accordingly

        local Δψj_rhoj::ITensor, Δψj_ψjp1::ITensor
        local contrj::ITensor, contrjp1::ITensor
        Δψj_rhoj_list = ITensor[]
        Δψj_ψjp1_list::Vector{ITensor} = [Δψt_vec[N]]
        for j in N-1:-1:1
            ψj12 = iψ12[j]
            rhoj, D, U = ieigh_trunc[j]
            site1 = isites[j]
            Ulinkind = ilinks[j]
            ΔBj = Δψt_vec[j]
            if j>1
                combj = icombiners[j-1]
            end

            ΔU = Array{ComplexF64}(j>1 ? combj*ΔBj : ΔBj, site1, Ulinkind)     # contribution from MPS([U1, U2, U3, ψ4])
            # now we need to add the contribution from |ψjp1⟩=Uj|ψj⟩, that is the contribution of Δψjp1 to ΔUj
            # which is obtained by multiplying ψj[1] with contrj ≡ contract(ψj[2:end], Δψjp1).

            if j==N-1
                # for j=N-1 this is simply
                contrj = ψj12[2]*dag(Δψt_vec[N])
            else
                # contrj can be constructed recursively by splitting Δψjp1 into Δψjp1_rhojp1 and Δψjp1_ψjp2
                # the first one is only present when j<N-1, as there's no reduced density matrix computed at the last step
                # it can be computed by reusing the right environment defined in the forward pass
                contr_rhoj = ψj12[2]*dag(prime(Δψj_rhoj, links[j+1]))*envR[N-j-1]

                # the second one can be defined recursively in terms of contrjp1
                contr_ψjp1 = ψj12[2]*dag(Δψj_ψjp1)*contrjp1

                # sum the two as they have the same inds
                contrj = contr_rhoj + contr_ψjp1
            end
            ΔU += Array{ComplexF64}(ψj12[1]*contrj, site1, Ulinkind)

            Δrhoj = zero(rhoj)
            MatrixAlgebraKit.eigh_trunc_pullback!(Δrhoj, rhoj, (D, U), (ZeroTangent(), ΔU))     # pullback of eigh_trunc
    
            Δψj_rhoj = noprime(ITensor(Δrhoj+Δrhoj', site1', site1)*ψj12[1]) # contribution from rhoj = Tr_(j+1..N)(ψ)
            Δψj_ψjp1 = ITensor(U, site1, Ulinkind)    #contribution from |ψj+1⟩=Uj|ψj⟩

            if j>1
                # decombine previous tensors: this has to be done after the sum of MPS, since the sum treats them as states
                Δψj_rhoj *= dag(combj)
                Δψj_ψjp1 *= dag(combj)
            end

            push!(Δψj_rhoj_list, Δψj_rhoj)
            push!(Δψj_ψjp1_list, Δψj_ψjp1)

            contrjp1 = contrj
        end

        # Reconstruct the final adjoint as a finite state machine
        Δψ_vect = ITensor[]

        l1t, l1 = ilinks[1], links[1]
        ls1 = directsum(l1t, l1)
        links_sum = [ls1]
        T1 = directsum(ls1, Δψj_ψjp1_list[N] => l1t, Δψj_rhoj_list[N-1] => l1)
        push!(Δψ_vect, T1)

        for j in 2:N-1
            lsjm1 = links_sum[j-1]
            Tj_col1 = directsum(lsjm1, Δψj_ψjp1_list[N-j+1] => ilinks[j-1], ITensor(ComplexF64, links[j-1], sites[j], ilinks[j]) => links[j-1]) 
            Tj_col2 = directsum(lsjm1, Δψj_rhoj_list[N-j] => ilinks[j-1], ψ[j] => links[j-1])

            lsj = directsum(ilinks[j], links[j])
            push!(links_sum, lsj)
            Tj = directsum(lsj, Tj_col1 => ilinks[j], Tj_col2 => links[j]) 
            push!(Δψ_vect, Tj)
        end

        TN = directsum(links_sum[N-1], Δψj_ψjp1_list[1] => ilinks[N-1], ψ[N] => links[N-1])
        push!(Δψ_vect, TN)

        Δψ_MPS = MPS(Δψ_vect)
        
        return (NoTangent(), Δψ_MPS)
    end
    return ψt, truncDM_pullback
end


"Given two Vector{ITensor}, MPS or MPO A and B, multiply A*B tensor by tensor, combining the linkinds with the combiners in combs"
function _apply(A::Union{MPS,MPO,Vector{ITensor}}, B::Union{MPS,MPO,Vector{ITensor}}, combs::Vector{ITensor})
    @assert length(A) == length(B) == length(combs)+1
    N = length(A)
    AB1 = (A[1]*B[1])*combs[1]
    ABbulk = [combs[j-1]*(A[j]*B[j])*combs[j] for j in 2:N-1]
    ABN = (A[N]*B[N])*combs[N-1]
    AB = [AB1; ABbulk; ABN]
    return AB
end

"Replace linkinds with new ones"
function new_linkinds(ψ::T) where {T<:Union{MPS,MPO}}
    N = length(ψ)
    links = linkinds(ψ)
    newlinks = [Index(links[j].space, "Link, l=$(j)") for j in 1:N-1]

    ψ1 = replaceind(ψ[1], links[1], newlinks[1])
    ψB = [replaceinds(ψ[j], links[j-1:j], newlinks[j-1:j]) for j in 2:N-1]
    ψN = replaceind(ψ[N], links[N-1], newlinks[N-1])
    return T([ψ1; ψB; ψN])
end


"""
Note the 1/sqrt(2) factor in the Pauli Strings/Vector
"""
function get_pauli_mps(ψ::MPS; sites_pauli_mps::Union{Nothing, Vector{<:Index}} = nothing)
    d = 2
    N = length(ψ)
    sites = siteinds(ψ)
    links = linkinds(ψ)
    
    if isnothing(sites_pauli_mps)
        sites_pauli_mps = siteinds(d^2,N) 
    end
    links_pauli_mps = [Index(links[j].space^2, "Link, l=$(j)") for j in 1:N-1]
    combs = [combiner(link', link) for link in links]
    combs = [replaceind(combs[j], combinedind(combs[j]), links_pauli_mps[j]) for j in 1:N-1]
    
    Pψvec = [ITensor(Ps/sqrt(2), sites_pauli_mps[j], sites[j]', sites[j])*ψ[j] for j in 1:N]
    ψdagPψvec = _apply(dag(ψ)', Pψvec, combs)
    Pmps = MPS(ψdagPψvec)
    return Pmps
end

function ChainRulesCore.rrule(::typeof(get_pauli_mps), ψ::MPS; sites_pauli_mps::Union{Nothing, Vector{<:Index}} = nothing)
    d = 2
    N = length(ψ)
    sites = siteinds(ψ)
    links = linkinds(ψ)
    
    if isnothing(sites_pauli_mps)
        sites_pauli_mps = siteinds(d^2,N) 
    end
    links_pauli_mps = [Index(links[j].space^2, "Link, l=$(j)") for j in 1:N-1]
    combs = [combiner(link, link') for link in links]
    combs = [replaceind(combs[j], combinedind(combs[j]), links_pauli_mps[j]) for j in 1:N-1]
    
    Pψvec = [ITensor(Ps/sqrt(2), sites_pauli_mps[j], sites[j], sites[j]')*(ψ[j]') for j in 1:N]
    ψdagPψvec = _apply(dag(ψ), Pψvec, combs)
    Pmps = MPS(ψdagPψvec)

    function pullback_get_pauli_mps(ΔPmps)
        ΔPmps = new_linkinds(ΔPmps)  # we need to make sure linkinds are different from links
        linksΔPmps = linkinds(ΔPmps) 
        combsΔPmps = [combiner(linksΔPmps[j], links[j]') for j in 1:N-1]
        Δψvec = _apply(2*ΔPmps, Pψvec, combsΔPmps)  # should be real(ΔPmps), but that is probably already real since it works
        Δψ = MPS(Δψvec)
        # consider compressing Δψ since bond dimension is now χ_ΔPmps * χ_ψ
        return (NoTangent(), Δψ)
    end

    return Pmps, pullback_get_pauli_mps
end


function get_W(pauli_mps::MPS)
    N = length(pauli_mps)
    sites = siteinds(pauli_mps)
    links = linkinds(pauli_mps)
    newlinks = [Index(links[j].space, "Link, l=$(j)") for j in 1:N-1]

    primed_pmps1 = replaceinds(pauli_mps[1], (sites[1], links[1]), (sites[1]'', newlinks[1]))
    primed_pmpsB = [replaceinds(pauli_mps[j], (links[j-1], sites[j], links[j]), (newlinks[j-1], sites[j]'', newlinks[j])) for j in 2:N-1]
    primed_pmpsN = replaceinds(pauli_mps[N], (links[N-1], sites[N]), (newlinks[N-1], sites[N]''))
    primed_pmps = [primed_pmps1; primed_pmpsB; primed_pmpsN]
    Wvec = [delta(sites[j]'', sites[j], sites[j]')*primed_pmps[j] for j in 1:N]
    return MPO(Wvec)
end

function ChainRulesCore.rrule(::typeof(get_W), pauli_mps::MPS)
    N = length(pauli_mps)
    sites = siteinds(pauli_mps)
    links = linkinds(pauli_mps)
    newlinks = [Index(links[j].space, "Link, l=$(j)") for j in 1:N-1]

    primed_pmps1 = replaceinds(pauli_mps[1], (sites[1], links[1]), (sites[1]'', newlinks[1]))
    primed_pmpsB = [replaceinds(pauli_mps[j], (links[j-1], sites[j], links[j]), (newlinks[j-1], sites[j]'', newlinks[j])) for j in 2:N-1]
    primed_pmpsN = replaceinds(pauli_mps[N], (links[N-1], sites[N]), (newlinks[N-1], sites[N]''))
    primed_pmps = [primed_pmps1; primed_pmpsB; primed_pmpsN]
    Wvec = [delta(sites[j], sites[j]', sites[j]'')*primed_pmps[j] for j in 1:N]
    W = MPO(Wvec)

    function pullback_get_W(ΔW)
        Δpauli_mps_vec = [replaceind(delta(sites[j]'', sites[j]', sites[j])*ΔW[j], sites[j]'', sites[j]) for j in 1:N]
        Δpauli_mps = MPS(Δpauli_mps_vec)
        return (NoTangent(), Δpauli_mps)
    end

    return W, pullback_get_W
end


"Make sure that the linkinds are different before using it"
function applyAD(W::MPO, ψ::MPS)
    N = length(ψ)
    linksψ = linkinds(ψ)
    linksW = linkinds(W)
    combs = [combiner(linksW[j], linksψ[j]) for j in 1:N-1]

    return MPS(_apply(W, ψ, combs))
end

function ChainRulesCore.rrule(::typeof(applyAD), W::MPO, ψ::MPS)
    N = length(ψ)
    sites = siteinds(ψ)
    linksψ = linkinds(ψ)
    linksW = linkinds(W)
    combs = [combiner(linksW[j], linksψ[j]) for j in 1:N-1]
    
    Wψvect = _apply(W, ψ, combs)
    Wψ = MPS(Wψvect)
    Wψ = replaceprime(Wψ, 1 => 0; inds = sites')
    
    function applyAD_pullback(ΔWψ)
        ΔWψ = new_linkinds(ΔWψ)
        linksWψ = linkinds(ΔWψ)
        ΔWψ = replaceprime(ΔWψ, 0 => 1; inds = sites)

        combsΔW = [combiner(linksWψ[j], dag(linksψ[j])) for j in 1:N-1]
        ΔWvect = _apply(ΔWψ, dag(ψ), combsΔW)
        ΔW = MPO(ΔWvect)

        combsΔψ = [combiner(dag(linksW[j]), linksWψ[j]) for j in 1:N-1]
        Δψvect = _apply(dag(W), ΔWψ, combsΔψ)
        Δψ = MPS(Δψvect)
        # again consider compressing both ΔW and Δψ
        return (NoTangent(), ΔW, Δψ)
    end

    return Wψ, applyAD_pullback
end


function sre2(M::MPS; trunc=NamedTuple())
    Pψ = get_pauli_mps(M)
    Pψt = truncDM(Pψ; trunc=trunc)
    W = get_W(Pψt) 
    P2 = applyAD(W, Pψt)
    return -log(2,real(ITensorMPS.inner(P2,P2))) - length(M)
end