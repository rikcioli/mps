using ITensors, ITensorMPS
using LinearAlgebra
using ChainRulesCore
import MatrixAlgebraKit as MAK
using MatrixAlgebraKit
using Zygote

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


function ChainRulesCore.rrule(::Type{Array{T}}, x::ITensor, linds, rinds) where {T}
    y = Array{T}(x, linds, rinds)
    function Array_pullback(ȳ)
        # Convert gradient back to ITensor with the proper indices
        x̄ = ITensor(unthunk(ȳ), linds, rinds)
        return (NoTangent(), x̄, NoTangent(), NoTangent())
    end
    return y, Array_pullback
end

"Given an array of ITensors, contracts all of them in left-associative order
and computes a truncated SVD of the results with the left indices given by linds."
function SVDcontract(tensors::Vector{ITensor}, linds; move_ogc=:right, kargs...)
    M_ten = reduce(*, tensors)
    rinds = uniqueinds(M_ten, linds)

    M = Array{ComplexF64}(M_ten, linds, rinds)
    ldims = map(space, linds)
    rdims = map(space, rinds)
    M = reshape(M, prod(ldims), prod(rdims))
    U, S, Vdg = svd_trunc(M; kargs...)
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link_trunc, u") : Index(size(Vdg)[1], "Link_trunc, v")

    M1 = move_ogc==:right ? U : U*S
    M1 = reshape(M1, tuple(ldims..., space(bondind)))
    M2 = move_ogc==:right ? S*Vdg : Vdg
    M2 = reshape(M2, tuple(space(bondind), rdims...))

    M1_ten = ITensor(M1, linds..., bondind)
    M2_ten = ITensor(M2, bondind, rinds...)

    return (M1_ten, M2_ten)
end

function ChainRulesCore.rrule(::typeof(SVDcontract), tensors::Vector{ITensor}, linds; move_ogc=:right, kargs...)
    prods = cumprod(tensors)
    M_ten = prods[end]
    rinds = uniqueinds(M_ten, linds)

    M = Array{ComplexF64}(M_ten, linds, rinds)
    ldims = map(space, linds)
    rdims = map(space, rinds)
    M = reshape(M, prod(ldims), prod(rdims))
    U, S, Vdg = svd_trunc(M; kargs...)
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link_trunc, u") : Index(size(Vdg)[1], "Link_trunc, v")

    M1 = move_ogc==:right ? U : U*S
    M1 = reshape(M1, tuple(ldims..., space(bondind)))
    M2 = move_ogc==:right ? S*Vdg : Vdg
    M2 = reshape(M2, tuple(space(bondind), rdims...))

    M1_ten = ITensor(M1, linds..., bondind)
    M2_ten = ITensor(M2, bondind, rinds...)
    Mf = (M1_ten, M2_ten)

    function SVDcontract_pullback(ΔMf)
        ΔM1_ten, ΔM2_ten = ΔMf
        ΔM1 = Array{ComplexF64}(ΔM1_ten, linds..., bondind)
        ΔM2 = Array{ComplexF64}(ΔM2_ten, bondind, rinds...)

        ΔM1 = reshape(ΔM1, prod(ldims), space(bondind))
        ΔM2 = reshape(ΔM2, space(bondind), prod(rdims))

        local ΔU, ΔS, ΔVdg
        if move_ogc==:right
            ΔU = ΔM1 
            ΔS = ΔM2*Vdg'
            ΔVdg = S'*ΔM2
        else
            ΔU = ΔM1*S'
            ΔS = U'*ΔM1
            ΔVdg = ΔM2
        end

        ΔM = zero(M)
        MatrixAlgebraKit.svd_trunc_pullback!(ΔM, M, (U, S, Vdg), (ΔU, ΔS, ΔVdg))
        ΔM = reshape(ΔM, tuple(ldims..., rdims...))

        ΔM_ten = ITensor(ΔM, linds, rinds)

        n_tensors = length(tensors)
        # we compute the pullback of the intermediate product
        # each prod[j+1] = prod[j] * tensors[j+1] with prod[1] = tensors[1]
        # so Δprod[j] = Δprod[j+1] * tensors[j+1]'
        # and Δtensors[j+1] = prod[j]' * Δprod[j+1]
        revΔprods = cumprod(vcat([ΔM_ten], conj(tensors[end:-1:2])))
        Δtensorsp1 = [conj(prods[j])*revΔprods[end-j] for j in 1:n_tensors-1]
        Δtensors = vcat([revΔprods[end]], Δtensorsp1)
        
        return (NoTangent(), Δtensors, NoTangent())
    end

    return Mf, SVDcontract_pullback
end


function vecToITensor(V::Vector{<:AbstractArray}, ogc)
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor
    sites = siteinds("Qubit", N)
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    links = [Index(dimlinks[j], "Link, l=$j") for j in 1:N-1]

    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]
    allinds = [inds1; indsbulk; indsN]

    V1 = [V[1]]
    VB = [reshape(V[j], (dimlinks[j-1], 2, dimlinks[j])) for j in 2:N-1]
    VN = [V[N]]

    Vresh = [V1; VB; VN]
    Vtensors = [ITensor(Vresh[j], allinds[j]) for j in 1:N]
    return Vtensors
end

function ChainRulesCore.rrule(::typeof(vecToITensor), V::Vector{<:AbstractArray}, ogc::Int)
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor
    sites = siteinds("Qubit", N)
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    links = [Index(dimlinks[j], "Link, l=$j") for j in 1:N-1]

    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]
    allinds = [inds1; indsbulk; indsN]

    V1 = [V[1]]
    VB = [reshape(V[j], (dimlinks[j-1], 2, dimlinks[j])) for j in 2:N-1]
    VN = [V[N]]

    Vresh = [V1; VB; VN]
    Vtensors = [ITensor(Vresh[j], allinds[j]) for j in 1:N]

    function vecToITensor_pullback(ΔVtensors)
        ΔVresh = [Array{ComplexF64}(ΔVtensors[j], allinds[j]) for j in 1:N]

        ΔVL = [reshape(ΔVresh[j], (j==1 ? 2 : dimlinks[j-1]*2, dimlinks[j])) for j in 1:ogc-1]
        ΔVC = [ΔVresh[ogc]]
        ΔVR = [reshape(ΔVresh[j], (dimlinks[j-1], j==N ? 2 : 2*dimlinks[j])) for j in ogc+1:N]
        ΔV = [ΔVL; ΔVC; ΔVR]

        return (NoTangent(), ΔV, NoTangent())
    end

    return Vtensors, vecToITensor_pullback
end



function ITensors.norm(ψ::Vector{ITensor}, ogc::Int)
    center_ten = ψ[ogc]
    center = Array{ComplexF64}(center_ten, inds(center_ten))
    return norm(center)
end

function ChainRulesCore.rrule(::typeof(ITensors.norm), ψ::Vector{ITensor}, ogc::Int)
    center_ten = ψ[ogc]
    center = Array{ComplexF64}(center_ten, inds(center_ten))
    nrm = norm(center)

    function norm_pullback(ΔN)
        Δψ = [zero(A) for A in ψ]
        Δψ[ogc] = nrm < eps() ? zero(center_ten) : center_ten*(ΔN/norm(center))
        return (NoTangent(), Δψ, NoTangent())
    end

    return norm(center), norm_pullback
end

function norm2(ψ::Vector{ITensor}, ogc::Int)
    ogc_ten = ψ[ogc]
    return real(dot(ogc_ten, ogc_ten))
end

function ChainRulesCore.rrule(::typeof(norm2), ψ::Vector{ITensor}, ogc::Int)
    ogc_ten = ψ[ogc]
    nrm2 = real(dot(ogc_ten, ogc_ten))

    function norm_pullback(ΔN)
        Δψ = [zero(A) for A in ψ]
        Δψ[ogc] = 2*ΔN*ogc_ten
        return (NoTangent(), Δψ, NoTangent())
    end

    return nrm2, norm_pullback
end

# function to orthogonalize an mps to the right that is compatible with AD
function ITensorMPS.orthogonalize(ψ::Vector{<:ITensor}, b::Int; current_center = 1, trunc=NamedTuple())
    @assert b>=current_center

    if b==current_center
        return ψ
    end

    N = length(ψ)
    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))

    maxranks = [link.space for link in links]
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

    cog = current_center
    Ulinkinds = Index[]

    cache = Array{ITensor, 1}(undef, b-cog+1)
    WLten_new = ψ[cog]
    for j in cog:b-1
        WLten = WLten_new
        WRten = ψ[j+1]

        if j > 1
            combL = combiner(sites[j], Ulinkinds[j-1])
            cLind = combinedind(combL)
            WLten = combL*WLten
        else
            cLind = sites[1]
        end
        if j < N-1
            combR = combiner(sites[j+1], links[j+1])
            cRind = combinedind(combR)
            WRten = WRten*combR
        else
            cRind = sites[N]
        end
        
        W = Matrix(WLten*WRten, cLind, cRind)
        U, S, Vdg = svd_trunc(W; trunc=(trunc..., maxrank=maxranks[j]))

        Ulinkind = Index(size(U)[2], "Link, u")
        push!(Ulinkinds, Ulinkind)
        Uten = ITensor(U, cLind, Ulinkind)
        Rten = ITensor(S*Vdg, Ulinkind, cRind)
        
        Uten = j==1 ? Uten : Uten*dag(combL)
        Rten = j==N-1 ? Rten : Rten*dag(combR)

        cache[j-cog+1] = Uten
        WLten_new = Rten
        if j==b-1
            cache[end] = Rten
        end
    end
    ψfinal = vcat(ψ[1:cog-1], cache, ψ[b+1:end])

    return ψfinal
end


function ChainRulesCore.rrule(::typeof(ITensorMPS.orthogonalize), ψ::Vector{<:ITensor}, b::Int; current_center = 1, trunc=NamedTuple())
    @assert b>=current_center
    if b==current_center
        return ψ, Δψ -> (NoTangent(), Δψ, NoTangent())
    end

    N = length(ψ)
    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))

    maxranks = [link.space for link in links]
    if maximum(maxranks) == 1
        return ψ, Δψ -> (NoTangent(), Δψ, NoTangent())
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

    cog = current_center

    WLRlist::Vector{Vector{Matrix{ComplexF64}}} = []
    Wlist = Matrix{ComplexF64}[]
    USVlist = []
    Ulinkinds = Index[]
    combs::Vector{@NamedTuple{cL::ITensor, cR::ITensor}} = []
    combinds = []

    ψcache = Array{ITensor, 1}(undef, b-cog+1)
    WLten_new = ψ[cog]
    for j in cog:b-1
        WLten = WLten_new
        WRten = ψ[j+1]
        combs_j = ITensor[]
        if j > 1
            combL = combiner(sites[j], Ulinkinds[j-1])
            push!(combs_j, combL)
            cLind = combinedind(combL)
            WLten = combL*WLten
        else
            cLind = sites[1]
            push!(combs_j, ITensor(1))
        end
        if j < N-1
            combR = combiner(sites[j+1], links[j+1])
            push!(combs_j, combR)
            cRind = combinedind(combR)
            WRten = WRten*combR
        else
            cRind = sites[N]
            push!(combs_j, ITensor(1))
        end
        
        push!(combs, (cL=combs_j[1], cR=combs_j[2]))   # store combiners for pullback
        push!(combinds, (cLind=cLind, cRind=cRind))     # store combined inds
        # convert WLten and WRten into matrices and store them for pullback
        push!(WLRlist, [Array{ComplexF64}(WLten, cLind, links[j]), Array{ComplexF64}(WRten, links[j], cRind)])

        W = Matrix(WLten*WRten, cLind, cRind)
        push!(Wlist, W)
        U, S, Vdg = svd_trunc(W; trunc=(trunc..., maxrank=maxranks[j]))
        push!(USVlist, (U, S, Vdg))

        Ulinkind = Index(size(U)[2], "Link_trunc, l=$j")
        push!(Ulinkinds, Ulinkind)
        Uten = ITensor(U, cLind, Ulinkind)
        Rten = ITensor(S*Vdg, Ulinkind, cRind)
        
        Uten = j==1 ? Uten : Uten*dag(combL)
        Rten = j==N-1 ? Rten : Rten*dag(combR)

        ψcache[j-cog+1] = Uten
        WLten_new = Rten
        if j==b-1
            ψcache[b-cog+1] = Rten
        end
    end
    ψfinal = vcat(ψ[1:cog-1], ψcache, ψ[b+1:end])

    function orthogonalize_pullback(Δψfinal)

        Δψcache = Array{ITensor, 1}(undef, b-cog+1)
        ΔRten_new = Δψfinal[b]
        for j in b-1:-1:cog
            combs_j = combs[j]
            combinds_j = combinds[j]
            Ulinkind_j = Ulinkinds[j]
            WL, WR = WLRlist[j]
            W = Wlist[j]
            U, S, Vdg = USVlist[j]

            ΔUten = Δψfinal[j]
            ΔUten = j==1 ? ΔUten : ΔUten*combs_j[:cL]
            ΔRten = ΔRten_new
            ΔRten = j==N-1 ? ΔRten : ΔRten*combs_j[:cR]

            ΔU = Array{ComplexF64}(ΔUten, combinds_j[:cLind], Ulinkind_j)
            ΔR = Array{ComplexF64}(ΔRten, Ulinkind_j, combinds_j[:cRind])
            ΔS = ΔR*Vdg'; ΔVdg = S'*ΔR

            ΔW = zero(W)
            MatrixAlgebraKit.svd_trunc_pullback!(ΔW, W, (U, S, Vdg), (ΔU, ΔS, ΔVdg))
            ΔWL = ΔW*WR'; ΔWR = WL'*ΔW

            ΔWRten = ITensor(ΔWR, links[j], combinds_j[:cRind])
            ΔWRten = j==N-1 ? ΔWRten : ΔWRten*dag(combs_j[:cR])
            Δψcache[j-cog+2] = ΔWRten

            ΔWLten = ITensor(ΔWL, combinds_j[:cLind], links[j])
            ΔWLten = j==1 ? ΔWLten : ΔWLten*dag(combs_j[:cL]) 
            ΔRten_new = ΔWLten
            if j==cog
                Δψcache[1] = ΔWLten
            end
        end

        Δψ = vcat(Δψfinal[1:cog-1], Δψcache, Δψfinal[b+1:end])
        
        return (NoTangent(), Δψ, NoTangent())
    end
    return ψfinal, orthogonalize_pullback
end

function is_orthogonal(psi::Vector{ITensor}, ogc::Int)
    N = length(psi)
    links = linkinds(MPS(psi))
    for j in 1:ogc-1
        UdgU = (conj(psi[j])*delta(links[j], links[j]'))*psi[j]
        norm(UdgU - delta(links[j], links[j]')) > 1e-12 && return false
    end
    for j in ogc+1:N
        VVdg = (delta(links[j-1], links[j-1]')*conj(psi[j]))*psi[j]
        norm(VVdg - delta(links[j-1], links[j-1]')) > 1e-12 && return false
    end
    return true
end

# function to orthogonalize an mps that is compatible with AD
function move_center(ψ::Vector{<:ITensor}, b0::Int, b::Int; check_b0=true, trunc=NamedTuple())
    if check_b0
        !is_orthogonal(ψ, b0) && throw(ErrorException("ψ is NOT orthogonal at b0"))
    end

    cog = b0
    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    N = length(ψ)
    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))

    # Preparing the maxranks for svd trunc
    maxranks = [link.space for link in links]
    if maximum(maxranks) == 1
        return ψ
    end
    if haskey(trunc, :maxrank)
        kwarg_maxrank = trunc[:maxrank]
        # remove maxrank from trunc tuple
        trunc = (; filter(p -> first(p) != :maxrank, collect(pairs(trunc)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks = [min(kwarg_maxrank, maxranks[j]) for j in 1:N-1]
    end
    if !haskey(trunc, :atol)
        trunc = (trunc..., atol=eps())
    end

    
    cache = Array{ITensor, 1}(undef, abs(b-cog)+1)
    Rten_new = ψ[cog]
    local ψfinal
    if to_right
        Ulinkinds = Index[]
        for j in cog:b-1
            WLten = Rten_new
            WRten = ψ[j+1]

            linds = if j > cog
                (sites[j], Ulinkinds[j-cog])
            else
                cog==1 ? (sites[j],) : (sites[j], links[cog-1]) 
            end
            W1, W2 = SVDcontract([WLten, WRten], linds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
            push!(Ulinkinds, commonind(W1, W2))

            cache[j-cog+1] = W1
            Rten_new = W2
            if j==b-1
                cache[end] = W2
            end
        end
        ψfinal = vcat(ψ[1:cog-1], cache, ψ[b+1:end])
        
    else
        for j in cog-1:-1:b
            WLten = ψ[j]
            WRten = Rten_new

            linds = j > 1 ? (sites[j], links[j-1]) : (sites[j],)
            W1, W2 = SVDcontract([WLten, WRten], linds; move_ogc=:left, trunc=(trunc..., maxrank=maxranks[j]))

            Rten_new = W1
            cache[j-b+2] = W2
            if j==b
                cache[1] = W1
            end
        end
        ψfinal = vcat(ψ[1:b-1], cache, ψ[cog+1:end])
    end

    return ψfinal
end

function ChainRulesCore.rrule(::typeof(move_center), ψ::Vector{<:ITensor}, b0::Int, b::Int; check_b0=true, trunc=NamedTuple())
    if check_b0
        !is_orthogonal(ψ, b0) && throw(ErrorException("ψ is NOT orthogonal at b0"))
    end

    cog = b0
    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    N = length(ψ)
    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))

    # Preparing the maxranks for svd trunc
    maxranks = [link.space for link in links]
    if maximum(maxranks) == 1
        return ψ
    end
    if haskey(trunc, :maxrank)
        kwarg_maxrank = trunc[:maxrank]
        # remove maxrank from trunc tuple
        trunc = (; filter(p -> first(p) != :maxrank, collect(pairs(trunc)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks = [min(kwarg_maxrank, maxranks[j]) for j in 1:N-1]
    end
    if !haskey(trunc, :atol)
        trunc = (trunc..., atol=eps())
    end

    
    cache = Array{ITensor, 1}(undef, abs(b-cog)+1)
    backs = Function[]
    Rten_new = ψ[cog]
    local ψfinal
    if to_right
        Ulinkinds = Index[]
        for j in cog:b-1
            WLten = Rten_new
            WRten = ψ[j+1]

            linds = if j > cog
                (sites[j], Ulinkinds[j-cog])
            else
                cog==1 ? (sites[j],) : (sites[j], links[cog-1]) 
            end
            SVDcontract_j = (tensors, leftinds) -> SVDcontract(tensors, leftinds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
            (W1, W2), back_j = Zygote.pullback(SVDcontract_j, [WLten, WRten], linds)
            push!(backs, back_j)
            push!(Ulinkinds, commonind(W1, W2))

            cache[j-cog+1] = W1
            Rten_new = W2
            if j==b-1
                cache[end] = W2
            end
        end
        ψfinal = vcat(ψ[1:cog-1], cache, ψ[b+1:end])
        
    else
        for j in cog-1:-1:b
            WLten = ψ[j]
            WRten = Rten_new

            linds = j > 1 ? (sites[j], links[j-1]) : (sites[j],)
            SVDcontract_j = (tensors, leftinds) -> SVDcontract(tensors, leftinds; move_ogc=:left, trunc=(trunc..., maxrank=maxranks[j]))
            (W1, W2), back_j = Zygote.pullback(SVDcontract_j, [WLten, WRten], linds)
            push!(backs, back_j)

            Rten_new = W1
            cache[j-b+2] = W2
            if j==b
                cache[1] = W1
            end
        end
        ψfinal = vcat(ψ[1:b-1], cache, ψ[cog+1:end])
    end

    function move_center_pullback(Δψfinal)

        Δψcache = Array{ITensor, 1}(undef, abs(b-cog)+1)
        ΔR_new = Δψfinal[b]
        local Δψ
        if to_right
            for j in b-1:-1:cog
                ΔW1 = Δψfinal[j]
                ΔW2 = ΔR_new

                (ΔWL, ΔWR), _ = backs[j-cog+1]((ΔW1, ΔW2))  # start from the last

                Δψcache[j-cog+2] = ΔWR
                ΔR_new = ΔWL
                if j==cog
                    Δψcache[1] = ΔWL
                end
            end
            Δψ = vcat(Δψfinal[1:cog-1], Δψcache, Δψfinal[b+1:end])
        else
            for j in b:cog-1
                ΔW1 = ΔR_new
                ΔW2 = Δψfinal[j+1]

                (ΔWL, ΔWR), _ = backs[cog-j]((ΔW1, ΔW2)) # start from the last again, since pullbacks are appended

                Δψcache[j-b+1] = ΔWL
                ΔR_new = ΔWR
                if j==cog-1
                    Δψcache[end] = ΔWR
                end
            end
            Δψ = vcat(Δψfinal[1:b-1], Δψcache, Δψfinal[cog+1:end])
        end
        
        return (NoTangent(), Δψ, NoTangent(), NoTangent())
    end
    return ψfinal, move_center_pullback
end


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
        D, U, ϵ = eigh_trunc(rhoj, trunc=(trunc..., maxrank=maxranks[j]))
        push!(errs, ϵ)
        if j==1
            @show U
        end
        @show norm(rhoj - U*D*U')
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
    println("Beginning forward pass")
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
        println("Entering pullback")
        for j in N-1:-1:1
            @show j
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
            eigh_trunc_pullback!(Δrhoj, rhoj, (D, U), (ZeroTangent(), ΔU))     # pullback of eigh_trunc
    
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


function truncPauliMPS(ψ::MPS; truncS=NamedTuple(), truncA=NamedTuple())
    N = length(ψ)
    maxranks_S = linkdims(ψ)
    maxranks_A = maxranks_S
    if maximum([maxranks_S; maxranks_A]) == 1
        return ψ
    end

    if haskey(truncS, :maxrank)
        # remove maxrank from trunc tuple
        kwarg_maxrank = truncS[:maxrank]

        truncS = (; filter(p -> first(p) != :maxrank, collect(pairs(truncS)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks_S = [min(kwarg_maxrank, maxranks_S[j]) for j in 1:N-1]
    end
    if !haskey(truncS, :atol)
        truncS = (truncS..., atol=eps())
    end

    if haskey(truncA, :maxrank)
        kwarg_maxrank = truncA[:maxrank]
        truncA = (; filter(p -> first(p) != :maxrank, collect(pairs(truncA)))...)
        maxranks_A = [min(kwarg_maxrank, maxranks_A[j]) for j in 1:N-1]
    end
    if !haskey(truncA, :atol)
        truncA = (truncA..., atol=eps())
    end

    sites = siteinds(ψ)
    links = linkinds(ψ)

    envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    for j in N-1:-1:2
        push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    end

    vect = ITensor[]    # store tensors that make the final mps
    errsS = Float64[]    # store truncation errors for symmetric part
    errsA = Float64[]
    isites = [sites[1]]     # store intermediate site1
    iψ12 = [ψ[1:2]]
    
    icombiners = ITensor[]      # store intermediate combiners
    
    Symmetry = [1., 1., 1., -1.]
    isymm = [Symmetry]  # store intermediate swap symmetry
    for j in 1:N-1
        ψj1, ψj2 = iψ12[j]
        site1 = isites[j]
        rhoj_ten = ψj1*envR[N-j]*dag(ψj1)'
        Symmj = Diagonal(isymm[j])

        rhoj = Matrix{ComplexF64}(rhoj_ten, site1, site1')
        @show norm(rhoj*Symmj - Symmj*rhoj)
        ΠS = (I+Symmj)/2
        ΠA = (I-Symmj)/2

        rhoj_S = ΠS*rhoj*ΠS
        Ds, Us, ϵ = eigh_trunc(rhoj_S, trunc=(truncS..., maxrank=maxranks_S[j]))
        push!(errsS, ϵ)

        rhoj_A = ΠA*rhoj*ΠA
        Da, Ua, ϵ = eigh_trunc(rhoj_A, trunc=(truncA..., maxrank=maxranks_A[j]))
        push!(errsA, ϵ)

        #@show norm(rhoj - rhoj_S - rhoj_A)

        U = hcat(Us, Ua)

        Ulinkind = Index(size(U)[2], "Link_trunc, l=$(j)")
        Uten = ITensor(U, site1, Ulinkind)
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

            Symmj_ten = diag_itensor(isymm[j], site1'', site1)
            S_through = dag(Uten)''*Symmj_ten*Uten   # symmetry operator has to go through Udag
            Snew_ten = S_through*diag_itensor(Symmetry, sites[j+1]'', sites[j+1])    # kronecker product of the pull through and the new site one
            Snew = Array{Float64}(real(c12''*Snew_ten*c12), site1new'', site1new)   # should be already real

            @show diag(Snew)
            @show norm(Snew-diagm(diag(Snew)))
            push!(isymm, diag(Snew))
        end
    end

    ψt = MPS(vect)
    ψt.llim = N-1
    ψt.rlim = N+1

    return ψt
end


function ChainRulesCore.rrule(::typeof(truncPauliMPS), ψ::MPS; truncS=NamedTuple(), truncA=NamedTuple())
    N = length(ψ)
    maxranks_S = linkdims(ψ)
    maxranks_A = maxranks_S
    if maximum([maxranks_S; maxranks_A]) == 1
        return ψ
    end

    if haskey(truncS, :maxrank)
        # remove maxrank from trunc tuple
        kwarg_maxrank = truncS[:maxrank]

        truncS = (; filter(p -> first(p) != :maxrank, collect(pairs(truncS)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks_S = [min(kwarg_maxrank, maxranks_S[j]) for j in 1:N-1]
    end
    if !haskey(truncS, :atol)
        truncS = (truncS..., atol=eps())
    end

    if haskey(truncA, :maxrank)
        kwarg_maxrank = truncA[:maxrank]
        truncA = (; filter(p -> first(p) != :maxrank, collect(pairs(truncA)))...)
        maxranks_A = [min(kwarg_maxrank, maxranks_A[j]) for j in 1:N-1]
    end
    if !haskey(truncA, :atol)
        truncA = (truncA..., atol=eps())
    end

    sites = siteinds(ψ)
    links = linkinds(ψ)

    envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    for j in N-1:-1:2
      push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    end

    vect = ITensor[]    # store tensors that make the final mps
    errsS = Float64[]    # store truncation errors
    errsA = Float64[]
    ilinks = Index[]    # store intermediate (truncated) links
    isites = [sites[1]]     # store intermediate site1
    icombiners = ITensor[]      # store intermediate combiners
    ieigh_trunc = Tuple{Matrix{ComplexF64}, 
                        Matrix{ComplexF64},
                        Matrix{ComplexF64},
                        Matrix{ComplexF64},
                        Diagonal{Float64, Vector{Float64}}, 
                        Matrix{ComplexF64},
                        Diagonal{Float64, Vector{Float64}},
                        Matrix{ComplexF64}}[]    # store intermediate rdm and eigh_trunc results
    iψ12 = [ψ[1:2]]    # store ψ[1] and ψ[2]


    Symmetry = [1., 1., 1., -1.]
    isymm = [Symmetry]  # store intermediate swap symmetry
    for j in 1:N-1
        ψj1, ψj2 = iψ12[j]
        site1 = isites[j]
        rhoj_ten = ψj1*envR[N-j]*dag(ψj1)'

        rhoj = Matrix{ComplexF64}(rhoj_ten, site1, site1')
        Symmj = Diagonal(isymm[j])
        ΠS = (I+Symmj)/2
        ΠA = (I-Symmj)/2

        rhoj_S = ΠS*rhoj*ΠS
        Ds, Us, ϵ = eigh_trunc(rhoj_S, trunc=(truncS..., maxrank=maxranks_S[j]))
        push!(errsS, ϵ)

        rhoj_A = ΠA*rhoj*ΠA
        Da, Ua, ϵ = eigh_trunc(rhoj_A, trunc=(truncA..., maxrank=maxranks_A[j]))
        push!(errsA, ϵ)

        U = hcat(Us, Ua)
        push!(ieigh_trunc, (rhoj, rhoj_S, rhoj_A, U, Ds, Us, Da, Ua))

        Ulinkind = Index(size(U)[2], "Link_trunc, l=$(j)")
        push!(ilinks, Ulinkind)
        Uten = ITensor(U, site1, Ulinkind)
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

            Symmj_ten = diag_itensor(isymm[j], site1'', site1)
            S_through = dag(Uten)''*Symmj_ten*Uten   # symmetry operator has to go through Udag
            Snew_ten = S_through*diag_itensor(Symmetry, sites[j+1]'', sites[j+1])    # kronecker product of the pull through and the new site one
            Snew = Array{Float64}(real(c12''*Snew_ten*c12), site1new'', site1new)   # should be already real

            @show norm(Snew-diagm(diag(Snew)))
            push!(isymm, diag(Snew))
        end
    end

    ψt, MPS_pullback = ChainRulesCore.rrule(MPS, vect)
    ψt.llim = N-1
    ψt.rlim = N+1

    function truncPauliMPS_pullback(Δψt)
        _, Δψt_vec = MPS_pullback(Δψt)  #vector to MPS
        # we call its elements [ΔB1, ΔB2, ..., ΔBN-1, ΔψN] as the last element is just ΔψN
        # every B is actually just U with split indices, that must be combined accordingly

        local Δψj_rhoj::ITensor, Δψj_ψjp1::ITensor
        local contrj::ITensor, contrjp1::ITensor
        local ΔSnew_ten::ITensor
        Δψj_rhoj_list = ITensor[]
        Δψj_ψjp1_list::Vector{ITensor} = [Δψt_vec[N]]
        for j in N-1:-1:1
            ψj12 = iψ12[j]
            rhoj, rhoj_S, rhoj_A, U, Ds, Us, Da, Ua = ieigh_trunc[j]
            site1 = isites[j]
            Ulinkind = ilinks[j]
            ΔBj = Δψt_vec[j]
            if j>1
                combj = icombiners[j-1]
            end
            Symmj = Diagonal(isymm[j])
            ΠS = (I+Symmj)/2
            ΠA = (I-Symmj)/2

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

            if j<N-1    # then ΔU recieves an additional contribution from the symmetry op at step j+1 (called Snew in the forward pass)
                ΔS_through_ten = ΔSnew_ten*diag_itensor(Symmetry, sites[j+1]'', sites[j+1])
                # this product is needed to invert the kron product used to extend the symmetry in the forward pass
                ΔS_through = Array{ComplexF64}(ΔS_through_ten, Ulinkind'', Ulinkind)
                ΔU += Symmj*U*2*real(ΔS_through)
            end

            ΔUs = @view U[:, 1:size(Us,2)]
            ΔUa = @view U[:, 1:size(Ua,2)]

            Δrhoj_S = zero(rhoj_S)
            eigh_trunc_pullback!(Δrhoj_S, rhoj_S, (Ds, Us), (ZeroTangent(), ΔUs))     # pullback of eigh_trunc
            
            Δrhoj_A = zero(rhoj_A)
            eigh_trunc_pullback!(Δrhoj_A, rhoj_A, (Da, Ua), (ZeroTangent(), ΔUa))

            # recombine the two into Δrhoj (in ITensor form)
            Δrhoj_ten = ITensor(ΠS*Δrhoj_S*ΠS + ΠA*Δrhoj_A*ΠA, site1, site1')

            if j>1
                # adjoint of symmetry (needed for ΔU_j-1)
                ΔSnew = real((Δrhoj_S*ΠS - Δrhoj_A*ΠA)*rhoj)
                ΔSnew_ten = dag(combj)''*ITensor(ΔSnew, site1'', site1)*dag(combj)
            end

            # adjoints of ψj
            Δψj_rhoj = noprime((Δrhoj_ten+dag(Δrhoj_ten))*ψj12[1]) # contribution from rhoj = Tr_(j+1..N)(ψ)
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
    return ψt, truncPauliMPS_pullback
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