include("helpers.jl")
using ITensors, ITensorMPS
using LinearAlgebra
using ChainRulesCore
using MatrixAlgebraKit
using Zygote


### HELPER TAPE STRUCT FOR SVDCONTRACT, THE SUBROUTINE WHICH IS PRESENT
### IN ALL THE FUNCTIONS THAT REQUIRE TRUNCATION
struct SVDcontractTape
    # ITensor data
    move_ogc::Symbol
    tensors::Vector{ITensor}
    prods::Vector{ITensor}
    cL::ITensor
    cR::ITensor
    bondind::Index
    # SVD data
    M::Matrix{ComplexF64}
    U::Matrix{ComplexF64}
    S::Diagonal{Float64, Vector{Float64}}
    Vdg::Matrix{ComplexF64}
end
   

"Contracts all the tensors in a Vector{ITensor} in order
and computes a truncated SVD of the result with the left indices specified by linds.
Also returns truncation error."
function SVDcontract(tensors::Vector{<:ITensor}, linds::Vector{<:Index}; move_ogc=:right, kwargs...)
    n = length(tensors)
    tensors = copy(tensors)
    prods = Array{ITensor, 1}(undef, n)
    prods[1] = tensors[1]
    for j in 2:n
        prods[j] = prods[j-1]*tensors[j]
    end
    M_ten = prods[end]
    rinds = uniqueinds(M_ten, linds)

    cL = combiner(linds); cR = combiner(rinds);
    cLind = combinedind(cL); cRind = combinedind(cR)
    M_ten = (cL*M_ten)*cR

    M = Matrix{ComplexF64}(M_ten, cLind, cRind)
    # ldims = map(space, linds)
    # rdims = map(space, rinds)
    # M = reshape(M, prod(ldims), prod(rdims))
    U, S, Vdg, err = svd_trunc(M; kwargs...)
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link, u") : Index(size(Vdg)[1], "Link, v")

    ML = move_ogc==:right ? U : U*S
    #ML = reshape(ML, tuple(ldims..., space(bondind)))
    MR = move_ogc==:right ? S*Vdg : Vdg
    #MR = reshape(MR, tuple(space(bondind), rdims...))

    ML_ten = ITensor(ML, cLind, bondind)
    ML_ten *= dag(cL)
    MR_ten = ITensor(MR, bondind, cRind)
    MR_ten *= dag(cR)

    out = ((ML_ten, MR_ten), err)
    tape = SVDcontractTape(move_ogc, tensors, prods, cL, cR, bondind, M, U, S, Vdg)

    return out, tape
end


function SVDcontract_pullback(ΔMf, tape::SVDcontractTape)
    ΔML_ten, ΔMR_ten = ΔMf
    (; move_ogc, tensors, prods, cL, cR, bondind, M, U, S, Vdg) = tape

    ΔML_ten *= cL
    ΔMR_ten *= cR
    cLind = combinedind(cL); cRind = combinedind(cR);

    ΔML = Matrix{ComplexF64}(ΔML_ten, cLind, bondind)
    ΔMR = Matrix{ComplexF64}(ΔMR_ten, bondind, cRind)

    #ΔM1 = reshape(ΔM1, prod(ldims), space(bondind))
    #ΔM2 = reshape(ΔM2, space(bondind), prod(rdims))

    local ΔU, ΔS, ΔVdg
    if move_ogc==:right
        ΔU = ΔML 
        ΔS = ΔMR*Vdg'
        ΔVdg = S'*ΔMR
    else
        ΔU = ΔML*S'
        ΔS = U'*ΔML
        ΔVdg = ΔMR
    end

    ΔM = zero(M)
    MatrixAlgebraKit.svd_trunc_pullback!(ΔM, M, (U, S, Vdg), (ΔU, ΔS, ΔVdg))
    #ΔM = reshape(ΔM, tuple(ldims..., rdims...))

    ΔM_ten = ITensor(ΔM, cLind, cRind)
    ΔM_ten *= dag(cR)
    ΔM_ten *= dag(cL)

    n = length(tensors)
    Δtensors = Array{ITensor, 1}(undef, n)
    # we compute the pullback of the intermediate product
    # each prod[j+1] = prod[j] * tensors[j+1] with prod[1] = tensors[1]
    # so Δtensors[j+1] = prod[j]' * Δprod[j+1]
    # and Δprod[j] = Δprod[j+1] * tensors[j+1]'
    Δprodjp1 = ΔM_ten
    for j in n-1:-1:1
        Δtensors[j+1] = dag(prods[j]) * Δprodjp1
        Δprodjp1 = Δprodjp1 * dag(tensors[j+1])
    end
    Δtensors[1] = Δprodjp1

    return Δtensors
end

function ChainRulesCore.rrule(::typeof(SVDcontract), tensors::Vector{<:ITensor}, linds::Vector{<:Index}; kwargs...)
    out, tape = SVDcontract(tensors, linds; kwargs...)

    function SVDcontract_pullback_Zygote(Δall)
        Δout, Δtape = Δall
        ΔMf, Δerr = Δout
        Δtensors = SVDcontract_pullback(ΔMf, tape)
        return (NoTangent(), Δtensors, NoTangent())
    end
    
    return (out, tape), SVDcontract_pullback_Zygote
end



##### EXTENSION OF ITensorMPS FUNCTIONS

### VECTOR OF ISOMETRIES TO MPS

"Convert vector of isometries with orthogonality center ogc into an MPS."
function ITensorMPS.MPS(V::Vector{<:AbstractArray}, ogc; check_og=true, sites=nothing)
    check_og && check_orthogonal(V, ogc)
    
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor

    if isnothing(sites)
        sites = siteinds("Qubit", N)
    end
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    @assert size(V[1]) == (2, dimlinks[1])
    for j in 2:ogc-1
        @assert size(V[j]) == (dimlinks[j-1]*2, dimlinks[j])
    end
    if 1<ogc<N
        @assert size(V[ogc]) == (dimlinks[ogc-1], 2, dimlinks[ogc])
    end
    for j in ogc+1:N-1
        @assert size(V[j]) == (dimlinks[j-1], 2*dimlinks[j])
    end
    @assert size(V[N]) == (dimlinks[N-1], 2)

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
    Vmps = MPS(Vtensors)
    set_ortho_lims!(Vmps, ogc:ogc)
    return Vmps
end

"Convert vector of isometries with orthogonality center ogc into an MPS.
The pullback treats the adjoint of the MPS as if it was a vector of isometries."
function ChainRulesCore.rrule(::typeof(ITensorMPS.MPS), V::Vector{<:AbstractArray}, ogc::Int; check_og=true, sites=nothing)
    check_og && check_orthogonal(V, ogc)
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor
    if isnothing(sites)
        sites = siteinds("Qubit", N)
    end
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    @assert size(V[1]) == (2, dimlinks[1])
    for j in 2:ogc-1
        @assert size(V[j]) == (dimlinks[j-1]*2, dimlinks[j])
    end
    if 1<ogc<N
        @assert size(V[ogc]) == (dimlinks[ogc-1], 2, dimlinks[ogc])
    end
    for j in ogc+1:N-1
        @assert size(V[j]) == (dimlinks[j-1], 2*dimlinks[j])
    end
    @assert size(V[N]) == (dimlinks[N-1], 2)

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
    Vmps = MPS(Vtensors)
    set_ortho_lims!(Vmps, ogc:ogc)

    function MPS_pullback(ΔVmps)
        ΔVresh = [Array{ComplexF64}(ΔVmps[j], allinds[j]) for j in 1:N]

        ΔVL = [reshape(ΔVresh[j], (j==1 ? 2 : dimlinks[j-1]*2, dimlinks[j])) for j in 1:ogc-1]
        ΔVC = [ΔVresh[ogc]]
        ΔVR = [reshape(ΔVresh[j], (dimlinks[j-1], j==N ? 2 : 2*dimlinks[j])) for j in ogc+1:N]
        ΔV = [ΔVL; ΔVC; ΔVR]

        return (NoTangent(), ΔV, NoTangent())
    end

    return Vmps, MPS_pullback
end

"Convert vector of isometries with orthogonality center ogc into a vector of ITensors."
function toITensors(V::Vector{<:AbstractArray}, ogc; kwargs...)
    Vmps = MPS(V, ogc; kwargs...)
    return Vmps[:]
end

"Convert vector of isometries with orthogonality center ogc into a vector of ITensors."
function ChainRulesCore.rrule(::typeof(toITensors), V::Vector{<:AbstractArray}, ogc::Int; kwargs...)
    Vmps, back = Zygote.pullback((vec, oc) -> MPS(vec, oc; kwargs...), V, ogc)
    Vtensors = Vmps[:]

    function toITensors_pullback(ΔVmps)
        ΔV = back(ΔVmps)[1]
        @assert isa(ΔV, Vector{<:AbstractArray})

        return (NoTangent(), ΔV, NoTangent())
    end

    return Vtensors, toITensors_pullback
end


### MPS TO VECTOR OF ISOMETRIES

function Base.vec(psi::MPS)
    return psi.data
end

function ChainRulesCore.rrule(::typeof(Base.vec), ψ::MPS)
    return vec(ψ), Δvec -> (NoTangent(), MPS(Δvec))
end


### MPS TO MPO

function ITensorMPS.MPO(ψ::MPS)
    isog = isortho(ψ)
    ψ = replace_linkinds(ψ)
    isog && @assert isortho(ψ)
    
    N = length(ψ)
    sites = siteinds(ψ)
    ψmpo = [delta(sites[j], sites[j]', sites[j]'')*ψ[j] for j in 1:N]
    ψmpo = replaceprime.(ψmpo, 1 => 0, 2 => 1)
    ψf = MPO(ψmpo)
    set_ortho_lims!(ψf, ortho_lims(ψ))
    return ψf
end

function ChainRulesCore.rrule(::typeof(ITensorMPS.MPO), ψ::MPS)
    isog = isortho(ψ)
    oldlinks = linkinds(ψ)
    ψ = replace_linkinds(ψ)
    isog && @assert isortho(ψ)

    N = length(ψ)
    sites = siteinds(ψ)
    ψmpo = [delta(sites[j], sites[j]', sites[j]'')*ψ[j] for j in 1:N]
    ψmpo = replaceprime.(ψmpo, 1 => 0, 2 => 1)
    ψf = MPO(ψmpo)
    set_ortho_lims!(ψf, ortho_lims(ψ))

    function MPO_pullback(Δψmpo)
        ogc = orthocenter(Δψmpo)
        Δψ_vec = [dag(delta(sites[j], sites[j]', sites[j]''))*Δψmpo[j] for j in 1:N]
        Δψ_vec = replaceprime.(Δψ_vec, 2 => 0)
        Δψ = MPS(Δψ_vec)
        set_ortho_lims!(Δψ, ogc:ogc)

        Δψ = replace_linkinds(Δψ; newlinks=oldlinks)
        return (NoTangent(), Δψ)
    end
    return ψf, MPO_pullback
end


### OVERLAP BETWEEN TWO MPS, TREATING THEM AS VECTORS OF ITENSORS

"Compute the scalar product of two MPS. Same as ITensorMPS.inner, but different pullback."
function sproduct(ψ::T, ϕ::T) where {T<:Union{MPS, Vector{<:ITensor}}}
    N = length(ψ)
    @assert length(ϕ)==N
    @assert siteinds(ψ)==siteinds(ϕ)
    ψ_bra = dag.(replace_linkinds(ψ))
    c1 = ψ_bra[1] * ϕ[1]
    for j in 2:N
        c1 *= ψ_bra[j]
        c1 *= ϕ[j]
    end
    return only(Array{ComplexF64}(c1))
end

"Compute the scalar product of two MPS. Same as ITensorMPS.inner, but different pullback."
function ChainRulesCore.rrule(::typeof(sproduct), ψ::T, ϕ::T) where {T<:Union{MPS, Vector{<:ITensor}}}
    N = length(ψ)
    @assert length(ϕ)==N
    @assert siteinds(ψ)==siteinds(ϕ)
    oldlinks = linkinds(ψ)
    ψ_bra = dag.(replace_linkinds(ψ))

    envL = [ψ_bra[1] * ϕ[1]]
    envR = [ψ_bra[N] * ϕ[N]]
    for j in 2:N-1
        push!(envL, envL[j-1] * ψ_bra[j] * ϕ[j])
        push!(envR, envR[j-1] * ψ_bra[N+1-j] * ϕ[N+1-j])
    end
    C = only(Array{ComplexF64}(envL[end] * ψ_bra[N] *ϕ[N]))

    function sproduct_pullback(ΔC)
        Δϕ_vec = ITensor[]
        Δψ_vec = ITensor[]

        push!(Δϕ_vec, ΔC*dag(ψ_bra[1] * envR[N-1]))
        push!(Δψ_vec, dag(ΔC)*ϕ[1]*envR[N-1])
        for j in 2:N-1
            push!(Δϕ_vec, ΔC*dag(envL[j-1] * ψ_bra[j] * envR[N-j]))
            push!(Δψ_vec, dag(ΔC)*(envL[j-1] * ϕ[j] * envR[N-j]))
        end
        push!(Δϕ_vec, ΔC*dag(envL[N-1] * ψ_bra[N]))
        push!(Δψ_vec, dag(ΔC)*(envL[N-1] * ϕ[N]))
        Δψ_vec = replace_linkinds(Δψ_vec; newlinks = oldlinks)
            
        Δψ = T(Δψ_vec)
        Δϕ = T(Δϕ_vec)
        if T <: MPS
            set_ortho_lims!(Δψ, ortho_lims(ψ))
            set_ortho_lims!(Δϕ, ortho_lims(ϕ))
        end

        return (NoTangent(), Δψ, Δϕ)
    end

    return C, sproduct_pullback
end


### PRODUCT OF AN MPO WITH AN MPS, TENSOR BY TENSOR

function product(W::MPO, ψ::MPS)
    N = length(ψ)

    Wlinks = linkinds(W)
    ψlinks = linkinds(ψ)
    sites = siteinds(ψ)
    
    Wψ_vec = (W[:]) .* (ψ[:])
    
    combs = combiner.(Wlinks, ψlinks)
    combinds = combinedind.(combs)
    Wψlinks = [Index(space(ψlinks[j])*space(Wlinks[j]), "Link,l=$j") for j in 1:N-1]
    replaceind!.(combs, combinds, Wψlinks)

    Wψ_vec[1] *= combs[1]
    for j in 2:N-1
        Wψ_vec[j] *= combs[j-1]
        Wψ_vec[j] *= combs[j]
    end
    Wψ_vec[N] *= combs[N-1]

    Wψ_vec = [replaceprime(Wψ_vec[j], 1 => 0, inds=sites[j]') for j in 1:N]
    Wψ = MPS(Wψ_vec)
    reset_ortho_lims!(Wψ)

    return Wψ
end

function ChainRulesCore.rrule(::typeof(product), W::MPO, ψ::MPS)
    N = length(ψ)

    Wlinks = linkinds(W)
    ψlinks = linkinds(ψ)
    sites = siteinds(ψ)
    
    Wψ_vec = (W[:]) .* (ψ[:])
    
    combs = combiner.(Wlinks, ψlinks)
    combinds = combinedind.(combs)
    Wψlinks = [Index(space(ψlinks[j])*space(Wlinks[j]), "Link,l=$j") for j in 1:N-1]


    Wψ_vec[1] = replaceind(combs[1]*Wψ_vec[1], combinds[1], Wψlinks[1])
    for j in 2:N-1
        Wψ_vec[j] = replaceinds(combs[j-1]*Wψ_vec[j]*combs[j], combinds[j-1:j], Wψlinks[j-1:j])
    end
    Wψ_vec[N] = replaceind(combs[N-1]*Wψ_vec[N], combinds[N-1], Wψlinks[N-1])
    Wψ_vec = [replaceprime(Wψ_vec[j], 1 => 0, inds=sites[j]') for j in 1:N]
    Wψ = MPS(Wψ_vec)
    reset_ortho_lims!(Wψ)
    
    function product_pullback(ΔWψ)

        ΔWψ = [replaceprime(ΔWψ[j], 0 => 1; inds = sites[j]) for j in 1:N]
        ΔWψ[1] = replaceind(ΔWψ[1], Wψlinks[1], combinds[1])*dag(combs[1])
        for j in 2:N-1
            ΔWψ[j] = dag(combs[j-1])*replaceinds(ΔWψ[j], Wψlinks[j-1:j], combinds[j-1:j])*dag(combs[j])
        end
        ΔWψ[N] = replaceind(ΔWψ[N], Wψlinks[N-1], combinds[N-1])*dag(combs[N-1])

        Δψ_vec = dag.(W)[:] .* ΔWψ
        Δψ = MPS(Δψ_vec)
        set_ortho_lims!(Δψ, ortho_lims(ψ))

        ΔW_vec = ΔWψ .* dag.(ψ)[:]
        ΔW = MPO(ΔW_vec)
        set_ortho_lims!(ΔW, ortho_lims(W))
        
        return (NoTangent(), ΔW, Δψ)
    end

    return Wψ, product_pullback
end




##### THE FOLLOWING FUNCTIONS EXTEND STANDARD ITensorMPS METHODS TO WORK WITH Zygote
##### BY TREATING THE MPS AS VECTORS OF ITENSORS

### ORTHOGONALIZE WITH TRUNCATION

function move_center(ψ::T, b::Int; trunc=NamedTuple(), post_factorize_callback=identity) where {T<:Union{MPS, MPO}}
    N = length(ψ)
    cog = only(ortho_lims(ψ)) #current orthogonality center
    @assert 1 <= cog <= N
    @assert 1 <= b <= N

    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ))
    
    sites = siteinds(ψ)
    links = linkinds(ψ)
    cache = Array{ITensor, 1}(undef, abs(b-cog)+1)
    errs = Float64[]
    Rten_new = ψ[cog]
    local ψf_vec
    if to_right
        Ulinkinds = Index[]
        for j in cog:b-1
            WLten = Rten_new
            WRten = ψ[j+1]

            linds = if j > cog
                [sites[j]; Ulinkinds[j-cog]]
            else
                cog==1 ? [sites[j];] : [sites[j]; links[cog-1]]
            end
            tensors = [WLten, WRten]
            ((W1, W2), err), _ = SVDcontract(tensors, linds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
            push!(Ulinkinds, commonind(W1, W2))
            push!(errs, err)

            cache[j-cog+1] = W1
            Rten_new = W2
            if j==b-1
                cache[end] = W2
            end
        end
        ψf_vec = vcat(ψ[1:cog-1], cache, ψ[b+1:end])
        
    else
        for j in cog-1:-1:b
            WLten = ψ[j]
            WRten = Rten_new

            linds = j > 1 ? [sites[j]; links[j-1]] : [sites[j];]
            tensors = [WLten, WRten]
            ((W1, W2), err), _ = SVDcontract(tensors, linds; move_ogc=:left, trunc=(trunc..., maxrank=maxranks[j]))
            push!(errs, err)

            Rten_new = W1
            cache[j-b+2] = W2
            if j==b
                cache[1] = W1
            end
        end
        ψf_vec = vcat(ψ[1:b-1], cache, ψ[cog+1:end])
    end

    post_factorize_callback(errs)
    ψf = T(ψf_vec)
    set_ortho_lims!(ψf, b:b)
    return ψf
end

function ChainRulesCore.rrule(::typeof(move_center), ψ::T, b::Int; trunc=NamedTuple(), post_factorize_callback=identity) where {T<:Union{MPS, MPO}}
    N = length(ψ)
    cog = only(ortho_lims(ψ)) #current orthogonality center
    @assert 1 <= cog <= N
    @assert 1 <= b <= N

    if b==cog
        return ψ, Δψf -> (NoTangent(), Δψf, NoTangent())
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ))

    sites = siteinds(ψ)
    links = linkinds(ψ)
    cache = Array{ITensor, 1}(undef, abs(b-cog)+1)
    tapes = SVDcontractTape[]  # store intermediate data for SVDcontract
    Rten_new = ψ[cog]
    errs = Float64[]
    local ψf_vec
    if to_right
        Ulinkinds = Index[]
        for j in cog:b-1
            WLten = Rten_new
            WRten = ψ[j+1]

            linds = if j > cog
                [sites[j]; Ulinkinds[j-cog]]
            else
                cog==1 ? [sites[j];] : [sites[j]; links[cog-1]] 
            end
            tensors = [WLten, WRten]

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
            push!(tapes, tape_j)
            push!(Ulinkinds, commonind(W1, W2))
            push!(errs, err)

            cache[j-cog+1] = W1
            Rten_new = W2
            if j==b-1
                cache[end] = W2
            end
        end
        ψf_vec = vcat(ψ[1:cog-1], cache, ψ[b+1:end])
        
    else
        for j in cog-1:-1:b
            WLten = ψ[j]
            WRten = Rten_new

            linds = j > 1 ? [sites[j]; links[j-1]] : [sites[j];]
            tensors = [WLten, WRten]

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; move_ogc=:left, trunc=(trunc..., maxrank=maxranks[j]))
            push!(tapes, tape_j)
            push!(errs, err)

            Rten_new = W1
            cache[j-b+2] = W2
            if j==b
                cache[1] = W1
            end
        end
        ψf_vec = vcat(ψ[1:b-1], cache, ψ[cog+1:end])
    end

    post_factorize_callback(errs)
    ψf = T(ψf_vec)
    set_ortho_lims!(ψf, b:b)

    function move_center_pullback(Δψf)
        check_orthogonal(Δψf, b)
        Δψcache = Array{ITensor, 1}(undef, abs(b-cog)+1)
        ΔR_new = Δψf[b]
        local Δψ_vec
        if to_right
            for j in b-1:-1:cog
                ΔW1 = Δψf[j]
                ΔW2 = ΔR_new
                ΔMf = (ΔW1, ΔW2)

                (ΔWL, ΔWR) = SVDcontract_pullback(ΔMf, tapes[j-cog+1])  # start from the last

                Δψcache[j-cog+2] = ΔWR
                ΔR_new = ΔWL
                if j==cog
                    Δψcache[1] = ΔWL
                end
            end
            Δψ_vec = vcat(Δψf[1:cog-1], Δψcache, Δψf[b+1:end])
        else
            for j in b:cog-1
                ΔW1 = ΔR_new
                ΔW2 = Δψf[j+1]
                ΔMf = (ΔW1, ΔW2)
                
                (ΔWL, ΔWR) = SVDcontract_pullback(ΔMf, tapes[cog-j])
                # start from the last again, since pullbacks are appended

                Δψcache[j-b+1] = ΔWL
                ΔR_new = ΔWR
                if j==cog-1
                    Δψcache[end] = ΔWR
                end
            end
            Δψ_vec = vcat(Δψf[1:b-1], Δψcache, Δψf[cog+1:end])
        end

        Δψ = T(Δψ_vec)
        set_ortho_lims!(Δψ, cog:cog)
        return (NoTangent(), Δψ, NoTangent(), NoTangent())
    end

    return ψf, move_center_pullback
end


### APPLY VECTOR OF UNITARIES IN A BRICKWORK PATTERN, SWEEPING LEFT TO RIGHT AND BACK

function apply_brickwork(Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, trunc=NamedTuple(), post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1

    ψ = move_center(ψ, 1)
    check_orthogonal(ψ, 1) # check that ψ is orthogonal at first site before applying unitaries

    to_right = true  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    
    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    local lastj
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)
        
        for j in jvals
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+to_right)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? [sites[j]'; commonind(ψfinal[j-1], WLten)] : [sites[j]';]
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? [sites[j]; commonind(ψfinal[j-1], WLten)] : [sites[j];]
                tensors = [WLten, WRten]
            end

            ((W1, W2), err), _ = SVDcontract(tensors, linds; move_ogc=(to_right ? :right : :left), trunc=(trunc..., maxrank=maxranks[j]))
            push!(errs, err)
            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if (j==N-1 && to_right) || (j==1 && !to_right)
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right      
            end
        end
    end
    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)
    return ψfinal
end

function ChainRulesCore.rrule(::typeof(apply_brickwork), Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, trunc=NamedTuple(), post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1

    ψ, move_center_back = Zygote.pullback(move_center, ψ, 1)
    check_orthogonal(ψ, 1) # check that ψ is orthogonal at first site before applying unitaries

    to_right = true  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    
    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    tapes = SVDcontractTape[]  # store intermediate data for the pullback
    local lastj      # store last j reached
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)

        for j in jvals
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+to_right)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? [sites[j]'; commonind(ψfinal[j-1], WLten)] : [sites[j]';]
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? [sites[j]; commonind(ψfinal[j-1], WLten)] : [sites[j];]
                tensors = [WLten, WRten]
            end

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; 
                                                move_ogc = (to_right ? :right : :left), 
                                                trunc = (trunc..., maxrank=maxranks[j]))
            push!(tapes, tape_j)
            push!(errs, err)
            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if (j==N-1 && to_right) || (j==1 && !to_right)
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right      
            end
        end
    end
    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)

    function apply_brickwork_pullback(Δψfinal)
        check_orthogonal(Δψfinal, final_ogc)

        Δψ = copy(Δψfinal)
        ΔUarray = [zeros(ComplexF64, size(U)) for U in Uarray]
        i = nU; pb_n = length(backs)
        while i>=1
            jvals = to_right ? (lastj:-1:1) : (lastj:N-1) 
            for j in jvals
                ΔW1 = Δψ[j]
                ΔW2 = Δψ[j+1]

                if iseven(j+shift+to_right)
                    ΔW1 = prime(ΔW1, tags="Site")
                    ΔW2 = prime(ΔW2, tags="Site")
                    ΔMf = (ΔW1, ΔW2)

                    (ΔWLten, ΔWRten, ΔUten) = SVDcontract_pullback(ΔMf, tapes[pb_n])  # start from the last

                    ΔU = Array{ComplexF64}(ΔUten, sites[j]', sites[j+1]', sites[j], sites[j+1])
                    ΔUarray[i] = reshape(ΔU, (4,4))
                    i -= 1
                else
                    ΔMf = (ΔW1, ΔW2)
                    (ΔWLten, ΔWRten) = SVDcontract_pullback(ΔMf, tapes[pb_n])
                end

                pb_n -= 1
                Δψ[j] = ΔWLten
                Δψ[j+1] = ΔWRten
            end
            lastj = to_right ? 1 : N-1
            to_right = !to_right
        end
        set_ortho_lims!(Δψ, 1:1)
        Δψ, _ = move_center_back(Δψ)
        
        return (NoTangent(), ΔUarray, Δψ)
    end
    return ψfinal, apply_brickwork_pullback
end


### MULTIPLY MPO WITH MPS WITH ZIP-UP ALGORITHM

function zipup(W::MPO, ψ::MPS; trunc=NamedTuple(), post_factorize_callback = identity)
    N = length(ψ)
    ψ = move_center(ψ, N)
    W = move_center(W, N)


    trunc, maxranks = adapt_truncarg(trunc, linkdims(W).*linkdims(ψ))
    Wlinks = linkinds(W)
    ψlinks = linkinds(ψ)

    errs = Float64[]    # store truncation errors
    Wψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final mps
    local Lten::ITensor
    for j in N:-1:2    
        linds = [Wlinks[j-1]; ψlinks[j-1]]
        tensors = j<N ? [Lten, ψ[j], W[j]] : [ψ[j], W[j]]

        ((Lten, V), err), _ = SVDcontract(tensors, linds; move_ogc=:left, trunc=(trunc..., maxrank=maxranks[j-1]))
        push!(errs, err)
        Wψ_vec[j] = V
    end
    Wψ_vec[1] = Lten*ψ[1]*W[1]

    reverse!(errs)
    post_factorize_callback(errs)
    Wψ = MPS(Wψ_vec)
    set_ortho_lims!(Wψ, 1:1)
    return Wψ
end

function ChainRulesCore.rrule(::typeof(zipup), W::MPO, ψ::MPS; trunc=NamedTuple(), post_factorize_callback = identity)
    N = length(ψ)
    ψ, move_center_back_ψ = Zygote.pullback(move_center, ψ, N)
    W, move_center_back_W = Zygote.pullback(move_center, W, N)

    trunc, maxranks = adapt_truncarg(trunc, linkdims(W).*linkdims(ψ))

    Wlinks = linkinds(W)
    ψlinks = linkinds(ψ)

    errs = Float64[]    # store truncation errors
    Wψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final mps
    tapes = SVDcontractTape[]
    local Lten::ITensor
    for j in N:-1:2    
        linds = [Wlinks[j-1]; ψlinks[j-1]]
        tensors = j<N ? [Lten, ψ[j], W[j]] : [ψ[j], W[j]]

        ((Lten, V), err), tape_j = SVDcontract(tensors, linds;
                                            move_ogc=:left, 
                                            trunc=(trunc..., maxrank=maxranks[j-1]))
        push!(errs, err)
        push!(tapes, tape_j)
        Wψ_vec[j] = V
    end
    Wψ_vec[1] = Lten*ψ[1]*W[1]

    post_factorize_callback(errs)
    Wψ = MPS(Wψ_vec)
    set_ortho_lims!(Wψ, 1:1)

    function zipup_pullback(ΔWψ)
        check_orthogonal(ΔWψ, 1)
        Δψ_vec = Array{ITensor, 1}(undef, N)
        ΔW_vec = Array{ITensor, 1}(undef, N)

        ΔLten = ΔWψ[1]*dag(ψ[1])*dag(W[1])
        Δψ_vec[1] = dag(Lten)*ΔWψ[1]*dag(W[1])
        ΔW_vec[1] = dag(Lten)*dag(ψ[1])*ΔWψ[1]

        for j in 2:N    
            ΔV = ΔWψ[j]
            ΔMf = (ΔLten, ΔV)
            if j<N
                (ΔLten, Δψj, ΔWj) = SVDcontract_pullback(ΔMf, tapes[N-j+1])
            else
                (Δψj, ΔWj) = SVDcontract_pullback(ΔMf, tapes[N-j+1])
            end
            Δψ_vec[j] = Δψj
            ΔW_vec[j] = ΔWj
        end

        Δψ = MPS(Δψ_vec)
        set_ortho_lims!(Δψ, N:N)
        ΔW = MPO(ΔW_vec)
        set_ortho_lims!(ΔW, N:N)

        Δψ, _ = move_center_back_ψ(Δψ)
        ΔW, _ = move_center_back_W(ΔW)
        return (NoTangent(), ΔW, Δψ)
    end

    return Wψ, zipup_pullback
end


##### FUNCTIONS FOR MAGIC EXTRACTION

### EXTRACT PAULI MPS FROM MPS

function get_pauli_mps(ψ::MPS; trunc=NamedTuple(), sites=nothing, post_factorize_callback = identity)
    N = length(ψ)
    ψ = move_center(ψ, 1)
    ψbra = bra(ψ)

    trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ).^2)

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites_pauli_mps = isnothing(sites) ? siteinds(d^2, N) : sites 
    sites = siteinds(ψ)
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], bra(sites[1]), sites[1])

    errs = Float64[]    # store truncation errors
    Pψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    Bp = ψbra[1]*Pten1*ψ[1]
    for j in 1:N-1    
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], bra(sites[j+1]), sites[j+1])

        linds = j>1 ? [sites_pauli_mps[j]; commonind(Pψ_vec[j-1], Bp)] : [sites_pauli_mps[j];] 
        tensors = [Bp, ψ[j+1], ψbra[j+1], Pten]

        ((Up, Rp), err), _ = SVDcontract(tensors, linds; 
                                        move_ogc=:right, 
                                        trunc=(trunc..., maxrank=maxranks[j]))
        push!(errs, err)
        
        Pψ_vec[j] = Up
        Bp = Rp
        if j==N-1
            Pψ_vec[N] = Rp
        end
    end

    post_factorize_callback(errs)
    Pψ = MPS(Pψ_vec)
    set_ortho_lims!(Pψ, N:N)

    return Pψ
end

function ChainRulesCore.rrule(::typeof(get_pauli_mps), ψ::MPS; trunc=NamedTuple(), sites=nothing, post_factorize_callback = identity)
    N = length(ψ)
    ψ, move_center_back = Zygote.pullback(move_center, ψ, 1)
    ψbra = bra(ψ)

    trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ).^2)

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites_pauli_mps = isnothing(sites) ? siteinds(d^2, N) : sites 
    sites = siteinds(ψ)
    
    errs = Float64[]    # store truncation errors
    Pψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], bra(sites[1]), sites[1])
    
    Bp = ψbra[1]*Pten1*ψ[1]
    tapes = SVDcontractTape[]
    for j in 1:N-1    
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], bra(sites[j+1]), sites[j+1])

        linds = j>1 ? [sites_pauli_mps[j]; commonind(Pψ_vec[j-1], Bp)] : [sites_pauli_mps[j];] 
        @show j, linds
        tensors = [Bp, ψ[j+1], ψbra[j+1], Pten]
        
        ((Up, Rp), err), tape_j = SVDcontract(tensors, linds; 
                                        move_ogc=:right, 
                                        trunc=(trunc..., maxrank=maxranks[j]))
        push!(errs, err)
        push!(tapes, tape_j)

        Pψ_vec[j] = Up
        Bp = Rp
        if j==N-1
            Pψ_vec[N] = Rp
        end
    end

    post_factorize_callback(errs)
    Pψ = MPS(Pψ_vec)
    set_ortho_lims!(Pψ, N:N)

    function get_pauli_mps_pullback(ΔPψ)
        @assert isa(ΔPψ, MPS)
        check_orthogonal(ΔPψ, N)

        Δψ_vec = Array{ITensor, 1}(undef, N)
        ΔRp = ΔPψ[N]
        for j in N-1:-1:1
            ΔUp = ΔPψ[j]
            ΔMf = (ΔUp, ΔRp)

            (ΔBp, Δψ_jp1, Δψbra_jp1, _) = SVDcontract_pullback(ΔMf, tapes[j])
            
            #Δψ_vec[j+1] = Δψ_jp1 + removetags(dag(Δψbra_jp1), "bra")
            @show Δψ_jp1
            @show Δψbra_jp1
            Δψ_vec[j+1] = 2*Δψ_jp1
            ΔRp = ΔBp
        end

        #Δψ_vec[1] = dag(ψbra[1])*dag(Pten1)*ΔPψ[1] + removetags(dag(ΔPψ[1])*Pten1*ψ[1], "bra")
        Δψ_vec[1] = 2*dag(ψbra[1])*dag(Pten1)*ΔRp
        Δψ = MPS(Δψ_vec)
        set_ortho_lims!(Δψ, 1:1)

        (Δψ,) = move_center_back(Δψ)

        return (NoTangent(), Δψ, NoTangent())
    end
    
    return Pψ, get_pauli_mps_pullback
end


### EXACT MAGIC COMPUTATION

function FWHT!(v)
    n = length(v); h = 1
    @inbounds while h < n
        for i in 0:2h:n-1, j in 0:h-1
            x = v[i+j+1]; y = v[i+j+h+1]
            v[i+j+1] = x + y; v[i+j+h+1] = x - y
        end; h *= 2
    end
end

function fastEDMagic(psi)
    d = length(psi)
    Nkvec = zeros(Float64, d)
    Threads.@threads for k in 0:d-1
        A = [conj(psi[xor(x,k)+1]) * psi[x+1] for x in 0:d-1]
        FWHT!(A)
        Nkvec[k+1] = sum(abs2.(A).^2)
    end
    return -log2(sum(Nkvec)/d)
end

function ChainRulesCore.rrule(::typeof(fastEDMagic), psi)
    d = length(psi)

    As = [Vector{ComplexF64}(undef, d) for _ in 1:d]
    Threads.@threads for k in 0:d-1
        A = [conj(psi[xor(x,k)+1]) * psi[x+1] for x in 0:d-1]
        FWHT!(A)
        As[k+1] = A    
    end

    T::Float64 = sum(sum(abs2.(A).^2) for A in As)
    m2 = -log2(T/d)

    function fastEDMagic_pullback(Δm2)
        ΔT = -Δm2/(T*log(2))

        nthreads = Threads.nthreads()
        Δψ_threads = [zeros(ComplexF64, d) for _ in 1:nthreads]
        ΔS_threads = [Vector{ComplexF64}(undef, d) for _ in 1:nthreads]

        Threads.@threads for k in 0:d-1
            tid = Threads.threadid()
            ΔS = ΔS_threads[tid]
            Δψt = Δψ_threads[tid]
            Ak = As[k+1]

            # in-place ΔS computation
            @inbounds @simd for x in 0:d-1
                a2 = abs2(Ak[x+1])
                ΔS[x+1] = 4*a2*Ak[x+1]
            end
            FWHT!(ΔS)

            @inbounds for x in 0:d-1
                xk = xor(x,k)+1     # correct permutation
                Δψt[x+1] += (conj(ΔS[xk])+ΔS[x+1])*psi[xk]
            end
        end

        Δψ = zeros(ComplexF64, d)
        @inbounds for t in 1:nthreads
            Δψ .+= Δψ_threads[t]
        end
        Δψ .*= ΔT
        return (NoTangent(), Δψ)
    end

    return m2, fastEDMagic_pullback
end



### RRULES THAT ONLY INVOLVE SINGLE TENSOR MANIPULATION, NOT NEEDED CURRENTLY


# function ChainRulesCore.rrule(::Type{Array{T}}, x::ITensor) where {T}
#     y = Array{T}(x)
#     function Array_pullback(ȳ)
#         # Convert gradient back to ITensor with the proper indices
#         x̄ = ITensor(unthunk(ȳ))
#         return (NoTangent(), x̄)
#     end
#     return y, Array_pullback
# end
# 
# function ChainRulesCore.rrule(::Type{Array{T}}, x::ITensor, linds, rinds) where {T}
#     y = Array{T}(x, linds, rinds)
#     function Array_pullback(ȳ)
#         # Convert gradient back to ITensor with the proper indices
#         x̄ = ITensor(unthunk(ȳ), linds, rinds)
#         return (NoTangent(), x̄, NoTangent(), NoTangent())
#     end
#     return y, Array_pullback
# end


# function ITensors.norm(ψ::Vector{ITensor}, ogc::Int; check_og=true)
#     check_og && check_orthogonal(ψ, ogc)
#     center_ten = ψ[ogc]
#     center = Array{ComplexF64}(center_ten, inds(center_ten))
#     return norm(center)
# end
# 
# function ChainRulesCore.rrule(::typeof(ITensors.norm), ψ::Vector{ITensor}, ogc::Int; check_og=true)
#     check_og && check_orthogonal(ψ, ogc)
#     center_ten = ψ[ogc]
#     center = Array{ComplexF64}(center_ten, inds(center_ten))
#     nrm = norm(center)
# 
#     function norm_pullback(ΔN)
#         Δψ = [zero(A) for A in ψ]
#         Δψ[ogc] = nrm < eps() ? zero(center_ten) : center_ten*(ΔN/norm(center))
#         return (NoTangent(), Δψ, NoTangent())
#     end
# 
#     return norm(center), norm_pullback
# end
# 
# function norm2(ψ::Vector{ITensor}, ogc::Int; check_og=true)
#     check_og && check_orthogonal(ψ, ogc)
#     ogc_ten = ψ[ogc]
#     return real(dot(ogc_ten, ogc_ten))
# end
# 
# function ChainRulesCore.rrule(::typeof(norm2), ψ::Vector{ITensor}, ogc::Int; check_og=true)
#     check_og && check_orthogonal(ψ, ogc)
#     ogc_ten = ψ[ogc]
#     nrm2 = real(dot(ogc_ten, ogc_ten))
# 
#     function norm_pullback(ΔN)
#         Δψ = [zero(A) for A in ψ]
#         Δψ[ogc] = 2*ΔN*ogc_ten
#         return (NoTangent(), Δψ, NoTangent())
#     end
# 
#     return nrm2, norm_pullback
# end