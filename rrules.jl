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

bra(T) = addtags(T,"bra")
bralinks(T) = addtags(T, "bra", tags="Link") 
ITensorMPS.siteinds(ψ::Vector{ITensor}) = siteinds(MPS(ψ))
ITensorMPS.linkinds(ψ::Vector{ITensor}) = linkinds(MPS(ψ))

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

function is_orthogonal(psi::Vector{<:AbstractArray}, ogc::Int)
    N = length(psi)
    assert_lengths = fill(2, N)
    assert_lengths[ogc] = 3
    (length.(size.(psi)) != assert_lengths) && return false
    for j in 1:ogc-1
        U = psi[j]
        norm(U'*U - I) > 1e-12 && return false
    end
    for j in ogc+1:N
        V = psi[j]
        norm(V*V' - I) > 1e-12 && return false
    end
    return true
end

function check_orthogonal(psi::AbstractVector, ogc::Int)
    !is_orthogonal(psi, ogc) && throw(ErrorException("ψ is NOT orthogonal at specified orthogonality center=$(ogc)"))
    return true
end


function inner(ψ::Vector{ITensor}, ϕ::Vector{ITensor})
    N = length(ψ)
    @assert length(ϕ)==N
    @assert siteinds(ψ)==siteinds(ϕ)
    ψ_bra = @. dag(bralinks(ψ))
    c1 = ψ_bra[1] * ϕ[1]
    for j in 2:N
        c1 *= ψ_bra[j]
        c1 *= ϕ[j]
    end
    return real(c1)[1]
end

# We write it in the form that is already in the tangent space since it's easier
function ChainRulesCore.rrule(::typeof(inner), ψ::Vector{ITensor}, ϕ::Vector{ITensor})
    N = length(ψ)
    @assert length(ϕ)==N
    @assert siteinds(ψ)==siteinds(ϕ)
    ψ_bra = @. dag(bralinks(ψ))
    envL = [ψ_bra[1] * ϕ[1]]
    envR = [ψ_bra[N] * ϕ[N]]
    for j in 2:N-1
        push!(envL, envL[j-1] * ψ_bra[j] * ϕ[j])
        push!(envR, envR[j-1] * ψ_bra[N+1-j] * ϕ[N+1-j])
    end
    C = Array{ComplexF64}(envL[end] * ψ_bra[N] *ϕ[N])[]

    function inner_pullback(ΔC)
        Δϕ = ITensor[]
        Δψ = ITensor[]

        push!(Δϕ, ΔC*conj(ψ_bra[1] * envR[N-1]))
        push!(Δψ, conj(ΔC)*ϕ[1]*envR[N-1])
        for j in 2:N-1
            push!(Δϕ, ΔC*conj(envL[j-1] * ψ_bra[j] * envR[N-j]))
            push!(Δψ, conj(ΔC)*(envL[j-1] * ϕ[j] * envR[N-j]))
        end
        push!(Δϕ, ΔC*conj(envL[N-1] * ψ_bra[N]))
        push!(Δψ, conj(ΔC)*(envL[N-1] * ϕ[N]))
        Δψ = map(T -> removetags(T, "bra"), Δψ)

        return (NoTangent(), Δψ, Δϕ)
    end

    return C, inner_pullback

end



function ChainRulesCore.rrule(::Type{Array{T}}, x::ITensor, linds, rinds) where {T}
    y = Array{T}(x, linds, rinds)
    function Array_pullback(ȳ)
        # Convert gradient back to ITensor with the proper indices
        x̄ = ITensor(unthunk(ȳ), linds, rinds)
        return (NoTangent(), x̄, NoTangent(), NoTangent())
    end
    return y, Array_pullback
end

"Given an array of ITensors, contracts all of them in the chosen order
and computes a truncated SVD of the results with the left indices given by linds."
function SVDcontract(tensors::Vector{ITensor}, linds; move_ogc=:right, kargs...)
    M_ten = tensors[1]
    for ten in tensors[2:end]
        M_ten *= ten
    end
    rinds = uniqueinds(M_ten, linds)

    M = Array{ComplexF64}(M_ten, linds, rinds)
    ldims = map(space, linds)
    rdims = map(space, rinds)
    M = reshape(M, prod(ldims), prod(rdims))
    U, S, Vdg = svd_trunc(M; kargs...)
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link, u") : Index(size(Vdg)[1], "Link, v")

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
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link, u") : Index(size(Vdg)[1], "Link, v")

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


function vecToITensor(V::Vector{<:AbstractArray}, ogc; check_og=true, sites=nothing)
    check_og && check_orthogonal(V, ogc)
    
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor

    sites = isnothing(sites) ? siteinds("Qubit", N) : sites
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
    return Vtensors
end

function ChainRulesCore.rrule(::typeof(vecToITensor), V::Vector{<:AbstractArray}, ogc::Int; check_og=true, sites=nothing)
    check_og && check_orthogonal(V, ogc)
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor
    sites = isnothing(sites) ? siteinds("Qubit", N) : sites
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



function ITensors.norm(ψ::Vector{ITensor}, ogc::Int; check_og=true)
    check_og && check_orthogonal(ψ, ogc)
    center_ten = ψ[ogc]
    center = Array{ComplexF64}(center_ten, inds(center_ten))
    return norm(center)
end

function ChainRulesCore.rrule(::typeof(ITensors.norm), ψ::Vector{ITensor}, ogc::Int; check_og=true)
    check_og && check_orthogonal(ψ, ogc)
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

function norm2(ψ::Vector{ITensor}, ogc::Int; check_og=true)
    check_og && check_orthogonal(ψ, ogc)
    ogc_ten = ψ[ogc]
    return real(dot(ogc_ten, ogc_ten))
end

function ChainRulesCore.rrule(::typeof(norm2), ψ::Vector{ITensor}, ogc::Int; check_og=true)
    check_og && check_orthogonal(ψ, ogc)
    ogc_ten = ψ[ogc]
    nrm2 = real(dot(ogc_ten, ogc_ten))

    function norm_pullback(ΔN)
        Δψ = [zero(A) for A in ψ]
        Δψ[ogc] = 2*ΔN*ogc_ten
        return (NoTangent(), Δψ, NoTangent())
    end

    return nrm2, norm_pullback
end

"Helper function to modify the maxrank argument in svd_trunc. 
Removes the :maxrank key in trunc if present, and returns an array maxranks such that
maxranks[j] = min(trunc(:maxrank), dims[j])"
function adapt_truncarg(trunc::NamedTuple, dims::Vector{<:Int})
    maxranks = dims
    if haskey(trunc, :maxrank)
        kwarg_maxrank = trunc[:maxrank]
        # remove maxrank from trunc tuple
        trunc = (; filter(p -> first(p) != :maxrank, collect(pairs(trunc)))...)
        # choose for maxranks the minimum between input one and linkdims
        maxranks = [min(kwarg_maxrank, maxranks[j]) for j in 1:N-1]
    end
    if !haskey(trunc, :atol)    # adds an eps() tolerance if not present
        trunc = (trunc..., atol=eps())
    end
    return trunc, maxranks
end

# function to orthogonalize an mps that is compatible with AD
function move_center(ψ::Vector{<:ITensor}, b0::Int, b::Int; check_og=true, trunc=NamedTuple())
    N = length(ψ)
    @assert 1 <= b0 <= N
    @assert 1 <= b <= N
    check_og && check_orthogonal(ψ, b0) # check that ψ is orthogonal at b0

    cog = b0
    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, linkdims(MPS(ψ)))
    
    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))
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

function ChainRulesCore.rrule(::typeof(move_center), ψ::Vector{<:ITensor}, b0::Int, b::Int; check_og=true, trunc=NamedTuple())
    N = length(ψ)
    @assert 1 <= b0 <= N
    @assert 1 <= b <= N
    check_og && check_orthogonal(ψ, b0) # check that ψ is orthogonal at b0

    cog = b0
    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, linkdims(MPS(ψ)))

    sites = siteinds(MPS(ψ))
    links = linkinds(MPS(ψ))
    cache = Array{ITensor, 1}(undef, abs(b-cog)+1)
    backs = Function[]  # store intermediate pullbacks
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


function apply(Uarray::Vector{<:AbstractMatrix}, ψ::Vector{ITensor}; check_og=false, shift=0, trunc=NamedTuple())
    N = length(ψ)
    @assert shift==0 || shift==1
    check_og && check_orthogonal(ψ, 1) # check that ψ is orthogonal at first site before applying unitaries

    to_right = true  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    
    sites = siteinds(MPS(ψ))
    ψfinal = copy(ψ)
    i = 1; nU = length(Uarray)
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)
        
        for j in jvals
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+to_right)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? (sites[j]', commonind(ψfinal[j-1], WLten)) : (sites[j]',)
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? (sites[j], commonind(ψfinal[j-1], WLten)) : (sites[j],)
                tensors = [WLten, WRten]
            end

            W1, W2 = SVDcontract(tensors, linds; move_ogc=(to_right ? :right : :left), trunc=(trunc..., maxrank=maxranks[j]))
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

    return ψfinal
end


function ChainRulesCore.rrule(::typeof(apply), Uarray::Vector{<:AbstractMatrix}, ψ::Vector{ITensor}; check_og=false, shift=0, trunc=NamedTuple())
    N = length(ψ)
    @assert shift==0 || shift==1
    check_og && check_orthogonal(ψ, 1) # check that ψ is orthogonal at first site before applying unitaries


    to_right = true  # left-to-right mode

    # Preparing the maxranks for svd trunc
    trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    
    sites = siteinds(MPS(ψ))
    ψfinal = copy(ψ)
    i = 1; nU = length(Uarray)
    backs = Function[]  # store intermediate pullbacks
    local lastj, prev_left_linkind      # store last j reached and left linkind of previous step
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)

        for j in jvals
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+to_right)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? (sites[j]', commonind(ψfinal[j-1], WLten)) : (sites[j]',)
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? (sites[j], commonind(ψfinal[j-1], WLten)) : (sites[j],)
                tensors = [WLten, WRten]
            end

            SVDcontract_j = (tens, leftinds) -> SVDcontract(tens, leftinds; move_ogc=(to_right ? :right : :left), trunc=(trunc..., maxrank=maxranks[j]))
            (W1, W2), back_j = Zygote.pullback(SVDcontract_j, tensors, linds)
            push!(backs, back_j)
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

    function apply_pullback(Δψfinal)

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
                    (ΔWLten, ΔWRten, ΔUten), _ = backs[pb_n]((ΔW1, ΔW2))  # start from the last
                    ΔU = Array{ComplexF64}(ΔUten, sites[j]', sites[j+1]', sites[j], sites[j+1])
                    ΔUarray[i] = reshape(ΔU, (4,4))
                    i -= 1
                else
                    (ΔWLten, ΔWRten), _ = backs[pb_n]((ΔW1, ΔW2))
                end
                pb_n -= 1
                Δψ[j] = ΔWLten
                Δψ[j+1] = ΔWRten
            end
            lastj = to_right ? 1 : N-1
            to_right = !to_right
        end
        
        return (NoTangent(), ΔUarray, Δψ)
    end
    return ψfinal, apply_pullback
end


function get_Ps()
    Id = (1.0+0.0im)*[1.0 0; 0 1.0]
    X = (1.0+0.0im)*[0 1.0 ; 1.0 0]
    Z = (1.0+0.0im)*[1.0 0; 0 -1.0]
    Y = [0.0 -1.0im; 1.0im 0.0]

    Psm = zeros(ComplexF64,4,2,2)
    Psm[1,:,:] = Id
    Psm[2,:,:] = X
    Psm[3,:,:] = Z
    Psm[4,:,:] = Y
    return Psm
end


function pauliMPS(ψ::Vector{ITensor}, ogc::Int; trunc=NamedTuple(), check_og=true, post_factorize_callback = identity)
    N = length(ψ)
    ψ = move_center(ψ, ogc, 1; check_og)

    ψbra = @. dag(bra(ψ))

    trunc, maxranks = adapt_truncarg(trunc, linkdims(MPS(ψ)).^2)

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites = siteinds(MPS(ψ))
    sites_pauli_mps = siteinds(d^2,N) 
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], bra(sites[1]), sites[1])

    errs = Float64[]    # store truncation errors
    Pψ = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    Pψ[1] = ψ[1]*Pten1*ψbra[1]

    for j in 1:N-1    
        Bp = Pψ[j]
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], bra(sites[j+1]), sites[j+1])

        linds = j>1 ? (sites_pauli_mps[j], commonind(Pψ[j-1], Bp)) : (sites_pauli_mps[j],) 

        Up, Rp = SVDcontract([Bp, ψ[j+1], ψbra[j+1], Pten], linds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
        Pψ[j] = Up
        Pψ[j+1] = Rp
    end

    post_factorize_callback(errs)

    return Pψ
end



function ChainRulesCore.rrule(::typeof(pauliMPS), ψ::Vector{ITensor}, ogc; trunc=NamedTuple(), check_og=true, post_factorize_callback = identity)
    N = length(ψ)
    og1 = (psi, b0) -> move_center(psi, b0, 1; check_og)
    ψ, og_back = Zygote.pullback(og1, ψ, ogc)

    ψbra = @. dag(bra(ψ))

    trunc, maxranks = adapt_truncarg(trunc, linkdims(MPS(ψ)).^2)

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites = siteinds(MPS(ψ))
    sites_pauli_mps = siteinds(d^2,N) 
    
    errs = Float64[]    # store truncation errors
    Pψ = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], bra(sites[1]), sites[1])
    
    Pψ[1] = ψbra[1]*Pten1*ψ[1]
    backs = Function[]
    for j in 1:N-1    
        Bp = Pψ[j]
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], bra(sites[j+1]), sites[j+1])

        linds = j>1 ? (sites_pauli_mps[j], commonind(Pψ[j-1], Bp)) : (sites_pauli_mps[j],) 
        
        SVDcontract_j = (tensors, leftinds) -> SVDcontract(tensors, leftinds; move_ogc=:right, trunc=(trunc..., maxrank=maxranks[j]))
        (Up, Rp), back_j = Zygote.pullback(SVDcontract_j, [Bp, ψ[j+1], ψbra[j+1], Pten], linds)
        push!(backs, back_j)

        Pψ[j] = Up
        Pψ[j+1] = Rp
    end


    function pauliMPS_pullback(ΔPψ)

        ΔPψ = copy(ΔPψ)
        Δψ = Array{ITensor, 1}(undef, N)
        for j in N-1:-1:1
            ΔUp, ΔRp = ΔPψ[j:j+1]

            (ΔBp, Δψ_jp1, Δψbra_jp1, _), _ = backs[j]((ΔUp, ΔRp))
            
            ΔPψ[j] = ΔBp
            Δψbra_jp1 = removetags(Δψbra_jp1, "bra")
            Δψ[j+1] = Δψ_jp1 + conj(Δψbra_jp1)
        end
        Δψ[1] = 2*conj(ψbra[1]*Pten1)*ΔPψ[1]

        Δψ, _ = og_back(Δψ)

        return (NoTangent(), Δψ, NoTangent())
    end

    post_factorize_callback(errs)
    return Pψ, pauliMPS_pullback
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