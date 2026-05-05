using LinearAlgebra
using ITensors, ITensorMPS

### CONSTANTS

const Id = (1.0+0.0im)*[1.0 0; 0 1.0]
const X = (1.0+0.0im)*[0 1.0 ; 1.0 0]
const Z = (1.0+0.0im)*[1.0 0; 0 -1.0]
const Y = [0.0 -1.0im; 1.0im 0.0]

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


### FUNCTIONS TO REPLACE LINKINDS

bra(T) = addtags(T,"bra")
ITensorMPS.siteinds(ψ::Vector{ITensor}) = siteinds(MPS(ψ))
ITensorMPS.linkinds(ψ::Vector{ITensor}) = linkinds(MPS(ψ))
ITensorMPS.linkdims(ψ::Vector{ITensor}) = linkdims(MPS(ψ))

function bralinks(ψ::Vector{ITensor})
    N = length(ψ)
    ψ = copy(ψ)
    links = linkinds(ψ)
    blinks = bra.(links)
    ψ[1] = replaceind(ψ[1], links[1], blinks[1])
    for j in 2:N-1
        ψ[j] = replaceinds(ψ[j], links[j-1:j], blinks[j-1:j])
    end
    ψ[N] = replaceind(ψ[N], links[N-1], blinks[N-1])
    return ψ
end

function bralinks(ψ::T) where {T<:Union{MPS, MPO}}
    olims = ortho_lims(ψ)
    ψ_vec = bralinks(ψ[:])
    ψ = T(ψ_vec)
    set_ortho_lims!(ψ, olims)
    return ψ
end

"Replace linkinds with new ones"
function replace_linkinds(ψ::T; newlinks=nothing) where {T<:Union{MPS,MPO}}
    ψ = copy(ψ)
    N = length(ψ)
    links = linkinds(ψ)

    if isnothing(newlinks)
        newlinks = [Index(links[j].space, "Link, l=$(j)") for j in 1:N-1]
    else
        @assert length(newlinks) == N-1
    end

    @preserve_ortho (ψ) begin
        ψ[1] = replaceind(ψ[1], links[1], newlinks[1])
        for j in 2:N-1
            ψ[j] = replaceinds(ψ[j], links[j-1:j], newlinks[j-1:j])
        end
        ψ[N] = replaceind(ψ[N], links[N-1], newlinks[N-1])
    end
    return ψ
end


### FUNCTIONS TO CHECK ORTHOGONALITY

function is_orthogonal(psi::Vector{ITensor}, ogc::Int)
    N = length(psi)
    links = linkinds(psi)
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

function is_orthogonal(psi::MPS, ogc::Int)
    return ortho_lims(psi) === ogc:ogc
end

function is_orthogonal(psi::Vector{<:AbstractArray}, ogc::Int)
    N = length(psi)
    assert_lengths = fill(2, N)
    if 1 < ogc < N
        assert_lengths[ogc] = 3
    end
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

function check_orthogonal(psi::Union{AbstractVector, MPS}, ogc::Int)
    !is_orthogonal(psi, ogc) && throw(ErrorException("ψ is NOT orthogonal at specified orthogonality center=$(ogc)"))
    return true
end


### FUNCTIONS TO EDIT KWARGS FOR TRUNCATION 

"Helper function to modify the maxrank argument in svd_trunc. 
Removes the :maxrank key in trunc if present, and returns an array maxranks such that
maxranks[j] = min(trunc(:maxrank), dims[j])"
function adapt_truncarg(trunc::NamedTuple, dims::Vector{<:Int})
    N = length(dims)+1
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