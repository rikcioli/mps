include("helpers.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using LinearAlgebra
using KrylovKit
using OptimKit
using ChainRulesCore
using MatrixAlgebraKit
using MatrixAlgebraKit: default_pullback_rank_atol, default_pullback_gauge_atol,
                        iszerotangent, project_antihermitian!, inv_safe
using Zygote


### OVERLOADING OF MatrixAlgebraKit svd_trunc_pullback to apply patch

using MatrixAlgebraKit: MatrixAlgebraKit, TruncationStrategy

struct TruncationDegenerate{Strategy <: TruncationStrategy, T <: Real} <: TruncationStrategy
    strategy::Strategy
    atol::T
    rtol::T
end

"""
    truncdegen(trunc::TruncationStrategy; atol::Real=0, rtol::Real=0)

Modify a truncation strategy so that if the truncation falls within
a degenerate subspace, the entire subspace gets truncated as well.
A value `val` is considered degenerate if
`norm(val - truncval) ≤ max(atol, rtol * norm(truncval))`
where `truncval` is the largest value truncated by the original
truncation strategy `trunc`.

For now, this truncation strategy assumes the spectrum being truncated
has already been reverse sorted and the strategy being wrapped
outputs a contiguous subset of values including the largest one. It
also only truncates for now, so may not respect if a minimum dimension
was requested in the strategy being wrapped. These restrictions may
be lifted in the future or provided through a different truncation strategy.
"""
function truncdegen(strategy::TruncationStrategy; atol::Real = 0, rtol::Real = 0)
    return TruncationDegenerate(strategy, promote(atol, rtol)...)
end


using MatrixAlgebraKit: findtruncated

function MatrixAlgebraKit.findtruncated(
        values::AbstractVector, strategy::TruncationDegenerate
    )
    Base.require_one_based_indexing(values)
    issorted(values; rev = true) || throw(ArgumentError("Values must be reverse sorted."))
    indices_collection = findtruncated(values, strategy.strategy)
    indices = Base.OneTo(maximum(indices_collection))
    indices_collection == indices ||
        throw(ArgumentError("Truncation must be a contiguous range."))
    if length(indices_collection) == length(values)
        # No truncation occurred.
        return indices
    end
    # The largest truncated value.
    truncval = values[last(indices) + 1]
    # Tolerance of determining if a value is degenerate.
    atol = max(strategy.atol, strategy.rtol * abs(truncval))
    for rank in reverse(indices)
        ≈(values[rank], truncval; atol, rtol = 0) || return Base.OneTo(rank)
    end
    return Base.OneTo(0)
end


using MatrixAlgebraKit: default_pullback_rank_atol, default_pullback_gauge_atol,
                            iszerotangent, project_antihermitian!, inv_safe, diagview

"""
    svd_trunc_pullback!(
        ΔA, A, USVᴴ, ΔUSVᴴ;
        rank_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ[1], ΔUSVᴴ[3])
    )

Adds the pullback from the truncated SVD of `A` to `ΔA`, given the output `USVᴴ` and the
cotangent `ΔUSVᴴ` of `svd_trunc`.

In particular, it is assumed that `A * Vᴴ' ≈ U * S` and `U' * A = S * Vᴴ`, with `U` and `Vᴴ`
rectangular matrices of left and right singular vectors, and `S` diagonal. For the
cotangents, it is assumed that if `ΔU` and `ΔVᴴ` are not zero, then they have the same size
as `U` and `Vᴴ` (respectively), and if `ΔS` is not zero, then it is a diagonal matrix of the
same size as `S`. For this method to work correctly, it is also assumed that the remaining
singular values (not included in `S`) are (sufficiently) separated from those in `S`.

A warning will be printed if the cotangents are not gauge-invariant, i.e. if the
anti-hermitian part of `U' * ΔU + Vᴴ * ΔVᴴ'`, restricted to rows `i` and columns `j` for
which `abs(S[i] - S[j]) < degeneracy_atol`, is not small compared to `gauge_atol`.
"""
function svd_trunc_pullback!(
        ΔA::AbstractMatrix, A, USVᴴ, ΔUSVᴴ;
        rank_atol::Real = 0,
        degeneracy_atol::Real = default_pullback_rank_atol(USVᴴ[2]),
        gauge_atol::Real = default_pullback_gauge_atol(ΔUSVᴴ[1], ΔUSVᴴ[3])
    )

    # Extract the SVD components
    U, Smat, Vᴴ = USVᴴ
    m, n = size(U, 1), size(Vᴴ, 2)
    (m, n) == size(ΔA) || throw(DimensionMismatch())
    p = size(U, 2)
    p == size(Vᴴ, 1) || throw(DimensionMismatch())
    S = diagview(Smat)
    p == length(S) || throw(DimensionMismatch())

    # Extract and check the cotangents
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    UΔU = fill!(similar(U, (p, p)), 0)
    VΔV = fill!(similar(Vᴴ, (p, p)), 0)
    if !iszerotangent(ΔU)
        (m, p) == size(ΔU) || throw(DimensionMismatch())
        mul!(UΔU, U', ΔU)
    end
    if !iszerotangent(ΔVᴴ)
        (p, n) == size(ΔVᴴ) || throw(DimensionMismatch())
        mul!(VΔV, Vᴴ, ΔVᴴ')
        # ΔVᴴ -= VΔVp' * Vᴴr but one less allocation without overwriting ΔVᴴ
        ΔVᴴ = mul!(copy(ΔVᴴ), VΔV', Vᴴ, -1, 1)
    end

    # Project onto antihermitian part; hermitian part outside of Grassmann tangent space
    aUΔU = project_antihermitian!(UΔU)
    aVΔV = project_antihermitian!(VΔV)

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(S' .- S) .< degeneracy_atol
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    Δgauge ≤ gauge_atol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    UdΔAV = (aUΔU .+ aVΔV) .* inv_safe.(S' .- S, degeneracy_atol) .+
        (aUΔU .- aVΔV) .* inv_safe.(S' .+ S, degeneracy_atol)
    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        p == length(ΔS) || throw(DimensionMismatch())
        diagview(UdΔAV) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, U, UdΔAV * Vᴴ, 1, 1) # add the contribution to ΔA

    # add contribution from orthogonal complement
    Ũ = qr_null(U)
    Ṽᴴ = lq_null(Vᴴ)
    m̃ = m - p
    ñ = n - p
    Ã = Ũ' * A * Ṽᴴ'
    ÃÃ = similar(A, (m̃ + ñ, m̃ + ñ))
    fill!(ÃÃ, 0)
    view(ÃÃ, (1:m̃), m̃ .+ (1:ñ)) .= Ã
    view(ÃÃ, m̃ .+ (1:ñ), 1:m̃) .= Ã'

    rhs = similar(Ũ, (m̃ + ñ, p))
    if !iszerotangent(ΔU)
        mul!(view(rhs, 1:m̃, :), Ũ', ΔU)
    else
        fill!(view(rhs, 1:m̃, :), 0)
    end
    if !iszerotangent(ΔVᴴ)
        mul!(view(rhs, m̃ .+ (1:ñ), :), Ṽᴴ, ΔVᴴ')
    else
        fill!(view(rhs, m̃ .+ (1:ñ), :), 0)
    end
    #XY = sylvester(ÃÃ, -Smat, rhs)     # INSTEAD WE USE KrilovKit LINSOLVE
    # replace XY = sylvester(ÃÃ, -Smat, rhs) with linsolve
    Smat⁻¹ = diagm(inv_safe.(S, degeneracy_atol))
    f(xy) = ÃÃ * xy * Smat⁻¹ - xy
    XY₀ = zeros(KrylovKit.scalartype(ÃÃ), size(ÃÃ, 2), size(Smat⁻¹, 1))
    XY, info = linsolve(f, -rhs * Smat⁻¹, XY₀; maxiter=100)

    X = view(XY, 1:m̃, :)
    Y = view(XY, m̃ .+ (1:ñ), :)
    ΔA = mul!(ΔA, Ũ, X * Vᴴ, 1, 1)
    ΔA = mul!(ΔA, U, Y' * Ṽᴴ, 1, 1)
    return ΔA
end




### HELPER TAPE STRUCT FOR SVDCONTRACT, THE SUBROUTINE WHICH IS PRESENT
### IN ALL THE FUNCTIONS THAT REQUIRE TRUNCATION
struct SVDcontractTape
    # ITensor data
    move_ogc::Symbol
    normalize::Bool
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
    Snorm::Float64
end
   

"Contracts all the tensors in a Vector{ITensor} in order
and computes a truncated SVD of the result with the left indices specified by linds.
Also returns truncation error."
function SVDcontract(tensors::Vector{<:ITensor}, linds::Vector{<:Index}; move_ogc=:right, normalize=false, kwargs...)
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
    #_, S, _ = svd_compact(M)  #Only needed for diagnostics
    #if length(diag(S)) > 2
    #    if S[2,2] - S[3,3] < 1e-8 && S[3,3] > 1e-12
    #        @warn "Found degeneracy"
    #        @show diag(S)
    #    end 
    #end
    U, S, Vdg, err = svd_trunc(M; kwargs...)
    Snorm = norm(S)
    if normalize
        S /= Snorm
    end
    
    bondind = move_ogc==:right ? Index(size(U)[2], "Link, u") : Index(size(Vdg)[1], "Link, v")

    ML = move_ogc==:right ? U : U*S
    MR = move_ogc==:right ? S*Vdg : Vdg

    ML_ten = ITensor(ML, cLind, bondind)
    ML_ten *= dag(cL)
    MR_ten = ITensor(MR, bondind, cRind)
    MR_ten *= dag(cR)

    out = ((ML_ten, MR_ten), err)
    tape = SVDcontractTape(move_ogc, normalize, tensors, prods, cL, cR, bondind, M, U, S, Vdg, Snorm)

    return out, tape
end


function SVDcontract_pullback(ΔMf, tape::SVDcontractTape)
    ΔML_ten, ΔMR_ten = ΔMf
    (; move_ogc, normalize, tensors, prods, cL, cR, bondind, M, U, S, Vdg, Snorm) = tape

    ΔML_ten *= cL
    ΔMR_ten *= cR
    cLind = combinedind(cL); cRind = combinedind(cR);

    ΔML = Matrix{ComplexF64}(ΔML_ten, cLind, bondind)
    ΔMR = Matrix{ComplexF64}(ΔMR_ten, bondind, cRind)

    local ΔU, ΔS, ΔVdg
    if move_ogc==:right
        ΔU = ΔML 
        ΔS = Diagonal(diag(ΔMR*Vdg'))
        ΔVdg = S'*ΔMR
    else
        ΔU = ΔML*S'
        ΔS = Diagonal(diag(U'*ΔML))
        ΔVdg = ΔMR
    end

    if normalize
        ΔS = ΔS/Snorm - S*dot(S, ΔS)/Snorm
        S *= Snorm
    end

    ΔM = zero(M)
    svd_trunc_pullback!(ΔM, M, (U, S, Vdg), (ΔU, ΔS, ΔVdg), gauge_atol = 1e-8)

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
function ITensorMPS.MPS(V::Vector{<:AbstractArray}, ogc; check_og=true, sites=nothing, links=nothing)
    check_og && check_orthogonal(V, ogc)
    
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor

    if isnothing(sites)
        sites = siteinds("Qubit", N)
    end
    d = only(Set(space.(sites)))
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    @assert size(V[1]) == (d, dimlinks[1])
    for j in 2:ogc-1
        @assert size(V[j]) == (dimlinks[j-1]*d, dimlinks[j])
    end
    if 1<ogc<N
        @assert size(V[ogc]) == (dimlinks[ogc-1], d, dimlinks[ogc])
    end
    for j in ogc+1:N-1
        @assert size(V[j]) == (dimlinks[j-1], d*dimlinks[j])
    end
    @assert size(V[N]) == (dimlinks[N-1], d)

    if isnothing(links)
        links = [Index(dimlinks[j], "Link, l=$j") for j in 1:N-1]
    end

    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]
    allinds = [inds1; indsbulk; indsN]

    V1 = [V[1]]
    VB = [reshape(V[j], (dimlinks[j-1], d, dimlinks[j])) for j in 2:N-1]
    VN = [V[N]]

    Vresh = [V1; VB; VN]
    Vtensors = [ITensor(Vresh[j], allinds[j]) for j in 1:N]
    Vmps = MPS(Vtensors)
    set_ortho_lims!(Vmps, ogc:ogc)
    return Vmps
end

"Convert vector of isometries with orthogonality center ogc into an MPS.
The pullback treats the adjoint of the MPS as if it was a vector of matrices."
function ChainRulesCore.rrule(::typeof(ITensorMPS.MPS), V::Vector{<:AbstractArray}, ogc::Int; check_og=true, sites=nothing, links=nothing)
    check_og && check_orthogonal(V, ogc)
    N = length(V)
    V = copy.(V) # eliminates adjoint type before converting to ITensor
    if isnothing(sites)
        sites = siteinds("Qubit", N)
    end
    d = only(Set(space.(sites)))
    dimlinksL = [size(V[j], 2) for j in 1:ogc-1]
    dimlinksR = [size(V[j], 1) for j in ogc+1:N]
    dimlinks = [dimlinksL; dimlinksR]

    @assert size(V[1]) == (d, dimlinks[1])
    for j in 2:ogc-1
        @assert size(V[j]) == (dimlinks[j-1]*d, dimlinks[j])
    end
    if 1<ogc<N
        @assert size(V[ogc]) == (dimlinks[ogc-1], d, dimlinks[ogc])
    end
    for j in ogc+1:N-1
        @assert size(V[j]) == (dimlinks[j-1], d*dimlinks[j])
    end
    @assert size(V[N]) == (dimlinks[N-1], d)

    if isnothing(links)
        links = [Index(dimlinks[j], "Link, l=$j") for j in 1:N-1]
    end

    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]
    allinds = [inds1; indsbulk; indsN]

    V1 = [V[1]]
    VB = [reshape(V[j], (dimlinks[j-1], d, dimlinks[j])) for j in 2:N-1]
    VN = [V[N]]

    Vresh = [V1; VB; VN]
    Vtensors = [ITensor(Vresh[j], allinds[j]) for j in 1:N]
    Vmps = MPS(Vtensors)
    set_ortho_lims!(Vmps, ogc:ogc)

    function MPS_pullback(ΔVmps)
        ΔVresh = [Array{ComplexF64}(ΔVmps[j], allinds[j]) for j in 1:N]

        ΔVL = [reshape(ΔVresh[j], (j==1 ? d : dimlinks[j-1]*d, dimlinks[j])) for j in 1:ogc-1]
        ΔVC = [ΔVresh[ogc]]
        ΔVR = [reshape(ΔVresh[j], (dimlinks[j-1], j==N ? d : d*dimlinks[j])) for j in ogc+1:N]
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

function toMatrices(V::Union{MPS, Vector{<:ITensor}}, ogc::Int)
    N = length(V)
    sites = siteinds(V)
    links = linkinds(V)
    d = only(Set(space.(sites)))    # all sites must have equal index
    dimlinks = space.(links)

    inds1 = [(sites[1], links[1])]
    indsbulk = [(links[j-1], sites[j], links[j]) for j in 2:N-1]
    indsN = [(links[N-1], sites[N])]
    allinds = vcat(inds1, indsbulk, indsN)

    Vmat = [Array{ComplexF64}(V[j], allinds[j]) for j in 1:N]

    Vfinal = Vector{Array{ComplexF64}}(undef, N)
    for j in 1:ogc-1
        Vfinal[j] = reshape(Vmat[j], (j==1 ? d : dimlinks[j-1]*d, dimlinks[j]))
    end
    for j in ogc+1:N
        Vfinal[j] = reshape(Vmat[j], (dimlinks[j-1], j==N ? d : d*dimlinks[j]))
    end
    Vfinal[ogc] = Vmat[ogc]
    return Vfinal
end

"Convert a left-canonical ITensors.MPS object ψ into a Vector{Matrix{ComplexF64}} of left isometries + one generic matrix at site N.
Can be also used to convert Vector{ITensor} object representing tangent vectors, in which case check_og must be set to false."
function toMatricesLC(ψ::Union{MPS, Vector{<:ITensor}}; check_og = true)
    N = length(ψ)
    check_og && !is_orthogonal(ψ, N) && throw(ErrorException("Trying to convert MPS to vector of left isometries, but the MPS is NOT orthogonal at site N"))
    
    sites = siteinds(ψ)
    d = only(Set(space.(sites)))    # all sites must have equal physical space
    dimlinks = space.(linkinds(ψ))
    allinds = ordered_inds(ψ)

    Vmat = [Array{ComplexF64}(ψ[j], allinds[j]) for j in 1:N]

    Vfinal = Vector{Matrix{ComplexF64}}(undef, N)
    for j in 1:N-1
        Vfinal[j] = reshape(Vmat[j], (j==1 ? d : dimlinks[j-1]*d, dimlinks[j]))
    end
    Vfinal[N] = Vmat[N]

    return Vfinal
end

"Convert a right-canonical ITensors.MPS object ψ into a Vector{Matrix{ComplexF64}} of right isometries + one generic matrix at site 1."
function toMatricesRC(ψ::Union{MPS, Vector{<:ITensor}}; check_og = true)
    N = length(ψ)
    check_og && !is_orthogonal(ψ, 1) && throw(ErrorException("Trying to convert MPS to vector of right isometries, but the MPS is NOT orthogonal at site 1"))
    
    sites = siteinds(ψ)
    d = only(Set(space.(sites)))    # all sites must have equal physical space
    dimlinks = space.(linkinds(ψ))
    allinds = ordered_inds(ψ)

    Vmat = [Array{ComplexF64}(ψ[j], allinds[j]) for j in 1:N]

    Vfinal = Vector{Matrix{ComplexF64}}(undef, N)
    Vfinal[1] = Vmat[1]
    for j in 2:N
        Vfinal[j] = reshape(Vmat[j], (dimlinks[j-1], j==N ? d : d*dimlinks[j]))
    end

    return Vfinal
end


### CURRENTLY NOT USED ANYWHERE, AS IT SEEMS IT'S NOT NEEDED
"Project vector D onto tangent space in V"
function project(V::Union{MPS, Vector{<:ITensor}}, D::Union{MPS, Vector{<:ITensor}}, ogc::Int)
    sites = siteinds(D); links = linkinds(D);
    Vmat = toMatrices(V, ogc)
    Dmat = toMatrices(D, ogc)
    Dproj = projectMixed(Vmat, Dmat, ogc)
    DprojT = toITensors(Dproj, ogc; sites, links, check_og = false)     # it will NOT be orthogonalized in general
    return DprojT
end


##### THE FOLLOWING FUNCTIONS EXTEND STANDARD ITensorMPS METHODS TO WORK WITH Zygote
##### BY TREATING THE MPS AS VECTORS OF ITENSORS

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
        Δψ_vec = [dag(delta(sites[j], sites[j]', sites[j]''))*Δψmpo[j] for j in 1:N]
        Δψ_vec = replaceprime.(Δψ_vec, 2 => 0)

        Δψ_vec = replace_linkinds(Δψ_vec; newlinks=oldlinks)
        return (NoTangent(), Δψ_vec)
    end
    return ψf, MPO_pullback
end


### OVERLAP BETWEEN TWO MPS, TREATING THEM AS VECTORS OF ITENSORS

"Compute the scalar product of two MPS. Same as ITensorMPS.inner, but different pullback."
function sproduct(ψ::MPS, ϕ::MPS)
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
function ChainRulesCore.rrule(::typeof(sproduct), ψ::MPS, ϕ::MPS)
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

        #if isortho(ψ)
        #    ogc1 = orthocenter(ψ)
        #    Δψ_vec = project(ψ, Δψ_vec, ogc1)
        #end
        #if isortho(ϕ)
        #    ogc2 = orthocenter(ϕ)
        #    Δϕ_vec = project(ϕ, Δϕ_vec, ogc2)
        #end
            
        return (NoTangent(), Δψ_vec, Δϕ_vec)
    end

    return C, sproduct_pullback
end


function sproduct(ψ::Vector{<:ITensor}, ϕ::Vector{<:ITensor}; ogc1 = 0, ogc2 = 0)
    ψmps = MPS(ψ)
    ϕmps = MPS(ϕ)
    if !iszero(ogc1)
        set_ortho_lims!(ψmps, ogc1:ogc1)
    end
    if !iszero(ogc2)
        set_ortho_lims!(ϕmps, ogc2:ogc2)
    end
    return sproduct(ψmps, ϕmps)
end

function ChainRulesCore.rrule(::typeof(sproduct), ψ::Vector{<:ITensor}, ϕ::Vector{<:ITensor}; ogc1 = 0, ogc2 = 0)
    ψmps = MPS(ψ)
    ϕmps = MPS(ϕ)
    if !iszero(ogc1)
        set_ortho_lims!(ψmps, ogc1:ogc1)
    end
    if !iszero(ogc2)
        set_ortho_lims!(ϕmps, ogc2:ogc2)
    end
    res, back = pullback(sproduct, ψmps, ϕmps)

    function sproduct_pullback(Δres)
        Δψ_vec, Δϕ_vec = back(Δres)
        return (NoTangent(), Δψ_vec, Δϕ_vec)
    end

    return res, sproduct_pullback
end


"Computes the norm of the MPS."
function snorm(ψ::MPS)
    return isortho(ψ) ? norm(ψ[only(ortho_lims(ψ))]) : sqrt(sproduct(ψ, ψ))
end

function ChainRulesCore.rrule(::typeof(snorm), ψ::MPS)

    if isortho(ψ)
        ogc = only(ortho_lims(ψ))
        C = ψ[ogc]
        n = norm(C)
        function snorm_pullback_og(Δn)
            ΔC = Δn*C/n
            Δψ = [ITensor(inds(T)) for T in ψ]
            Δψ[ogc] = ΔC
            return (NoTangent(), Δψ)
        end
        return n, snorm_pullback_og
    else
        n2, back_sproduct = pullback(sproduct, ψ, ψ)
        n = sqrt(n2)
        function snorm_pullback_all(Δn)
            Δn2 = Δn/(2n)
            Δψ1, Δψ2 = back_sproduct(Δn2)
            Δψ = Δψ1 + Δψ2
            return (NoTangent(), Δψ)
        end
        return n, snorm_pullback_all
    end
end

"Computes the squared norm of the MPS."
function snorm_sq(ψ::MPS)
    return isortho(ψ) ? norm(ψ[only(ortho_lims(ψ))])^2 : sproduct(ψ, ψ)
end

function ChainRulesCore.rrule(::typeof(snorm_sq), ψ::MPS)

    if isortho(ψ)
        ogc = only(ortho_lims(ψ))
        C = ψ[ogc]
        n2 = norm(C)^2
        function snorm_sq_pullback_og(Δn2)
            ΔC = 2*Δn2*C
            Δψ = [ITensor(inds(T)) for T in ψ]
            Δψ[ogc] = ΔC
            return (NoTangent(), Δψ)
        end
        return n, snorm_pullback_og
    else
        n2, back_sproduct = pullback(sproduct, ψ, ψ)
        function snorm_sq_pullback_all(Δn2)
            Δψ1, Δψ2 = back_sproduct(Δn2)
            Δψ = Δψ1 + Δψ2
            return (NoTangent(), Δψ)
        end
        return n2, snorm_sq_pullback_all
    end
end

function slognorm(ψ::MPS)
    return log(norm(ψ))/2
end

function normalize_logn(ψ::MPS)
    @assert isortho(ψ)
    ogc = only(ortho_lims(ψ))
    ψn = copy(ψ)
    n = snorm(ψ)
    ψn[ogc] /= n
    return ψn, log(n)
end

function ChainRulesCore.rrule(::typeof(normalize_logn), ψ::MPS)
    @assert isortho(ψ)
    ogc = only(ortho_lims(ψ))
    ψn = copy(ψ)
    n = snorm(ψ)
    ψn[ogc] /= n

    function pullback_normalize_logn(Δy)
        Δψn, Δlogn = Δy
        Δψ = copy(Δψn)

        # scalar inner product between the cotangent of An and An itself
        alpha = real(dot(Δψn[ogc], ψn[ogc]))

        # derivative w.r.t. A
        Δψ[ogc] = (Δψn[ogc] - ψn[ogc]*alpha + Δlogn*ψn[ogc])/n

        return (NoTangent(), Δψ)
    end

    return (ψn, log(n)), pullback_normalize_logn
end

### PRODUCT OF AN MPO WITH AN MPS, TENSOR BY TENSOR

function direct(W::MPO, ψ::MPS)
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

function ChainRulesCore.rrule(::typeof(direct), W::MPO, ψ::MPS)
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
    
    function direct_pullback(ΔWψ)

        ΔWψ = [replaceprime(ΔWψ[j], 0 => 1; inds = sites[j]) for j in 1:N]
        ΔWψ[1] = replaceind(ΔWψ[1], Wψlinks[1], combinds[1])*dag(combs[1])
        for j in 2:N-1
            ΔWψ[j] = dag(combs[j-1])*replaceinds(ΔWψ[j], Wψlinks[j-1:j], combinds[j-1:j])*dag(combs[j])
        end
        ΔWψ[N] = replaceind(ΔWψ[N], Wψlinks[N-1], combinds[N-1])*dag(combs[N-1])

        Δψ_vec = dag.(W)[:] .* ΔWψ
        ΔW_vec = ΔWψ .* dag.(ψ)[:]
        return (NoTangent(), ΔW_vec, Δψ_vec)
    end

    return Wψ, direct_pullback
end


### MULTIPLY MPO WITH MPS WITH ZIP-UP ALGORITHM

function zipup(W::MPO, ψ::MPS; trunc=NamedTuple(), post_factorize_callback = identity)
    N = length(ψ)
    ψ = move_center(ψ, N)
    W = move_center(W, N)

    #trunc, maxranks = adapt_truncarg(trunc, linkdims(W).*linkdims(ψ))
    Wlinks = linkinds(W)
    ψlinks = linkinds(ψ)

    errs = Float64[]    # store truncation errors
    Wψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final mps
    local Lten::ITensor
    for j in N:-1:2    
        linds = [Wlinks[j-1]; ψlinks[j-1]]
        tensors = j<N ? [Lten, ψ[j], W[j]] : [ψ[j], W[j]]

        ((Lten, V), err), _ = SVDcontract(tensors, linds; move_ogc=:left, trunc=trunc)
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

    #trunc, maxranks = adapt_truncarg(trunc, linkdims(W).*linkdims(ψ))

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
                                            trunc=trunc)
        push!(errs, err)
        push!(tapes, tape_j)
        Wψ_vec[j] = V
    end
    Wψ_vec[1] = Lten*ψ[1]*W[1]

    post_factorize_callback(errs)
    Wψ = MPS(Wψ_vec)
    set_ortho_lims!(Wψ, 1:1)

    function zipup_pullback(ΔWψ)
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

        (Δψ_vec,) = move_center_back_ψ(Δψ_vec)
        (ΔW_vec,) = move_center_back_W(ΔW_vec)
        return (NoTangent(), ΔW_vec, Δψ_vec)
    end

    return Wψ, zipup_pullback
end


function product(W::MPO, ψ::MPS, alg::Symbol; kwargs...)
    if alg == :direct
        return direct(W, ψ)
    elseif alg == :zipup
        return zipup(W, ψ; kwargs...)
    else
        throw(DomainError(alg, "Invalid algorithm. Supported: direct, zipup."))
    end
end


### ORTHOGONALIZE WITH TRUNCATION

function move_center(ψ::T, b::Int; trunc=NamedTuple(), normalize=false, post_factorize_callback=identity) where {T<:Union{MPS, MPO}}
    N = length(ψ)
    cog = only(ortho_lims(ψ)) #current orthogonality center
    @assert 1 <= cog <= N
    @assert 1 <= b <= N

    if b==cog
        return ψ
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    #trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ))
    
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
            ((W1, W2), err), _ = SVDcontract(tensors, linds; 
                                    move_ogc=:right,
                                    normalize=normalize,
                                    trunc=trunc)
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
            ((W1, W2), err), _ = SVDcontract(tensors, linds; 
                                    move_ogc=:left,
                                    normalize=normalize,
                                    trunc=trunc)
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

function ChainRulesCore.rrule(::typeof(move_center), ψ::T, b::Int; trunc=NamedTuple(), normalize=false, post_factorize_callback=identity) where {T<:Union{MPS, MPO}}
    N = length(ψ)
    cog = only(ortho_lims(ψ)) #current orthogonality center
    @assert 1 <= cog <= N
    @assert 1 <= b <= N

    if b==cog
        return ψ, Δψf -> (NoTangent(), Δψf, NoTangent())
    end
    to_right = b>cog  # left-to-right mode

    # Preparing the maxranks for svd trunc
    #trunc, maxranks = adapt_truncarg(trunc, linkdims(ψ))

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

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; 
                                        move_ogc=:right,
                                        normalize=normalize,
                                        trunc=trunc)
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

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; 
                                        move_ogc=:left,
                                        normalize=normalize,
                                        trunc=trunc)
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

        return (NoTangent(), Δψ_vec, NoTangent())
    end

    return ψf, move_center_pullback
end


### APPLY VECTOR OF UNITARIES IN A BRICKWORK PATTERN, SWEEPING LEFT TO RIGHT AND BACK

function apply_brickwork(Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, to_right=true, trunc=NamedTuple(), normalize=true, post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1
    @assert length(Uarray)>0

    # Preparing the maxranks for svd trunc
    #trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    strat = to_strategy(trunc)
    if !(strat isa MatrixAlgebraKit.NoTruncation)
        strat = truncdegen(strat; atol=1e-12)
    end
    
    # by default to_right == true, meaning we sweep from left to right
    ψ = move_center(ψ, to_right ? 1 : N)

    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    current_layer_odd = true
    local lastj
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)
        
        for j in jvals
            end_of_sweep = (j==N-1 && to_right) || (j==1 && !to_right)
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+current_layer_odd)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? [sites[j]'; commonind(ψfinal[j-1], WLten)] : [sites[j]';]
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? [sites[j]; commonind(ψfinal[j-1], WLten)] : [sites[j];]
                tensors = [WLten, WRten]
            end

            ((W1, W2), err), _ = SVDcontract(tensors, linds; 
                                    move_ogc = (to_right ? :right : :left), 
                                    normalize = normalize,
                                    trunc = strat)
            push!(errs, err)
            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if end_of_sweep
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right    
                current_layer_odd = !current_layer_odd
            end
        end
    end
    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)
    return ψfinal
end

function ChainRulesCore.rrule(::typeof(apply_brickwork), Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, to_right=true, trunc=NamedTuple(), normalize=true, post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1
    @assert length(Uarray)>0

    ψ, move_center_back = Zygote.pullback(move_center, ψ, to_right ? 1 : N)

    # Preparing the maxranks for svd trunc
    #trunc, maxranks = adapt_truncarg(trunc, [min(2^j, 2^(N-j)) for j in 1:N])
    strat = to_strategy(trunc)
    if !(strat isa MatrixAlgebraKit.NoTruncation)
        strat = truncdegen(strat; atol=1e-12)
    end
    
    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    current_layer_odd = true
    tapes = SVDcontractTape[]  # store intermediate data for the pullback
    local lastj      # store last j reached
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)

        for j in jvals
            end_of_sweep = (j==N-1 && to_right) || (j==1 && !to_right)
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+current_layer_odd)
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
                                                normalize = normalize,
                                                trunc = strat)
            push!(tapes, tape_j)
            push!(errs, err)
            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if end_of_sweep
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right
                current_layer_odd = !current_layer_odd
            end
        end
    end
    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)

    function apply_brickwork_pullback(Δψfinal)

        Δψ = copy(Δψfinal)
        ΔUarray = [zeros(ComplexF64, size(U)) for U in Uarray]
        i = nU; pb_n = length(tapes);
        while i>=1
            jvals = to_right ? (lastj:-1:1) : (lastj:N-1) 
            for j in jvals
                ΔW1 = Δψ[j]
                ΔW2 = Δψ[j+1]

                if iseven(j+shift+current_layer_odd)
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
            current_layer_odd = !current_layer_odd
        end

        (Δψ,) = move_center_back(Δψ)
        # note it's a VECTOR, not an MPS, since it will NOT be orthogonalized in general
        return (NoTangent(), ΔUarray, Δψ)
    end
    return ψfinal, apply_brickwork_pullback
end


"""
TODO:: custom chain rule?
"""
function lognorm(A::ITensor)
    n = norm(A)
    return log(n)
end

"""
TODO: safeinv
"""
function diven(A::ITensor, logn::Number)
    inv = exp(-logn)
    return A*inv
end

"""
TODO: custom rrule, safeinv
"""
#function normalize_logn!(A::ITensor)
#    logn = lognorm(A::ITensor)
#    An = diven(A,logn)
#    return An, logn
#end

function normalize_logn!(A::ITensor)
    n = norm(A)
    An = A/n
    return An, log(n)
end

function ChainRulesCore.rrule(::typeof(normalize_logn!), A::ITensor)
    n = norm(A)
    An = A/n
    logn = log(n)

    function pullback(Δy)
        ΔAn, Δlogn = Δy

        # scalar inner product between the cotangent of An and An itself
        alpha = real(dot(ΔAn, An))

        # derivative w.r.t. A
        ΔA = (ΔAn - An*alpha + Δlogn*An)/n

        return (NoTangent(), ΔA)
    end

    return (An, logn), pullback
end

"""
- APPLY VECTOR OF UNITARIES IN A BRICKWORK PATTERN, SWEEPING LEFT TO RIGHT AND BACK
- Normalize after every SVD.
- return normalizes state, and logs = log(s) where s is the norm
"""
function apply_brickwork_normalize(Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, to_right=true, trunc=NamedTuple(), post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1
    @assert length(Uarray)>0
  
    ψ = move_center(ψ, to_right ? 1 + shift : N)

    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    lognorm_factors = Float64[]
    current_layer_odd = true
    local lastj
    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)
        
        for j in jvals
            end_of_sweep = (j==N-1 && to_right) || (j==1 && !to_right)
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+current_layer_odd)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? [sites[j]'; commonind(ψfinal[j-1], WLten)] : [sites[j]';]
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? [sites[j]; commonind(ψfinal[j-1], WLten)] : [sites[j];]
                tensors = [WLten, WRten]
            end
            move_ogc = (to_right ? :right : :left)
            ((W1, W2), err), tape = SVDcontract(tensors, linds; 
                                    move_ogc = move_ogc, 
                                    normalize = false,
                                    trunc = trunc)

            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")
            if move_ogc == :right
                W2, logn = normalize_logn!(W2)
            else
                W1, logn = normalize_logn!(W1)
            end

            push!(lognorm_factors,logn)
            push!(errs, err)

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if end_of_sweep
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right
                current_layer_odd = !current_layer_odd
            end
        end
    end
    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)
    return ψfinal, lognorm_factors
end

function ChainRulesCore.rrule(::typeof(apply_brickwork_normalize), Uarray::Vector{<:AbstractMatrix}, ψ::MPS; shift=0, to_right=true, trunc=NamedTuple(), post_factorize_callback=identity)
    N = length(ψ)
    @assert shift==0 || shift==1
    @assert length(Uarray)>0

    ψ, move_center_back = Zygote.pullback(move_center, ψ, to_right ? 1 +shift : N)

    # Preparing the maxranks for svd trunc
    d_loc = space(siteind(ψ,1))

    sites = siteinds(ψ)
    ψfinal = copy(ψ)
    errs = Float64[]
    i = 1; nU = length(Uarray)
    current_layer_odd = true
    tapes = SVDcontractTape[]  # store intermediate data for the pullback
    local lastj      # store last j reached
    lognorm_factors = Float64[]
    pull_lognorms = []

    while i<=nU
        jvals = to_right ? (1:N-1) : (N-1:-1:1)

        for j in jvals
            end_of_sweep = (j==N-1 && to_right) || (j==1 && !to_right)
            lastj = j
            WLten, WRten = ψfinal[j:j+1]

            if iseven(j+shift+current_layer_odd)
                Uten = ITensor(Uarray[i], sites[j]', sites[j+1]', sites[j], sites[j+1])
                linds = j > 1 ? [sites[j]'; commonind(ψfinal[j-1], WLten)] : [sites[j]';]
                tensors = [WLten, WRten, Uten]
                i += 1
            else
                linds = j > 1 ? [sites[j]; commonind(ψfinal[j-1], WLten)] : [sites[j];]
                tensors = [WLten, WRten]
            end
            move_ogc = (to_right ? :right : :left)

            ((W1, W2), err), tape_j = SVDcontract(tensors, linds; 
                                                move_ogc = move_ogc,
                                                normalize = false, 
                                                trunc = trunc)

                                                
            W1 = noprime(W1, tags="Site")
            W2 = noprime(W2, tags="Site")
            if move_ogc == :right
                (W2, logn), pull_logn = pullback(normalize_logn!,W2)
            else
                (W1, logn), pull_logn = pullback(normalize_logn!,W1)
            end

            push!(lognorm_factors,logn)
            push!(pull_lognorms,pull_logn)

            push!(tapes, tape_j)
            push!(errs, err)

            ψfinal[j] = W1
            ψfinal[j+1] = W2
            i>nU && break # before to_right changes
            if end_of_sweep
                # this has to be here, because we want to_right to remain as it is
                # if the endpoint of the sweep is reached exactly at i==nU
                to_right = !to_right
                current_layer_odd = !current_layer_odd
            end
        end
    end
    #logs,pull_sumlogs  = pullback(sum,lognorm_factors)

    post_factorize_callback(errs)
    final_ogc = to_right ? lastj+1 : lastj
    set_ortho_lims!(ψfinal, final_ogc:final_ogc)
 
    function apply_brickwork_normalize_pullback(Δout)
        Δψfinal, Δlognorm_factors = Δout
        Δψ = copy(Δψfinal)
        ΔUarray = [zeros(ComplexF64, size(U)) for U in Uarray]
        i = nU; pb_n = length(tapes);
        while i>=1
            jvals = to_right ? (lastj:-1:1) : (lastj:N-1) 
            for j in jvals
                ΔW1 = Δψ[j]
                ΔW2 = Δψ[j+1]
                if to_right
                    ΔW2, = pull_lognorms[pb_n]((ΔW2, Δlognorm_factors[pb_n]))
                else
                    ΔW1, = pull_lognorms[pb_n]((ΔW1, Δlognorm_factors[pb_n]))
                end

                if iseven(j+shift+current_layer_odd)
                    ΔW1 = prime(ΔW1, tags="Site")
                    ΔW2 = prime(ΔW2, tags="Site")
                    ΔMf = (ΔW1, ΔW2)

                    (ΔWLten, ΔWRten, ΔUten) = SVDcontract_pullback(ΔMf, tapes[pb_n])  # start from the last

                    ΔU = Array{ComplexF64}(ΔUten, sites[j]', sites[j+1]', sites[j], sites[j+1])
                    ΔUarray[i] = reshape(ΔU, size(Uarray[1]))
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
            current_layer_odd = !current_layer_odd
        end
        (Δψ,) = move_center_back(Δψ)
        # note it's a VECTOR, not an MPS, since it will NOT be orthogonalized in general
        return (NoTangent(), ΔUarray, Δψ)
    end
    return (ψfinal, lognorm_factors), apply_brickwork_normalize_pullback
end


##### FUNCTIONS FOR MAGIC EXTRACTION

### EXTRACT PAULI MPS FROM MPS

using MatrixAlgebraKit: trunctol, truncrank

function get_pauli_mps(ψ::MPS; trunc=NamedTuple(), sites=nothing, post_factorize_callback = identity)
    N = length(ψ)
    ψ = move_center(ψ, 1)
    ψbra, unbra = bra(ψ)

    strat = to_strategy(trunc)
    if !(strat isa MatrixAlgebraKit.NoTruncation)
        strat = truncdegen(strat; atol=2*eps())
    end

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites_pauli_mps = isnothing(sites) ? siteinds(d^2, N) : sites 
    sites = siteinds(ψ)
    brasites = siteinds(ψbra)
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], brasites[1], sites[1])

    errs = Float64[]    # store truncation errors
    Pψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    Bp = ψbra[1]*Pten1*ψ[1]
    for j in 1:N-1    
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], brasites[j+1], sites[j+1])

        linds = j>1 ? [sites_pauli_mps[j]; commonind(Pψ_vec[j-1], Bp)] : [sites_pauli_mps[j];] 
        tensors = [Bp, ψ[j+1], ψbra[j+1], Pten]

        ((Up, Rp), err), _ = SVDcontract(tensors, linds; 
                                        move_ogc=:right, 
                                        trunc=strat)
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
    ψbra, unbra = bra(ψ)

    strat = to_strategy(trunc)
    if !(strat isa MatrixAlgebraKit.NoTruncation)
        strat = truncdegen(strat; atol=2*eps())
    end

    # Build compressed Pauli MPS iteratively from left
    # bra is conjugated tensor in pauli mps, prime is conjugated Pauli mps
    d = 2
    sites_pauli_mps = isnothing(sites) ? siteinds(d^2, N) : sites 
    sites = siteinds(ψ)
    brasites = siteinds(ψbra)
    
    errs = Float64[]    # store truncation errors
    Pψ_vec = Array{ITensor, 1}(undef, N)    # store tensors that make the final Pauli mps
    
    Ps = get_Ps()
    Pten1 = ITensor(Ps/sqrt(2), sites_pauli_mps[1], brasites[1], sites[1])
    
    Bp = ψbra[1]*Pten1*ψ[1]
    tapes = SVDcontractTape[]
    for j in 1:N-1    
        Pten = ITensor(Ps/sqrt(2), sites_pauli_mps[j+1], brasites[j+1], sites[j+1])

        linds = j>1 ? [sites_pauli_mps[j]; commonind(Pψ_vec[j-1], Bp)] : [sites_pauli_mps[j];] 
        tensors = [Bp, ψ[j+1], ψbra[j+1], Pten]
        
        ((Up, Rp), err), tape_j = SVDcontract(tensors, linds; 
                                        move_ogc=:right, 
                                        trunc=strat)
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

        Δψ_vec = Array{ITensor, 1}(undef, N)
        ΔRp = ΔPψ[N]
        for j in N-1:-1:1
            ΔUp = ΔPψ[j]
            ΔMf = (ΔUp, ΔRp)

            (ΔBp, Δψ_jp1, Δψbra_jp1, _) = SVDcontract_pullback(ΔMf, tapes[j])
            
            Δψ_vec[j+1] = Δψ_jp1 + unbra(Δψbra_jp1)
            ΔRp = ΔBp
        end

        Δψ_vec[1] = dag(ψbra[1])*dag(Pten1)*ΔRp + unbra(ΔRp*dag(Pten1)*dag(ψ[1]))

        (Δψ_vec,) = move_center_back(Δψ_vec)

        return (NoTangent(), Δψ_vec, NoTangent())
    end
    
    return Pψ, get_pauli_mps_pullback
end

### COMPUTE SRE2 FOR VECTOR OF ISOMETRIES

function sre2(arrV::Vector{<:AbstractArray}, ogc::Int, alg::Symbol; trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(arrV)
    ψ = MPS(arrV, ogc)
    Pψ = get_pauli_mps(ψ; trunc = trunc_pauli)
    W = MPO(Pψ)
    WP = product(W, Pψ, alg; trunc = trunc_product)
    m2 = -log2(real(sproduct(WP, WP))) - N
    return m2
end

# WORKS, BUT WE ARE FORCED TO USE OUR CUSTOM CHAINRULE BECAUSE MPS NEED TO BE SUMMED
# AS IF THEY WERE VECTORS OF ITENSORS
function ChainRulesCore.rrule(::typeof(sre2), arrV::Vector{<:AbstractArray}, ogc::Int, alg::Symbol; trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(arrV)
    ψ, MPS_back = pullback(MPS, arrV, ogc)
    Pψ, get_pauli_mps_pullback = pullback(psi -> get_pauli_mps(psi; trunc=trunc_pauli), ψ)
    W, MPO_back = pullback(MPO, Pψ)    # at this point Pψ and W have same ortho lims
    WP, product_back = pullback((mpo, mps) -> product(mpo, mps, alg; trunc=trunc_product), W, Pψ)
    res, sproduct_back = pullback(sproduct, WP, WP)
    
    m2, m2_back = -log2(real(res))-N, Δm2 -> (NoTangent(), -Δm2/(log(2)*real(res)))

    function sre2_pullback(Δm2)
        _, Δres = m2_back(Δm2)

        ΔWP_1, ΔWP_2 = sproduct_back(Δres)
        ΔWP = ΔWP_1 .+ ΔWP_2

        ΔW, ΔPψ_1 = product_back(ΔWP)
        ΔPψ_2 = MPO_back(ΔW)[1]
        ΔPψ = ΔPψ_1 .+ ΔPψ_2
        Δψ = get_pauli_mps_pullback(ΔPψ)[1]
        ΔarrV, _ = MPS_back(Δψ)

        return (NoTangent(), ΔarrV, NoTangent(), NoTangent())
    end
    return m2, sre2_pullback
end

### SRE2 OF AN MPS ψ

function sre2( ψ::MPS, alg::Symbol; trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(ψ)
    # ψf = apply_brickwork(arrU, ψ; trunc=trunc_bw)
    Pψ = get_pauli_mps(ψ; trunc=trunc_pauli)
    W = MPO(Pψ)
    P2 = product(W, Pψ, alg; trunc=trunc_product)
    m2 = -log2(real(sproduct(P2, P2))) - N
    return m2
end
### SRE2 OF A BRICKWORK CIRCUIT APPLIED ON AN MPS ψ

function sre2(arrU::Vector{<:AbstractMatrix}, ψ::MPS, alg::Symbol; trunc_bw = NamedTuple(), trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(ψ)
    ψf = apply_brickwork(arrU, ψ; trunc=trunc_bw)
    Pψ = get_pauli_mps(ψf; trunc=trunc_pauli)
    W = MPO(Pψ)
    P2 = product(W, Pψ, alg; trunc=trunc_product)
    m2 = -log2(real(sproduct(P2, P2))) - N
    return m2
end




function ChainRulesCore.rrule(::typeof(sre2), arrU::Vector{<:AbstractMatrix}, ψ::MPS, alg::Symbol; trunc_bw = NamedTuple(), trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(ψ)
    ψf, apply_brickwork_back = pullback((Uarr, psi) -> apply_brickwork(Uarr, ψ; trunc=trunc_bw), arrU, ψ)
    Pψ, get_pauli_mps_pullback = pullback(psi -> get_pauli_mps(psi; trunc=trunc_pauli), ψf)
    W, MPO_back = pullback(MPO, Pψ)
    P2, product_back = pullback((mpo, psi) -> product(mpo, psi, alg; trunc=trunc_product), W, Pψ)
    res, sproduct_back = pullback(sproduct, P2, P2)
    m2, m2_back = -log2(real(res))-N, Δm2 -> (NoTangent(), -Δm2/(log(2)*real(res)))

    function sre2_pullback(Δm2)
        _, Δres = m2_back(Δm2)

        ΔP2_1, ΔP2_2 = sproduct_back(Δres)
        ΔP2 = ΔP2_1 .+ ΔP2_2
        ΔW, ΔPψ_1 = product_back(ΔP2)
        (ΔPψ_2,) = MPO_back(ΔW)
        ΔPψ = ΔPψ_1 .+ ΔPψ_2
        (Δψ2,) = get_pauli_mps_pullback(ΔPψ)

        ΔarrU, Δψ = apply_brickwork_back(Δψ2)

        return (NoTangent(), ΔarrU, NoTangent(), NoTangent())
    end
    return m2, sre2_pullback
end

### M_lin

function m_lin(ψ::MPS, alg::Symbol; trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    N = length(ψ)
    # ψf = apply_brickwork(arrU, ψ; trunc=trunc_bw)
    Pψ = get_pauli_mps(ψ; trunc=trunc_pauli)
    W = MPO(Pψ)
    P2 = product(W, Pψ, alg; trunc=trunc_product)
    m = 1-real(sproduct(P2, P2))*2^N
    return m
end

function m_lin(arrU::Vector{<:AbstractMatrix}, ψ::MPS, alg::Symbol; trunc_bw = NamedTuple(), trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
    truncerr = Ref(0.0)
    post_factorize_callback(errs) = (truncerr[] += sum(errs))
    N = length(ψ)
    ψf = apply_brickwork(arrU, ψ; trunc=trunc_bw,post_factorize_callback)
    Pψ = get_pauli_mps(ψf; trunc=trunc_pauli,post_factorize_callback)
    W = MPO(Pψ)
    P2 = product(W, Pψ, alg; trunc=trunc_product,post_factorize_callback)
    m = 1-real(sproduct(P2, P2))*2^N
    err = truncerr[]*(2^N)
    println("truncerr: $err")
    flush(stdout)
    return m
end

# function ChainRulesCore.rrule(::typeof(m_lin), arrU::Vector{<:AbstractMatrix}, ψ::MPS, alg::Symbol; trunc_bw = NamedTuple(), trunc_pauli = NamedTuple(), trunc_product = NamedTuple())
#     N = length(ψ)
#     ψf, apply_brickwork_back = pullback((Uarr, psi) -> apply_brickwork(Uarr, ψ; trunc=trunc_bw), arrU, ψ)
#     Pψ, get_pauli_mps_pullback = pullback(psi -> get_pauli_mps(psi; trunc=trunc_pauli), ψf)
#     W, MPO_back = pullback(MPO, Pψ)
#     P2, product_back = pullback((mpo, psi) -> product(mpo, psi, alg; trunc=trunc_product), W, Pψ)
#     res, sproduct_back = pullback(sproduct, P2, P2)
#     m, m_back = 1-real(res)*2^N, Δm -> (NoTangent(), -Δm*2^N)

#     function m_lin_pullback(Δm)
#         _, Δres = m_back(Δm)

#         ΔP2_1, ΔP2_2 = sproduct_back(Δres)
#         ΔP2 = ΔP2_1 .+ ΔP2_2
#         ΔW, ΔPψ_1 = product_back(ΔP2)
#         (ΔPψ_2,) = MPO_back(ΔW)
#         ΔPψ = ΔPψ_1 .+ ΔPψ_2
#         (Δψ2,) = get_pauli_mps_pullback(ΔPψ)

#         ΔarrU, Δψ = apply_brickwork_back(Δψ2)

#         return (NoTangent(), ΔarrU, NoTangent(), NoTangent())
#     end
#     return m, m_lin_pullback
# end

# ### EXACT MAGIC COMPUTATION

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



### MPS COMPRESSION


# =========================================================
# 1. Environments (you likely already have these from `compress`)
# =========================================================

"""
Build cumulative L⁽⁰⁾[j], R⁽⁰⁾[j]: ⟨ψ|ϕ⟩ transfer matrices,
contracted up to (excluding) site j.
"""
function build_environments(ψ::MPS, ϕ::MPS)
    @assert siteinds(ψ) == siteinds(ϕ)
    @assert linkinds(ψ) != linkinds(ϕ)

    N = length(ψ)
    L0 = Vector{ITensor}(undef, N+1)   # L0[j] = contraction of sites 1..j-1
    R0 = Vector{ITensor}(undef, N+1)   # R0[j] = contraction of sites j+1..N
    L0[1] = ITensor(1)
    R0[N+1] = ITensor(1)
    for j in 1:N
        L0[j+1] = L0[j]*dag(ψ[j])*ϕ[j] 
    end
    for j in N:-1:1
        R0[j] = R0[j+1]*dag(ψ[j])*ϕ[j]
    end
    return L0, R0
end


"""
Σ_j = A_j† E_j* for j < N (free at the converged point; reused as Hessian's rank-1 correction)
Where E_j has been reshaped to an isometry of the same shape as A_j.
This currently only works for MPS ψ = (B1, ..., BN) and ϕ = (A1, ..., AN) orthogonalized at site N"
"""
function compute_sigma(ψ::MPS, ϕ::MPS, arrϕ::Vector{<:AbstractArray}, L0::Vector{ITensor}, R0::Vector{ITensor})
    N = length(ϕ)
    check_orthogonal(ϕ, N)

    # Construct environment of tensor Aj for each j by leaving a hole at Aj
    Ej_tensors = [L0[j]*dag(ψ[j])*R0[j+1] for j in 1:N]
    Ej = toMatricesLC(Ej_tensors; check_og=false) # Convert from ITensors to array of matrices

    Σ = [arrϕ[j]' * conj(Ej[j]) for j in 1:N-1]     # last site is not an isometry, doesn't have this term

    return Σ
end

# =========================================================
# 2. Hessian-vector product, O(N) via tagged ("one-defect") sweeps
# =========================================================

"""
Build L1[j], R1[j]: cumulative environments with exactly one ξ-defect inserted
somewhere to the left (L1) / right (R1) of site j.
The ψ is conjugated since the environment is computed from <ψ|ϕ>.
"""
function build_defect_environments(ψ::MPS, ϕ::MPS, ξ::Vector{<:AbstractArray}, L0::Vector{ITensor}, R0::Vector{ITensor})
    N = length(ψ)
    L1 = Vector{ITensor}(undef, N+1)
    R1 = Vector{ITensor}(undef, N+1)
    L1[1] = ITensor(0)
    R1[N+1] = ITensor(0)
    ξten = [ITensor(ξ[j], inds(ϕ[j])) for j in eachindex(ξ)]
    for j in 1:N
        L1[j+1] = L1[j]*dag(ψ[j])*ϕ[j] + L0[j]*dag(ψ[j])*ξten[j]
    end
    for j in N:-1:1
        R1[j] = R1[j+1]*dag(ψ[j])*ϕ[j] + R0[j+1]*dag(ψ[j])*ξten[j]
    end
    return L1, R1
end

"""
Riemannian Hessian-vector product: Hess f(A,B)[ξ] = {P_{A_j}(Σ_k≠j E_j(A;A_k→ξ_k)) - ξ_j Σ_j}_j
where ψ = (B1, ..., BN) and ϕ = (A1, ..., AN). Currently works only if orthogonality center of ϕ is at N.
"""
function hessian_vector_product(ψ::MPS, ϕ::MPS, arrϕ::Vector{<:AbstractArray}, ξ::Vector{<:AbstractArray}, L0::Vector{ITensor}, R0::Vector{ITensor}, Σ::Vector{<:AbstractArray})
    N = length(ψ)
    check_orthogonal(ϕ, N)

    L1, R1 = build_defect_environments(ψ, ϕ, ξ, L0, R0)
    
    # Construct the defect environment of the tensor Aj for each j (leaving a hole at Aj)
    Aj_defect_envs_tensors = [L1[j]*dag(ψ[j])*R0[j+1] + L0[j]*dag(ψ[j])*R1[j+1] for j in eachindex(ψ)]
    Aj_defect_envs = toMatricesLC(Aj_defect_envs_tensors; check_og=false)   # converts ITensors to array of matrices

    Hξ = similar(ξ)     # Hessian vector product will be a tangent vector
    for j in 1:N-1      # first the term that comes by differentiating the projector on Aj
        Hξ[j] = 2*ξ[j]*Σ[j]
    end
    Hξ[N] = 2*ξ[N]
    Hξ -= 2*projectLC(arrϕ, conj(Aj_defect_envs))   # then the projector of the derivative of the environment Ej

    return Hξ
end

# =========================================================
# 3. Block-diagonal (site-local) preconditioner
# =========================================================

"""
Block-diagonal (Jacobi) preconditioner for the Hessian.

The site-diagonal block of H is exactly the curvature term:
  M_j[ξ_j] = 2 ξ_j Σ_j          (j < N,  right-multiplication)
  M_N[ξ_N] = 2 ξ_N              (center: the norm-term identity)
We store (2Σ_j)^{-1} per isometric site so that applying M_j^{-1} is one
small (χ' × χ') matrix multiply. Σ_j is Hermitian PSD at the optimum; we
symmetrize and floor the eigenvalues in absolute value to guarantee the
preconditioner is SPD (required for PCG) and robust to small off-optimum noise.
"""
function build_preconditioner_blocks(arrA, Σ; reg = 1e-8)
    N = length(arrA)
    blocks = Vector{Any}(undef, N)
    for j in 1:N-1
        S = 2 .* Σ[j]
        S = (S + S') / 2                          # Hermitian up to roundoff at convergence
        vals, vecs = eigen(Hermitian(S))
        scale = maximum(abs, vals)
        floor = reg * max(scale, one(scale))
        invvals = 1 ./ max.(abs.(vals), floor)    # abs ⇒ SPD even if slightly indefinite
        blocks[j] = vecs * Diagonal(invvals) * vecs'   # (2Σ_j)^{-1}, regularized
    end
    blocks[N] = nothing                            # center handled analytically in apply
    return blocks
end

"""
Apply M^{-1} to a tangent vector ξ. Solves M_j[out_j] = ξ_j site-by-site:
  out_j = ξ_j (2Σ_j)^{-1}       (j < N)
  out_N = ξ_N / 2               (center)
"""
function apply_preconditioner(blocks, ξ)
    N = length(ξ)
    out = similar(ξ)
    for j in 1:N-1
        out[j] = ξ[j] * blocks[j]
    end
    out[N] = ξ[N] / 2
    return out
end

# =========================================================
# 4. Warm start (Rayleigh-quotient scaled — no call history available)
# =========================================================

function warm_start(ψ, ψA, arrψA, ΔarrψA, L0, R0, Σ)
    HΔarrψA = hessian_vector_product(ψ, ψA, arrψA, ΔarrψA, L0, R0, Σ)
    μ = innerLC(ΔarrψA, HΔarrψA) / innerLC(ΔarrψA, ΔarrψA)
    μ = abs(μ) < 1e-12 ? one(μ) : μ  # guard against degenerate Rayleigh quotient
    return [-a/μ for a in ΔarrψA]
end

# =========================================================
# 5. Preconditioned CG (matrix-free), self-adjoint H
# =========================================================

function pcg_solve(hvp, precond, b, x0; tol=1e-8, maxiter=100)
    x = deepcopy(x0)
    r = b .- hvp(x)
    z = precond(r)
    p = deepcopy(z)
    rz_old = innerLC(r, z)
    b_norm = max(sqrt(innerLC(b,b)), 1e-30)

    for iter in 1:maxiter
        sqrt(innerLC(r,r)) / b_norm < tol && return x, iter
        Hp = hvp(p)
        α = rz_old / innerLC(p, Hp)
        x = x .+ α .* p
        r = r .- α .* Hp
        z = precond(r)
        rz_new = innerLC(r, z)
        β = rz_new / rz_old
        p = z .+ β .* p
        rz_old = rz_new
    end
    return x, maxiter
end

# =========================================================
# 6. Mixed partial adjoint (D_B g)* [λ], then project onto T_B
# =========================================================

"""
(D_B g(A,B))*[λ]: same "one-defect" sweep structure as the Hessian, but the defect
is inserted into the B-slot (held open) instead of propagated into another A_k.
"""
function mixed_partial_adjoint(ψ::MPS, arrψ::Vector{<:AbstractArray}, ϕ::MPS, λ::Vector{<:AbstractArray}, L0::Vector{ITensor}, R0::Vector{ITensor})
    N = length(ψ)
    L1, R1 = build_defect_environments(ψ, ϕ, λ, L0, R0)
    λten = [ITensor(λ[j], inds(ϕ[j])) for j in 1:N]
    # Construct the defect environment of the tensor Bk for each k (leaving a hole at Bk)
    Bk_defect_envs_tensors = [L1[k]*ϕ[k]*R0[k+1] + L0[k]*ϕ[k]*R1[k+1] + L0[k]*λten[k]*R0[k+1] for k in 1:N]
    # last term is the j=k diagonal term, which was absent in the hessian since differentiating twice by the same Aj gives zero
    # instead here we can have a hole both in Ak and Bk for the same k
    Bk_defect_envs = toMatricesLC(Bk_defect_envs_tensors; check_og=false)   # convert to matrices

    ΔB = -2*Bk_defect_envs  # for all j = 1, ..., N
    # we DO NOT project on Grassmann tangent space here, as all the other functions are defined for Stiefel
    return ΔB
end

# =========================================================
# 7. The rrule itself
# =========================================================

function compress(ψ::MPS, χ::Int; maxiter::Int = 10000, gradtol::Float64 = 1e-8, verbosity::Int = 2)
    N = length(ψ)
    sites = siteinds(ψ)

    # Build warm start to speed up compression
    ψapprox = move_center(ψ, 1)
    ψapprox = move_center(ψapprox, N; trunc=(maxrank=χ,))
    arrA0 = toMatricesLC(ψapprox)

    cost_func = arrA::Vector{<:AbstractArray} -> begin
        ψA = MPS(arrA, N; sites = sites)
        return snorm(ψA)^2 - 2*real(sproduct(ψ, ψA))
    end

    fg = arrA::Vector{<:AbstractArray} -> begin
        func, grad = withgradient(cost_func, arrA)
        grad = projectLC(arrA, grad[1])
        return func, grad
    end

    m = 5
    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = verbosity)

    # optimize and store results
    arrAmin, fmin, gradmin, numfg, normgradhistory = optimize(fg, arrA0, algorithm; 
                                                            retract = retractLC, 
                                                            transport! = transportLC!, 
                                                            isometrictransport = true, 
                                                            inner = innerLC)
    
    ψcompr = MPS(arrAmin, N; sites=sites)
    convergence_info = (fmin=fmin, gradmin=gradmin, numfg=numfg, normgradhistory=normgradhistory)

    return (ψcompr, convergence_info)
end

function compress_pullback(Δψcompr::Vector{ITensor}, ψ::MPS, ψcompr::MPS; gradtol = 1e-8)
    N = length(ψ)
    ψA = copy(ψcompr)
    ΔψA = copy(Δψcompr)

    arrψ = toMatricesLC(ψ)
    arrψA = toMatricesLC(ψA)
    ΔarrψA = toMatricesLC(ΔψA; check_og=false)

    projΔarrψA = projectLC(arrψA, ΔarrψA)
    #normal_comp = norm(projΔarrψA - ΔarrψA)
    #normal_comp > 1e-12 && @warn "compress_pullback expects a tangent incoming adjoint ΔψA, but a normal component was found.\nnorm(projΔψA - ΔψA) = $(normal_comp).\nProjecting onto tangent space."
    ΔarrψA = projΔarrψA

    L0, R0 = build_environments(ψ, ψA)
    Σ = compute_sigma(ψ, ψA, arrψA, L0, R0)

    hvp(ξ::Vector{<:AbstractMatrix}) = hessian_vector_product(ψ, ψA, arrψA, ξ, L0, R0, Σ)
    blocks = build_preconditioner_blocks(arrψA, Σ; reg=1e-8)  # see note in step 3
    precond(r) = apply_preconditioner(blocks, r)

    λ0 = warm_start(ψ, ψA, arrψA, ΔarrψA, L0, R0, Σ)
    λ, _ = pcg_solve(hvp, precond, -ΔarrψA, λ0; tol = gradtol)

    Δarrψ = mixed_partial_adjoint(ψ, arrψ, ψA, λ, L0, R0)
    Δψ = toITensors(Δarrψ, N; check_og=false, sites=siteinds(ψ), links=linkinds(ψ))
    return Δψ
end


function ChainRulesCore.rrule(::typeof(compress), ψ::MPS, χ::Int; maxiter::Int = 10000, gradtol::Float64 = 1e-8, verbosity::Int = 2)
    N = length(ψ)
    sites = siteinds(ψ)

    # Build warm start to speed up compression
    ψapprox = move_center(ψ, 1)
    ψapprox = move_center(ψapprox, N; trunc=(maxrank=χ,))   # TODO: replace with randomized svd
    arrA0 = toMatricesLC(ψapprox)

    ψ, back_og = Zygote.pullback(move_center, ψ, N) # needed cause the fixed point is solved in left canonical form

    cost_func = arrA::Vector{<:AbstractMatrix} -> begin
        ψA = MPS(arrA, N; sites = sites)
        return snorm(ψA)^2 - 2*real(sproduct(ψ, ψA))
    end

    fg = arrA::Vector{<:AbstractMatrix} -> begin
        func, grad = withgradient(cost_func, arrA)
        grad = projectLC(arrA, grad[1])
        return func, grad
    end

    m = 5
    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = verbosity)

    # optimize and store results
    arrAmin, fmin, gradmin, numfg, normgradhistory = optimize(fg, arrA0, algorithm; 
                                                            retract = retractLC, 
                                                            transport! = transportLC!, 
                                                            isometrictransport = true, 
                                                            inner = innerLC)
    
    ψcompr = MPS(arrAmin, N; sites=sites)    
    convergence_info = (fmin=fmin, gradmin=gradmin, numfg=numfg, normgradhistory=normgradhistory)

    function compress_pullback_Zygote(Δout)
        Δψcompr, _ = Δout
        Δψ = compress_pullback(Δψcompr, ψ, ψcompr; gradtol)
        (Δψ,) = back_og(Δψ) 
        return (NoTangent(), Δψ, NoTangent())
    end

    return (ψcompr, convergence_info), compress_pullback_Zygote
end


"""
TODO: normalize option after each compression, saving the discarded norm in between
"""
function apply_brickwork_variational(Uarray::Vector{<:AbstractMatrix}, ψ::MPS, χ::Int; kwargs...)
    N = length(ψ)
    Uarrs = group(Uarray, N, 2)
    lognorm_factors = Float64[]
    for (batch_no, arr) in enumerate(Uarrs)
        ψt = apply_brickwork(arr, ψ; normalize=false, 
                                    shift=Int(isodd(compression_depth)),
                                    to_right=iseven(compression_depth) || (isodd(compression_depth) && iseven(batch_no)), 
                                    kwargs...)
        ψtcompr, convergence_info = compress(ψt, χ)
        ψtcompr_normalized, logn = normalize_logn(ψtcompr)
        ψ = ψtcompr_normalized
        push!(lognorm_factors, logn)
    end

    return ψ, lognorm_factors
end


function ChainRulesCore.rrule(::typeof(apply_brickwork_variational), Uarray::Vector{<:AbstractMatrix}, ψ::MPS, χ::Int; kwargs...)
    N = length(ψ)
    compression_depth = 2       # number of layers after which to compress
    Uarrs = group(Uarray, N, compression_depth)
    lognorm_factors = Float64[]
    pullback_apply_brickworks = []
    pullback_lognorms = []
    pullback_compress = []

    for (batch_no, arr) in enumerate(Uarrs)
        apply_brickwork_local = (Uarr, ϕ) -> apply_brickwork(Uarr, ϕ; 
                                                normalize=false, 
                                                shift=Int(isodd(compression_depth)),
                                                to_right=iseven(compression_depth) || (isodd(compression_depth) && iseven(batch_no)), 
                                                kwargs...)
        ψt, back_brick = pullback(apply_brickwork_local, arr, ψ)
        push!(pullback_apply_brickworks, back_brick)

        (ψtcompr, convergence_info), back_compress = pullback(compress, ψt, χ)
        push!(pullback_compress, back_compress)

        (ψtcompr_normalized, logn), back_logn = pullback(normalize_logn, ψtcompr)
        push!(lognorm_factors, logn)
        push!(pullback_lognorms, back_logn)

        ψ = ψtcompr_normalized
    end

    function apply_brickwork_variational_pullback(Δout)
        Δψ, Δlognorm_factors = Δout
        Δψ = copy(Δψ)
        ΔUarrs::Vector{Vector{Matrix{ComplexF64}}} = []

        for j in eachindex(Uarrs)
            k = lastindex(Uarrs) - j + 1
            Δψtcompr_normalized = Δψ
            (Δψtcompr,) = pullback_lognorms[k]((Δψtcompr_normalized, Δlognorm_factors[k]))
            (Δψt,) = pullback_compress[k]((Δψtcompr, NoTangent()))
            (Δarr, Δψ) = pullback_apply_brickworks[k](Δψt)
            push!(ΔUarrs, Δarr)
        end
        ΔUarray = reduce(vcat, reverse(ΔUarrs))

        return (NoTangent(), ΔUarray, Δψ, NoTangent())
    end

    return (ψ, lognorm_factors), apply_brickwork_variational_pullback
end