using ITensors, ITensorMPS
using LinearAlgebra

function random_unitary(D::Int)
    x = (randn(D,D) + randn(D,D)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end 

"""
    renormalize_mps(psi, q; cutoff=1e-14)

Blocks `psi` q sites at a time (the last block takes the remainder if q ∤ N). Each block tensor A
(physical legs × links (lL, lR)) is polar-decomposed A = W P, with P = √(A†A) mapping (lL, lR) to
new site legs (uL, uR) of the same dimensions; the isometries W are dropped. The P's, contracted along
the original links, form a state with the same norm and the same entanglement across block boundaries.
Bulk P's are split by SVD into (uL, lL) | (uR, lR), so the result is an MPS with sites
uR₁, uL₂, uR₂, …, uL_nb, where each pair (2i-1, 2i) straddles a block boundary (dims = that link's dim).
"""
function renormalize_mps(psi::MPS, q::Integer; cutoff::Real = 1e-14)
    N = length(psi)
    ranges = [(i-1)*q+1 : min(i*q, N) for i in 1:cld(N, q)]
    nblocks = length(ranges)
    nblocks > 1 || throw(ArgumentError("need at least two blocks, got N=$N, q=$q"))
    links = linkinds(psi)
    sites = siteinds(psi)

    rg_mps = ITensor[]
    for (i, rng) in enumerate(ranges)
        # Free links of the block: lL = link to the previous block, lR = to the next one
        lows = Index[]
        i > 1 && push!(lows, links[first(rng)-1])
        i < nblocks && push!(lows, links[last(rng)])

        # Transfer matrix E = Σ_s A[s] conj(A[s])', i.e. A†A with primed links as row indices (all links primed on conj copy)
        E = ITensor(1.0)
        for j in rng
            E *= psi[j] * prime(dag(psi[j]), filter(!=(sites[j]), collect(inds(psi[j]))))
        end
        dims = dim.(lows)
        dimP = prod(dims)
        M = reshape(Array{ComplexF64}(E, prime.(lows)..., lows...), dimP, dimP)
        F = eigen(Hermitian(M))
        P = F.vectors * Diagonal(sqrt.(max.(F.values, 0))) * F.vectors'

        # New site legs uL, uR: same dimensions as the corresponding links
        ups = [Index(dim(l), "Site,n=$(length(rg_mps) + k)") for (k, l) in enumerate(lows)]
        Pten = ITensor(reshape(P, dims..., dims...), ups..., lows...)

        if i == 1 || i == nblocks
            push!(rg_mps, Pten)
        else
            U, S, V = svd(Pten, (ups[1], lows[1]); cutoff, lefttags = "Link,l=$(length(rg_mps)+1)")
            push!(rg_mps, U)
            push!(rg_mps, S * V)
        end
    end

    return MPS(rg_mps)
end

"""
    invert(ψ, arrU0; maxiter=1000, tol=1e-12, verbose=false)

Finds the depth-1 layer of 2-qudit gates (acting on pairs (1,2), (3,4), ...) that maximizes
|⟨ψ| ⊗ᵢ Uᵢ |00⟩|. Gate i is a (d₂ᵢ₋₁d₂ᵢ)×(d₂ᵢ₋₁d₂ᵢ) matrix in kron convention (kron(A,B) = A on the left site).
Each gate is updated a la Evenbly-Vidal with all other gates fixed: the environment of gate i is
computed from cached left/right environments. Only the first column v = U|00⟩ enters the cost,
overlap = Σₖ e[k] v[k], so the optimal update is v = conj(e)/‖e‖ with overlap ‖e‖ (the
Evenbly-Vidal SVD of the rank-1 environment, without forming it). Sweeps back and forth.
For odd N the last site is left in |0⟩. Returns (arrU, |overlap|): the gates are completed
to unitaries with first column v only at the end.

The MPS tensors are contracted one at a time (never a pair tensor (link, s, s, link), which for
blocked states of bond dimension χ has χ²·χ²·d² entries: tens of GB for χ ~ 24).
"""
function invert(ψ::MPS, arrU0::Vector{<:AbstractMatrix}; maxiter::Int = 1000, tol::Real = 1e-12, verbose::Bool = false)
    N = length(ψ)
    npairs = N ÷ 2
    sites = siteinds(ψ)
    dims = dim.(sites)
    gdim(i) = dims[2i-1] * dims[2i]
    length(arrU0) == npairs || throw(ArgumentError("expected $npairs gates, got $(length(arrU0))"))
    all(size(arrU0[i]) == (gdim(i), gdim(i)) for i in 1:npairs) || throw(ArgumentError("gate i must be d₂ᵢ₋₁d₂ᵢ × d₂ᵢ₋₁d₂ᵢ"))

    ψ = ψ / norm(ψ)
    ψd = [dag(ψ[j]) for j in 1:N]
    # First columns U|00⟩ of the gates, the only part that enters the cost
    vs = [Vector{ComplexF64}(U[:, 1]) for U in arrU0]

    # The state of gate i as an ITensor on (s_{2i-1}, s_{2i}) (kron convention: s_{2i} is the fast index)
    gate_state(i) = ITensor(reshape(vs[i], dims[2i], dims[2i-1]), sites[2i], sites[2i-1])

    # L[i+1] = left environment of pairs 1..i, R[i] = right environment of pairs i..npairs (+ dangling site)
    L = Vector{ITensor}(undef, npairs + 1)
    R = Vector{ITensor}(undef, npairs + 1)
    L[1] = ITensor(1.0)
    R[npairs+1] = isodd(N) ? ψd[N] * onehot(sites[N] => 1) : ITensor(1.0)
    # left to right inside a product: the largest intermediate is (link, s, s)
    grow_left(i)  = L[i] * ψd[2i-1] * ψd[2i] * gate_state(i)
    grow_right(i) = R[i+1] * ψd[2i] * ψd[2i-1] * gate_state(i)
    for i in npairs:-1:1
        R[i] = grow_right(i)
    end

    # Evenbly-Vidal update of gate i, returns |overlap| after the update
    function update!(i)
        E = L[i] * ψd[2i-1] * ψd[2i] * R[i+1]
        e = vec(Array(E, sites[2i], sites[2i-1]))   # overlap = Σₖ e[k] v[k]
        n = norm(e)
        vs[i] = conj(e) / n
        return n
    end

    overlap = abs(scalar(L[1] * R[1]))
    verbose && println("Initial overlap: $overlap")
    for iter in 1:maxiter
        overlap_old = overlap
        # Right sweep
        for i in 1:npairs
            overlap = update!(i)
            L[i+1] = grow_left(i)
        end
        # Left sweep
        for i in npairs:-1:1
            overlap = update!(i)
            R[i] = grow_right(i)
        end
        verbose && println("Sweep $iter: overlap = $overlap, 1 - overlap = $(1 - overlap)")
        if abs(overlap - overlap_old) < tol
            break
        end
    end

    # complete every state to a unitary with that first column
    arrU = map(vs) do v
        U = qr(reshape(v, :, 1)).Q * Matrix{ComplexF64}(I, length(v), length(v))
        U[:, 1] .*= dot(U[:, 1], v)          # Q's first column is v up to a phase
        U
    end
    return arrU, overlap
end

function blocking_scaling(ψ::MPS, qlist = [2, 4, 8, 16, 32]; kargs...)
    err_list = []
    for q in qlist
        ψrg = renormalize_mps(ψ, q)
        newN = length(ψrg)
        sites = siteinds(ψrg)
        arrU0 = [random_unitary(dim(sites[2i-1]) * dim(sites[2i])) for i in 1:newN÷2]

        ## Start variational optimization to approximate P blocks
        arrU, overlap = invert(ψrg, arrU0; kargs...)
        push!(err_list, abs(1 - overlap))
    end
    return err_list
end


function blocking_scaling_err(ψ::MPS, err::Float64; qstart=2, kargs...)
    q = qstart; found = false
    while !found
        ψrg = renormalize_mps(ψ, q)
        newN = length(ψrg)
        sites = siteinds(ψrg)
        arrU0 = [random_unitary(dim(sites[2i-1]) * dim(sites[2i])) for i in 1:newN÷2]

        ## Start variational optimization to approximate P blocks
        arrU, overlap = invert(ψrg, arrU0; kargs...)
        err_q = abs(1 - overlap)
        if err_q < err
            found = true
        else
            q += 1
        end
    end
    return q
end