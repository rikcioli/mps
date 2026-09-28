using ITensors, ITensorMPS
using KrylovKit
using SparseArrays, LinearAlgebra
using Plots
using HDF5
using LaTeXStrings
using Printf
using Statistics
ITensors.set_warn_order(28)

function entropy!(psi::MPS, b::Integer)  
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log2(p)
    end
    return SvN
end

function spectrum!(psi::MPS, b::Integer)
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)

    spec = diag(Matrix{Float64}(S, inds(S)))
    return spec
end

function H_spin(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hy::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jx,"Sx",j,"Sx",j+1
        os += Jy,"Sy",j,"Sy",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hx,"Sx",j
        os += hy,"Sy",j
        os += hz,"Sz",j
    end
    os += hx,"Sx",N
    os += hy,"Sy",N
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_spin(sites, Jxy::Real, Jz::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jxy/2,"S+",j,"S-",j+1
        os += Jxy/2,"S-",j,"S+",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hz,"Sz",j
    end
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_XY(sites, g::Real, hx::Real)
    return H_spin(sites, -(1+g), -(1-g), 0., hx, 0., 0.) 
end

function H_heisenberg(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hz::Real)
    return H_spin(sites, Jx, Jy, Jz, hx, 0., hz)
end

function H_heisenberg(sites, Jxy::Real, Jz::Real, hz::Real)
    return H_spin(sites, Jxy, Jz, hz)
end

function initialize_gs(H::MPO, sites; nsweeps = 5, maxdim = [10,20,100,100,200], cutoff = 1e-15, linkdims=2, kwargs...)
    psi0 = random_mps(ComplexF64, sites; linkdims=linkdims)
    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff,kwargs...)
    return energy, psi
end

function XXZ(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_heisenberg(sites, -1., -1., -0.5, -0.1, -0.1)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function XY(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_XY(sites, 0.0, 0.5)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function ising(N::Int)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end

function ising(N::Int, g::Real)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., g)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end

function H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.0, hz=0.0)
    N = length(sites)
    os = OpSum()
    for j in 1:N-1
        os += Jxy/2, "S+", j, "S-", j+1
        os += Jxy/2, "S-", j, "S+", j+1     # same coefficient => Hermitian
        os += Jz,    "Sz", j, "Sz", j+1
    end
    for j in 1:N
        os += -hs*(-1)^j, "Sz", j           # STAGGERED — this is what was missing
        os += hz,         "Sz", j
    end
    return MPO(os, sites)
end


function transfer_matrix(ψ::MPS, b::Int)
    sites = siteinds(ψ)
    T = dag(ψ[b])' * delta(dag(sites[b]), sites[b]') * ψ[b]
    return T
end

function transfer_matrix(ψ::MPS)
    sites = siteinds(ψ)
    Tlist = [dag(ψ[i])' * delta(dag(sites[i]), sites[i]') * ψ[i] for i in eachindex(ψ)]
    return Tlist
end


"Computes polar decomposition but only returning P (efficient)"
function polar_P(block::Vector{ITensor}, sites::Vector{<:Index})
    M = ITensor(1.0)
    for j in eachindex(block)
        M *= block[j] * delta(dag(sites[j]), sites[j]') * dag(block[j])'
    end

    low_inds = filter(i -> plev(i) == 0, inds(M))

    # Hermitian eigendecomposition: M ≈ U' * D * dag(U), done block by block
    D, U = eigen(M, prime.(low_inds), low_inds; ishermitian=true)

    # square root of the eigenvalues, clamping tiny negative values from rounding
    sqrtD = ITensors.map_diag(x -> sqrt(max(real(x), 0.0)), D)

    return U' * sqrtD * dag(U)
end



"""
Four-index tensor T(lL, lR, lL', lR') = Σ_s B(lL,s,lR) conj(B(lL',s,lR'))
for the block of sites j..j+q-1. Reshaped one way it is B†B (polar
decomposition); reshaped the other way it is the blocked transfer matrix.
"""
function blockT(ψ::MPS, j::Int, q::Int)
    phi = ITensorMPS.orthogonalize(ψ, length(ψ))
    T = phi[j] * dag(prime(phi[j], "Link"))
    for k in (j + 1):(j + q - 1)
        T = T * phi[k] * dag(prime(phi[k], "Link"))
    end
    return T, linkind(phi, j - 1), linkind(phi, j + q - 1)
end


Ttrial, iL, iR = blockT(ψ, 100, 100)
cL, cR = combiner(iL, iL'), combiner(iR, iR')
cLind, cRind = combinedind(cL), combinedind(cR)
Tmat = Matrix{ComplexF64}(cL*Ttrial*cR, (cLind, cRind))
evals, evecs = eigen(Tmat)
evals

pos_1 = findall(x -> isapprox(x, 1.0; atol=1e-10), evals)
right_evec = evecs[:, pos_1]

rho = reshape(right_evec, (dim(iR), dim(iR)))
alpha = 1/tr(rho)
sqrt_rho = sqrt(alpha*rho)

"Given a transfer matrix T and its left and right indices iL, iR,
computes the right eigenstate of the transfer matrix B otimes B^*, and returns
the square root of its matrix form"
function extract_rho(T::ITensor, iL, iR)

    # Convert T to matrix and diagonalize to extract right eigenvector of eigenval=1
    T_matrix = reshape(Array{ComplexF64}(T, iL, iL', iR, iR'), (dim(iL)^2, dim(iR)^2))

    # Extract eigenvalues and right eigenvectors
    eig = eigen(T_matrix)
    pos_1 = findmax(abs.(eig.values))
    if pos_1[1] < 0.99
        throw(DomainError(pos_1[1], "Max eigenvalue less than 1"))
    end
    pos_1 = pos_1[2]
    right_eig = eig.vectors[:, pos_1]

    # Reshape eigenvec and rescale to unit norm
    # the result is an operator that acts vertically, and must act on bell pairs as (Id \otimes \sqrt{\rho})
    rho = reshape(right_eig, (iR.space, iR.space))
    alpha = 1/tr(rho)
    sqrt_rho = sqrt(alpha*rho)

    return sqrt_rho
end



"""
    connected_corr(psi, op1, op2; i0=nothing)

Connected correlator C(r) = ⟨op1_i0 op2_(i0+r)⟩ - ⟨op1_i0⟩⟨op2_(i0+r)⟩
for r = 1:N-i0, computed from a single reference site i0 (default: center).
Returns (rs, C).
"""
function connected_corr(psi::MPS, op1::String, op2::String; i0::Union{Nothing,Int}=nothing)
    N = length(psi)
    i0 = isnothing(i0) ? div(N, 2) : i0

    corrmat = correlation_matrix(psi, op1, op2)
    onept1 = expect(psi, op1)
    onept2 = expect(psi, op2)

    rs = 1:(N - i0)
    C = [corrmat[i0, i0 + r] - onept1[i0] * onept2[i0 + r] for r in rs]
    return collect(rs), C
end

"""
    connected_corr_avg(psi, op1, op2; rmax)

Connected correlator averaged over all reference sites i0, for distances
r = 1:rmax (default N/4), to reduce noise from a single reference site.
Returns (rs, Cavg).
"""
function connected_corr_avg(psi::MPS, op1::String, op2::String; rmax::Union{Nothing,Int}=nothing)
    N = length(psi)
    rmax = isnothing(rmax) ? div(N, 4) : rmax

    corrmat = correlation_matrix(psi, op1, op2)
    onept1 = expect(psi, op1)
    onept2 = expect(psi, op2)

    rs = 1:rmax
    Cavg = Float64[]
    for r in rs
        vals = [real(corrmat[i, i + r] - onept1[i] * onept2[i + r]) for i in 1:(N - r)]
        push!(Cavg, mean(vals))
    end
    return collect(rs), Cavg
end






# ---------------------------------------------------------------------------
# Load MPS from testdata/xxz/Jz2.5 for several system sizes and plot how the
# Sz-Sz connected correlator decays with distance (semi-log => exponential
# decay = short range, straight line on log-log => power law = long range).
# ---------------------------------------------------------------------------

sizes = 60:40:300
plt = plot(xlabel="Distance r", ylabel="|C(r)|", yscale=:log10,
           title="Connected ⟨Sz Sz⟩ correlator, XXZ Jz=2.5", legend=:topright)

for N in sizes
    f = h5open("testdata\\xxz\\Jz2.5\\$(N)_mps.h5", "r")
    psi = read(f, "psi", MPS)
    close(f)

    rs, C = connected_corr_avg(psi, "Sz", "Sz")
    plot!(plt, rs, abs.(C), marker=:circle, ms=2, label="N=$N")
end
display(plt)



sizes = 20:20:100
plt = plot(xlabel="Distance r", ylabel="|C(r)|", yscale=:log10,
           title="Connected ⟨Sz Sz⟩ correlator, XXZ Jz=2.5", legend=:topright)

for N in sizes
    sites = siteinds("S=1/2", N; conserve_qns=true)
    H = H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.05)
    psi0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n=1:N])
    #psi0 = random_mps(ComplexF64, sites; linkdims=2)

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    rs, C = connected_corr_avg(psi, "Sz", "Sz")
    plot!(plt, rs, abs.(C), marker=:circle, ms=2, label="N=$N")
end
display(plt)



sizes = 20:20:100
plt = plot(xlabel="Distance r", ylabel="|C(r)|", yscale=:log10,
           title="Connected ⟨Sz Sz⟩ correlator, XXZ Jz=2.5", legend=:topright)

for N in sizes
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    rs, C = connected_corr_avg(psi, "Sz", "Sz")
    plot!(plt, rs, abs.(C), marker=:circle, ms=2, label="N=$N")
end
display(plt)




### EXTRACT ALPHA AND BETA FOR CAT STATES


# P = ∏ X_j applied to an MPS (needs non-QN indices, since X changes S^z)
function flip_all(psi::MPS)
    phi = copy(psi)
    for j in eachindex(phi)
        s = siteind(phi, j)
        phi[j] = noprime(2 * op("Sx", s) * phi[j])
    end
    return phi
end

psi_d = dense(psi)          # drop QNs so X is allowed
H_d   = dense(H)
Ppsi  = flip_all(psi_d)

for (sgn, name) in ((+1, "+"), (-1, "-"))
    raw = add(psi_d, sgn * Ppsi; cutoff=1e-12)
    w = norm(raw)
    phi = normalize(raw)
    Pexp = real(inner(phi, flip_all(phi)))
    E = real(inner(phi', H_d, phi))
    println("sector $name: weight = $(w^2/4), <P> = $Pexp, E = $E")
    @show maxlinkdim(phi)
    ξ, ev = corr_length(phi; ncell=2)
    @show ev[1]-ev[2]
end
println("E(DMRG) = ", real(inner(psi_d', H_d, psi_d)))
maxlinkdim(psi_d)


"""
Four-index tensor T(lL, lR, lL', lR') = Σ_s B(lL,s,lR) conj(B(lL',s,lR'))
for the block of sites j..j+q-1. Reshaped one way it is B†B (polar
decomposition); reshaped the other way it is the blocked transfer matrix.
"""
function block_gram(phi::MPS, j::Int, q::Int)
    T = phi[j] * dag(prime(phi[j], "Link"))
    for k in (j + 1):(j + q - 1)
        T = T * phi[k] * dag(prime(phi[k], "Link"))
    end
    return T, linkind(phi, j - 1), linkind(phi, j + q - 1)
end

"""
    corr_length_blocked(psi; qs=1:20, j=nothing, eig_pos=1, floor=1e-12)

For each block size q, builds the blocked transfer matrix E^B between bonds
j-1 and j+q-1 and computes its singular values. r(q) = σ_{eig_pos+1}/σ₁
measures ‖Δ‖ in E^B ≈ fixed-point part + Δ (eig_pos = 1 injective, 2 for a cat).
Local estimates ξ(q) = -Δq / log(r(q+Δq)/r(q)) should converge as q grows.
Dense SVD of a D_L² × D_R² matrix: use for D ≲ 50.
"""
function corr_length_blocked(psi::MPS; qs=1:20, j=nothing, eig_pos::Int=1,
                             nsv::Int=eig_pos + 3, floor::Float64=1e-12)
    phi = ITensorMPS.orthogonalize(dense(psi), length(psi))     # left-canonical, no QNs
    N = length(phi)
    jj = isnothing(j) ? N ÷ 2 - maximum(qs) ÷ 2 : j
    (jj ≥ 2 && jj + maximum(qs) ≤ N) || error("block does not fit in the chain")

    qdone, ratios, svals = Int[], Float64[], Vector{Vector{Float64}}()
    for q in qs
        T, lL, lR = block_gram(phi, jj, q)
        dL, dR = dim(lL), dim(lR)
        max(dL, dR) > 60 && @warn "D = $(max(dL, dR)): dense SVD will be slow"
        M = reshape(Array(T, lL, lL', lR, lR'), dL^2, dR^2)   # transfer-matrix grouping
        s = svdvals(M)
        s = s[1:min(nsv, length(s))]
        r = s[eig_pos + 1] / s[1]
        push!(qdone, q); push!(ratios, r); push!(svals, s)
        r < floor && break
    end

    ξloc = [-(qdone[i + 1] - qdone[i]) / log(ratios[i + 1] / ratios[i])
            for i in 1:(length(ratios) - 1)]
    return (; q=qdone, ratios, ξloc, svals)
end

"""
Fit |C(r)| ≈ A r^(-p) exp(-r/ℓ). Pass p to fix the power-law prefactor
(recommended), or p=nothing to fit it. Keeps points with rmin ≤ r and
|C| > floor, to avoid short-distance effects and numerical noise.
"""
function fit_decay(r, C; p=nothing, rmin=2, floor=1e-11)
    r = collect(Float64, r); y = abs.(collect(C))
    keep = (r .>= rmin) .& (y .> floor)
    r, y = r[keep], log.(y[keep])
    length(r) ≥ 3 || error("too few points above floor")
    if p === nothing
        X = hcat(ones(length(r)), -r, -log.(r))
        a, invℓ, pfit = X \ y
    else
        X = hcat(ones(length(r)), -r)
        a, invℓ = X \ (y .+ p .* log.(r))
        pfit = p
    end
    # local decay lengths with the prefactor removed: should plateau
    ℓloc = [1 / (-(y[i+1]-y[i]) / (r[i+1]-r[i]) - pfit*log(r[i+1]/r[i]) / (r[i+1]-r[i]))
            for i in 1:length(r)-1]
    return (; ℓ = 1/invℓ, p = pfit, A = exp(a), r, ℓloc)
end





N = 100; g=1.0
sites = siteinds("S=1/2", N; conserve_szparity=true)
H = H_spin(sites, -1., 0., 0., 0., 0., -g)
psi0 = MPS(sites,"Up")

E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
           cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])

res = corr_length_blocked(psi; qs=1:25, eig_pos=1)
for (q, r, ξ) in zip(res.q[2:end], res.ratios[2:end], res.ξloc)
    println("q = $q   σ₂/σ₁ = $(round(r, sigdigits=3))   ξ(q) = $(round(ξ, digits=4))")
end
1/log(2*g)


plt = plot(xlabel="Distance r", ylabel="|C(r)|", yscale=:log10,
           title="Connected ⟨Sz Sz⟩ correlator, Ising g=3", legend=:topright)
rlist, corrs = connected_corr(psi, "Sz", "Sz")
plot!(plt, rlist[1:40], abs.(corrs[1:40]), marker=:circle, ms=2, label="N=$N")

# --- with your Sz correlator: decay length is ξ/2 ---
fz = fit_decay(rlist, corrs; p=2)
println("ξ from SzSz  = ", 2fz.ℓ)
println("local ξ(r)   = ", round.(2 .* fz.ℓloc, digits=4))

# --- recommended: Sx correlator, decay length is ξ ---
ψd = dense(psi)
Cxx = correlation_matrix(ψd, "Sx", "Sx")
i0 = length(ψd) ÷ 2 - 15
rx = 1:30
cx = [Cxx[i0, i0 + d] for d in rx]      # ⟨Sx⟩ = 0 in the paramagnet
fx = fit_decay(rx, cx; p=0.5)
println("ξ from SxSx  = ", fx.ℓ)
println("local ξ(r)   = ", round.(fx.ℓloc, digits=4))




# block array
q = 30
D = maxlinkdim(psi)
tensors_list, tensors_sites = blocking(ψ, q)
newN = length(tensors_list)
# polar decomp and store P matrices in array
blockMPS = [polar_P(tensors_list[i], tensors_sites[i]) for i in eachindex(tensors_list)]

# save linkinds and create new siteinds
block_linkinds = linkinds(ψ)[q:q:end]
block_linkinds_dims = [dim(ind) for ind in block_linkinds]
        
new_siteinds = siteinds(D, 2*(newN-1))

# replace primed linkinds with new siteinds
replaceind!(blockMPS[1], block_linkinds[1]', new_siteinds[1])
replaceind!(blockMPS[end], block_linkinds[end]', new_siteinds[end])
for i in 2:(newN-1)
    replaceind!(blockMPS[i], block_linkinds[i-1]', new_siteinds[2*i-2])
    replaceind!(blockMPS[i], block_linkinds[i]', new_siteinds[2*i-1])
end




### TUNABLE CORR LENGTH 

#=
D = 2, d = 2 MPS class with tunable correlation length (arXiv:2307.01696, Eq. S39-S40)

    A^0 = [0 0; 1 1],   A^1 = [1 g; 0 0]      (bulk sites j = 2..N-1)
    boundary tensors = 2x2 identity           (A_[1]^{i}_b = δ_{i,b},  A_[N]^{i}_a = δ_{a,i})

Transfer matrix E = Σ_i A^i ⊗ A^i has nonzero eigenvalues 1 ± g, so
    ξ = 1 / |ln((1-g)/(1+g))|    <=>    g = tanh(1/(2ξ))
=#


g_from_xi(ξ::Real) = tanh(1 / (2ξ))
xi_from_g(g::Real) = 1 / abs(log((1 - g) / (1 + g)))

"""
    correlated_mps(sites, g; L = I(2), R = I(2), normalize = true)

Build the open-boundary MPS of Eq. (S39)-(S40).
Physical value i ∈ {0,1} maps to site state 1,2 (for "Qubit" sites: |0>,|1>;
for "S=1/2" sites: Up, Dn).

`L[i, b]` is the left boundary tensor A_[1]^{i}_b and `R[a, i]` the right one A_[N]^{i}_a;
both default to the 2x2 identity as in the paper.
"""
function correlated_mps(sites::Vector{<:Index}, g::Number;
                        L = Matrix(1.0I, 2, 2), R = Matrix(1.0I, 2, 2),
                        normalize::Bool = true)
    N = length(sites)
    N >= 2 || error("Need N ≥ 2")
    all(dim.(sites) .== 2) || error("All site indices must have dimension 2")
    D = 2
    T = promote_type(typeof(g), eltype(L), eltype(R), Float64)

    A = zeros(T, D, 2, D)                # A[a, i, b] = (A^{i-1})_{ab}
    A[:, 1, :] = [0 0; 1 1]              # A^0
    A[:, 2, :] = [1 g; 0 0]              # A^1

    links = [Index(D, "Link,l=$j") for j in 1:(N - 1)]
    psi = MPS(N)
    psi[1] = ITensor(T.(L), sites[1], links[1])
    for j in 2:(N - 1)
        psi[j] = ITensor(A, links[j - 1], sites[j], links[j])
    end
    psi[N] = ITensor(T.(R), links[N - 1], sites[N])

    normalize && normalize!(psi)
    return psi
end

correlated_mps_xi(sites, ξ::Real; kwargs...) = correlated_mps(sites, g_from_xi(ξ); kwargs...)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

"Nonzero transfer-matrix eigenvalues (should be 1+g and 1-g)."
function transfer_spectrum(g)
    A0 = [0 0; 1 1.0]
    A1 = [1 g; 0 0.0]
    E = kron(A0, conj(A0)) + kron(A1, conj(A1))
    λ = sort(eigvals(E); by = abs, rev = true)
    return λ
end


N = 100
sites = siteinds("Qubit", N)
psi = correlated_mps_xi(sites, 4)

res = corr_length_blocked(psi; qs=1:25, eig_pos=1)
for (q, r, ξ) in zip(res.q[2:end], res.ratios[2:end], res.ξloc)
    println("q = $q   σ₂/σ₁ = $(round(r, sigdigits=3))   ξ(q) = $(round(ξ, digits=4))")
end

for xi in []



### ED STUFF

const PAULI = Dict(
    "Sz" => sparse([0.5 0.0; 0.0 -0.5]),
    "Sx" => sparse([0.0 0.5; 0.5 0.0]),
    "Sy" => sparse([0.0 -0.5im; 0.5im 0.0]),
    "S+" => sparse([0.0 1.0; 0.0 0.0]),
    "S-" => sparse([0.0 0.0; 1.0 0.0]),
    "Id" => sparse(1.0I, 2, 2),
)
 
# op(name, site, N): the operator `name` on `site` (1-based), identity elsewhere,
# as a 2^N x 2^N sparse matrix.
function op(name::String, site::Int, N::Int)
    mats = [k == site ? PAULI[name] : PAULI["Id"] for k in 1:N]
    M = mats[1]
    for k in 2:N
        M = kron(M, mats[k])
    end
    return M
end
 
# two-site operator name1_site1 * name2_site2, identity elsewhere.
function op2(name1::String, site1::Int, name2::String, site2::Int, N::Int)
    mats = [PAULI["Id"] for _ in 1:N]
    mats[site1] = PAULI[name1]
    mats[site2] = PAULI[name2]         # assumes site1 != site2, true for all our terms
    M = mats[1]
    for k in 2:N
        M = kron(M, mats[k])
    end
    return M
end
 
"""
    xxz_sparse(N; Jxy=1.0, Jz=2.0, hs=0.0) -> SparseMatrixCSC
 
Exactly the Hamiltonian from xxz_mpo, as a sparse matrix, built without going
through ITensors at all.
"""
function xxz_sparse(N::Int; Jxy = 1.0, Jz = 2.0, hs = 0.0)
    dim = 2^N
    H = spzeros(ComplexF64, dim, dim)
    for j in 1:(N - 1)
        H += (Jxy / 2) * op2("S+", j, "S-", j + 1, N)
        H += (Jxy / 2) * op2("S-", j, "S+", j + 1, N)
        H += Jz        * op2("Sz", j, "Sz", j + 1, N)
    end
    if hs != 0.0
        for j in 1:N
            H += (-hs * (-1)^j) * op("Sz", j, N)
        end
    end
    return (H + H') / 2
end
 
function ED_sparse(N::Int; Jxy = 1.0, Jz = 2.0, hs = 0.0, nev = 5)
    H = xxz_sparse(N; Jxy = Jxy, Jz = Jz, hs = hs)
    println("dim = ", size(H,1), "   nnz = ", nnz(H),
            "   density = ", nnz(H) / size(H,1)^2)
    vals, vecs, info = eigsolve(H, size(H, 1), nev, :SR;
                                ishermitian = true, krylovdim = 30, tol = 1e-13)
    println("Ground state energy: ", real(vals[1]))
    nev > 1 && println("Gap E1 - E0:         ", real(vals[2] - vals[1]))
    return vals, vecs, info
end
 
# self-test against the known-exact numbers used throughout this thread
function selftest()
    for (N, E_ref) in [(12, -7.2049809970), (14, -8.4756888020)]
        vals, _, _ = ED_sparse(N; hs = 0.05, nev = 1)
        err = abs(real(vals[1]) - E_ref)
        println("  N=$N  err=$err  ", err < 1e-8 ? "OK" : "MISMATCH")
    end
end
 
selftest()




N = 140
sites = siteinds("S=1/2", N)
H = H_heisenberg(sites, 1.0, 1.0, 2.5, 0.0)
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0 = random_mps(ComplexF64, sites; linkdims=2)

E0, psi = dmrg(H, psi0; nsweeps=10, cutoff=1e-12)
xis, ratios = corr_lengths(psi; ncell=2)
sz = expect(psi, "Sz");
StagSz = sum((-1)^(n+1) * sz[n] for n in 1:N)/N

plot(1:N, sz, marker=:circle, xlabel="Site", ylabel="⟨Sz⟩", title="Sz expectation value", legend=false)

Evals, Evecs, conv_info = ED_sparse(N; Jxy = 1.0, Jz = 2.5, hs = 0.0, nev = 5)
Evals[1]
Evals[2] - Evals[1]
Evals[3] - Evals[1]
[Evecs[1]'*op("Sz",i,20)*Evecs[1] for i in 1:20]