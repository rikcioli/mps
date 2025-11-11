
### CUSTOM ITENSOR METHODS

"Converts order-4 ITensor to N x N matrix following the order of indices in 'order'"
function tensor_to_matrix(T::ITensor, order = [1,2,3,4])
    inds = inds(T)
    ord_inds = [inds[i] for i in order]
    M = reshape(Array(T, ord_inds), (ord_inds[1].space^2, ord_inds[1].space^2))
    return M
end

"Extends m x n isometry to M x M unitary, where M is the power of 2 which bounds max(m, n) from above"
function iso_to_unitary(V)
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


"Computes full polar decomposition"
function polar(block::ITensor, inds::Vector{Index{Int64}})
    spaceU = reduce(*, [ind.space for ind in inds])
    spaceV = reduce(*, [ind.space for ind in uniqueinds(block, inds)])
    #if spaceU > spaceV
    #    throw("Polar decomposition of block failed: make sure link space is bigger than physical space")
    #end
    U, S, V = svd(block, inds)
    u, v = inds(S)
    
    Vdag = replaceind(conj(V)', v', u)
    P = Vdag * S * V

    V = replaceind(V', v', u)
    W = U * V

    return W, P
end

"Computes polar decomposition but only returning P (efficient)"
function polar_P(block::Vector{ITensor}, inds::Vector{Index{Int64}})
    q = length(block)
    block_conj = [conj(tens)' for tens in block]
    left_tensor = block[1] * block_conj[1] * delta(inds[1], inds[1]')
    for j in 2:q
        left_tensor *= block[j] * block_conj[j] * delta(inds[j], inds[j]')
    end

    low_inds = reduce(noncommoninds, [block; inds])
    dim = reduce(*, [ind.space for ind in low_inds])
    mat = reshape(Array(left_tensor, (low_inds', low_inds)), (dim, dim))
    P = sqrt(mat)
    P = ITensor(P, (low_inds', low_inds))
    return P
end





### CUSTOM MPS METHODS ###

"Apply 2-qubit matrix U to sites b and b+1 of MPS psi"
function apply(U::Matrix, psi::MPS, b::Integer; cutoff = 1E-14)
    psi = deepcopy(psi)
    orthogonalize!(psi, b)
    s = siteinds(psi)
    operator = op(U, s[b+1], s[b])

    wf = noprime((psi[b]*psi[b+1])*operator)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
    return psi
end

"Apply 2-qubit matrix U to sites b and b+1 of MPS psi"
function apply!(U::Matrix, psi::MPS, b::Integer; cutoff = 1E-15)
    orthogonalize!(psi, b)
    s = siteinds(psi)
    operator = op(U, s[b+1], s[b])

    wf = noprime((psi[b]*psi[b+1])*operator)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
end

"Apply 2-qubit ITensors.ITensor U to sites b and b+1 of Vector{ITensors.ITensor} psi"
function contract!(psi::Vector{ITensor}, U::ITensor, b; cutoff = 1E-15)
    summed_inds = commoninds(U, unioninds(psi[b], psi[b+1]))
    outinds = uniqueinds(U, summed_inds)
    W = U*psi[b]*psi[b+1]
    replaceinds!(W, outinds, summed_inds)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(W, indsb, cutoff = cutoff)
    psi[b] = replaceinds(U, summed_inds, outinds)
    psi[b+1] = replaceinds(S*V, summed_inds, outinds)
end

"Multiply two mps/mpo respecting their indices"
function contract(psi1::Union{MPS, Vector{ITensor}}, psi2::Union{MPS, Vector{ITensor}})
    left_tensor = psi1[1] * psi2[1]
    N = length(psi1)
    for i in 2:N
        left_tensor *= psi1[i]
        left_tensor *= psi2[i]
    end
    return left_tensor
end


function entropy(psi::MPS, b::Integer)  
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb, cutoff = 1E-15)
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    return SvN
end

"Truncate mps at position b until bond dimension of link (b, b+1) becomes 1"
function cut!(psi::MPS, b::Integer)
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b]*psi[b+1], indsb, cutoff = 1E-18)

    u, v = inds(S)
    w = Index(1)
    projU = ITensor([1; [0 for _ in 1:u.space-1]], (u,w))
    projV = ITensor([1; [0 for _ in 1:v.space-1]], (w,v))
    psi[b] = U*projU
    psi[b+1] = projV*V
end


function brickwork(psi::MPS, brick_odd::Matrix, brick_even::Matrix, t::Integer; cutoff = 1E-15)
    N = length(psi)
    sites = siteinds(psi)
    layerOdd = [op(brick_odd, sites[b+1], sites[b]) for b in 1:2:(N-1)]
    layerEven = [op(brick_even, sites[b+1], sites[b]) for b in 2:2:(N-1)]
    for i in 1:t
        layer = isodd(i) ? layerOdd : layerEven
        psi = ITensors.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end


"Apply depth-t brickwork of 2-local random unitaries"
function brickwork(psi::MPS, t::Integer; cutoff = 1E-15)
    N = length(psi)
    sites = siteinds(psi)
    d = sites[1].space
    for i in 1:t
        layer = [op(random_unitary(d^2), sites[b+1], sites[b]) for b in (isodd(i) ? 1 : 2):2:(N-1)]
        psi = ITensors.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end

function initialize_vac(N::Integer, sites = nothing)
    if isnothing(sites)
        sites = siteinds("Qubit", N)
    end
    states = ["0" for _ in 1:N]
    vac = MPS(sites, states)
    return vac
end

function initialize_ghz(N::Integer)
    ghz = initialize_vac(N)
    brick = CX * kron(H, Id)
    ghz = apply(brick, ghz, 1)
    for i in 2:N-1
        ghz = apply(CX, ghz, i)
    end
    return ghz
end

function initialize_fdqc(N::Integer, tau::Integer; kargs...)
    fdqc = initialize_vac(N)
    fdqc = brickwork(fdqc, tau)
    return fdqc
end

function initialize_fdqc(N::Integer, tau::Integer, brick_odd::Matrix, brick_even::Matrix; kargs...)
    fdqc = initialize_vac(N)
    fdqc = brickwork(fdqc, brick_odd, brick_even, tau)
    return fdqc
end

function H_spin(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hy::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jx,"Sx",j,"Sz",j+1
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

function H_ising(sites, J::Real, hx::Real, hz::Real)::MPO
    return H_spin(sites, 0., 0., J, hx, 0., hz)
end

function H_XY(sites, g::Real, hx::Real)
    return H_spin(sites, -(1+g), -(1-g), 0., hx, 0., 0.) 
end

function H_heisenberg(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hz::Real)
    return H_spin(sites, Jx, Jy, Jz, hx, 0., hz)
end

function initialize_gs(H::MPO, sites; nsweeps = 5, maxdim = [10,20,100,100,200], cutoff = 1e-15, linkdims=2, kwargs...)
    psi0 = random_mps(sites; linkdims=linkdims)
    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff,kwargs...)
    return energy, psi
end

function initialize_ising(N::Integer, hx::Real, hz::Real; nsweeps = 10, maxdim = 200, cutoff = 1e-12)
    sites = siteinds("Qubit",N)

    os = OpSum()
    for j=1:N-1
      os += -1.,"Z",j,"Z",j+1
      os += -hx,"X",j
      os += -hz,"Z",j
    end
    os += -hx,"X",N
    os += -hz,"Z",N
    H = MPO(os,sites)

    psi0 = random_mps(sites; linkdims=2)

    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    return energy, psi
end

function initialize_2Dheisenberg(Nx::Integer, Ny::Integer, Jx::Real, Jy::Real, Jz::Real; nsweeps = 10)

    N = Nx*Ny
    sites = siteinds("Qubit",N)

    # Obtain an array of LatticeBond structs
    # which define nearest-neighbor site pairs
    # on the 2D square lattice
    lattice = square_lattice(Nx, Ny)

    os = OpSum()
    for b in lattice
        os += -Jz, "Z", b.s1, "Z", b.s2
        os += -Jx, "X", b.s1, "X", b.s2
        os += -Jy, "Y", b.s1, "Y", b.s2
    end
    H = MPO(os,sites)

    psi0 = random_mps(sites; linkdims=2)
    maxdim = [64]
    cutoff = [1E-15]

    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    return energy, psi
end

# function initialize_fdqc(N::Integer, tau::Integer, lightbounds, gate = nothing, d = 2)
#     mps = initialize_vac(N)
#     sites = siteinds(mps)
#     lc = newLightcone(sites, tau, U_array = U_array, lightbounds = lightbounds)
#     prime!(mps, tau)
#     contract!(mps, lc)
#     mps = noprime(mps)
#     return mps
# end

"Returns mps of Haar random isometries with bond dimension D"
function randMPS(sites::Vector{<:Index}, D::Integer)
    N = length(sites)
    d = sites[1].space

    mps = MPS(sites, linkdims = D)
    links = linkinds(mps)

    U0 = ITensor(random_unitary(d*D), (sites[1], links[1]', links[1], sites[1]'))
    U_list = [ITensor(random_unitary(d*D), (sites[i], links[i-1], links[i], sites[i]')) for i in 2:N-1]
    UN = ITensor(random_unitary(d*D), (sites[N], links[N-1], links[N-1]', sites[N]'))

    zero_projs::Vector{ITensor} = [ITensor([1; [0 for _ in 2:d]], site') for site in sites]
    zero_L = ITensor([1; [0 for _ in 2:D]], links[1]')
    zero_R = ITensor([1; [0 for _ in 2:D]], links[N-1]')

    U0 *= zero_L
    UN *= zero_R
    U_list = [U0; U_list; UN]

    tensors = [zero_projs[i]*U_list[i] for i in 1:N]

    for i in 1:N
        mps[i] = tensors[i]
    end
    orthogonalize!(mps, div(N,2))
    normalize!(mps)
    orthogonalize!(mps, N)   # never touch again
    return mps
end

"Returns N-qubit random MPS with bond dimension D"
function randMPS(N::Integer, D::Integer)
    return randMPS(siteinds("Qubit", N), D)
end


"Project sites indexed by 'positions' array to zero. Normalizes at the end"
function project_tozero(psi::MPS, positions::Vector{Int64})
    psi = deepcopy(psi)
    sites = siteinds(psi)
    for b in positions
        orthogonalize!(psi, b)
        ind = sites[b]
        zero_vec = [1; [0 for _ in 1:ind.space-1]]
        zero_proj = ITensor(kron(zero_vec, zero_vec'), ind, ind')
        new_psib = psi[b]*zero_proj
        norm_psi = real(Array(new_psib * conj(new_psib))[1])
        psi[b] = noprime(new_psib/sqrt(norm_psi))
    end
    return psi
end


"Project sites indexed by 'positions' array to zero. Normalizes at the end"
function project_tozero!(psi::MPS, positions::Vector{Int64})
    sites = siteinds(psi)
    for b in positions
        orthogonalize!(psi, b)
        ind = sites[b]
        zero_vec = [1; [0 for _ in 1:ind.space-1]]
        zero_proj = ITensor(kron(zero_vec, zero_vec'), ind, ind')
        new_psib = psi[b]*zero_proj
        norm_psi = real(Array(new_psib * conj(new_psib))[1])
        psi[b] = noprime(new_psib/sqrt(norm_psi))
    end
end




### CUSTOM MPO METHODS ###
"Convert unitary to MPO via repeated SVD"
function unitary_to_mpo(U::Union{Matrix, Vector{AbstractFloat}}; d = 2, sites = nothing, skip_qudits = 0, orthogonalize = true)
    D = size(U, 1)
    N = length(digits(D-1, base=d))     # N = logd(D)

    if isnothing(sites)
        sites = siteinds(d, N)
        skip_qudits = 0
    else # check that you're not skipping too many qubit
        N + skip_qudits > length(sites) &&
            throw(DomainError(skip_qudits, "Skipping too many qubits for the siteinds given."))
    end


    sites_active = sites[1+skip_qudits:N+skip_qudits]
    mpo::Vector{ITensor} = []

    block = ITensor(U, sites_active', sites_active)
    local link
    for i in 1:N-1
        Uinds = (i==1 ? (sites_active[i]', sites_active[i]) : (link, sites_active[i]', sites_active[i]))
        Us, Ss, Vsdag = svd(block, Uinds, cutoff = 1e-16)
        push!(mpo, Us)
        link = commonind(Us, Ss)
        block = Ss*Vsdag
    end
    push!(mpo, block)
    

    # add identities left and right for every siteind that is not in sites_active
    left_deltas::Vector{ITensor} = []
    right_deltas::Vector{ITensor} = []
    for left_site in sites[1:skip_qudits]
        push!(left_deltas, delta(left_site', left_site))
    end
    for right_site in sites[N+1+skip_qudits:end]
        push!(right_deltas, delta(right_site', right_site))
    end

    mpo = [left_deltas; mpo; right_deltas]
    mpo_final = MPO(mpo)
    if orthogonalize
        orthogonalize!(mpo_final, length(sites))
    end

    return mpo_final
end




"multiply Vector{ITensors.ITensor} object together. Indices have to match in advance." 
function Base.:*(mpo1::Vector{ITensor}, mpo2::Vector{ITensor})
    if length(mpo1) != length(mpo2)
        throw(DomainError, "The two objects have different length")
    end
    N = length(mpo1)
    mpo = [mpo1[i]*mpo2[i] for i in 1:N]
    for j in 1:N-1
        linkinds = commoninds(mpo[j], mpo[j+1])
        if length(linkinds) > 1
            combiner = combiner(linkinds)
            mpo[j] *= combiner
            mpo[j+1] *= combiner
        end
    end

    return mpo
end


### RG METHODS ###

"Performs blocking on an MPS, q sites at a time. Returns the blocked MPS as an array of it, together with the siteinds."
function blocking(mps::Union{Vector{ITensor},MPS}, q::Integer)
    N = length(mps)
    newN = div(N, q)
    r = mod(N, q)

    block_mps = [mps[q*(i-1)+1 : q*i] for i in 1:newN] 
    if r != 0
        push!(block_mps, r > 1 ? mps[q * newN : end] : [mps[end]])
    end

    sites = reduce(noncommoninds, mps[1:end])
    sitegroups = [sites[q*(i-1)+1 : q*i] for i in 1:newN]
    if r != 0
        push!(sitegroups, r > 1 ? sites[q * newN : end] : [sites[end]])
        newN += 1
    end

    return block_mps, sitegroups
end


"Given a block B and a 3-Tuple containing the left, central and right indices,
computes the right eigenstate of the transfer matrix B otimes B^*, and returns
the square root of its matrix form"
function extract_rho(block::ITensor, inds::NTuple{3, Index{Int64}})
    iL, i, iR = inds[1:3]
    block_star = conj(block)'
    delta_i = delta(i, i')

    # Combine into transfer matrix
    T = delta_i * block * block_star

    # Convert T to matrix and diagonalize to extract right eigenvector of eigenval=1
    T_matrix = reshape(Array(T, iL, iL', iR, iR'), (iL.space^2, iR.space^2))

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