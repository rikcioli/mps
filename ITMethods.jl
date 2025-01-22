
### CUSTOM ITENSOR METHODS

"Converts order-4 ITensor to N x N matrix following the order of indices in 'order'"
function tensor_to_matrix(T::it.ITensor, order = [1,2,3,4])
    inds = it.inds(T)
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
function polar(block::it.ITensor, inds::Vector{it.Index{Int64}})
    spaceU = reduce(*, [ind.space for ind in inds])
    spaceV = reduce(*, [ind.space for ind in it.uniqueinds(block, inds)])
    #if spaceU > spaceV
    #    throw("Polar decomposition of block failed: make sure link space is bigger than physical space")
    #end
    U, S, V = it.svd(block, inds)
    u, v = it.inds(S)
    
    Vdag = it.replaceind(conj(V)', v', u)
    P = Vdag * S * V

    V = it.replaceind(V', v', u)
    W = U * V

    return W, P
end

"Computes polar decomposition but only returning P (efficient)"
function polar_P(block::Vector{it.ITensor}, inds::Vector{it.Index{Int64}})
    q = length(block)
    block_conj = [conj(tens)' for tens in block]
    left_tensor = block[1] * block_conj[1] * it.delta(inds[1], inds[1]')
    for j in 2:q
        left_tensor *= block[j] * block_conj[j] * it.delta(inds[j], inds[j]')
    end

    low_inds = reduce(it.noncommoninds, [block; inds])
    dim = reduce(*, [ind.space for ind in low_inds])
    mat = reshape(Array(left_tensor, (low_inds', low_inds)), (dim, dim))
    P = sqrt(mat)
    P = it.ITensor(P, (low_inds', low_inds))
    return P
end





### CUSTOM MPS METHODS ###

"Apply 2-qubit matrix U to sites b and b+1 of MPS psi"
function apply(U::Matrix, psi::itmps.MPS, b::Integer; cutoff = 1E-14)
    psi = deepcopy(psi)
    it.orthogonalize!(psi, b)
    s = it.siteinds(psi)
    op = it.op(U, s[b+1], s[b])

    wf = it.noprime((psi[b]*psi[b+1])*op)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
    return psi
end

"Apply 2-qubit matrix U to sites b and b+1 of MPS psi"
function apply!(U::Matrix, psi::itmps.MPS, b::Integer; cutoff = 1E-15)
    it.orthogonalize!(psi, b)
    s = it.siteinds(psi)
    op = it.op(U, s[b+1], s[b])

    wf = it.noprime((psi[b]*psi[b+1])*op)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
end

"Apply 2-qubit ITensors.ITensor U to sites b and b+1 of Vector{ITensors.ITensor} psi"
function contract!(psi::Vector{it.ITensor}, U::it.ITensor, b; cutoff = 1E-15)
    summed_inds = it.commoninds(U, it.unioninds(psi[b], psi[b+1]))
    outinds = it.uniqueinds(U, summed_inds)
    W = U*psi[b]*psi[b+1]
    it.replaceinds!(W, outinds, summed_inds)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(W, indsb, cutoff = cutoff)
    psi[b] = it.replaceinds(U, summed_inds, outinds)
    psi[b+1] = it.replaceinds(S*V, summed_inds, outinds)
end

"Multiply two mps/mpo respecting their indices"
function contract(psi1::Union{itmps.MPS, itmps.MPO, Vector{it.ITensor}}, psi2::Union{itmps.MPS, itmps.MPO, Vector{it.ITensor}})
    left_tensor = psi1[1] * psi2[1]
    for i in 2:length(psi1)
        left_tensor *= psi1[i]
        left_tensor *= psi2[i]
    end
    return left_tensor
end


function entropy(psi::itmps.MPS, b::Integer)  
    it.orthogonalize!(psi, b)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(psi[b], indsb, cutoff = 1E-15)
    SvN = 0.0
    for n in 1:it.dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    return SvN
end

"Truncate mps at position b until bond dimension of link (b, b+1) becomes 1"
function cut!(psi::itmps.MPS, b::Integer)
    it.orthogonalize!(psi, b)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(psi[b], indsb, cutoff = 1E-15)

    u, v = it.inds(S)
    w = it.Index(1)
    projU = it.ITensor([1; [0 for _ in 1:u.space-1]], (u,w))
    projV = it.ITensor([1; [0 for _ in 1:v.space-1]], (w,v))
    psi[b] = U*projU*projV*V
end


function brickwork(psi::itmps.MPS, brick_odd::Matrix, brick_even::Matrix, t::Integer; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    layerOdd = [it.op(brick_odd, sites[b+1], sites[b]) for b in 1:2:(N-1)]
    layerEven = [it.op(brick_even, sites[b+1], sites[b]) for b in 2:2:(N-1)]
    for i in 1:t
        layer = isodd(i) ? layerOdd : layerEven
        psi = it.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end


"Apply depth-t brickwork of 2-local random unitaries"
function brickwork(psi::itmps.MPS, t::Int; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    d = sites[1].space
    for i in 1:t
        layer = [it.op(random_unitary(d^2), sites[b+1], sites[b]) for b in (isodd(i) ? 1 : 2):2:(N-1)]
        psi = it.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end

function initialize_vac(N::Int, siteinds = nothing)
    if isnothing(siteinds)
        siteinds = it.siteinds("Qubit", N)
    end
    states = ["0" for _ in 1:N]
    vac = itmps.MPS(siteinds, states)
    return vac
end

function initialize_ghz(N::Int)
    ghz = initialize_vac(N)
    brick = CX * kron(H, Id)
    ghz = apply(brick, ghz, 1)
    for i in 2:N-1
        ghz = apply(CX, ghz, i)
    end
    return ghz
end

function initialize_fdqc(N::Int, tau::Int)
    fdqc = initialize_vac(N)
    fdqc = brickwork(fdqc, tau)
    return fdqc
end

function initialize_fdqc(N::Int, tau::Int, brick_odd::Matrix, brick_even::Matrix)
    fdqc = initialize_vac(N)
    fdqc = brickwork(fdqc, brick_odd, brick_even, tau)
    return fdqc
end

# function initialize_fdqc(N::Int, tau::Int, lightbounds, gate = nothing, d = 2)
#     mps = initialize_vac(N)
#     siteinds = it.siteinds(mps)
#     lc = newLightcone(siteinds, tau, U_array = U_array, lightbounds = lightbounds)
#     it.prime!(mps, tau)
#     contract!(mps, lc)
#     mps = it.noprime(mps)
#     return mps
# end

"Returns mps of Haar random isometries with bond dimension D"
function randMPS(sites::Vector{<:it.Index}, D::Int)
    N = length(sites)
    d = sites[1].space

    mps = itmps.MPS(sites, linkdims = D)
    links = it.linkinds(mps)

    U0 = it.ITensor(random_unitary(d*D), (sites[1], links[1]', links[1], sites[1]'))
    U_list = [it.ITensor(random_unitary(d*D), (sites[i], links[i-1], links[i], sites[i]')) for i in 2:N-1]
    UN = it.ITensor(random_unitary(d*D), (sites[N], links[N-1], links[N-1]', sites[N]'))

    zero_projs::Vector{it.ITensor} = [it.ITensor([1; [0 for _ in 2:d]], site') for site in sites]
    zero_L = it.ITensor([1; [0 for _ in 2:D]], links[1]')
    zero_R = it.ITensor([1; [0 for _ in 2:D]], links[N-1]')

    U0 *= zero_L
    UN *= zero_R
    U_list = [U0; U_list; UN]

    tensors = [zero_projs[i]*U_list[i] for i in 1:N]

    for i in 1:N
        mps[i] = tensors[i]
    end
    it.orthogonalize!(mps, div(N,2))
    it.normalize!(mps)
    it.orthogonalize!(mps, N)   # never touch again
    return mps
end

"Returns N-qubit random MPS with bond dimension D"
function randMPS(N::Int, D::Int)
    return randMPS(it.siteinds("Qubit", N), D)
end


"Project sites indexed by 'positions' array to zero. Normalizes at the end"
function project_tozero(psi::itmps.MPS, positions::Vector{Int64})
    psi = deepcopy(psi)
    sites = it.siteinds(psi)
    for b in positions
        it.orthogonalize!(psi, b)
        ind = sites[b]
        zero_vec = [1; [0 for _ in 1:ind.space-1]]
        zero_proj = it.ITensor(kron(zero_vec, zero_vec'), ind, ind')
        new_psib = psi[b]*zero_proj
        norm_psi = real(Array(new_psib * conj(new_psib))[1])
        psi[b] = it.noprime(new_psib/sqrt(norm_psi))
    end
    return psi
end


"Project sites indexed by 'positions' array to zero. Normalizes at the end"
function project_tozero!(psi::itmps.MPS, positions::Vector{Int64})
    sites = it.siteinds(psi)
    for b in positions
        it.orthogonalize!(psi, b)
        ind = sites[b]
        zero_vec = [1; [0 for _ in 1:ind.space-1]]
        zero_proj = it.ITensor(kron(zero_vec, zero_vec'), ind, ind')
        new_psib = psi[b]*zero_proj
        norm_psi = real(Array(new_psib * conj(new_psib))[1])
        psi[b] = it.noprime(new_psib/sqrt(norm_psi))
    end
end




### CUSTOM MPO METHODS ###
"Convert unitary to MPO via repeated SVD"
function unitary_to_mpo(U::Union{Matrix, Vector{Float64}}; d = 2, siteinds = nothing, skip_qudits = 0, orthogonalize = true)
    D = size(U, 1)
    N = length(digits(D-1, base=d))     # N = logd(D)

    if isnothing(siteinds)
        siteinds = it.siteinds(d, N)
        skip_qudits = 0
    else # check that you're not skipping too many qubit
        N + skip_qudits > length(siteinds) &&
            throw(DomainError(skip_qudits, "Skipping too many qubits for the siteinds given."))
    end


    sites = siteinds[1+skip_qudits:N+skip_qudits]
    mpo::Vector{it.ITensor} = []

    block = it.ITensor(U, sites', sites)
    local link
    for i in 1:N-1
        Uinds = (i==1 ? (sites[i]', sites[i]) : (link, sites[i]', sites[i]))
        Us, Ss, Vsdag = it.svd(block, Uinds, cutoff = 1e-16)
        push!(mpo, Us)
        link = it.commonind(Us, Ss)
        block = Ss*Vsdag
    end
    push!(mpo, block)
    

    # add identities left and right for every siteind that is not in sites
    left_deltas::Vector{it.ITensor} = []
    right_deltas::Vector{it.ITensor} = []
    for left_site in siteinds[1:skip_qudits]
        push!(left_deltas, it.delta(left_site', left_site))
    end
    for right_site in siteinds[N+1+skip_qudits:end]
        push!(right_deltas, it.delta(right_site', right_site))
    end

    mpo = [left_deltas; mpo; right_deltas]
    mpo_final = itmps.MPO(mpo)
    if orthogonalize
        it.orthogonalize!(mpo_final, length(siteinds))
    end

    return mpo_final
end




"multiply Vector{ITensors.ITensor} object together. Indices have to match in advance." 
function Base.:*(mpo1::Vector{it.ITensor}, mpo2::Vector{it.ITensor})
    if length(mpo1) != length(mpo2)
        throw(DomainError, "The two objects have different length")
    end
    N = length(mpo1)
    mpo = [mpo1[i]*mpo2[i] for i in 1:N]
    for j in 1:N-1
        linkinds = it.commoninds(mpo[j], mpo[j+1])
        if length(linkinds) > 1
            combiner = it.combiner(linkinds)
            mpo[j] *= combiner
            mpo[j+1] *= combiner
        end
    end

    return mpo
end


### RG METHODS ###

"Performs blocking on an MPS, q sites at a time. Returns the blocked MPS as an array of it, together with the siteinds."
function blocking(mps::Union{Vector{it.ITensor},itmps.MPS}, q::Int)
    N = length(mps)
    newN = div(N, q)
    r = mod(N, q)

    block_mps = [mps[q*(i-1)+1 : q*i] for i in 1:newN] 
    if r != 0
        push!(block_mps, r > 1 ? mps[q * newN : end] : [mps[end]])
    end

    siteinds = reduce(it.noncommoninds, mps[1:end])
    sitegroups = [siteinds[q*(i-1)+1 : q*i] for i in 1:newN]
    if r != 0
        push!(sitegroups, r > 1 ? siteinds[q * newN : end] : [siteinds[end]])
        newN += 1
    end

    return block_mps, sitegroups
end


"Given a block B and a 3-Tuple containing the left, central and right indices,
computes the right eigenstate of the transfer matrix B otimes B^*, and returns
the square root of its matrix form"
function extract_rho(block::it.ITensor, inds::NTuple{3, it.Index{Int64}})
    iL, i, iR = inds[1:3]
    block_star = conj(block)'
    delta_i = it.delta(i, i')

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