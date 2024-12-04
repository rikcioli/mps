module MPSMethods

import ITensorMPS as itmps
import ITensors as it
using LinearAlgebra, Statistics, Random

const H = [1 1
           1 -1]/sqrt(2)
const Id = [1 0
            0 1]
const T = [1 0
            0 exp(1im*pi/4)]
const X = [0 1
            1 0]
const CX = [1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0]

"Returns random N x N unitary matrix sampled with Haar measure"
function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end     




### LIGHTCONE METHODS ###

mutable struct Lightcone
    circuit::Vector{Vector{it.ITensor}}     # circuit[i] is the i-th layer, circuit[odd][1] acts on sites (1, 2), circuit[even][1] acts on sites (2, 3)
    d::Int64                                # dimension of the physical space
    size::Int64                             # number of qubits the whole circuit acts on
    depth::Int64                            # circuit depth
    lightbounds::Tuple{Bool, Bool}          # (leftslope == 'light', rightslope == 'light')
    sitesAB::Vector{it.Index}               # siteinds of AB region
    coords                                  # coordinates of all unitaries
    id_coords                               # coordinates of all identities
    layers                                  # coordinates of all gates ordered by layer
    range::Vector{Tuple{Int64, Int64}}      # leftmost and rightmost sites on which each layer acts non-trivially
end


function newLightcone(siteinds::Vector{<:it.Index}, depth; U_array = nothing, lightbounds = (true, true))

    # extract number of sites on which lightcone acts
    sizeAB = length(siteinds)
    isodd(sizeAB) &&
        throw(DomainError(sizeAB, "Choose an even number for sizeAB"))

    # count the minimum sizeAB has to be to construct a circuit of depth 'depth'
    n_light = count(lightbounds)    # count how many boundaries have 'light' type structure
    min_size = n_light*(depth+1) - 2*div(n_light,2)
    if sizeAB < min_size
        @warn "Cannot construct a depth $depth lightcone with lightbounds $lightbounds with only $sizeAB sites. Attempting to change lightbounds..."
        lightbounds = (true, false)
        n_light = count(lightbounds)    
        min_size = n_light*(depth+1) - 2*div(n_light,2)
        if sizeAB >= min_size
            @warn "Lightbounds changed to $lightbounds"
        else
            throw(DomainError(sizeAB, "Cannot construct a depth $depth lightcone with lightbounds $lightbounds with only $sizeAB sites."))
        end
    end
    
    # if U_array is not given, construct an array of random d-by-d unitaries
    d = siteinds[1].space
    if isnothing(U_array)
        n_unitaries = (sizeAB-1)*div(depth,2) + div(sizeAB,2)
        if depth > 2
            dep_m2 = depth-2
            n_unitaries -= n_light*(div(dep_m2+1,2)^2 + mod(dep_m2+1,2)*div(dep_m2+1,2))
        end
        U_array = [random_unitary(d^2) for _ in 1:n_unitaries]
    end

    # finally convert the U_array 
    circuit::Vector{Vector{it.ITensor}} = []
    coords = []
    id_coords = []
    layers_coords = []
    range = []

    k = 1
    for i in 1:depth
        layer_i::Vector{it.ITensor} = []
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1,2)
        lslope, rslope = llim, rlim
        lrange, rrange = 1+mod(i+1,2), sizeAB-mod(i+1,2)
        if lightbounds[1]
            lslope += div(i-1,2)
            lrange += 2*div(i-1,2)
        end
        if lightbounds[2]
            rslope -= div(i-1,2)
            rrange -= 2*div(i-1,2)
        end
        push!(range, (lrange, rrange))

        layer_i_coords = []
        # sweep over unitaries, from left to right if depth odd, otherwise from right to left
        # this will be helpful with the og center
        for j in (isodd(i) ? (llim:rlim) : (rlim:-1:llim))
            # prepare inds for current unitary
            inds = it.prime((siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1]), depth-i)
            fullinds = (inds[1], inds[2], inds[1]', inds[2]')

            # add a clause for boundary terms: for even layers, a delta should be placed at the extremities
            # we simulate this by increasing the lower index of the first and last gate of the odd layers
            # so that they are connected directly to the odd layer below
            if isodd(i) && i>1
                if j == llim
                    fullinds = (inds[1], inds[2], inds[1]'', inds[2]')
                elseif j == rlim
                    fullinds = (inds[1], inds[2], inds[1]', inds[2]'')
                end
            end

            # insert unitaries according to lightbounds, inserting identities if we are outside the lightcone
            if j < lslope || j > rslope
                brick = it.delta(inds[1], inds[1]') * it.delta(inds[2], inds[2]')
                push!(id_coords, ((i,j), fullinds))
            else
                brick = it.ITensor(U_array[k], fullinds)
                push!(coords, ((i,j), fullinds))
                k += 1
            end
            push!(layer_i, brick)
            push!(layer_i_coords, ((i,j), fullinds))
        end
        if iseven(i)
            reverse!(layer_i)
        end
        push!(circuit, layer_i)
        push!(layers_coords, layer_i_coords)
    end

    return Lightcone(circuit, d, sizeAB, depth, lightbounds, siteinds, coords, id_coords, layers_coords, range)

end

"Update Lightcone in-place"
function updateLightcone!(lightcone::Lightcone, U_array::Vector{<:Matrix})
    n_unitaries = length(U_array)
    if length(lightcone.coords) != n_unitaries
        throw(BoundsError(n_unitaries, "Number of updated unitaries does not match lightcone structure"))
    end

    for k in 1:n_unitaries
        (i,j), inds = lightcone.coords[k]
        U = U_array[k]
        U_tensor = it.ITensor(U, inds)
        lightcone.circuit[i][j] = U_tensor
    end
end

"Flatten Lightcone to 1D array, from bottom to top"
function Base.Array(lightcone::Lightcone)
    arr::Vector{Matrix} = []
    d = lightcone.d
    for ((i,j), inds) in lightcone.coords
        push!(arr, reshape(Array(lightcone.circuit[i][j], inds), (d^2,d^2)))
    end
    return arr
end

"Apply k-th unitary of lightcone to MPS psi. Lightcone.size must be equal to length(psi)"
function contract!(psi::itmps.MPS, lightcone::Lightcone, k::Int64; cutoff = 1E-15)
    l1, l2 = length(psi), lightcone.size
    if l1 != l2
        raise(DomainError(l1, "Cannot apply lightcone of size $l2 to mps of length $l1: the two lengths must be equal"))
    end
    (i,j), inds = lightcone.coords[k]
    U = lightcone.circuit[i][j]
    b = 2*j-mod(i,2)    # where to apply the unitary
    #it.orthogonalize!(psi, b+1)

    W = psi[b]*psi[b+1]
    suminds = it.commoninds(inds, W)
    outinds = it.uniqueinds(U, suminds)
    W *= U
    
    it.replaceinds!(W, outinds, suminds)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(W, indsb, cutoff = cutoff)
    psi[b] = it.replaceinds(U, suminds, outinds)
    psi[b+1] = it.replaceinds(S*V, suminds, outinds) 
end

"Apply lightcone to mps psi, from base to top. Lightcone.size must be equal to length(psi)"
function contract!(psi::itmps.MPS, lightcone::Lightcone; cutoff = 1E-15)
    for k in 1:length(lightcone.coords)
        contract!(psi, lightcone, k)
    end
end

"Apply vector of Lightcones to mps, from base to top. The lightcones positions are specified by the ints contained in initial_pos. Prime level of mps siteinds must be the same as lightcones'."
function contract!(mps::itmps.MPS, lightcones::Vector{Lightcone}, initial_pos::Vector{<:Int}; cutoff = 1E-15)
    
    n_lightcones = length(initial_pos)
    
    for l in 1:n_lightcones
        lc = lightcones[l]
        initpos = initial_pos[l]

        for k in 1:length(lc.coords)
            (i,j), inds = lc.coords[k]
            U = lc.circuit[i][j]
            b = initpos-1 + 2*j-mod(i,2)
            it.orthogonalize!(mps, b+1)
            norm0 = norm(mps)

            W = mps[b]*mps[b+1]
            suminds = it.commoninds(inds, W)
            outinds = it.uniqueinds(U, suminds)
            W *= U
            
            it.replaceinds!(W, outinds, suminds)
            indsb = it.uniqueinds(mps[b], mps[b+1])
            U, S, V = it.svd(W, indsb, cutoff = cutoff)
            mps[b] = it.replaceinds(U, suminds, outinds)
            mps[b+1] = it.replaceinds(S*V, suminds, outinds) 

            norm1 = norm(mps)
            if abs(norm0-norm1)>1E-14
                throw(DomainError(norm1, "MPS norm has changed during the application of lightcone of initpos $initpos"))
            end
        end
    end

end

"Convert lightcone to MPO"
function MPO(lightcone::Lightcone)
    # convert lightcone to mpo by using the ij_dict and the itmps.MPO(it.ITensor) function, keeping track of the deltas for even depths

    circ = lightcone.circuit
    depth = lightcone.depth
    siteinds = lightcone.siteinds

    mpo_list = []
    for layer in lightcone.layers
        qr_list::Vector{it.ITensor} = []
        i = layer[1][1][1]
        if iseven(i)    # add a delta at the edges for even layers
            push!(qr_list, it.prime(it.delta(siteinds[1], siteinds[1]'), depth-i))
        end
        for (pos, inds) in layer
            Q, R = it.qr(circ[pos[1]][pos[2]], inds[1], inds[3])
            push!(qr_list, Q)
            push!(qr_list, R)
        end
        if iseven(i)    # add a delta at the edges for even layers
            push!(qr_list, it.prime(it.delta(siteinds[end], siteinds[end]'), depth-i))
        end
        layer_mpo = itmps.MPO(qr_list)
        push!(mpo_list, layer_mpo)
    end

    return mpo_list
end





### CUSTOM ITENSOR METHODS

"Converts order-4 ITensor to N x N matrix following the order of indices in 'order'"
function tensor_to_matrix(T::it.ITensor, order = [1,2,3,4])
    inds = it.inds(T)
    ord_inds = [inds[i] for i in order]
    M = reshape(Array(T, ord_inds), (ord_inds[1].space^2, ord_inds[1].space^2))
    return M
end

"Extends m x n isometry to M x M unitary, where M is the power of 2 which bounds max(m, n) from above"
function iso_to_unitary(V::Union{Matrix, Vector{Float64}})
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
    psi = it.orthogonalize(psi, b)
    s = it.siteinds(psi)
    op = it.op(U, s[b+1], s[b])

    wf = it.noprime((psi[b]*psi[b+1])*op)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
    return psi
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

"Multiply two mps respecting their indices"
function contract(psi1::Union{itmps.MPS, Vector{it.ITensor}}, psi2::Union{itmps.MPS, Vector{it.ITensor}})
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


function brickwork(psi::itmps.MPS, brick_odd::Matrix, brick_even::Matrix, t::Integer; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    layerOdd = [it.op(brick_odd, sites[b+1], sites[b]) for b in 1:2:(N-1)]
    layerEven = [it.op(brick_even, sites[b+1], sites[b]) for b in 2:2:(N-1)]
    for i in 1:t
        println("Evolving step $i")
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

function initialize_vac(N::Int)
    sites = it.siteinds("Qubit", N)
    states = ["0" for _ in 1:N]
    vac = itmps.MPS(sites, states)
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

    block = U
    dR = div(D,d)
    
    # reshape into d x d^N-1 x d x d^N-1 matrix
    Uresh = reshape(block, (d, dR, d, dR))
    # permute so that the first two indices are those related to site 1
    Uresh = permutedims(Uresh, (1, 3, 2, 4))
    # reshape for svd
    Uresh = reshape(Uresh, (d^2, dR^2))
    F = svd(Uresh)
    Us, Ss, Vs = F.U, F.S, F.Vt
    # save link dimension and create link index
    dlink = size(Ss,1)
    link = it.Index(dlink; tags = "l=1")
    # store Us as an ITensor
    U_tensor = it.ITensor(Us, sites[1]', sites[1], link)
    push!(mpo, U_tensor)
    # update block and right dimension dR for next iteration
    block = diagm(Ss) * Vs
    dR = div(dR,d)
    
    for i in 2:N-1
        Uresh = reshape(block, (dlink, d, dR, d, dR))
        Uresh = permutedims(Uresh, (1, 2, 4, 3, 5))
        Uresh = reshape(Uresh, (dlink*d^2, dR^2))
        F = svd(Uresh)
        Us, Ss, Vs = F.U, F.S, F.Vt
        oldlink = link
        dlink = size(Ss,1)
        link = it.Index(dlink; tags = "l=$i")
        U_tensor = it.ITensor(Us, oldlink, sites[i]', sites[i], link)
        push!(mpo, U_tensor)
        block = diagm(Ss) * Vs
        dR = div(dR,d)
    end

    end_tensor = it.ITensor(block, link, sites[end]', sites[end])
    push!(mpo, end_tensor)

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




### MALZ-RELATED METHODS ###

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

"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertBW(mpo::Vector{it.ITensor}, tau, input_inds::Vector{<:it.Index}; d = 2, output_inds = nothing, conv_err = 1E-6, n_sweeps = 1E6)
    N = length(mpo)
    siteinds = input_inds
    mpo_mode = !isnothing(output_inds)

    L_blocks::Vector{it.ITensor} = []
    R_blocks::Vector{it.ITensor} = []

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    circuit::Vector{Vector{it.ITensor}} = []
    for i in 1:tau
        layer_i = [it.prime(it.ITensor(random_unitary(d^2), siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]'), tau-i) for j in 1:(div(N,2)-mod(N+1,2)*mod(i+1,2))]
        push!(circuit, layer_i)
    end

    # construct projectors for all sites
    # if state is mps, construct projectors onto |0>
    # if output_inds are given, construct deltas connecting output_inds to beginning of brickwork (trace)
    zero_projs::Vector{it.ITensor} = []

    if mpo_mode
        dim = output_inds[1].space
        new_outinds = it.siteinds(dim, N)
        for i in 1:N
            it.replaceind!(mpo[i], output_inds[i], new_outinds[i])
            push!(zero_projs, it.delta(new_outinds[i], it.prime(siteinds[i], tau)))
        end
    else
        for ind in siteinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, it.ITensor(vec, it.prime(ind, tau)))
        end
    end


    if N == 2
        left = zero_projs[1] * mpo[1]
        right = zero_projs[2] * mpo[2]
        env = conj(left*right)

        inds = siteinds
        U, S, Vdag = it.svd(env, inds, cutoff = 1E-14)
        u, v = it.commonind(U, S), it.commonind(Vdag, S)

        # evaluate fidelity
        newfid = real(tr(Array(S, (u, v))))
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        circuit[1][1] = gate_ji_opt

        if mpo_mode
            newfid /= 2^N # normalize if mpo mode
            for i in 1:N
                it.replaceind!(mpo[i], new_outinds[i], output_inds[i])
            end
        end

        println("Matrix is 2-local, converged to fidelity $newfid immediately")
        return circuit, newfid, 0
    end
            

    # prepare gates on the edges, which are just identities
    left_deltas = [it.prime(it.delta(siteinds[1], siteinds[1]'), tau-i) for i in 2:2:tau]
    right_deltas = [it.prime(it.delta(siteinds[N], siteinds[N]'), tau-i) for i in 2-mod(N,2):2:tau]

    # construct L_1
    # first item is zero projector, then there are the gates in ascending order, then the mpo site
    leftmost_block = [zero_projs[1]; left_deltas; mpo[1]]
    leftmost_block = reduce(*, leftmost_block)  # t+2 indices
    push!(L_blocks, leftmost_block)

    # construct R_N
    rightmost_block = [zero_projs[N]; right_deltas; mpo[N]]
    rightmost_block = reduce(*, rightmost_block)
    push!(R_blocks, rightmost_block)

    # contract everything on the right and save rightmost_block at each intermediate step
    # must be done only the first time, when j=2 (so contract up to j=3)
    for k in (N-1):-1:3
        # extract right gates associated with site k
        right_gates_k = [circuit[i][div(k,2)+mod(k,2)] for i in (2-mod(k,2)):2:tau]
        all_blocks = [zero_projs[k]; right_gates_k; mpo[k]]

        # if k even contract in descending order, else ascending
        all_blocks = iseven(k) ? reverse(all_blocks) : all_blocks
        for block in all_blocks
            rightmost_block *= block
        end
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # start the loop
    rev = false
    first_sweep = true
    fid, newfid = 0, 0
    j = 2  
    sweep = 0

    while sweep < n_sweeps

        # extract all gates touching site j
        gates_j = [circuit[i][div(j,2) + mod(i,2)*mod(j,2)] for i in 1:tau]
        # determine which gates are on left
        is_onleft = [isodd(j+i) for i in 1:tau]

        # optimize each gate
        for i in 1:tau
            gate_ji = gates_j[i]    # extract gate j
            is_j_onleft = isodd(i+j)

            sameside_gates = [gates_j[2-mod(i,2):2:i-2]; gates_j[i+2:2:end]]
            otherside_gates = gates_j[1+mod(i,2):2:end]
            
            left_gates = is_j_onleft ? sameside_gates : otherside_gates
            right_gates = is_j_onleft ? otherside_gates : sameside_gates

            contract_left = []
            contract_right = []
            if isodd(1+j)
                push!(contract_left, zero_projs[j])
            else
                push!(contract_right, zero_projs[j])
            end
            if isodd(tau+j)
                push!(contract_left, mpo[j])
            else
                push!(contract_right, mpo[j])
            end
            contract_left = [contract_left; left_gates]
            contract_right = [contract_right; right_gates]
            
            env_left = leftmost_block
            for gate in contract_left
                env_left *= gate
            end
            env_right = rightmost_block
            for gate in contract_right
                env_right *= gate
            end

            env = conj(env_left*env_right)

            inds = it.commoninds(it.prime(siteinds, tau-i), gate_ji)
            U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
            u, v = it.commonind(U, S), it.commonind(Vdag, S)

            # evaluate fidelity as Tr(Env*Gate), i.e. the overlap (NOT SQUARED)
            newfid = real(tr(Array(S, (u, v))))
            if mpo_mode     # normalize if mpo mode
                newfid /= 2^N
            end
            #println("Step $j: ", newfid)

            #replace gate_ji with optimized one, both in gates_j (used in this loop) and in circuit
            gate_ji_opt = U * it.replaceind(Vdag, v, u)
            gates_j[i] = gate_ji_opt    
            circuit[i][div(j,2) + mod(i,2)*mod(j,2)] = gate_ji_opt
        end

        # while loop end conditions
        if newfid >= 1
            newfid = 1
            break
        end

        # compare the relative increase in frobenius norm
        ratio = 1 - sqrt((1-newfid)/(1-fid))
        if -conv_err < ratio < conv_err && first_sweep == false
            break
        end

        if ratio >= 0   # only register if the ratio has increased, otherwise it's useless
            fid = newfid
        end

        if j == 2 && first_sweep == false
            rev = false
            sweep += 1
        end

        if j == N-1
            first_sweep = false
            rev = true
            sweep += 1
        end
        
        if N > 3
            # update L_blocks or R_blocks depending on sweep direction
            if rev == false
                gates_to_append = gates_j[(1+mod(j,2)):2:end]     # follow fig. 6
                all_blocks = [zero_projs[j]; gates_to_append; mpo[j]]
                all_blocks = isodd(j) ? reverse(all_blocks) : all_blocks
                for block in all_blocks
                    leftmost_block *= block
                end
                if first_sweep == true
                    push!(L_blocks, leftmost_block)
                else
                    L_blocks[j] = leftmost_block
                end
                rightmost_block = R_blocks[j]       #it should be j+2 but R_blocks starts from site 3
            else
                gates_to_append = gates_j[(2-mod(j,2)):2:end]
                all_blocks = [zero_projs[j]; gates_to_append; mpo[j]]
                all_blocks = iseven(j) ? reverse(all_blocks) : all_blocks
                for block in all_blocks
                    rightmost_block *= block
                end
                R_blocks[j-2] = rightmost_block         #same argument
                leftmost_block = L_blocks[j-2]
            end
            j = rev ? j-1 : j+1
        end

    end

    if mpo_mode
        for i in 1:N
            it.replaceind!(mpo[i], new_outinds[i], output_inds[i])
        end
    end

    println("Converged to fidelity $newfid with $sweep sweeps")

    return circuit, newfid, sweep

end

"Calls invertBW for Vector{ITensor} mpo input with increasing inversion depth tau until it converges with fidelity F = 1-err_to_one"
function invertBW(mpo::Vector{it.ITensor}, input_inds::Vector{<:it.Index}; err_to_one = 1E-6, start_tau = 1, n_runs = 10, kargs...)
    println("Tolerance $err_to_one, starting from depth $start_tau")
    tau = start_tau
    found = false

    local bw_best, sweep_best
    fid_best = 0
    while !found
        println("Attempting depth $tau with $n_runs runs...")
        fids = []
        for _ in 1:n_runs
            bw, fid, sweep = invertBW(mpo, tau, input_inds; kargs...)
            push!(fids, fid)
            if fid > fid_best
                fid_best = fid
                bw_best, sweep_best = bw, sweep
            end
        end
        avgfid = mean(fids)
        println("Avg fidelity = $avgfid")

        if abs(1-fid_best) < err_to_one
            found = true
            println("Convergence within desired error achieved with depth $tau\n")
            break
        end
        
        if tau > 9
            println("Attempt stopped at tau = $tau, ITensor cannot go above")
            break
        end

        tau += 1
    end
    return bw_best, fid_best, sweep_best, tau
end

"Wrapper for ITensorsMPS.MPS input. Calls invertBW by first conjugating mps (mps to invert must be above)"
function invertBW(mps::itmps.MPS; tau = 0, kargs...)
    obj = typeof(mps)
    println("Attempting inversion of $obj")
    mps = conj(mps)
    siteinds = it.siteinds(mps)

    if iszero(tau)
        results = invertBW(mps[1:end], siteinds; kargs...)
    else
        results = invertBW(mps[1:end], tau, siteinds; kargs...)
    end
    return results
end

"Wrapper for ITensorsMPS.MPO input. Calls invertBW by first conjugating and extracting upper and lower indices"
function invertBW(mpo::itmps.MPO; tau = 0, kargs...)
    obj = typeof(mpo)
    println("Attempting inversion of $obj")
    mpo = conj(mpo)
    allinds = reduce(vcat, it.siteinds(mpo))
    # determine primelevel of inputinds, which will be the lowest found in allinds
    first2inds = allinds[1:2]   
    plev_in = 0
    while true
        ind = it.inds(first2inds, plev = plev_in)
        if length(ind) > 0
            break
        end
        plev_in += 1
    end
    siteinds = it.inds(allinds, plev = plev_in)
    outinds = it.uniqueinds(allinds, siteinds)

    if iszero(tau)
        results = invertBW(mpo[1:end], siteinds; output_inds = outinds, kargs...)
    else
        results = invertBW(mpo[1:end], tau, siteinds; output_inds = outinds, kargs...)
    end
    return results
end

"Wrapper for ITensors.ITensor input. Calls invertBW on isometry V by first promoting it to a unitary matrix U"
function invertBW(V::it.ITensor, input_ind::it.Index; tau = 0, kargs...)
    obj = typeof(V)
    println("Attempting inversion of $obj")
    input_dim = input_ind.space
    output_inds = it.uniqueinds(V, input_ind)
    output_dim = reduce(*, [ind.space for ind in output_inds])

    Vmat = Array(V, output_inds, input_ind)
    Vmat = reshape(Vmat, output_dim, input_dim)
    Umat = iso_to_unitary(Vmat)
    Umpo, sites = unitary_to_mpo(Umat)

    if iszero(tau)
        results = invertBW(Umpo, sites; output_inds = sites', kargs...)
    else
        results = invertBW(Umpo, tau, sites; output_inds = sites', kargs...)
    end
    return results
end




function invertMPSMalz(mps::itmps.MPS; q = 0, eps_malz = 1E-2, eps_bell = 1E-2, eps_V = 1E-2, kargs...)
    N = length(mps)
    siteinds = it.siteinds(mps)
    linkinds = it.linkinds(mps)
    linkdims = [ind.space for ind in linkinds]
    D = linkdims[div(N,2)]
    d = siteinds[div(N,2)].space
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D

    local blockMPS, new_siteinds, newN, npairs, block_linkinds, block_linkinds_dims

    q_list = iszero(q) ? [i for i in 2:N if mod(N,i) == 0] : [q]

    local q_found, circuit, fid, sweepB
    for q in q_list
        println("Attempting blocking with q = $q...")
        q_found = q
        
        if mod(N, q) != 0
            throw(DomainError(q, "Inhomogeneous blocking is not supported, choose a q that divides N"))
        end

        if length(Set(linkdims)) > q
            throw(DomainError(linkdims, "MPS has non-constant bond dimension, use a greater blocking size"))
        end

       
        if q - 2*bitlength < 0
            throw(DomainError(D, "Malz inversion only works if q - 2n >= 0, with n: D = d^n\nMax bond dimension D = $D, blocking size q = $q."))
        end

        newN = div(N,q)
        npairs = newN-1

        # block array
        blocked_mps, blocked_siteinds = blocking(mps, q)
        # polar decomp and store P matrices in array
        blockMPS = [polar_P(blocked_mps[i], blocked_siteinds[i]) for i in 1:newN]

        # save linkinds and create new siteinds
        block_linkinds = it.linkinds(mps)[q:q:end]
        block_linkinds_dims = [ind.space for ind in block_linkinds]
        
        new_siteinds = it.siteinds(D, 2*(newN-1))

        # replace primed linkinds with new siteinds
        it.replaceind!(blockMPS[1], block_linkinds[1]', new_siteinds[1])
        it.replaceind!(blockMPS[end], block_linkinds[end]', new_siteinds[end])
        for i in 2:(newN-1)
            it.replaceind!(blockMPS[i], block_linkinds[i-1]', new_siteinds[2*i-2])
            it.replaceind!(blockMPS[i], block_linkinds[i]', new_siteinds[2*i-1])
        end

        # separate each block into 2 different sites for optimization
        sep_mps::Vector{it.ITensor} = [blockMPS[1]]
        for i in 2:newN-1
            iL, nL = block_linkinds[i-1], new_siteinds[2*i-2]
            bL, S, bR = it.svd(blockMPS[i], iL, nL)
            push!(sep_mps, bL, S*bR)
        end
        push!(sep_mps, blockMPS[end])

        ## Start variational optimization to approximate P blocks
        circuit, fid, sweepB = invertBW(conj(sep_mps), 1, new_siteinds; d = D, kargs...)
        
        if abs(1 - fid) < eps_malz
            break
        end
    end
    
    q = q_found
    W_list = circuit[1]

    ## Extract unitary matrices that must be applied on bell pairs
    # here we follow the sequential rg procedure
    V_list::Vector{Vector{it.ITensor}} = []
    V_tau_list = []
    
    for i in 1:newN

        block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]

        # if i=1 no svd must be done, entire block 1 is already a series of isometries
        if i == 1

            iR = block_linkinds[1]
            block_i[end] = it.replaceind(block_i[end], iR, new_siteinds[1])
            block_i_Ulist = block_i
            push!(V_list, block_i_Ulist)

        else

            # prepare left index of block i
            block_i_Ulist = []
            iL = block_linkinds[i-1]
            local prev_SV


            for j in 1:q-1
                A = block_i[j]                              # j-th tensor in block_i
                iR = it.commoninds(A, linkinds)[2]    # right link index
                Aprime = j > 1 ? prev_SV*A : A              # contract SV from previous svd (of the block on the left)
                Uprime, S, V = it.svd(Aprime, it.uniqueinds(Aprime, iR, iL))
                push!(block_i_Ulist, Uprime)
                prev_SV = S*V
            end
            # now we need to consider the last block C and apply P^-1
            C = block_i[end]
            P = blockMPS[i]

            if i < newN     #if we are not at the end of the whole spin chain
                # extract P indices, convert to matrix and invert
                PiL, PiR = block_linkinds[i-1:i]
                PnL, PnR = new_siteinds[2*i-2:2*i-1]
                dim = reduce(*, block_linkinds_dims[i-1:i])
                P_matrix = reshape(Array(P, [PnL, PnR, PiL, PiR]), (dim, dim))
                Pinv = inv(P_matrix)
            
                # convert back to tensor with inds ready to contract with Ctilde = prev_SV * C   
                # and with blockMPS[i] siteinds       
                CindR = it.commoninds(C, linkinds)[2]
                Pinv = it.ITensor(Pinv, [iL, CindR, PnL, PnR])
            else    #same here, only different indices
                PiL = block_linkinds[i-1]
                PnL = new_siteinds[2*i-2]
                dim = block_linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PnL, PiL]), (dim, dim))
                Pinv = inv(P_matrix)

                Pinv = it.ITensor(Pinv, [iL, PnL])
            end

            Ctilde = prev_SV * C * Pinv
            push!(block_i_Ulist, Ctilde)
            push!(V_list, block_i_Ulist)

        end

        # Approximate total V_i unitary with bw circuit
        # contract isometries in block_i_Ulist as long as dimension <= d^2bitlength (efficient for D low, independent from q)
        upinds = siteinds[q*(i-1)+1 : q*i]
        V = reduce(*, block_i_Ulist[1:2*bitlength])
        V_upinds = upinds[1:2*bitlength]
        V_downinds = it.uniqueinds(V, V_upinds)
        down_d = reduce(*, [ind.space for ind in V_downinds])  # will be D for i=1, D^2 elsewhere
        V_matrix = reshape(Array(V, V_upinds, V_downinds), (d^(2*bitlength), down_d))      # MUST BE MODIFIED WHEN i=1 or newN THE OUTPUT LEG IS ONLY D
        @show typeof(V_matrix)
        @show typeof(iso_to_unitary(V_matrix))
        V_mpo = unitary_to_mpo(iso_to_unitary(V_matrix), siteinds = upinds)
        it.prime!(V_mpo, q - 2*bitlength)
        
        # transform next isometries to mpo and contract with previous ones
        for k in 2*bitlength+1:q
            uk = block_i_Ulist[k]
            prev_u = block_i_Ulist[k-1]
            uk_upinds = [it.commoninds(uk, prev_u); upinds[k]]
            uk_downinds = it.uniqueinds(uk, uk_upinds)
            up_d = reduce(*, [ind.space for ind in uk_upinds])
            down_d = reduce(*, [ind.space for ind in uk_downinds])
            uk_matrix = reshape(Array(uk, uk_upinds, uk_downinds), (up_d, down_d))
            # need to decide how many identities to put to the left of uk unitaries based on up_d
            nskips = k - length(digits(up_d-1, base=d))
            uk_mpo = unitary_to_mpo(iso_to_unitary(uk_matrix), siteinds = upinds, skip_qudits = nskips)
            it.prime!(uk_mpo, q-k)
            V_mpo = V_mpo*uk_mpo
        end

        it.replaceprime!(V_mpo, q-2*bitlength+1 => 1)
        # invert final V
        _, _, _, tau = invertBW(V_mpo; d = d, err_to_one = eps_V/((newN^2)*d^q), kargs...)
        
        # account for the swap gates 
        if i > 1
            tau += q - bitlength - 1
        end
        push!(V_tau_list, tau)

    end

    zero_projs::Vector{it.ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, ind'))
    end

    # act with W on zeros
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]


    ## Compute scalar product to check fidelity
    # just a check, can be disabled

    # # group blocks together
    # block_list::Vector{it.ITensor} = []
    # mps_primed = conj(mps)'''''
    # siteinds = it.siteinds(mps_primed)
    # local left_tensor
    # for i in 1:newN
    #     block_i = i < newN ? mps_primed[q*(i-1)+1 : q*i] : mps_primed[q*(i-1)+1 : end]
    #     i_siteinds = i < newN ? siteinds[q*(i-1)+1 : q*i] : siteinds[q*(i-1)+1 : end]
    #     left_tensor = (i == 1 ? block_i[1] : block_i[1]*left_tensor)
    #     left_tensor *= it.replaceinds(V_list[i][1], it.noprime(i_siteinds), i_siteinds)

    #     for j in 2:q
    #         left_tensor *= block_i[j] * it.replaceinds(V_list[i][j], it.noprime(i_siteinds), i_siteinds)
    #     end
    #     if i < newN
    #         left_tensor *= Wlayer[i]
    #     end
    # end


    # fidelity = abs(Array(left_tensor)[1])
    # println(fidelity)

    #println(fid)

    # prepare W|0> states to turn into mps
    Wmps_list = []
    for i in 1:npairs
        W_ext = zeros(ComplexF64, (2^bitlength, 2^bitlength))   # embed W in n = bitlength qudits
        W = Wlayer[i]
        W = reshape(Array(W, it.inds(W)), (D, D))
        W_ext[1:D, 1:D] = W
        W_ext = reshape(W_ext, 2^(2*bitlength))
        sites = it.siteinds(d, 2*bitlength)
        W_mps = itmps.MPS(W_ext, sites)
        push!(Wmps_list, W_mps)
    end

    # invert each mps in the list
    results = [invertBW(Wmps; d = d, err_to_one = eps_bell/npairs) for Wmps in Wmps_list]

    W_tau_list = [res[4] for res in results]

    return Wlayer, V_list, fid, sweepB, W_tau_list, V_tau_list
end



end