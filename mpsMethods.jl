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

function entropy(psi::itmps.MPS, b::Integer)  
    it.orthogonalize!(psi, b)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(psi[b], indsb, cutoff = 1E-14)
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

"Project sites indexed by 'positions' array to zero. Normalizes at the end"
function project_tozero(psi::itmps.MPS, positions::Vector{Int64})
    psi = copy(psi)
    N = length(psi)
    sites = it.siteinds(psi)
    for b in positions
        ind = sites[b]
        zero_vec = [1; [0 for _ in 1:ind.space-1]]
        zero_proj = it.ITensor(kron(zero_vec, zero_vec'), ind, ind')
        psi[b] = it.noprime(psi[b]*zero_proj)
    end
    it.normalize!(psi)
    return psi
end


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

"Extends m x n isometry to M x M unitary, where M = max(m, n)"
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

    bitlenght = length(digits(nrows-1, base=2))     # represent nrows in base 2
    D = 2^bitlenght

    U = zeros(ComplexF64, (D, D))
    U[1:nrows, 1:ncols] = V
    kerU = nullspace(U')
    U[:, ncols+1:D] = kerU

    if dagger
        U = U'
    end

    return U
end

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
    it.orthogonalize!(mps, N)
    it.normalize!(mps)

    return mps
end

"Converts order-4 ITensor to N x N matrix following the order of indices in 'order'"
function tensor_to_matrix(T::it.ITensor, order = [1,2,3,4])
    inds = it.inds(T)
    ord_inds = [inds[i] for i in order]
    M = reshape(Array(T, ord_inds), (ord_inds[1].space^2, ord_inds[1].space^2))
    return M
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
        throw(BoundsError(pos_1[1], "Max eigenvalue less than 1"))
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


function invertMPS(mps::itmps.MPS, tau::Int64; n_sweeps::Int64)
    mps = conj(mps)
    N = length(mps)
    it.orthogonalize!(mps, N)
    siteinds = it.siteinds(mps)

    L_blocks::Vector{it.ITensor} = []
    R_blocks::Vector{it.ITensor} = []

    # create circuit of identities
    # circuit[i, j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    #circuit = [it.prime(it.delta(siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)]') * it.delta(siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)+1]'), tau-i) for i in 1:tau, j in 1:div(N,2)]
    # create random circuit instead
    circuit = [it.prime(it.ITensor(random_unitary(4), siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]'), tau-i) for i in 1:tau, j in 1:div(N,2)]

    zero_projs::Vector{it.ITensor} = []
    for ind in siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, it.prime(ind, tau)))
    end

    # prepare gates on the edges, which are just identities
    left_deltas = [it.prime(it.delta(siteinds[1], siteinds[1]'), tau-i) for i in 2:2:tau]
    right_deltas = [it.prime(it.delta(siteinds[N], siteinds[N]'), tau-i) for i in 1:2:tau]

    # construct L_1
    leftmost_block = mps[1] * zero_projs[1] 
    if tau > 1
        leftmost_block *= reduce(*, left_deltas)
    end

    push!(L_blocks, leftmost_block)

    # construct R_N
    rightmost_block = mps[N] * zero_projs[N] * reduce(*, right_deltas)
    push!(R_blocks, rightmost_block) 

    # contract everything on the right and save rightmost_block at each intermediate step
    # must be done only the first time, when j=2
    for k in (N-1):-1:3
        # extract right gates associated with site k
        right_gates_k = [circuit[i, div(k,2)+mod(k,2)] for i in (2-mod(k,2)):2:tau]
        if length(right_gates_k) > 0
            rightmost_block *= reduce(*, right_gates_k)
        end
        rightmost_block *= mps[k]*zero_projs[k]
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # start the loop
    reverse = false
    first_sweep = true
    fid, newfid = 0, 0
    j = 2  
    sweep = 0

    while sweep < n_sweeps

        # extract all gates touching site j
        gates_j = [circuit[i, div(j,2) + mod(i,2)*mod(j,2)] for i in 1:tau]
        L_jm1 = leftmost_block
        R_jp1 = rightmost_block

        # optimize each gate
        for i in 1:tau
            gate_ji = gates_j[i]
            other_gates = [gates_j[1:i-1]; gates_j[i+1:end]]

            env = L_jm1 * mps[j] * zero_projs[j] * R_jp1
            if length(other_gates) > 0
                env *= reduce(*, other_gates)
            end
            env = conj(env)

            inds = it.commoninds(it.prime(siteinds, tau-i), gate_ji)
            U, S, Vdag = it.svd(env, inds, cutoff = 1E-14)
            u, v = it.commonind(U, S), it.commonind(Vdag, S)

            # evaluate fidelity and stop if converged
            newfid = real(tr(Array(S, (u, v))))^2
            #println("Step $j: ", newfid)

            #replace gate_ji with optimized one, both in gates_j (used in this loop) and in circuit
            gate_ji_opt = U * it.replaceind(Vdag, v, u)
            gates_j[i] = gate_ji_opt    
            circuit[i, div(j,2) + mod(i,2)*mod(j,2)] = gate_ji_opt
        end

        ## while loop end conditions

        ## if isapprox(newfid, fid) && (first_sweep == false)
        ##     break
        ## end
        if abs(1 - newfid) < 1E-10
            break
        end
        fid = newfid

        if j == N-1
            first_sweep = false
            reverse = true
            sweep += 1
        end

        if j == 2 && first_sweep == false
            reverse = false
            sweep += 1
        end

        # update L_blocks or R_blocks depending on sweep direction
        if reverse == false
            gates_to_append = gates_j[(1+mod(j,2)):2:end]     # follow fig. 6
            if length(gates_to_append) > 0
                leftmost_block *= reduce(*, gates_to_append)
            end
            leftmost_block *= mps[j]*zero_projs[j]
            if first_sweep == true
                push!(L_blocks, leftmost_block)
            else
                L_blocks[j] = leftmost_block
            end
            rightmost_block = R_blocks[j]       #it should be j+2 but R_blocks starts from site 3
        else
            gates_to_append = gates_j[(2-mod(j,2)):2:end]
            if length(gates_to_append) > 0
                rightmost_block *= reduce(*, gates_to_append)
            end
            rightmost_block *= mps[j]*zero_projs[j]
            R_blocks[j-2] = rightmost_block         #same argument
            leftmost_block = L_blocks[j-2]
        end

        j = reverse ? j-1 : j+1

    end

    return newfid, sweep
end

end