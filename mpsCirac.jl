import ITensorMPS
import ITensors
import Plots
using LaTeXStrings, LinearAlgebra, Statistics

H = [1 1
    1 -1]/sqrt(2)
Id = [1 0
      0 1]
T = [1 0
    0 exp(1im*pi/4)]
X = [0 1
    1 0]
CX = [1 0 0 0
      0 1 0 0
      0 0 0 1
      0 0 1 0]

# Apply 2-site tensor to sites b and b+1
function apply(U::AbstractMatrix, psi::ITensorMPS.MPS, b::Integer; cutoff::AbstractFloat = 1E-8)
    psi = ITensors.orthogonalize(psi, b)
    s = ITensors.siteinds(psi)
    op = ITensors.op(U, s[b+1], s[b])

    wf = ITensors.noprime((psi[b]*psi[b+1])*op)
    indsb = ITensors.uniqueinds(psi[b], psi[b+1])
    U, S, V = ITensors.svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
    return psi
end

function initialize_vac(N::Int)
    nsites = N
    sites = ITensors.siteinds("Qubit", nsites)
    states = ["0" for _ in 1:N]
    psiMPS = ITensorMPS.MPS(sites, states)
    return psiMPS
end



function entropy(psi::ITensorMPS.MPS, b::Integer)  
    ITensors.orthogonalize!(psi, b)
    indsb = ITensors.uniqueinds(psi[b], psi[b+1])
    U, S, V = ITensors.svd(psi[b], indsb, cutoff = 1E-14)
    SvN = 0.0
    for n in 1:ITensors.dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    return SvN
  end

function layers(psi::ITensorMPS.MPS, brick_odd::Matrix, brick_even::Matrix)
    sites = ITensors.siteinds(psi)
    layerOdd = [ITensors.op(brick_odd, sites[b+1], sites[b]) for b in 1:2:(N-1)]
    layerEven = [ITensors.op(brick_even, sites[b+1], sites[b]) for b in 2:2:(N-1)]
    return layerOdd, layerEven
end

function brickwork(psi::ITensorMPS.MPS, brick_odd::Matrix, brick_even::Matrix, t::Integer; cutoff = 1E-14)
    layerOdd, layerEven = layers(psi, brick_odd, brick_even)
    for i in 1:t
        println("Step $i")
        layer = iseven(i) ? layerOdd : layerEven
        psi = ITensors.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end



function polar(block::ITensors.ITensor, inds::Vector{ITensors.Index{Int64}})
    spaceU = reduce(*, [ind.space for ind in inds])
    spaceV = reduce(*, [ind.space for ind in ITensors.uniqueinds(block, inds)])
    #if spaceU > spaceV
    #    throw("Polar decomposition of block failed: make sure link space is bigger than physical space")
    #end
    U, S, V = ITensors.svd(block, inds)
    u, v = ITensors.inds(S)
    
    Vdag = ITensors.replaceind(conj(V)', v', u)
    P = Vdag * S * V

    V = ITensors.replaceind(V', v', u)
    W = U * V

    return W, P
end


"Given a block B and a 3-Tuple containing the left, central and right indices,
computes the right eigenstate of the transfer matrix B otimes B^*, and returns
the square root of its matrix form"
function extract_rho(block::ITensors.ITensor, inds::NTuple{3, ITensors.Index{Int64}})
    iL, i, iR = inds[1:3]
    block_star = conj(block)'
    delta_i = ITensors.delta(i, i')

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

function iso_to_unitary(V::Matrix)
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

"Performs blocking on an MPS, q sites at a time. Returns the blocked MPS as an array of ITensors, together with the siteinds."
function blocking(mps::Union{Vector{ITensors.ITensor},ITensorMPS.MPS}, q::Int; combine_indices = false)
    N = length(mps)
    
    if q > 13   # itensor only works with tensors up to 13 indices
        qnew = 2
        while div(q, qnew) > 13
            qnew += 1
        end
        mps = blocking(mps, qnew, combine_indices = true)
        N = length(mps)
        q = div(q, qnew)
    end

    newN = div(N, q)
    r = mod(N, q)

    block_mps = [reduce(*, mps[q*(i-1)+1 : q*i]) for i in 1:newN] 
    if r != 0
        push!(block_mps, r > 1 ? reduce(*, mps[q * newN : end]) : mps[end])
    end

    
    siteinds = reduce(ITensors.noncommoninds, block_mps[1:end])
    sitegroups = [siteinds[q*(i-1)+1 : q*i] for i in 1:newN]
    if r != 0
        push!(sitegroups, r > 1 ? siteinds[q * newN : end] : siteinds[end])
        newN += 1
    end
    
    if combine_indices
        combiners = [ITensors.combiner(sitegroups[i]; tags="c$i") for i in 1:newN]
        block_mps = [combiners[i]*block_mps[i] for i in 1:newN]
    end

    return block_mps, sitegroups
end


# Prepare initial state

N = 4
q_list = [2,]
avg_list = []
err_list = []
Nruns = 1
rand = true 


test = ITensorMPS.random_mps(ITensors.siteinds("Qubit", 32), linkdims=2)

for q in q_list

    eps_per_block_array::Vector{Float64} = []

    for run in 1:Nruns
        if mod(run, 100) == 0
            println("$run done")
        end

        sites = ITensors.siteinds("Qubit", N)
        randMPS = ITensorMPS.random_mps(sites, linkdims=2)
        ITensors.orthogonalize!(randMPS, N)

        testMPS = initialize_vac(N)
        brick = CX * kron(H, Id)
        testMPS = brickwork(testMPS, brick, brick, 2)
        #testMPS = apply(CX * kron(H, Id), testMPS, 4)
        ITensors.orthogonalize!(testMPS, N)

        newN = div(N, q)
        mps = rand ? randMPS : testMPS
        sites = ITensors.siteinds(mps)
        data_array = [Dict{String,Any}() for _ in 1:newN]

        # block array, polar decomp and save W and P matrices
        for i in 1:newN
            block_i = reduce(*, mps[q*(i-1)+1 : q*i])       # extract q consecutive matrices
            block_siteinds = sites[q*(i-1)+1 : q*i]
            W, P = polar(block_i, block_siteinds)

            data_array[i]["W"] = W
            data_array[i]["P"] = P
        end

        # construct MPS of P matrices
        blockMPS = [copy(data_array[i]["P"]) for i in 1:newN]
        block_linkinds = ITensors.linkinds(mps)[q:q:end]
        linkinds_dims = [ind.space for ind in block_linkinds]

        # prime left index of each block to distinguish it from right index of previous block
        for i in 2:newN
            ind_to_increase = ITensors.uniqueinds(blockMPS[i], block_linkinds)
            ind_to_increase = ITensors.commoninds(ind_to_increase, blockMPS[i-1])
            ITensors.replaceinds!(blockMPS[i], ind_to_increase, ind_to_increase')
        end


        zero_projs = []
        for ind in block_linkinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, ITensors.ITensor(vec, ind'))
            push!(zero_projs, ITensors.ITensor(vec, ind''))
        end


        current_env = copy(blockMPS)
        npairs = newN-1
        W_list::Vector{ITensors.ITensor} = []
        first_sweep = true
        reverse = false

        j = 1   # W1 position (pair)
        fid = 0

        # Start iteration
        while true
            # contract on 0 everywhere but on pair j
            # contract everything on the right
            rightmost_block = current_env[end]
            if j < npairs   # move rightmost block until it reaches pair j
                for k in npairs:-1:j+1
                    # extract block_k from current env
                    block_k = current_env[k] 
                    zero_L, zero_R = zero_projs[2k-1:2k]
                    rightmost_block = zero_L * block_k * rightmost_block * zero_R
                end
            end

            # contract everything on the left
            leftmost_block = current_env[1]
            if j > 1    # move leftmost block until it reaches pair j
                for k in 1:j-1
                    block_kp1 = current_env[k+1]
                    zero_L, zero_R = zero_projs[2k-1:2k]
                    leftmost_block = zero_L * block_kp1 * leftmost_block * zero_R
                end
            end

            # build environment tensor by adding the two |0> projectors
            zero_j = [zero'' for zero in zero_projs[2j-1:2j]]       # add two primes since they must be traced at the end
            env_tensor = leftmost_block * rightmost_block
            Winds = ITensors.inds(env_tensor)           # extract indices to apply W
            env_tensor = env_tensor * zero_j[1] * zero_j[2]         # construct environment tensor Ej
            # svd environment and construct Wj
            Uenv, Senv, Venv_dg = ITensors.svd(env_tensor, Winds, cutoff = 1E-14)       # SVD environment
            u, v = ITensors.commonind(Uenv, Senv), ITensors.commonind(Venv_dg, Senv)
            W_j = Uenv * ITensors.replaceind(Venv_dg, v, u)       # Construct Wj as UVdag


            newfid = real(tr(Array(Senv, (u, v))))^2
            println("Step $j: ", newfid)

            if isapprox(newfid, fid) && (first_sweep == false)
                break
            end
            fid = newfid

            # dim = reduce(*, [ind.space for ind in Winds])
            # W_matrix = reshape(Array(W_j, [Winds; Winds'']), (dim, dim))
            # U_matrix = iso_to_unitary(W_matrix)
            # W_j = ITensors.ITensor(U_matrix, [Winds, Winds''])

            Pj, Pjp1 = current_env[j], current_env[j+1]
            update_env = conj(W_j) * Pj * Pjp1
            # replace zero_j inds with Winds (i.e. remove '' from each index)
            zero_j_inds = [ITensors.inds(zero)[1] for zero in zero_j]
            ITensors.replaceinds!(update_env, zero_j_inds, Winds)

            Pjnew, S, V = ITensors.svd(update_env, ITensors.uniqueinds(Pj, Pjp1), cutoff = 1E-14)
            Pjp1new = S*V
            current_env[j], current_env[j+1] = Pjnew, Pjp1new

            if first_sweep
                push!(W_list, W_j)
            else
                Wnew = ITensors.replaceinds(W_j, Winds, Winds'''')
                Wold = ITensors.replaceinds(W_list[j], Winds'', Winds'''')
                W_list[j] = Wold * Wnew
            end

            if npairs == 1
                break
            end

            if j == npairs
                first_sweep = false
                reverse = true
            end

            if j == 1
                reverse = false
            end

            j = reverse ? j-1 : j+1
        end

        # Save W matrices
        #W_matrices = [reshape(Array(W, ITensors.inds(W)), (ITensors.inds(W)[1].space^2, ITensors.inds(W)[1].space^2)) for W in W_list]
        #W_unitaries = [iso_to_unitary(W) for W in W_matrices]


        # Extract unitary matrices that must be applied on bell pairs
        U_list = []
        linkinds = ITensors.linkinds(mps)
# 
        for i in 1:newN
            block_i = mps[q*(i-1)+1 : q*i]
# 
            if i == 1
                iR = block_linkinds[1]
                block_i[end] = ITensors.replaceind(block_i[end], iR, iR')
                push!(U_list, block_i)
                continue
            end
# 
            block_i_Ulist = []
            iL = block_linkinds[i-1]
            local prev_SV
# 
            for j in 1:q-1
                A = block_i[j]
                iR = ITensors.commoninds(A, linkinds)[2]
                Aprime = j > 1 ? prev_SV*A : A
                Uprime, S, V = ITensors.svd(Aprime, ITensors.uniqueinds(Aprime, iR, iL))
                push!(block_i_Ulist, Uprime)
                prev_SV = S*V
            end
            C = block_i[end]
            P = data_array[i]["P"]
# 
            if i < newN     #if we are not at the end of the chain
                PiL, PiR = block_linkinds[i-1:i]
                dim = reduce(*, linkinds_dims[i-1:i])
                P_matrix = reshape(Array(P, [PiL', PiR', PiL, PiR]), (dim, dim))
                Pinv = inv(P_matrix)
# 
                CindR = ITensors.commoninds(C, linkinds)[2]
                Pinv = ITensors.ITensor(Pinv, [iL, CindR, iL'', CindR'])
            else
                PiL = block_linkinds[i-1]
                dim = linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PiL', PiL]), (dim, dim))
                Pinv = inv(P_matrix)
# 
                Pinv = ITensors.ITensor(Pinv, [iL, iL''])
            end
# 
            Ctilde = prev_SV * C * Pinv
            push!(block_i_Ulist, Ctilde)
            push!(U_list, block_i_Ulist)
        end
# 
# 
        ## Compute scalar product to check fidelity
# 
        # prepare zero initial states to act with W on them
        zero_projs = []
        for ind in block_linkinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, ITensors.ITensor(vec, ind^3))
            push!(zero_projs, ITensors.ITensor(vec, ind^4))
        end
        Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]
# 
        # group blocks together
        block_list = []
        mps_primed = conj(mps)'''''
        siteinds = ITensors.siteinds(mps_primed)
        for i in 1:newN
            block_i = mps_primed[q*(i-1)+1 : q*i]
            i_siteinds = siteinds[q*(i-1)+1 : q*i]
            left_tensor = block_i[1] * ITensors.replaceinds(U_list[i][1], ITensors.noprime(i_siteinds), i_siteinds)
            for j in 2:q
                left_tensor = left_tensor * block_i[j] * ITensors.replaceinds(U_list[i][j], ITensors.noprime(i_siteinds), i_siteinds)
            end
            push!(block_list, left_tensor)
        end
# 
        left_tensor = block_list[1]
        for i in 2:newN
            left_tensor = left_tensor * block_list[i] * Wlayer[i-1]
        end
# 
        fidelity = abs(Array(left_tensor)[1])
        eps = 1-sqrt(fidelity)
        eps_per_block = eps/newN
        push!(eps_per_block_array, eps_per_block)

    end

    avg_perq = mean(eps_per_block_array)
    err_perq = std(eps_per_block_array)/sqrt(1000)
# 
    push!(avg_list, avg_perq)
    push!(err_list, err_perq)

end

#A = randMPS[3]
#Astar = conj(A)'
#
#delta_n3 = ITensors.delta([ITensors.inds(A)[1], ITensors.inds(Astar)[1]])
#
#T = delta_n3 * A * Astar
#l2, l3 = ITensors.linkinds(randMPS)[2], ITensors.linkinds(randMPS)[3]
#
#U,S,V = ITensors.svd(T, (l2, l2'), cutoff = 1E-14)
#indsS = ITensors.inds(S)
#
#delta_l2 = ITensors.delta(l2, l2')
#@show delta_l2 * T
#
#u1 = reshape(Array(U, l2, l2', indsS[1])[:, :, 1], 4)
#v1 = reshape(Array(V, l3, l3', indsS[2])[:, :, 1], 4)
#proj = S[1] * u1 * v1'
#
#S[indsS[1] => 2, indsS[2] => 2] = 0
#S[indsS[1] => 3, indsS[2] => 3] = 0
#S[indsS[1] => 4, indsS[2] => 4] = 0
#
#T_trunc = U*S*V
#reshape(Array(T_trunc, l2, l2', l3, l3'), (4,4))
## see it is the same as proj
#
#delta_l3 = ITensors.delta(l3, l3')
#@show delta_l2 * T_trunc
#
## Convert T to matrix and diagonalize to extract right eigenvector of eigenval=1
#T_matrix = reshape(Array(T, l2, l2', l3, l3'), (4,4))
#
#Um, Sm, Vm = svd(T_matrix)
#
#eig = eigen(T_matrix)
#right_eig = eig.vectors[:,4]
#alpha = 1/tr(reshape(right_eig, (2,2)))
#rho = alpha*right_eig
#
## Construct the td limit transfer matrix
#T_inf = rho * [1 0 0 1]
#
#R = T_matrix - T_inf
#
## Extract rho and reshape
##sqrt_rho = sqrt(reshape(rho, (2,2)))