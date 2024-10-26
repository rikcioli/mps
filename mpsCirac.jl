import ITensorMPS
import ITensors
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles


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
function apply(U::AbstractMatrix, psi::ITensorMPS.MPS, b::Integer; cutoff::AbstractFloat = 1E-14)
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
        layer = isodd(i) ? layerOdd : layerEven
        psi = ITensors.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end

"Apply depth-t brickwork of 2-local random unitaries"
function brickwork(psi::itmps.MPS, t::Integer; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    d = sites[1].space
    for i in 1:t
        layer = [it.op(random_unitary(d^2), sites[b+1], sites[b]) for b in (isodd(i) ? 1 : 2):2:(N-1)]
        psi = it.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end


function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
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

"Computes polar decomposition but only returning P (efficient)"
function polar_P(block::Vector{ITensors.ITensor}, inds::Vector{ITensors.Index{Int64}})
    q = length(block)
    block_conj = [conj(tens)' for tens in block]
    left_tensor = block[1] * block_conj[1] * ITensors.delta(inds[1], inds[1]')
    for j in 2:q
        left_tensor *= block[j] * block_conj[j] * ITensors.delta(inds[j], inds[j]')
    end

    low_inds = reduce(ITensors.noncommoninds, [block; inds])
    dim = reduce(*, [ind.space for ind in low_inds])
    mat = reshape(Array(left_tensor, (low_inds', low_inds)), (dim, dim))
    P = sqrt(mat)
    P = ITensors.ITensor(P, (low_inds', low_inds))
    return P
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

"Performs blocking on an MPS, q sites at a time. Returns the blocked MPS as an array of ITensors, together with the siteinds."
function blocking(mps::Union{Vector{ITensors.ITensor},ITensorMPS.MPS}, q::Int)
    N = length(mps)
    newN = div(N, q)
    r = mod(N, q)

    block_mps = [mps[q*(i-1)+1 : q*i] for i in 1:newN] 
    if r != 0
        push!(block_mps, r > 1 ? mps[q * newN : end] : [mps[end]])
    end

    siteinds = reduce(ITensors.noncommoninds, mps[1:end])
    sitegroups = [siteinds[q*(i-1)+1 : q*i] for i in 1:newN]
    if r != 0
        push!(sitegroups, r > 1 ? siteinds[q * newN : end] : [siteinds[end]])
        newN += 1
    end

    return block_mps, sitegroups
end

function rand_MPS(sites::Vector{<:ITensors.Index}; linkdims=1)
    d = sites[1].space
    D = linkdims

    mps = ITensorMPS.MPS(sites, linkdims = D)
    links = ITensors.linkinds(mps)

    U0 = ITensors.ITensor(random_unitary(d*D), (sites[1], links[1]', links[1], sites[1]'))
    U_list = [ITensors.ITensor(random_unitary(d*D), (sites[i], links[i-1], links[i], sites[i]')) for i in 2:N-1]
    UN = ITensors.ITensor(random_unitary(d*D), (sites[N], links[N-1], links[N-1]', sites[N]'))

    zero_projs::Vector{ITensors.ITensor} = [ITensors.ITensor([1; [0 for _ in 2:d]], site') for site in sites]
    zero_L = ITensors.ITensor([1; [0 for _ in 2:D]], links[1]')
    zero_R = ITensors.ITensor([1; [0 for _ in 2:D]], links[N-1]')

    U0 *= zero_L
    UN *= zero_R
    U_list = [U0; U_list; UN]

    tensors = [zero_projs[i]*U_list[i] for i in 1:N]

    for i in 1:N
        mps[i] = tensors[i]
    end
    ITensors.orthogonalize!(mps, N)
    ITensors.normalize!(mps)

    return mps
end
    

# Prepare initial state

N = 120
q_list = [i for i in 2:2:12]
avg_list::Vector{Float64} = []
all_runs_list::Vector{Vector{Float64}} = []
err_list::Vector{Float64} = []
Nruns = 100
random = true

neg_eps_pos::Vector{Int64} = []
t_vs_q = []

t=4

for t in 1:6
    println("Evaluating depth t = $t...")

# Loop over different values of q
for q in q_list
    if mod(N, q) != 0
        throw(DomainError(q, "Inhomogeneous blocking is not supported, choose a q that divides N"))
    end
    println("Evaluating blocking size q = $q...")

    eps_per_block_array::Vector{Float64} = []

    # Loop over different random mps
    for run in 1:Nruns

        #sites = ITensors.siteinds("Qubit", N)
        #randMPS = rand_MPS(sites, linkdims=2)
        #ITensors.orthogonalize!(randMPS, N)

        testMPS = initialize_vac(N)
        testMPS = brickwork(testMPS, t)
        #testMPS = apply(CX * kron(H, Id), testMPS, 4)
        ITensors.orthogonalize!(testMPS, N)

        newN = mod(N,q) == 0 ? div(N,q) : div(N,q)+1
        mps = testMPS
        sites = ITensors.siteinds(mps)

        # block array
        blocked_mps, blocked_siteinds = blocking(mps, q)
        # polar decomp and store P matrices in array
        blockMPS = [polar_P(blocked_mps[i], blocked_siteinds[i]) for i in 1:newN]
        block_linkinds = ITensors.linkinds(mps)[q:q:end]
        linkinds_dims = [ind.space for ind in block_linkinds]

        # prime left index of each block to distinguish it from right index of previous block
        for i in 2:newN
            ind_to_increase = ITensors.uniqueinds(blockMPS[i], block_linkinds)
            ind_to_increase = ITensors.commoninds(ind_to_increase, blockMPS[i-1])
            ITensors.replaceinds!(blockMPS[i], ind_to_increase, ind_to_increase')
        end

        ## Start variational optimization to approximate P blocks

        # construct projectors onto |0> for all sites
        zero_projs = []
        for ind in block_linkinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, ITensors.ITensor(vec, ind'))
            push!(zero_projs, ITensors.ITensor(vec, ind''))
        end

        # prepare iteration
        current_env = copy(blockMPS)        # copy current environment
        npairs = newN-1                     # number of bell pairs
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
            Winds = ITensors.inds(env_tensor)                       # prepare W's input legs
            env_tensor = env_tensor * zero_j[1] * zero_j[2]         # construct environment tensor Ej
            # svd environment and construct Wj
            Uenv, Senv, Venv_dg = ITensors.svd(env_tensor, Winds, cutoff = 1E-14)       # SVD environment
            u, v = ITensors.commonind(Uenv, Senv), ITensors.commonind(Venv_dg, Senv)
            W_j = Uenv * ITensors.replaceind(Venv_dg, v, u)       # Construct Wj as UVdag

            newfid = real(tr(Array(Senv, (u, v))))^2
            # println("Step $j: ", newfid)

            # stop if fidelity converged
            if isapprox(newfid, fid) && (first_sweep == false)
                break
            end
            fid = newfid

            # store W_j in W_list
            if first_sweep
                push!(W_list, W_j)
            else
                Wnew = ITensors.replaceinds(W_j, Winds, Winds'''')
                Wold = ITensors.replaceinds(W_list[j], Winds'', Winds'''')
                W_list[j] = Wold * Wnew
            end

            # dim = reduce(*, [ind.space for ind in Winds])
            # W_matrix = reshape(Array(W_j, [Winds; Winds'']), (dim, dim))
            # U_matrix = iso_to_unitary(W_matrix)
            # W_j = ITensors.ITensor(U_matrix, [Winds, Winds''])

            # update current environment by applying Wdag
            Pj, Pjp1 = current_env[j], current_env[j+1]
            update_env = conj(W_j) * Pj * Pjp1
            # replace zero_j inds with Winds (i.e. remove '' from each index)
            zero_j_inds = [ITensors.inds(zero)[1] for zero in zero_j]
            ITensors.replaceinds!(update_env, zero_j_inds, Winds)

            Pjnew, S, V = ITensors.svd(update_env, ITensors.uniqueinds(Pj, Pjp1), cutoff = 1E-14)
            Pjp1new = S*V
            current_env[j], current_env[j+1] = Pjnew, Pjp1new
            
            ## while loop end conditions

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


        ## Extract unitary matrices that must be applied on bell pairs
        # here we follow the sequential rg procedure
        U_list::Vector{Vector{ITensors.ITensor}} = []
        linkinds = ITensors.linkinds(mps)
        
        for i in 1:newN

            block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]
   
            # if i=1 no svd must be done, entire block 1 is already a series of isometries
            if i == 1
                iR = block_linkinds[1]
                block_i[end] = ITensors.replaceind(block_i[end], iR, iR')
                push!(U_list, block_i)
                continue
            end
            
            # prepare left index of block i
            block_i_Ulist = []
            iL = block_linkinds[i-1]
            local prev_SV
# 
            for j in 1:q-1
                A = block_i[j]                              # j-th tensor in block_i
                iR = ITensors.commoninds(A, linkinds)[2]    # right link index
                Aprime = j > 1 ? prev_SV*A : A              # contract SV from previous svd (of the block on the left)
                Uprime, S, V = ITensors.svd(Aprime, ITensors.uniqueinds(Aprime, iR, iL))
                push!(block_i_Ulist, Uprime)
                prev_SV = S*V
            end
            # now we need to consider the last block C and apply P^-1
            C = block_i[end]
            P = blockMPS[i]
#           
            if i < newN     #if we are not at the end of the whole spin chain
                # extract P indices, convert to matrix and invert
                PiL, PiR = block_linkinds[i-1:i]
                dim = reduce(*, linkinds_dims[i-1:i])
                P_matrix = reshape(Array(P, [PiL'', PiR', PiL, PiR]), (dim, dim))
                Pinv = inv(P_matrix)
              
                # convert back to tensor with inds ready to contract with Ctilde = prev_SV * C          
                CindR = ITensors.commoninds(C, linkinds)[2]
                Pinv = ITensors.ITensor(Pinv, [iL, CindR, iL'', CindR'])
            else    #same here, only different indices
                PiL = block_linkinds[i-1]
                dim = linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PiL'', PiL]), (dim, dim))
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
        zero_projs::Vector{ITensors.ITensor} = []
        for ind in block_linkinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, ITensors.ITensor(vec, ind^3))
            push!(zero_projs, ITensors.ITensor(vec, ind^4))
        end
        # act with W on zeros
        Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]
# 
        # group blocks together
        block_list::Vector{ITensors.ITensor} = []
        mps_primed = conj(mps)'''''
        siteinds = ITensors.siteinds(mps_primed)
        local left_tensor
        for i in 1:newN
            block_i = i < newN ? mps_primed[q*(i-1)+1 : q*i] : mps_primed[q*(i-1)+1 : end]
            i_siteinds = i < newN ? siteinds[q*(i-1)+1 : q*i] : siteinds[q*(i-1)+1 : end]
            left_tensor = (i == 1 ? block_i[1] : block_i[1]*left_tensor)
            left_tensor *= ITensors.replaceinds(U_list[i][1], ITensors.noprime(i_siteinds), i_siteinds)

            for j in 2:q
                left_tensor *= block_i[j] * ITensors.replaceinds(U_list[i][j], ITensors.noprime(i_siteinds), i_siteinds)
            end
            if i < newN
                left_tensor *= Wlayer[i]
            end
        end


        fidelity = abs(Array(left_tensor)[1])

        eps = 1-sqrt(fidelity)
        if eps < 0
            println("error is negative, saving -err at position $run")
            push!(neg_eps_pos, run)
        end
        eps = abs(eps)
        eps_per_block = eps/newN
        push!(eps_per_block_array, eps_per_block)

        if mod(run, 100) == 0
            println("$run done")
        end

    end

    avg_perq = mean(eps_per_block_array)
    err_perq = std(eps_per_block_array)
# 
    push!(avg_list, avg_perq)
    push!(err_list, err_perq)
    push!(all_runs_list, eps_per_block_array)

    # break if it reached convergence
    if avg_perq < 1E-14
        push!(t_vs_q, (t, q))
        break
    end

end

end


swapped_results = [getindex.(all_runs_list,i) for i=1:length(all_runs_list[1])]

Plots.plot(q_list, swapped_results, lc=:gray90, legend=false)
Plots.plot!(q_list, swapped_results, seriestype=:scatter, mc=:gray90, markersize=:3, legend=false)
Plots.plot!(q_list, avg_list, lc=:green)
Plots.plot!(q_list, avg_list, seriestype=:scatter, mc=:green)
Plots.plot!(ylims = (1E-20, 1), yscale=:log)
Plots.plot!(title = L"N="*string(N), ylabel = L"\epsilon / M", xlabel = L"q")

#Plots.savefig("D:\\Julia\\MyProject\\Plots\\err_per_block_1600.pdf");

#writedlm( "all_runs_list_1600.csv",  all_runs_list, ',')