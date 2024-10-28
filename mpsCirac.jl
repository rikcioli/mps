include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles



# Prepare initial state

N = 24
q_list = [2, 3, 4, 6, 8, 12]
avg_list::Vector{Float64} = []
all_runs_list::Vector{Vector{Float64}} = []
err_list::Vector{Float64} = []
Nruns = 1

neg_eps_pos::Vector{Int64} = []
n_sweeps = 10


# Loop over different values of q
for q in q_list
    if mod(N, q) != 0
        throw(DomainError(q, "Inhomogeneous blocking is not supported, choose a q that divides N"))
    end
    println("Evaluating blocking size q = $q...")

    eps_per_block_array::Vector{Float64} = []

    # Loop over different random mps
    for run in 1:Nruns

        #sites = it.siteinds("Qubit", N)
        #randMPS = rand_MPS(sites, linkdims=2)
        #it.orthogonalize!(randMPS, N)

        testMPS = mt.initialize_fdqc(N, 4)
        #testMPS = apply(CX * kron(H, Id), testMPS, 4)
        it.orthogonalize!(testMPS, N)

        newN = mod(N,q) == 0 ? div(N,q) : div(N,q)+1
        mps = testMPS
        sites = it.siteinds(mps)

        # block array
        blocked_mps, blocked_siteinds = mt.blocking(mps, q)
        # polar decomp and store P matrices in array
        blockMPS = [mt.polar_P(blocked_mps[i], blocked_siteinds[i]) for i in 1:newN]
        block_linkinds = it.linkinds(mps)[q:q:end]
        linkinds_dims = [ind.space for ind in block_linkinds]

        # prime left index of each block to distinguish it from right index of previous block
        for i in 2:newN
            ind_to_increase = it.uniqueinds(blockMPS[i], block_linkinds)
            ind_to_increase = it.commoninds(ind_to_increase, blockMPS[i-1])
            it.replaceinds!(blockMPS[i], ind_to_increase, ind_to_increase')
        end

        ## Start variational optimization to approximate P blocks
        
        # prepare iteration
        L_blocks::Vector{it.ITensor} = []
        R_blocks::Vector{it.ITensor} = []
        npairs = newN-1     # number of bell pairs

        # prepare array of initial random W tensors
        Wdg_list::Vector{it.ITensor} = []     
        for j in 1:npairs
            ind = block_linkinds[j]
            d = ind.space
            push!(Wdg_list, it.ITensor(mt.random_unitary(d^2), ind''', ind'''', ind', ind''))
        end

        # construct projectors onto |0> for all sites
        zero_projs::Vector{it.ITensor} = []
        for ind in block_linkinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, it.ITensor(vec, ind'''))
            push!(zero_projs, it.ITensor(vec, ind''''))
        end

        # construct L_1
        leftmost_block = blockMPS[1]
        push!(L_blocks, leftmost_block)

        # construct R_N
        rightmost_block = blockMPS[end]
        push!(R_blocks, rightmost_block) 

        # contract everything on the right and save rightmost_block at each intermediate step
        # must be done only the first time, when j=1 (so contract up to j=2)
        for k in npairs:-1:2
            block_k = blockMPS[k]   # extract block k
            zero_L, zero_R = zero_projs[2k-1:2k]    # extract zero projectors for this pair
            Wdg_k = Wdg_list[k]     # extract Wdg_k
            rightmost_block *= block_k * Wdg_k * zero_L * zero_R 
            push!(R_blocks, rightmost_block)
        end
        reverse!(R_blocks)

        # start the loop        
        first_sweep = true
        reverse = false
        sweep = 0

        j = 1   # W1 position (pair)
        fid = 0

        # Start iteration
        while sweep < n_sweeps

            # build environment tensor by adding the two |0> projectors 
            env_tensor = leftmost_block * rightmost_block 
            Winds = it.inds(env_tensor)     # extract W's input legs
            zero_L, zero_R = zero_projs[2j-1:2j] 
            env_tensor *= zero_L * zero_R         # construct environment tensor Ej

            # svd environment and construct Wj
            Uenv, Senv, Venv_dg = it.svd(env_tensor, Winds, cutoff = 1E-14)       # SVD environment
            u, v = it.commonind(Uenv, Senv), it.commonind(Venv_dg, Senv)
            W_j = Uenv * it.replaceind(Venv_dg, v, u)       # Construct Wj as UVdag

            # Save optimized Wdg_j in the list
            Wdg_j = conj(W_j) 
            Wdg_list[j] = Wdg_j

            newfid = real(tr(Array(Senv, (u, v))))^2
            println("Step $j: ", newfid)

            # stop if fidelity converged
            if abs(1 - newfid) < 1E-10
                break
            end
            fid = newfid

            if npairs == 1
                break
            end

            if j == npairs
                first_sweep = false
                reverse = true
                sweep += 1
            end

            if j == 1 && first_sweep == false
                reverse = false
                sweep += 1
            end

            # update L_blocks or R_blocks depending on sweep direction
            if reverse == false
                next_block = blockMPS[j+1]
                leftmost_block *= next_block * Wdg_j * zero_L * zero_R 
                if first_sweep == true
                    push!(L_blocks, leftmost_block)
                else
                    L_blocks[j+1] = leftmost_block
                end
                rightmost_block = R_blocks[j+1]
            else
                prev_block = blockMPS[j]
                rightmost_block *= prev_block * Wdg_j * zero_L * zero_R
                R_blocks[j-1] = rightmost_block
                leftmost_block = L_blocks[j-1]
            end

            # alternate procedure: APPLY new W_j ON TOP OF old W_j
            #Wnew = it.replaceinds(W_j, Winds, Winds'''')
            #Wold = it.replaceinds(W_list[j], Winds'', Winds'''')
            #W_list[j] = Wold * Wnew

            # dim = reduce(*, [ind.space for ind in Winds])
            # W_matrix = reshape(Array(W_j, [Winds; Winds'']), (dim, dim))
            # U_matrix = iso_to_unitary(W_matrix)
            # W_j = it.ITensor(U_matrix, [Winds, Winds''])

            j = reverse ? j-1 : j+1
        end

        # Save W matrices
        #W_matrices = [reshape(Array(W, it.inds(W)), (it.inds(W)[1].space^2, it.inds(W)[1].space^2)) for W in W_list]
        #W_unitaries = [iso_to_unitary(W) for W in W_matrices]


        ## Extract unitary matrices that must be applied on bell pairs
        # here we follow the sequential rg procedure
        U_list::Vector{Vector{it.ITensor}} = []
        linkinds = it.linkinds(mps)
        
        for i in 1:newN

            block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]
   
            # if i=1 no svd must be done, entire block 1 is already a series of isometries
            if i == 1
                iR = block_linkinds[1]
                block_i[end] = it.replaceind(block_i[end], iR, iR')
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
                iR = it.commoninds(A, linkinds)[2]    # right link index
                Aprime = j > 1 ? prev_SV*A : A              # contract SV from previous svd (of the block on the left)
                Uprime, S, V = it.svd(Aprime, it.uniqueinds(Aprime, iR, iL))
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
                CindR = it.commoninds(C, linkinds)[2]
                Pinv = it.ITensor(Pinv, [iL, CindR, iL'', CindR'])
            else    #same here, only different indices
                PiL = block_linkinds[i-1]
                dim = linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PiL'', PiL]), (dim, dim))
                Pinv = inv(P_matrix)
# 
                Pinv = it.ITensor(Pinv, [iL, iL''])
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
        # act with W on zeros
        Wlayer = [conj(Wdg_list[i]) * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]
# 
        # group blocks together
        block_list::Vector{it.ITensor} = []
        mps_primed = conj(mps)'''''
        siteinds = it.siteinds(mps_primed)
        local left_tensor
        for i in 1:newN
            block_i = i < newN ? mps_primed[q*(i-1)+1 : q*i] : mps_primed[q*(i-1)+1 : end]
            i_siteinds = i < newN ? siteinds[q*(i-1)+1 : q*i] : siteinds[q*(i-1)+1 : end]
            left_tensor = (i == 1 ? block_i[1] : block_i[1]*left_tensor)
            left_tensor *= it.replaceinds(U_list[i][1], it.noprime(i_siteinds), i_siteinds)

            for j in 2:q
                left_tensor *= block_i[j] * it.replaceinds(U_list[i][j], it.noprime(i_siteinds), i_siteinds)
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

end



#swapped_results = [getindex.(all_runs_list,i) for i=1:length(all_runs_list[1])]
#
#Plots.plot(q_list, swapped_results, lc=:gray90, legend=false)
#Plots.plot!(q_list, swapped_results, seriestype=:scatter, mc=:gray90, markersize=:3, legend=false)
#Plots.plot!(q_list, avg_list, lc=:green)
#Plots.plot!(q_list, avg_list, seriestype=:scatter, mc=:green)
#Plots.plot!(ylims = (1E-20, 1), yscale=:log)
#Plots.plot!(title = L"N="*string(N), ylabel = L"\epsilon / M", xlabel = L"q")

#Plots.savefig("D:\\Julia\\MyProject\\Plots\\err_per_block_1600.pdf");

#writedlm( "all_runs_list_1600.csv",  all_runs_list, ',')