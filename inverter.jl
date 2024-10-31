include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles

"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qudit unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertBW(mpo::Vector{it.ITensor}, tau::Int64, input_inds::Vector{<:it.Index}; d::Int64 = 2, output_inds = nothing, err::Float64 = 1E-10, n_sweeps::Int64 = 1000)
    N = length(mpo)
    siteinds = input_inds

    L_blocks::Vector{it.ITensor} = []
    R_blocks::Vector{it.ITensor} = []

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    circuit::Vector{Vector{it.ITensor}} = []
    for i in 1:tau
        layer_i = [it.prime(it.ITensor(mt.random_unitary(d^2), siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]'), tau-i) for j in 1:(div(N,2)-mod(N+1,2)*mod(i+1,2))]
        push!(circuit, layer_i)
    end

    # construct projectors for all sites
    # if state is mps, construct projectors onto |0>
    # if output_inds are given, construct deltas connecting output_inds to beginning of brickwork (trace)
    zero_projs::Vector{it.ITensor} = []

    if isnothing(output_inds)
        for ind in siteinds
            vec = [1; [0 for _ in 1:ind.space-1]]
            push!(zero_projs, it.ITensor(vec, it.prime(ind, tau)))
        end
    else
        for inds_pair in zip(siteinds, output_inds)
            push!(zero_projs, it.delta(inds_pair[2], it.prime(inds_pair[1], tau)))
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
        newfid = real(tr(Array(S, (u, v))))^2
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        circuit[1][1] = gate_ji_opt
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
            U, S, Vdag = it.svd(env, inds, cutoff = 1E-14)
            u, v = it.commonind(U, S), it.commonind(Vdag, S)

            # evaluate fidelity
            newfid = real(tr(Array(S, (u, v))))^2
            #println("Step $j: ", newfid)

            #replace gate_ji with optimized one, both in gates_j (used in this loop) and in circuit
            gate_ji_opt = U * it.replaceind(Vdag, v, u)
            gates_j[i] = gate_ji_opt    
            circuit[i][div(j,2) + mod(i,2)*mod(j,2)] = gate_ji_opt
        end

        ## while loop end conditions

        ## if isapprox(newfid, fid) && (first_sweep == false)
        ##     break
        ## end
        if abs(1 - newfid) < err
            break
        end
        fid = newfid

        if j == N-1
            first_sweep = false
            rev = true
            sweep += 1
        end

        if j == 2 && first_sweep == false
            rev = false
            sweep += 1
        end

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

    return circuit, newfid, sweep

end


function invertMPSMalzNEW(mps::itmps.MPS, q::Int64; kargs...)

    N = length(mps)
    if mod(N, q) != 0
        throw(DomainError(q, "Inhomogeneous blocking is not supported, choose a q that divides N"))
    end
    linkdims = [ind.space for ind in it.linkinds(mps)]
    if length(Set(linkdims)) > 1
        throw(DomainError(linkdims, "MPS has non-constant bond dimension"))
    end
    d = linkdims[1]
    newN = div(N,q)
    npairs = newN-1

    # block array
    blocked_mps, blocked_siteinds = mt.blocking(mps, q)
    # polar decomp and store P matrices in array
    blockMPS = [mt.polar_P(blocked_mps[i], blocked_siteinds[i]) for i in 1:newN]

    # save linkinds and create new siteinds
    block_linkinds = it.linkinds(mps)[q:q:end]
    linkinds_dims = [ind.space for ind in block_linkinds]
    
    new_siteinds = it.siteinds(d, 2*(newN-1))

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
    circuit, fid, sweep = invertBW(conj(sep_mps), 1, new_siteinds; d = d, kargs...)
    W_list = circuit[1]

    ## Extract unitary matrices that must be applied on bell pairs
    # here we follow the sequential rg procedure
    U_list::Vector{Vector{it.ITensor}} = []
    linkinds = it.linkinds(mps)
    
    for i in 1:newN

        block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]

        # if i=1 no svd must be done, entire block 1 is already a series of isometries
        if i == 1
            iR = block_linkinds[1]
            block_i[end] = it.replaceind(block_i[end], iR, new_siteinds[1])
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
            PnL, PnR = new_siteinds[2*i-2:2*i-1]
            dim = reduce(*, linkinds_dims[i-1:i])
            P_matrix = reshape(Array(P, [PnL, PnR, PiL, PiR]), (dim, dim))
            Pinv = inv(P_matrix)
          
            # convert back to tensor with inds ready to contract with Ctilde = prev_SV * C   
            # and with blockMPS[i] siteinds       
            CindR = it.commoninds(C, linkinds)[2]
            Pinv = it.ITensor(Pinv, [iL, CindR, PnL, PnR])
        else    #same here, only different indices
            PiL = block_linkinds[i-1]
            PnL = new_siteinds[2*i-2]
            dim = linkinds_dims[i-1]
            P_matrix = reshape(Array(P, [PnL, PiL]), (dim, dim))
            Pinv = inv(P_matrix)
#           
            Pinv = it.ITensor(Pinv, [iL, PnL])
        end
# 
        Ctilde = prev_SV * C * Pinv
        push!(block_i_Ulist, Ctilde)
        push!(U_list, block_i_Ulist)
    end
# 
# 
    ## Compute scalar product to check fidelity

    zero_projs::Vector{it.ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, ind'))
    end
# 
    # act with W on zeros
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]
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


    fidelity = abs2(Array(left_tensor)[1])

    println(fid)
    println(fidelity)

    return fidelity, sweep
end


# Prepare initial state

N = 10
tau = 5
inverter_tau = 2 # 8 IS MAX TO CONTRACT ENVIRONMENT

"Select case:
1: FDQC of depth tau
2: Random MPS of bond dim tau
3: GHZ"
case = 2
proj = []

# GHZ

# Random bond D mps
# testMPS = rand_MPS(it.siteinds("Qubit", N), linkdims = 4)

let mps
    if case == 1
        # Random FDQC of depth tau
        mps = mt.initialize_fdqc(N, tau)

        if length(proj) > 0
            mps = mt.project_tozero(mps, proj)
        end

    elseif case == 2
        mps = mt.randMPS(it.siteinds("Qubit", N), tau)
    
    elseif case == 3
        mps = mt.initialize_ghz(N)
    end

    fid, sweep = mt.invertMPSMalzNEW(mps, inverter_tau, n_sweeps = 1000)
    println("Algorithm stopped after $sweep sweeps \nFidelity = $fid")
    fid, sweep = mt.invertMPSMalz(mps, inverter_tau, n_sweeps = 1000)
    println("Algorithm stopped after $sweep sweeps \nFidelity = $fid")
end





#unitaries = [reshape(Array(circ[i,j], it.inds(circ[i,j])), (4, 4)) for i in 1:tau, j in 1:div(N,2)]

## function execute(N::Int64; state_depth = 2, n_sweeps = 100, n_runs = 1000)
## 
##     fid1_depth_run::Vector{Vector{Float64}} = []
##     fid2_depth_run::Vector{Vector{Float64}} = []
##     avg_fid1::Vector{Float64} = []
##     avg_fid2::Vector{Float64} = []
## 
##     for depth in 1:5
##         println("Running depth $depth")
##         fid1_run::Vector{Float64} = []
##         fid2_run::Vector{Float64} = []
## 
##         for run in 1:n_runs
##             if mod(run, 100) == 0
##                 println("Run $run of n_runs")
##             end
##             # create random FDQC of depth 2
##             testMPS = initialize_vac(N)
##             siteinds = it.siteinds(testMPS)
##             testMPS = brickwork(testMPS, state_depth)
## 
##             # try to invert it
##             fid1, sweep1 = invertMPS(testMPS, depth, n_sweeps)
##             err1 = 1-sqrt(fid1)
##             push!(fid1_run, err1)
## 
##             # measure middle qubit
##             pos = div(N,2)
##             zero_projs::Vector{it.ITensor} = []
##             for ind in siteinds
##                 vec = [1; [0 for _ in 1:ind.space-1]]
##                 push!(zero_projs, it.ITensor(kron(vec, vec'), ind, ind'))
##             end
##             proj = zero_projs[pos]
##             testMPS[pos] = it.noprime(testMPS[pos]*proj)
##             it.normalize!(testMPS)
## 
##             # try to invert measured one
##             fid2, sweep2 = invertMPS(testMPS, depth, 100)
##             err2 = 1-sqrt(fid2)
##             push!(fid2_run, err2)
##         end
##         push!(fid1_depth_run, fid1_run)
##         push!(fid2_depth_run, fid2_run)
##         push!(avg_fid1, mean(fid1_run))
##         push!(avg_fid2, mean(fid2_run))
##     end
## 
##     return fid1_depth_run, fid2_depth_run, avg_fid1, avg_fid2
## 
## end
## 
## nruns = 10
## err1_all, err2_all, err1, err2 = execute(21, state_depth = 3, n_runs = nruns)
## 
## swapped_results1 = [getindex.(err1_all,i) for i=1:length(err1_all[1])]
## swapped_results2 = [getindex.(err2_all,i) for i=1:length(err2_all[1])]
## 
## Plots.plot(1:5, swapped_results1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err1, lc=:green, primary=false)
## Plots.plot!(1:5, err1, seriestype=:scatter, mc=:green, legend=true, label="Depth-3")
## Plots.plot!(yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"\epsilon = 1-|\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## 
## Plots.plot!(1:5, swapped_results2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err2, lc=:red, primary=false)
## Plots.plot!(1:5, err2, seriestype=:scatter, mc=:red, label="Depth-3 + measure", legend=:bottomright)
## #Plots.plot!(title = L"N="*string(N), ylabel = L"\epsilon / M", xlabel = L"\tau")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\err_depth3.pdf");
## 
## 
## fid1, fid2 = 1 .- err1, 1 .- err2
## 
## swapped_fid1 = [1 .- swapped_results1[i, 1] for i in 1:nruns]
## swapped_fid2 = [1 .- swapped_results2[i, 1] for i in 1:nruns]
## 
## Plots.plot(1:5, swapped_fid1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, swapped_fid2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, fid1, lc=:green, primary=false)
## Plots.plot!(1:5, fid1, seriestype=:scatter, mc=:green, label="Depth-3", legend=:bottomright)
## Plots.plot!(ylims = (1E-1, 1.2), yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"F = |\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## Plots.plot!(1:5, fid2, lc=:red, primary=false)
## Plots.plot!(1:5, fid2, seriestype=:scatter, mc=:red, label="Depth-3 + measure")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\fid_depth3.pdf");