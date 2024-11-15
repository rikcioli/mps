include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles


function invertMPSMalz(mps::itmps.MPS; q = 0, eps_malz = 1E-2, eps_bell = 1E-2, eps_V = 1E-2, kargs...)
    N = length(mps)
    siteinds = it.siteinds(mps)
    linkinds = it.linkinds(mps)
    linkdims = [ind.space for ind in linkinds]
    D = linkdims[div(N,2)]
    d = siteinds[div(N,2)].space
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D

    local circuit, blockMPS, new_siteinds, newN, npairs, block_linkinds, block_linkinds_dims

    q_list = iszero(q) ? [i for i in 2:N if mod(N,i) == 0] : [q]

    local q_found
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
        blocked_mps, blocked_siteinds = mt.blocking(mps, q)
        # polar decomp and store P matrices in array
        blockMPS = [mt.polar_P(blocked_mps[i], blocked_siteinds[i]) for i in 1:newN]

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
        circuit, fid, sweep = mt.invertBW(conj(sep_mps), 1, new_siteinds; d = D, kargs...)
        
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

            total_tau = 0
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
        V_mpo = mt.unitary_to_mpo(mt.iso_to_unitary(V_matrix), siteinds = upinds)
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
            uk_mpo = mt.unitary_to_mpo(mt.iso_to_unitary(uk_matrix), siteinds = upinds, skip_qudits = k-2*bitlength-1)
            it.prime!(uk_mpo, q-k)
            V_mpo = V_mpo*uk_mpo
        end

        it.replaceprime!(V_mpo, q-2*bitlength+1 => 1)
        # invert final V
        _, _, _, tau = mt.invertBW(V_mpo; d = d, err_to_one = eps_V/(newN*d^N), kargs...)
        
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

    println(fid)

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
    results = [mt.invertBW(Wmps; d = d, err_to_one = eps_bell/npairs) for Wmps in Wmps_list]

    W_tau_list = [res[4] for res in results]

    return Wlayer, V_list, fid, sweep, W_tau_list, V_tau_list
end

# Prepare initial state

N = 12
tau = 2
inverter_tau = 4 # 8 IS MAX TO CONTRACT ENVIRONMENT

"Select case:
1: FDQC of depth tau
2: Random MPS of bond dim tau
3: GHZ"
case = 2
proj = []

# unitary = mt.CX * kron(mt.H, mt.Id)
# sites = it.siteinds("Qubit", 4)
# mpo3 = mt.unitary_to_mpo(unitary; siteinds = sites, skip_qudits = 2)''
# mpo2 = mt.unitary_to_mpo(unitary; siteinds = sites, skip_qudits = 1)'
# mpo1 = mt.unitary_to_mpo(unitary; siteinds = sites)
# 
# mpo = mpo3 * mpo2 * mpo1
# results = mt.invertBW(mpo)

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

    # NOTE: if malz is used, inverter_tau plays the role of q
    global W_list, U_list, fid, sweep, W_tau_list, U_tau_list = invertMPSMalz(mps; eps_bell = 1E-2, eps_V = 0.1)
    println("Algorithm stopped after $sweep sweeps \nFidelity = $fid")

    #U = U_list[1][3]
    #inputind = it.inds(U_list[1][3])[end]
    #results = mt.invertBW(U, inputind, n_sweeps = 100)

    #W = reshape(Array(W_list[1], it.inds(W_list[1])), tau^2)
    #sites = it.siteinds(2, 4)
    #W = itmps.MPS(W, sites)

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