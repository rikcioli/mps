module MPSMethods

import ITensorMPS as itmps
import ITensors as it
using LinearAlgebra, Statistics, Random, OptimKit


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

const SWAP = [1 0 0 0
              0 0 1 0
              0 1 0 0
              0 0 0 1]

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
    
include("optFunctions.jl")
include("lightcone.jl")
include("ITMethods.jl")
include("varMethods.jl")


function invertMPSMalz(mps::itmps.MPS; q = 0, eps_malz = 1E-2, eps_bell = 1E-2, eps_V = 1E-2, kargs...)
    N = length(mps)
    siteinds = it.siteinds(mps)
    linkinds = it.linkinds(mps)
    linkdims = [ind.space for ind in linkinds]
    D = maximum(linkdims)
    d = siteinds[div(N,2)].space
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D

    # adjust bond dimension so that it's the same everywhere (except boundaries)
    for j in bitlength : N-bitlength
        link = linkinds[j]
        if link.space < D
            it.orthogonalize!(mps, j+1)
            block = mps[j]*mps[j+1]
            U, S, Vdag = it.svd(block, (siteinds[j], linkinds[j-1]), cutoff=1e-15)
            Sinds = (it.commonind(S, U), it.commonind(S, Vdag))
            Svec = diag(Array(S, Sinds))
            Svec_ext = [Svec; [0 for _ in 1:(D-link.space)]]
            S = it.ITensor(diagm(Svec_ext), Sinds)
            mps[j] = U
            mps[j+1] = S*Vdag
        end
    end

    @show mps


    local blockMPS, new_siteinds, newN, npairs, block_linkinds, block_linkinds_dims

    q_list = iszero(q) ? [i for i in 2:N if (mod(N,i) == 0 && i-2*bitlength >= 0)] : [q]

    local q_found, circuit, overlap, sweepB
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
            throw(DomainError(D, "Malz inversion only works if q - 2n >= 0, with n = logd(D) \nMax bond dimension D = $D, blocking size q = $q."))
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
        circuit, overlap, sweepB = invertSweep(conj(sep_mps), 1, new_siteinds; d = D, kargs...)
        
        if abs(1 - overlap) < eps_malz
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
        # @show typeof(V_matrix)
        # @show typeof(iso_to_unitary(V_matrix))
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
        _, _, _, tau = invertSweep(V_mpo; d = d, eps = eps_V/((newN^2)*d^q), kargs...)
        
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
    results = [invertSweep(Wmps; d = d, eps = eps_bell/npairs) for Wmps in Wmps_list]

    W_tau_list = [res[4] for res in results]

    return Wlayer, V_list, overlap, sweepB, W_tau_list, V_tau_list
end


# ONLY WORKS FOR d=2
function invertMPSMalz(mps::itmps.MPS, invertMethod; q = 0, eps = 1e-3, kargsP = NamedTuple(), kargsV = NamedTuple(), kargsW = NamedTuple())
    N = length(mps)
    siteinds = it.siteinds(mps)
    linkinds = it.linkinds(mps)
    linkdims = [ind.space for ind in linkinds]
    D = maximum(linkdims)
    d = siteinds[div(N,2)].space
    @assert d == 2
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D

    eps_malz = eps/9
    eps_V = eps/9
    eps_bell = eps/9
    println("Attempting inversion of MPS with invertMPSMalz and errors:\neps_malz = $eps_malz\neps_V = $eps_V\neps_bell = $eps_bell")

    local blockMPS, new_siteinds, newN, npairs, block_linkinds, block_linkinds_dims

    q_list = iszero(q) ? [i for i in 2:N-1 if (mod(N,i) == 0 && i-2*bitlength >= 0)] : [q]  # make sure q is large enough based on D

    # find out if input state has uniform bond dimension in the bulk; if not, it means it's a FDQC and we can only block an even number of sites
    # otherwise the different bond dimensions will cause problems
    bulkdims = linkdims[bitlength+1 : N-1-bitlength]
    if length(Set(bulkdims)) > 1
        @show Set(bulkdims)
        q_list = [i for i in q_list if iseven(i)]
    end

    local q_found, circuit, err
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
        circuit, overlap, sweepB = invertSweep(conj(sep_mps), 1, new_siteinds; d = D, kargsP...)
        err = 1-overlap

        if err < eps_malz
            break
        end
    end
    
    q = q_found
    W_list = circuit[1]

    ## Extract unitary matrices that must be applied on bell pairs
    # here we follow the sequential rg procedure
    V_list::Vector{Vector{it.ITensor}} = []
    V_mpo_list = []

    for i in 1:newN

        block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]
        block_i_Ulist::Vector{it.ITensor} = []

        # if i=1 no svd must be done, entire block 1 is already a series of isometries
        if i == 1
            iR = block_linkinds[1]
            block_i[end] = it.replaceind(block_i[end], iR, new_siteinds[1])
            block_i_Ulist = block_i
            push!(V_list, block_i_Ulist)

        else
            # prepare left index of block i
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
                cup, clow = it.combiner(PnL, PnR), it.combiner(PiL, PiR)
                P = (P*cup)*clow
                P_matrix = Array(P, (it.combinedind(cup), it.combinedind(clow)))
                Pinv = inv(P_matrix)
            
                # convert back to tensor with inds ready to contract with Ctilde = prev_SV * C   
                # and with blockMPS[i] siteinds       
                CindR = it.commoninds(C, linkinds)[2]
                Pinv = it.ITensor(Pinv, (iL, CindR, PnL, PnR))
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

        # Convert total V_i unitary to MPO efficiently
        # contract isometries in block_i_Ulist as long as dimension <= d^2bitlength (efficient for D low, independent from q)
        upinds = siteinds[q*(i-1)+1 : q*i]
        V = reduce(*, block_i_Ulist[1:2*bitlength])
        V_upinds = upinds[1:2*bitlength]
        V_downinds = it.uniqueinds(V, V_upinds)
        down_d = reduce(*, [ind.space for ind in V_downinds])  # will be D for i=1, D^2 elsewhere
        V_matrix = reshape(Array(V, V_upinds, V_downinds), (d^(2*bitlength), down_d))      # MUST BE MODIFIED WHEN i=1 or newN THE OUTPUT LEG IS ONLY D
        sizeV = size(V_matrix)
        @show sizeV

        if sizeV[1] != sizeV[2]
            # we promote the isometry to a unitary by appending orthonormal columns
            # note that the appended columns will end up on the lower right index, while we
            # want them to be associated with the lower left index since that is the index
            # that gets contracted with |0> ad the end;
            # the only exception is when i == newN and q == bitlength, where the contraction
            # with |0> will be on the lower right indices
            V_matrix = iso_to_unitary(V_matrix)
            if i < newN || 2*bitlength < q
                remainder = div(d^(2*bitlength), down_d)
                V_matrix = reshape(V_matrix, (d^(2*bitlength), down_d, remainder))
                V_matrix = permutedims(V_matrix, [1,3,2])
                V_matrix = reshape(V_matrix, (d^(2*bitlength), d^(2*bitlength)))
            end
        end
        
        V_mpo = unitary_to_mpo(V_matrix, siteinds = upinds)
        it.prime!(V_mpo, q - 2*bitlength)   # we prime to contract with the next MPOs
        
        # transform next isometries to mpo and contract with previous ones
        for k in 2*bitlength+1:q
            uk = block_i_Ulist[k]
            prev_u = block_i_Ulist[k-1]
            uk_upinds = [it.commoninds(uk, prev_u); upinds[k]]
            uk_downinds = it.uniqueinds(uk, uk_upinds)
            up_d = reduce(*, [ind.space for ind in uk_upinds])
            down_d = reduce(*, [ind.space for ind in uk_downinds])
            uk_matrix = reshape(Array(uk, uk_upinds, uk_downinds), (up_d, down_d))
            size_uk = size(uk_matrix)
            @show size_uk

            # # complete and permute uk_matrix if needed
            if size_uk[1] != size_uk[2]
                # same reasoning applied here, with the sole exception of the last gate of i == newN
                # where the contraction with |0> will be on the lower right indices
                uk_matrix = iso_to_unitary(uk_matrix)
                if i < newN || (i == newN && k < q)
                    remainder = div(up_d, down_d)
                    uk_matrix = reshape(uk_matrix, (up_d, down_d, remainder))
                    uk_matrix = permutedims(uk_matrix, [1,3,2])
                    uk_matrix = reshape(uk_matrix, (up_d, up_d))
                end
            end

            # need to decide how many identities to put to the left of uk unitaries based on up_d
            nskips = k - length(digits(up_d-1, base=d))
            uk_mpo = unitary_to_mpo(uk_matrix, siteinds = upinds, skip_qudits = nskips)
            it.prime!(uk_mpo, q-k)
            V_mpo = V_mpo*uk_mpo
        end

        it.replaceprime!(V_mpo, q-2*bitlength+1 => 1)
        push!(V_mpo_list, V_mpo)
    end
    

    # BELL PAIRS PREPARATION
    # prepare |0> state and act with W
    zero_projs::Vector{it.ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, ind'))
    end
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]

    # turn W|0> states into MPSs
    Wmps_list = []
    for i in 1:npairs
        sitesL = siteinds[q*i-bitlength+1 : q*i]
        sitesR = siteinds[q*i+1 : q*i+bitlength]
        cL = it.combiner(sitesL)
        it.replaceind!(cL, it.combinedind(cL), new_siteinds[2*i-1])
        cR = it.combiner(sitesR)
        it.replaceind!(cR, it.combinedind(cR), new_siteinds[2*i])

        Wi = (Wlayer[i]*cL)*cR
        W_mps = itmps.MPS(Wi, [sitesL; sitesR])
        push!(Wmps_list, W_mps)
    end

    # invert each mps in the list
    # NOTE: INVERTGLOBAL ONLY WORKS ON QUBITS
    results = [invert(Wmps, invertMethod; eps = eps_bell/npairs, kargsW...) for Wmps in Wmps_list]
    W_tau_list = [res["tau"] for res in results]
    W_lc_list::Vector{Lightcone} = [res["lightcone"] for res in results]


    # compute numerically total error by creating a 0 state and applying everything in sequence
    mps_final = initialize_vac(N, siteinds)
    apply!(mps_final, W_lc_list)

    # compare with exact W|0>
    Wexact = [it.ITensor([1; 0], siteinds[i]) for i in 1:N]
    for i in 1:npairs
        for j in q*i-bitlength+1 : q*i+bitlength
            jshift = j-(q*i-bitlength)
            Wexact[j] = Wmps_list[i][jshift]
        end
    end
    Wstep_mps = itmps.MPS(Wexact)
    err_W = 1-abs(Array(contract(conj(Wstep_mps), mps_final))[1])
    @show err_W 

    # apply swaps before the V unitaries
    spacing = q-2*bitlength
    for i in 1:npairs
        for j in i*q+bitlength : -1 : i*q+1
            for t in 1:(i < npairs ? spacing : spacing-1)   #one less swap for the last W gate, since we are putting the |0> contraction on the right instead of the left
                apply!(SWAP, Wstep_mps, j+t-1)
                apply!(SWAP, mps_final, j+t-1)
            end
        end
    end

    # apply V_mpo on Wstep_mps to check that conversion to mpo worked - must be equal to the fidelity obtained in the blocking part
    V_mpo_final = itmps.MPO(reduce(vcat, [mpo[1:end] for mpo in V_mpo_list]))
    res = it.apply(V_mpo_final, Wstep_mps)
    fidelity = abs(Array(contract(conj(mps), res))[1])
    @show fidelity


    # invert all the V_mpos to bw circuits
    eps_V_i = eps_V/(newN^2)
    # proper error should have a d^q factor at the denominator, but it's probably a very loose estimate, so we try not putting it
    # and adjust it iteratively if it does not converge to required fidelity
    start_taus = [2 for _ in 1:newN]
    local V_tau_list, V_lc_list, err_total
    for _ in 1:10
        V_tau_list = []         # final tau list, counting also swap gates
        V_taures_list = []      # intermediate tau list, storing the results of inversion of each mpo
        V_lc_list::Vector{Lightcone} = []
        mps_final_copy = deepcopy(mps_final)
        for i in 1:newN
            V_mpo = V_mpo_list[i]
            res = invert(V_mpo, invertMethod; eps = eps_V_i, start_tau = start_taus[i], kargsV...) # NOTE: FOR NOW invertGlobalSweep ONLY WORKS WITH QUBITS, AS IT OPTIMIZES OVER U(4) UNITARIES
            tau = res["tau"]
            push!(V_taures_list, tau)
            push!(V_lc_list, res["lightcone"])
            
            # account for the swap gates 
            if i > 1
                tau += q - bitlength - 1
            end
            push!(V_tau_list, tau)
        end

        # apply V unitaries as fdqc on mps_final to construct the final inversion circuit
        apply!(mps_final_copy, V_lc_list)
        err_total = 1-abs(Array(contract(conj(mps), mps_final_copy))[1])
        @show err_total
        if err_total < eps
            break
        else
            println("Convergence to required total error eps = $eps not found, decreasing eps_V")
            eps_V_i *= 0.5
            start_taus = V_taures_list
        end
    end


    return Dict([("V_lc", V_lc_list), ("V_tau", V_tau_list), ("W_lc", W_lc_list), ("W_tau", W_tau_list), ("err_total", err_total)])
end




"Cost function and gradient for invertGlobalSweep optimization"
function _fgLiu(U_array::Vector{<:Matrix}, lightcone, reduced_mps::Vector{it.ITensor})
    updateLightcone!(lightcone, U_array)
    d = lightcone.d
    N = lightcone.size
    plevconj = lightcone.depth+1    #inds of conj mps are primed to depth+1
    interval = lightcone.range[end]

    siteinds = it.prime(lightcone.sitesAB, lightcone.depth)
    reduced_mps = [it.prime(tensor, lightcone.depth) for tensor in reduced_mps]
    reduced_mps_conj = [conj(it.prime(tensor, plevconj)) for tensor in reduced_mps]   

    # prepare right blocks to save contractions
    R_blocks::Vector{it.ITensor} = []

    # prepare zero matrices for middle qubits
    d = lightcone.d
    zero_vec = [1; [0 for _ in 1:d-1]]
    zero_mat = kron(zero_vec, zero_vec')

    # construct leftmost_block, which will be updated sweeping from left to right
    # first item is the contraction of first tensor of reduced_mps, first tensor of reduced_mps_conj, 
    # a delta/zero projector connecting their siteind, a delta connecting their left link (if any).
    # insert a zero proj if we are in region A, else insert identity
    ind = it.noprime(siteinds[1])
    middle_op = (interval[1] <= 1 <= interval[2]) ? it.ITensor(zero_mat, ind, it.prime(ind, plevconj)) : it.delta(ind, it.prime(ind, plevconj))
    leftmost_block = reduced_mps[1]*middle_op*reduced_mps_conj[1]
    # connect left links if any
    l1 = it.uniqueind(reduced_mps[1], siteinds[1], reduced_mps[2])
    if !isnothing(l1)
        leftmost_block *= it.delta(l1, it.prime(l1, plevconj))
    end

    # prepare R_N by multipliying the tensors of the last site
    lN = it.uniqueind(reduced_mps[N], siteinds[N], reduced_mps[N-1])
    rightmost_block = reduced_mps[N]*reduced_mps_conj[N]
    if !isnothing(lN)
        rightmost_block *= it.delta(lN, it.prime(lN, plevconj))
    end

    # contract everything on the left and save rightmost_block at each intermediate step
    for k in N:-1:3
        # extract left gates associated with site k
        gates_k = lightcone.gates_by_site[k]
        coords_left = [gate["coords"] for gate in gates_k if gate["orientation"]=="L"]
        tensors_left = [lightcone.circuit[pos[1]][pos[2]] for pos in coords_left]

        # insert a zero proj if we are in region A, else insert identity
        ind = it.noprime(siteinds[k])
        middle_op = (interval[1] <= k <= interval[2]) ? it.ITensor(zero_mat, ind, it.prime(ind, plevconj)) : it.delta(ind, it.prime(ind, plevconj))

        # put left tensors and middle op together with conj reversed left tensors
        tensors_left = [tensors_left; middle_op; reverse([conj(it.prime(tensor, plevconj)) for tensor in tensors_left])]

        all_blocks = (k==N ? tensors_left : [reduced_mps[k]; tensors_left; reduced_mps_conj[k]]) # add sites
        for block in all_blocks
            rightmost_block *= block
        end
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # now sweep from left to right by removing each unitary at a time to compute gradient
    grad = [Array{ComplexF64}(undef, 0, 0) for _ in 1:length(lightcone.coords)]
    for j in 2:N
        # extract all gates on the left of site j
        gates_j = lightcone.gates_by_site[j]
        coords_left = [gate["coords"] for gate in gates_j if gate["orientation"]=="L"]
        tensors_left = [lightcone.circuit[pos[1]][pos[2]] for pos in coords_left]

        # insert a zero proj if we are in region A, else insert identity
        ind = it.noprime(siteinds[j])
        middle_op = (interval[1] <= j <= interval[2]) ? it.ITensor(zero_mat, ind, it.prime(ind, plevconj)) : it.delta(ind, it.prime(ind, plevconj))

        # evaluate gradient by removing each gate
        # prepare contraction of conj gates since they are the same for each lower gate
        # the order of contractions is chosen so that the number of indices does not increase
        # except for the last two terms: on j odds the index number will increase by 1, but we need to store the site tensors
        # on the leftmost_block anyway to proceed with the sweep
        contract_left_upper = [reverse([conj(it.prime(tensor, plevconj)) for tensor in tensors_left]); middle_op; reduced_mps_conj[j]; reduced_mps[j]]
        for gate in contract_left_upper
            leftmost_block *= gate
        end

        if j == N && !isnothing(lN)     # check whether the reduced mps has a link on the right or not and in case complete the contraction 
            leftmost_block *= it.delta(lN, it.prime(lN, plevconj))
        end

        upper_env = j<N ? leftmost_block*rightmost_block : leftmost_block
        for l in 1:length(tensors_left)
            not_l_tensors = [tensors_left[1:l-1]; tensors_left[l+1:end]]
            
            env = upper_env     # store current state of upper_env
            for gate in not_l_tensors
                env *= gate
            end

            gate_jl = filter(gate -> gate["orientation"] == "L", gates_j)[l]
            gate_jl_inds, gate_jl_num = gate_jl["inds"], gate_jl["number"]
            ddUjl = Array(env, gate_jl_inds)
            ddUjl = 2*conj(reshape(ddUjl, (d^2, d^2)))
            grad[gate_jl_num] = ddUjl
        end

        # update leftmost_block for next j and add it to L_blocks list
        for gate in tensors_left
            leftmost_block *= gate
        end

        # update rightmost_block for next j
        if j < N-1
            rightmost_block = R_blocks[j]       #R_blocks starts from site 3
        end
    end

    # compute environment now that we contracted all blocks, so that we are effectively computing the overlap
    # we use the absolute value as a cost function
    overlap_sq = abs(Array(leftmost_block)[1])
    riem_grad = project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -overlap_sq
    riem_grad = - riem_grad

    return cost, riem_grad

end



function invertMPSLiu(mps::itmps.MPS, invertMethod; d = 2, start_tau = 1, eps_trunc = 0.01, eps_inv = 0.01, kargs_inv = NamedTuple())

    N = length(mps)
    isodd(N) && throw(DomainError(N, "Choose an even number for N"))
    siteinds = it.siteinds(mps)
    #eps_liu = eps_trunc/N

    local mps_trunc, boundaries, rangesA, V_list, err_list, lc_list, err_trunc
    tau = start_tau
    while true
        mps_copy = deepcopy(mps)

        # for a given tau, we already know that both the sizeAB and the spacing have to be chosen
        # so that the final state is a tensor product of pure states
        sizeAB = 6*(tau-1)
        spacing = 2*(tau-1)
        if tau == 1
            sizeAB = 2
            spacing = 2
        end
        println("Attempting inversion of reduced density matrices with depth tau = $tau, imposing sizeAB = $sizeAB and spacing = $spacing for factorization")

        @assert tau > 0
        isodd(sizeAB) && throw(DomainError(sizeAB, "Choose an even number for sizeAB"))
        isodd(spacing) && throw(DomainError(spacing, "Choose an even number for the spacing between regions"))
        
        i = spacing+1
        initial_pos::Vector{Int64} = []
        while i < N-tau
            push!(initial_pos, i)
            i += sizeAB+spacing
        end
        rangesAB = [(i, min(i+sizeAB-1, N)) for i in initial_pos]
        @show rangesAB
        rangesA = []

        V_list = []
        lc_list::Vector{Lightcone} = []
        err_list = []
        println("Inverting reduced density matrices...")
        for i in initial_pos
            last_site = min(i+sizeAB-1, N)
            k_sites = siteinds[i:last_site]
            it.orthogonalize!(mps_copy, div(i+last_site, 2))

            # extract reduced mps on k_sites and construct lightcone structure of depth tau
            reduced_mps = mps_copy[i:last_site]
            # FOR NOW CAN ONLY DEAL WITH (TRUE, TRUE)
            #lightbounds != (true, true) && 
            #    throw(DomainError(lightbounds, "Try to choose spacing so that regions to invert are away from boundaries"))
            lightbounds = (true, last_site==N ? false : true)
            lightcone = newLightcone(k_sites, tau; lightbounds = lightbounds)
            rangeA = lightcone.range[end]
            if !lightbounds[2]
                rangeA = (rangeA[1], N)
            end
            push!(rangesA, (rangeA[1]+i-1, rangeA[2]+i-1))

            # setup optimization stuff
            arrU0 = Array(lightcone)
            fg = arrU -> _fgLiu(arrU, lightcone, reduced_mps)
            # Quasi-Newton method
            m = 5
            algorithm = LBFGS(m;maxiter = 20000, gradtol = 1E-8, verbosity = 1)
            # optimize and store results
            # note that arrUmin is already stored in current lightcone, ready to be applied to mps
            arrUmin, err, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport = true , inner = inner);
            
            push!(lc_list, lightcone)
            push!(V_list, arrUmin)
            push!(err_list, 1+err)
        end

        it.prime!(mps_copy, tau)
        contract!(mps_copy, lc_list, initial_pos)
        mps_copy = it.noprime(mps_copy)
        mps_trunc = deepcopy(mps_copy)

        @show rangesA
        boundaries = [0]
        for rangeA in rangesA
            if rangeA[1]>1
                cut!(mps_trunc, rangeA[1]-1)
                push!(boundaries, rangeA[1]-1)
            end
            if rangeA[end]<N
                cut!(mps_trunc, rangeA[end])
                push!(boundaries, rangeA[end])
            end
        end
        push!(boundaries, N)
        err_trunc = norm(mps_copy - mps_trunc)
        @show err_trunc

        if err_trunc <= eps_trunc
            break
        else
            println("Convergence not found with initial depth tau = $tau, increasing inversion depth")
            #eps_liu /= 2
            tau += 1
        end
    end

    println("Truncation reached within requested eps_trunc, inverting local states...")
    trunc_siteinds = it.siteinds(mps_trunc)
    trunc_linkinds = it.linkinds(mps_trunc)
    lc_list2::Vector{Lightcone} = []
    tau_list2 = []
    err_list2 = []

    ranges = []
    for l in 1:length(boundaries)-1
        push!(ranges, (boundaries[l]+1, boundaries[l+1]))
    end
    @show ranges

    for range in ranges
        # extract reduced mps and remove external linkind (which will be 1-dim)
        reduced_mps = mps_trunc[range[1]:range[2]]
        
        if range[1]>1
            comb1 = it.combiner(trunc_linkinds[range[1]-1], trunc_siteinds[range[1]])
            reduced_mps[1] *= comb1
            cind = it.combinedind(comb1)
            it.replaceind!(reduced_mps[1], cind, trunc_siteinds[range[1]])
        end
        if range[end]<N
            comb2 = it.combiner(trunc_linkinds[range[2]], trunc_siteinds[range[2]])
            reduced_mps[end] *= comb2
            cind = it.combinedind(comb2)
            it.replaceind!(reduced_mps[end], cind, trunc_siteinds[range[2]])
        end

        reduced_mps = itmps.MPS(reduced_mps)
        
        results2 = invert(reduced_mps, invertMethod; start_tau = (range in rangesA ? 1 : 2), eps = eps_inv, kargs_inv...)
        push!(lc_list2, results2["lightcone"])
        push!(tau_list2, results2["tau"])
        push!(err_list2, results2["err"])
    end


    # finally create a 0 state and apply all the V to recreate the original state for final total error
    mps_final = initialize_vac(N, trunc_siteinds)
    apply!(mps_final, lc_list2)
    err2tot = 1-abs(Array(contract(mps_final, conj(mps_trunc)))[1])
    @show err2tot
    it.replace_siteinds!(mps_final, siteinds)
    apply!(mps_final, lc_list, dagger=true)
    err_total = 1-abs(Array(contract(mps_final, conj(mps)))[1])
    @show err_total
    
    return Dict([("lc1", lc_list), ("tau1", tau), ("err1", err_list), ("lc2", lc_list2), ("tau2", tau_list2), ("err2", err_list2), ("mps_final", mps_final), ("err_trunc", err_trunc), ("err_inv", err2tot), ("err_total", err_total)])

end



end