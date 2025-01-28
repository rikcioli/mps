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
function invertMPSMalz(mps::itmps.MPS, invertMethod; q = 0, eps_malz = 1E-2, eps_bell = 1E-2, eps_V = 1E-2, kargsP = NamedTuple(), kargsV= NamedTuple(), kargsW = NamedTuple())
    N = length(mps)
    siteinds = it.siteinds(mps)
    linkinds = it.linkinds(mps)
    linkdims = [ind.space for ind in linkinds]
    D = maximum(linkdims)
    d = siteinds[div(N,2)].space
    @assert d == 2
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D


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
    # up until here all good

    ## Extract unitary matrices that must be applied on bell pairs
    # here we follow the sequential rg procedure
    V_list::Vector{Vector{it.ITensor}} = []
    V_tau_list = []
    V_lc_list::Vector{Lightcone} = []
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

        # Approximate total V_i unitary with bw circuit
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
        # invert final V
        res = invert(V_mpo, invertMethod; eps = eps_V/(newN^2), start_tau = 2, kargsV...)
        tau = res["tau"]
        lc = res["lightcone"]
        # NOTE: FOR NOW invertGlobalSweep ONLY WORKS WITH QUBITS, AS IT OPTIMIZES OVER U(4) UNITARIES
        
        # account for the swap gates 
        if i > 1
            tau += q - bitlength - 1
        end
        push!(V_tau_list, tau)
        push!(V_lc_list, lc)

    end


    zero_projs::Vector{it.ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, ind'))
    end

    # act with W on zeros
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]

    # prepare W|0> states to turn into mps
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

    # apply V unitaries as fdqc on mps_final to construct the final inversion circuit
    apply!(mps_final, V_lc_list)
    err_total = 1-abs(Array(contract(conj(mps), mps_final))[1])
    @show err_total


    return Dict([("V_lc", V_lc_list), ("V_tau", V_tau_list), ("W_lc", W_lc_list), ("W_tau", W_tau_list), ("err_total", err_total)])
end

end