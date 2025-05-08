module MPSMethods

using ITensors, ITensorMPS
using LinearAlgebra, Statistics, Random, OptimKit
using CSV, HDF5, JLD2, DataFrames

ITensors.set_warn_order(28)


const H = [1 1
           1 -1]/sqrt(2)
const Id = [1 0
            0 1]
const T = [1 0
            0 exp(1im*pi/4)]
const X = [0 1
            1 0]
const Z = [1 0
            0 -1]
const CX = [1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0]

const Id2 = [1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 1]

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


function invertMPSMalz(mps::MPS; q = 0, eps_malz = 1E-2, eps_bell = 1E-2, eps_V = 1E-2, kargs...)
    N = length(mps)
    sites = siteinds(mps)
    links = linkinds(mps)
    linkdims = [ind.space for ind in links]
    D = maximum(linkdims)
    d = sites[div(N,2)].space
    bitlength = length(digits(D-1, base=d))     # how many qudits of dim d to represent bond dim D

    # adjust bond dimension so that it's the same everywhere (except boundaries)
    for j in bitlength : N-bitlength
        link = links[j]
        if link.space < D
            orthogonalize!(mps, j+1)
            block = mps[j]*mps[j+1]
            U, S, Vdag = svd(block, (sites[j], links[j-1]), cutoff=1e-15)
            Sinds = (commonind(S, U), commonind(S, Vdag))
            Svec = diag(Array(S, Sinds))
            Svec_ext = [Svec; [0 for _ in 1:(D-link.space)]]
            S = ITensor(diagm(Svec_ext), Sinds)
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
        block_linkinds = linkinds(mps)[q:q:end]
        block_linkinds_dims = [ind.space for ind in block_linkinds]
        
        new_siteinds = siteinds(D, 2*(newN-1))

        # replace primed linkinds with new siteinds
        replaceind!(blockMPS[1], block_linkinds[1]', new_siteinds[1])
        replaceind!(blockMPS[end], block_linkinds[end]', new_siteinds[end])
        for i in 2:(newN-1)
            replaceind!(blockMPS[i], block_linkinds[i-1]', new_siteinds[2*i-2])
            replaceind!(blockMPS[i], block_linkinds[i]', new_siteinds[2*i-1])
        end

        # separate each block into 2 different sites for optimization
        sep_mps::Vector{ITensor} = [blockMPS[1]]
        for i in 2:newN-1
            iL, nL = block_linkinds[i-1], new_siteinds[2*i-2]
            bL, S, bR = svd(blockMPS[i], iL, nL)
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
    V_list::Vector{Vector{ITensor}} = []
    V_tau_list = []
    
    for i in 1:newN

        block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]

        # if i=1 no svd must be done, entire block 1 is already a series of isometries
        if i == 1

            iR = block_linkinds[1]
            block_i[end] = replaceind(block_i[end], iR, new_siteinds[1])
            block_i_Ulist = block_i
            push!(V_list, block_i_Ulist)

        else

            # prepare left index of block i
            block_i_Ulist = []
            iL = block_linkinds[i-1]
            local prev_SV


            for j in 1:q-1
                A = block_i[j]                              # j-th tensor in block_i
                iR = commoninds(A, links)[2]    # right link index
                Aprime = j > 1 ? prev_SV*A : A              # contract SV from previous svd (of the block on the left)
                Uprime, S, V = svd(Aprime, uniqueinds(Aprime, iR, iL))
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
                CindR = commoninds(C, links)[2]
                Pinv = ITensor(Pinv, [iL, CindR, PnL, PnR])
            else    #same here, only different indices
                PiL = block_linkinds[i-1]
                PnL = new_siteinds[2*i-2]
                dim = block_linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PnL, PiL]), (dim, dim))
                Pinv = inv(P_matrix)

                Pinv = ITensor(Pinv, [iL, PnL])
            end

            Ctilde = prev_SV * C * Pinv
            push!(block_i_Ulist, Ctilde)
            push!(V_list, block_i_Ulist)

        end

        # Approximate total V_i unitary with bw circuit
        # contract isometries in block_i_Ulist as long as dimension <= d^2bitlength (efficient for D low, independent from q)
        upinds = sites[q*(i-1)+1 : q*i]
        V = reduce(*, block_i_Ulist[1:2*bitlength])
        V_upinds = upinds[1:2*bitlength]
        V_downinds = uniqueinds(V, V_upinds)
        down_d = reduce(*, [ind.space for ind in V_downinds])  # will be D for i=1, D^2 elsewhere
        V_matrix = reshape(Array(V, V_upinds, V_downinds), (d^(2*bitlength), down_d))      # MUST BE MODIFIED WHEN i=1 or newN THE OUTPUT LEG IS ONLY D
        # @show typeof(V_matrix)
        # @show typeof(iso_to_unitary(V_matrix))
        V_mpo = unitary_to_mpo(iso_to_unitary(V_matrix), sites = upinds)
        prime!(V_mpo, q - 2*bitlength)
        
        # transform next isometries to mpo and contract with previous ones
        for k in 2*bitlength+1:q
            uk = block_i_Ulist[k]
            prev_u = block_i_Ulist[k-1]
            uk_upinds = [commoninds(uk, prev_u); upinds[k]]
            uk_downinds = uniqueinds(uk, uk_upinds)
            up_d = reduce(*, [ind.space for ind in uk_upinds])
            down_d = reduce(*, [ind.space for ind in uk_downinds])
            uk_matrix = reshape(Array(uk, uk_upinds, uk_downinds), (up_d, down_d))
            # need to decide how many identities to put to the left of uk unitaries based on up_d
            nskips = k - length(digits(up_d-1, base=d))
            uk_mpo = unitary_to_mpo(iso_to_unitary(uk_matrix), sites = upinds, skip_qudits = nskips)
            prime!(uk_mpo, q-k)
            V_mpo = V_mpo*uk_mpo
        end

        replaceprime!(V_mpo, q-2*bitlength+1 => 1)
        # invert final V
        _, _, _, tau = invertSweep(V_mpo; d = d, eps = eps_V/((newN^2)*d^q), kargs...)
        
        # account for the swap gates 
        if i > 1
            tau += q - bitlength - 1
        end
        push!(V_tau_list, tau)

    end

    zero_projs::Vector{ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, ITensor(vec, ind'))
    end

    # act with W on zeros
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]


    ## Compute scalar product to check fidelity
    # just a check, can be disabled

    # # group blocks together
    # block_list::Vector{ITensor} = []
    # mps_primed = conj(mps)'''''
    # sites = siteinds(mps_primed)
    # local left_tensor
    # for i in 1:newN
    #     block_i = i < newN ? mps_primed[q*(i-1)+1 : q*i] : mps_primed[q*(i-1)+1 : end]
    #     i_siteinds = i < newN ? sites[q*(i-1)+1 : q*i] : sites[q*(i-1)+1 : end]
    #     left_tensor = (i == 1 ? block_i[1] : block_i[1]*left_tensor)
    #     left_tensor *= replaceinds(V_list[i][1], noprime(i_siteinds), i_siteinds)

    #     for j in 2:q
    #         left_tensor *= block_i[j] * replaceinds(V_list[i][j], noprime(i_siteinds), i_siteinds)
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
        W = reshape(Array(W, inds(W)), (D, D))
        W_ext[1:D, 1:D] = W
        W_ext = reshape(W_ext, 2^(2*bitlength))
        sites = siteinds(d, 2*bitlength)
        W_mps = MPS(W_ext, sites)
        push!(Wmps_list, W_mps)
    end

    # invert each mps in the list
    results = [invertSweep(Wmps; d = d, eps = eps_bell/npairs) for Wmps in Wmps_list]

    W_tau_list = [res[4] for res in results]

    return Wlayer, V_list, overlap, sweepB, W_tau_list, V_tau_list
end


# ONLY WORKS FOR d=2
function invertMPSMalz(mps::MPS, invertMethod; q = 0, eps = 1e-3, kargsP = NamedTuple(), kargsV = NamedTuple(), kargsW = NamedTuple())
    N = length(mps)
    sites = siteinds(mps)
    links = linkinds(mps)
    linkdims = [ind.space for ind in links]
    D = maximum(linkdims)
    d = sites[div(N,2)].space
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
        block_linkinds = linkinds(mps)[q:q:end]
        block_linkinds_dims = [ind.space for ind in block_linkinds]
        
        new_siteinds = siteinds(D, 2*(newN-1))

        # replace primed linkinds with new siteinds
        replaceind!(blockMPS[1], block_linkinds[1]', new_siteinds[1])
        replaceind!(blockMPS[end], block_linkinds[end]', new_siteinds[end])
        for i in 2:(newN-1)
            replaceind!(blockMPS[i], block_linkinds[i-1]', new_siteinds[2*i-2])
            replaceind!(blockMPS[i], block_linkinds[i]', new_siteinds[2*i-1])
        end

        # separate each block into 2 different sites for optimization
        sep_mps::Vector{ITensor} = [blockMPS[1]]
        for i in 2:newN-1
            iL, nL = block_linkinds[i-1], new_siteinds[2*i-2]
            bL, S, bR = svd(blockMPS[i], iL, nL)
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
    V_list::Vector{Vector{ITensor}} = []
    V_mpo_list = []

    for i in 1:newN

        block_i = i < newN ? mps[q*(i-1)+1 : q*i] : mps[q*(i-1)+1 : end]
        block_i_Ulist::Vector{ITensor} = []

        # if i=1 no svd must be done, entire block 1 is already a series of isometries
        if i == 1
            iR = block_linkinds[1]
            block_i[end] = replaceind(block_i[end], iR, new_siteinds[1])
            block_i_Ulist = block_i
            push!(V_list, block_i_Ulist)

        else
            # prepare left index of block i
            iL = block_linkinds[i-1]
            local prev_SV
            for j in 1:q-1
                A = block_i[j]                              # j-th tensor in block_i
                iR = commoninds(A, links)[2]    # right link index
                Aprime = j > 1 ? prev_SV*A : A              # contract SV from previous svd (of the block on the left)
                Uprime, S, V = svd(Aprime, uniqueinds(Aprime, iR, iL))
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
                cup, clow = combiner(PnL, PnR), combiner(PiL, PiR)
                P = (P*cup)*clow
                P_matrix = Array(P, (combinedind(cup), combinedind(clow)))
                Pinv = inv(P_matrix)
            
                # convert back to tensor with inds ready to contract with Ctilde = prev_SV * C   
                # and with blockMPS[i] siteinds       
                CindR = commoninds(C, links)[2]
                Pinv = ITensor(Pinv, (iL, CindR, PnL, PnR))
            else    #same here, only different indices
                PiL = block_linkinds[i-1]
                PnL = new_siteinds[2*i-2]
                dim = block_linkinds_dims[i-1]
                P_matrix = reshape(Array(P, [PnL, PiL]), (dim, dim))
                Pinv = inv(P_matrix)

                Pinv = ITensor(Pinv, [iL, PnL])
            end

            Ctilde = prev_SV * C * Pinv
            push!(block_i_Ulist, Ctilde)
            push!(V_list, block_i_Ulist)

        end

        # Convert total V_i unitary to MPO efficiently
        # contract isometries in block_i_Ulist as long as dimension <= d^2bitlength (efficient for D low, independent from q)
        upinds = sites[q*(i-1)+1 : q*i]
        V = reduce(*, block_i_Ulist[1:2*bitlength])
        V_upinds = upinds[1:2*bitlength]
        V_downinds = uniqueinds(V, V_upinds)
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
        
        V_mpo = unitary_to_mpo(V_matrix, sites = upinds)
        prime!(V_mpo, q - 2*bitlength)   # we prime to contract with the next MPOs
        
        # transform next isometries to mpo and contract with previous ones
        for k in 2*bitlength+1:q
            uk = block_i_Ulist[k]
            prev_u = block_i_Ulist[k-1]
            uk_upinds = [commoninds(uk, prev_u); upinds[k]]
            uk_downinds = uniqueinds(uk, uk_upinds)
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
            uk_mpo = unitary_to_mpo(uk_matrix, sites = upinds, skip_qudits = nskips)
            prime!(uk_mpo, q-k)
            V_mpo = V_mpo*uk_mpo
        end

        replaceprime!(V_mpo, q-2*bitlength+1 => 1)
        push!(V_mpo_list, V_mpo)
    end
    

    # BELL PAIRS PREPARATION
    # prepare |0> state and act with W
    zero_projs::Vector{ITensor} = []
    for ind in new_siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, ITensor(vec, ind'))
    end
    Wlayer = [W_list[i] * zero_projs[2*i-1] * zero_projs[2*i] for i in 1:npairs]

    # turn W|0> states into MPSs
    Wmps_list = []
    for i in 1:npairs
        sitesL = sites[q*i-bitlength+1 : q*i]
        sitesR = sites[q*i+1 : q*i+bitlength]
        cL = combiner(sitesL)
        replaceind!(cL, combinedind(cL), new_siteinds[2*i-1])
        cR = combiner(sitesR)
        replaceind!(cR, combinedind(cR), new_siteinds[2*i])

        Wi = (Wlayer[i]*cL)*cR
        W_mps = MPS(Wi, [sitesL; sitesR])
        push!(Wmps_list, W_mps)
    end

    # invert each mps in the list
    # NOTE: INVERTGLOBAL ONLY WORKS ON QUBITS
    results = [invert(Wmps, invertMethod; eps = eps_bell/npairs, kargsW...) for Wmps in Wmps_list]
    W_tau_list = [res["tau"] for res in results]
    W_lc_list::Vector{Lightcone} = [res["lightcone"] for res in results]


    # compute numerically total error by creating a 0 state and applying everything in sequence
    mps_final = initialize_vac(N, sites)
    apply!(mps_final, W_lc_list)

    # compare with exact W|0>
    Wexact = [ITensor([1; 0], sites[i]) for i in 1:N]
    for i in 1:npairs
        for j in q*i-bitlength+1 : q*i+bitlength
            jshift = j-(q*i-bitlength)
            Wexact[j] = Wmps_list[i][jshift]
        end
    end
    Wstep_mps = MPS(Wexact)
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
    V_mpo_final = MPO(reduce(vcat, [mpo[1:end] for mpo in V_mpo_list]))
    res = apply(V_mpo_final, Wstep_mps)
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
function _fgLiu(U_array::Vector{Matrix{T}}, lightcone::Lightcone, reduced_mps::Vector{ITensor}) where {T}
    updateLightcone!(lightcone, U_array)
    d = lightcone.d
    N = lightcone.size
    plevconj = lightcone.depth+1    #inds of conj mps are primed to depth+1
    interval = lightcone.range[end]

    sites = prime(lightcone.siteinds, lightcone.depth)
    reduced_mps = [prime(tensor, lightcone.depth) for tensor in reduced_mps]
    reduced_mps_conj = [conj(prime(tensor, plevconj)) for tensor in reduced_mps]   

    # prepare right blocks to save contractions
    R_blocks = ITensor[]

    # prepare zero matrices for middle qubits
    d = lightcone.d
    zero_vec = [1; [0 for _ in 1:d-1]]
    zero_mat = kron(zero_vec, zero_vec')

    # construct leftmost_block, which will be updated sweeping from left to right
    # first item is the contraction of first tensor of reduced_mps, first tensor of reduced_mps_conj, 
    # a delta/zero projector connecting their siteind, a delta connecting their left link (if any).
    # insert a zero proj if we are in region A, else insert identity
    ind = noprime(sites[1])
    middle_op = (interval[1] <= 1 <= interval[2]) ? ITensor(zero_mat, ind, prime(ind, plevconj)) : delta(ind, prime(ind, plevconj))
    leftmost_block = reduced_mps[1]*middle_op*reduced_mps_conj[1]
    # connect left links if any
    l1 = uniqueind(reduced_mps[1], sites[1], reduced_mps[2])
    if !isnothing(l1)
        leftmost_block *= delta(l1, prime(l1, plevconj))
    end

    # prepare R_N by multipliying the tensors of the last site
    lN = uniqueind(reduced_mps[N], sites[N], reduced_mps[N-1])
    rightmost_block = reduced_mps[N]*reduced_mps_conj[N]
    if !isnothing(lN)
        rightmost_block *= delta(lN, prime(lN, plevconj))
    end

    # contract everything on the left and save rightmost_block at each intermediate step
    for k in N:-1:3
        # extract left gates associated with site k
        gates_k = lightcone.gates_by_site[k]
        tensors_left = [lightcone.circuit[gate[:pos]] for gate in gates_k if gate[:orientation]=="L"]

        # insert a zero proj if we are in region A, else insert identity
        ind = noprime(sites[k])
        middle_op = (interval[1] <= k <= interval[2]) ? ITensor(zero_mat, ind, prime(ind, plevconj)) : delta(ind, prime(ind, plevconj))

        # put left tensors and middle op together with conj reversed left tensors
        tensors_left = [tensors_left; middle_op; reverse([conj(prime(tensor, plevconj)) for tensor in tensors_left])]

        all_blocks = (k==N ? tensors_left : [reduced_mps[k]; tensors_left; reduced_mps_conj[k]]) # add sites
        for block in all_blocks
            rightmost_block *= block
        end
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # now sweep from left to right by removing each unitary at a time to compute gradient
    grad = Vector{Matrix{T}}(undef, lightcone.n_unitaries)
    for j in 2:N
        # extract all gates on the left of site j
        gates_j = lightcone.gates_by_site[j]
        tensors_left = [lightcone.circuit[gate[:pos]] for gate in gates_j if gate[:orientation]=="L"]

        # insert a zero proj if we are in region A, else insert identity
        ind = noprime(sites[j])
        middle_op = (interval[1] <= j <= interval[2]) ? ITensor(zero_mat, ind, prime(ind, plevconj)) : delta(ind, prime(ind, plevconj))

        # evaluate gradient by removing each gate
        # prepare contraction of conj gates since they are the same for each lower gate
        # the order of contractions is chosen so that the number of indices does not increase
        # except for the last two terms: on j odds the index number will increase by 1, but we need to store the site tensors
        # on the leftmost_block anyway to proceed with the sweep
        contract_left_upper = [reverse([conj(prime(tensor, plevconj)) for tensor in tensors_left]); middle_op; reduced_mps_conj[j]; reduced_mps[j]]
        for gate in contract_left_upper
            leftmost_block *= gate
        end

        if j == N && !isnothing(lN)     # check whether the reduced mps has a link on the right or not and in case complete the contraction 
            leftmost_block *= delta(lN, prime(lN, plevconj))
        end

        upper_env = j<N ? leftmost_block*rightmost_block : leftmost_block
        for l in 1:length(tensors_left)
            not_l_tensors = [tensors_left[1:l-1]; tensors_left[l+1:end]]
            
            env = upper_env     # store current state of upper_env
            for gate in not_l_tensors
                env *= gate
            end

            gate_jl = filter(gate -> gate[:orientation] == "L", gates_j)[l]
            gate_jl_inds, gate_jl_pos = gate_jl[:inds], gate_jl[:pos]
            ddUjl = Array{T}(env, gate_jl_inds)
            ddUjl = 2*conj(reshape(ddUjl, (d^2, d^2)))
            grad[gate_jl_pos] = ddUjl
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
    overlap = abs(Array{T}(leftmost_block)[1])
    riem_grad = project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -overlap
    riem_grad = - riem_grad

    return cost, riem_grad

end



function invertMPSLiu(mps::MPS, invertMethod; start_tau = 1, eps = 1e-5)

    N = length(mps)
    isodd(N) && throw(DomainError(N, "Choose an even number for N"))
    sites = siteinds(mps)
    eps1 = eps # error of the whole first part, that is inversion of reduced dm + truncation
    eps2 = eps # error of the second part, that is inversion of the pure states
    println("Attempting inversion of MPS with invertMPSLiu and errors:\neps1 = $eps1\neps2 = $eps2")

    local mps_trunc, boundaries, rangesA, err_list, lc_list, err1, ltg_map
    tau = start_tau

    lc_list_old = Array{Lightcone, 1}()
    local ltg_map_old
    while true
        mps_copy = deepcopy(mps)
        use_previous = (tau > 2 && !isempty(lc_list_old)) #use knowledge of previous inversions for initial start 

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
        
        # the first region will always be a C type
        i = spacing+1
        # select initial positions 
        initial_pos::Vector{Int64} = []
        while i < N-tau
            push!(initial_pos, i)
            i += sizeAB+spacing
        end
        rangesAB = [(j, min(j+sizeAB-1, N)) for j in initial_pos]
        @show rangesAB

        # Construct map that associated to each site the region name (C vs AB), the lightcone that will end up in lc_list, and the local site in that lightcone
        ltg_map = [("C", 1, l) for l in 1:spacing]
        for num in eachindex(initial_pos[1:end-1])
            for l in 1:sizeAB
                push!(ltg_map, ("AB", num, l))
            end
            for l in 1:spacing
                push!(ltg_map, ("C", num+1, l))
            end
        end
        remainder = N-length(ltg_map)
        lastABsize = min(sizeAB, remainder)
        for l in 1:lastABsize
            push!(ltg_map, ("AB", length(initial_pos), l))
        end
        remainder -= lastABsize
        for l in 1:remainder
            push!(ltg_map, ("C", length(initial_pos)+1, l))
        end


        rangesA = Array{Tuple, 1}(undef, length(initial_pos))

        lc_list = Array{Lightcone, 1}(undef, length(initial_pos))
        #lc_list::Vector{Lightcone} = []
        err_list = Array{Float64, 1}(undef, length(initial_pos))
        #err_list = []

        reduced_mps_list = []
        k_sites_list = []
        for (first_site, last_site) in rangesAB
            orthogonalize!(mps_copy, div(first_site+last_site, 2))
            push!(reduced_mps_list, mps_copy[first_site:last_site])
            push!(k_sites_list, sites[first_site:last_site])
        end

        println("Inverting reduced density matrices...")
        # all variables defined inside each thread have a _
        Threads.@threads for l in eachindex(rangesAB)
            (_first_site, _last_site) = rangesAB[l]
            _reduced_mps = reduced_mps_list[l]
            _k_sites = k_sites_list[l]
            
            _lightbounds = (true, _last_site==N ? false : true)   # first region will always be left for step 2, last region could end up in step 1
            _lightcone = newLightcone(_k_sites, tau; lightbounds = _lightbounds, U_array = use_previous ? Vector{Matrix{ComplexF64}}(undef, 0) : nothing)

            if use_previous #use knowledge of previous inversions for initial start 
                for j in _first_site:_last_site
                    _local_new = ltg_map[j]
                    _local_old = ltg_map_old[j]
                    if _local_old[1] == "AB"  # if it was in one of the previous step's lightcones
                        # extract gates to reuse from old lightcone
                        _lc = lc_list_old[_local_old[2]]  # which lightcone
                        _gates_old = [gate for gate in _lc.gates_by_site[_local_old[3]] if gate[:orientation] == "R"] # gates touching that site, only those on the right
                        _unitaries_old = [Matrix(_lc, gate[:pos]) for gate in _gates_old]

                        _gates_new = [gate for gate in _lightcone.gates_by_site[_local_new[3]] if gate[:orientation] == "R"]
                        _gates_newpos = [gate[:pos] for gate in _gates_new]

                        for m in 1:min(length(_gates_old), length(_gates_new))
                            _U = _unitaries_old[m]
                            updateLightcone!(_lightcone, _U, _gates_newpos[m])
                        end
                    end
                end
            end

            _rangeA_rel = _lightcone.range[end]       # region which will be inverted to 0, relative to first site of lightcone
            _rangeA_abs = (_rangeA_rel[1]+_first_site-1, _last_site==N ? N : _rangeA_rel[2]+_first_site-1)      # same region relative to total mps sites
            rangesA[l] = _rangeA_abs

            # setup optimization stuff
            _arrU0 = Array(_lightcone)
            _fg = arrU -> _fgLiu(arrU, _lightcone, _reduced_mps)
            # Quasi-Newton method
            _algorithm = LBFGS(5; maxiter = 20000, gradtol = 1E-8, verbosity = 1)
            # optimize and store results
            # note that arrUmin is already stored in current lightcone, ready to be applied to mps
            _, _neg_overlap, _ = optimize(_fg, _arrU0, _algorithm; retract = retract, transport! = transport!, isometrictransport = true , inner = inner);
            
            lc_list[l] = _lightcone
            err_list[l] = 1+_neg_overlap
        end

        
        # now that we inverted we apply the circuits, truncate and apply back to compute err1
        apply!(mps_copy, lc_list)
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

        mps_reconstr = deepcopy(mps_trunc)
        apply!(mps_reconstr, lc_list, dagger=true)
        err1 = 1-abs(dot(mps, mps_reconstr))
        @show err1

        if -eps1 <= err1 <= eps1
            break
        else
            println("Inversion and truncation to required error failed with initial depth tau = $tau, increasing inversion depth")
            tau += 1
            lc_list_old = lc_list
            ltg_map_old = ltg_map
        end
    end

    println("Inversion and truncation reached within requested eps_trunc, inverting local states with local error eps2/N_regions^2 = $(eps2/length(rangesA)^2)")
    
    # Now we proceed with inversion of all pure states
    epsinv2 = 2*eps2/length(rangesA)^2     # heuristic, the scaling is 1/L^2
    trunc_siteinds = siteinds(mps_trunc)
    trunc_linkinds = linkinds(mps_trunc)
    ranges = []
    for l in 1:length(boundaries)-1
        push!(ranges, (boundaries[l]+1, boundaries[l+1]))
    end
    @show ranges

    # construct local-to-global map for second part of inversion, could be useful later
    # WARNING: here region_n counts all regions as different, so it's gonna be BC-1, A-2, BC-3, A-4 and so on
    # this is different from ltg_map, where the pos counts differently for the two regions, so C-1, AB-1, C-2, AB-2, and so on
    curr_type = "BC"
    region_n = 1
    ltg_map2 = []
    for range in ranges
        for l in 1:range[2]-range[1]+1
            push!(ltg_map2, (curr_type, region_n, l))
        end
        region_n += 1
        curr_type = (curr_type == "A" ? "BC" : "A")
    end
    
    # prepare list of MPS to invert
    reduced_mps_list = []
    for l in eachindex(ranges)
        range = ranges[l]
        # extract reduced mps and remove external linkind (which will be 1-dim)
        reduced_mps = mps_trunc[range[1]:range[2]]
        
        if range[1]>1
            comb1 = combiner(trunc_linkinds[range[1]-1], trunc_siteinds[range[1]])
            reduced_mps[1] *= comb1
            cind = combinedind(comb1)
            replaceind!(reduced_mps[1], cind, trunc_siteinds[range[1]])
        end
        if range[end]<N
            comb2 = combiner(trunc_linkinds[range[2]], trunc_siteinds[range[2]])
            reduced_mps[end] *= comb2
            cind = combinedind(comb2)
            replaceind!(reduced_mps[end], cind, trunc_siteinds[range[2]])
        end
        
        reduced_mps = MPS(reduced_mps)
        push!(reduced_mps_list, reduced_mps)
    end
    
    local lc_list2, tau_list2, err_list2, mps_final, err2, err_total
    for attempt in 1:5
        lc_list2 = Array{Lightcone, 1}(undef, length(ranges))
        tau_list2 = Array{Integer, 1}(undef, length(ranges))
        err_list2 = Array{Float64, 1}(undef, length(ranges))

        Threads.@threads for l in eachindex(ranges)
            _range = ranges[l]
            _reduced_mps = reduced_mps_list[l]
            _start_tau2 = (_range in rangesA ? 1 : 2)
            _site1_empty = false
            if iseven(_range[2]-_range[1])
                _start_tau2 = 2
                # if the first region has odd number of sites we need to start the lightcone
                # in the site1_empty mode, in order to have a coherent global brickwork structure at the end
                if l == 1   
                    _site1_empty = true
                end
            end
            results2 = invert(_reduced_mps, invertMethod; start_tau = _start_tau2, eps = epsinv2, site1_empty = _site1_empty, reuse_previous = false, nruns = 1)
            #if _site1_empty
            #    @show results2["lightcone"]
            #end
            lc_list2[l] = results2["lightcone"]
            tau_list2[l] = results2["tau"]
            err_list2[l] = results2["err"]
        end

        # finally create a 0 state and apply all the V to recreate the original state for final total error
        mps_final = initialize_vac(N, trunc_siteinds)
        apply!(mps_final, lc_list2)
        err2 = 1-abs(dot(mps_final, mps_trunc))
        @show err2

        replace_siteinds!(mps_final, sites)
        apply!(mps_final, lc_list, dagger=true)
        
        err_total = 1-abs(dot(mps_final, mps))
        @show err_total
        if err_total < eps
            break
        else
            println("Convergence up to desired total error not found with initial eps2 = $epsinv2, reducing the error...")
            epsinv2 *= 0.5
        end

    end
    
    return Dict([("lc1", lc_list), ("tau1", tau), ("errinv1", err_list), ("lc2", lc_list2), ("tau2", tau_list2), 
    ("errinv2", err_list2), ("mps_final", mps_final), ("err1", err1), ("err2", err2), ("err_total", err_total), ("ltg_map", ltg_map), ("ltg_map2", ltg_map2)])
    # NOTE: lc_list is the lightcone that inverts the state to 0, so in order to create the state from 0
    # you need to take the DAGGER; this can be done via apply!(state, lc_list, dagger = true)

end


function invertMPS1(mps::MPS, invertMethod; eps = 1e-5, pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\")
    N = length(mps)

    results = invertMPSLiu(mps, invertMethod; eps = 0.1)

    tau1 = results["tau1"]
    tau2 = maximum(results["tau2"])
    tau = tau1+tau2
    lc_list = results["lc1"]    # REMEMBER WE MUST TAKE THE DAGGER OF THIS BEFORE APPLYING IT TO ZERO
    lc_list2 = results["lc2"]
    ltg_map = results["ltg_map"]
    ltg_map2 = results["ltg_map2"]

    @show ltg_map, ltg_map2, tau1, tau2

    mps_final = results["mps_final"]
    sites = siteinds(mps_final)
    
    reg1_size = lc_list2[1].size
    reduced = isodd(tau2)    # WILL ONLY WORK AS LONG AS SPACING OF THE LC_LIST1 IS EVEN, WHICH IS THE DEFAULT IN OPTLIU AND CANNOT BE MODIFIED
    if reduced
        tau2 -= 1
        tau -= 1
    end

    site1_empty = isodd(reg1_size)
    lightcone = newLightcone(sites, tau; U_array = Vector{Matrix{ComplexF64}}(undef, 0), lightbounds = (false, false), site1_empty = site1_empty)

    for j in 1:N
        # extract gates from second part (i.e. those that need to act first on 0)
        reg_type2, pos2, local_site2 = ltg_map2[j]
        lc_local2 = lc_list2[pos2]
        gates2 = [gate for gate in lc_local2.gates_by_site[local_site2] if gate[:orientation] == "R"]
        unitaries = [Matrix(lc_local2, gate[:pos]) for gate in gates2]   #take the unitaries from lc_list2
        depths = [gate[:depth] for gate in gates2]

        reg_type, pos, local_site = ltg_map[j]
        if reg_type == "AB"
            lc_local = lc_list[pos]
            gates = [gate for gate in lc_local.gates_by_site[local_site] if gate[:orientation] == "R"]
            if !isempty(gates)
                unitaries1 = [copy((Matrix(lc_local, gate[:pos]))') for gate in gates]   #take the unitaries from lc_list
                depths1 = [tau1+1-gate[:depth]+tau2 for gate in gates]

                if !isempty(depths)
                    if depths[end] == depths1[end]  # we can remove the last gate of lc2 by multipliying it with the one from lc1 above
                        @show j
                        unitaries1[end] = unitaries1[end]*unitaries[end]
                        unitaries = unitaries[1:end-1]
                        depths = depths[1:end-1]
                    end
                end
                
                unitaries = [unitaries; unitaries1]
                depths = [depths; depths1]
            end
        end

        for m in eachindex(depths)
            U = unitaries[m]
            tau_U = depths[m]
            updateLightcone!(lightcone, U, (j, tau_U))
        end
    end

    zero = initialize_vac(N, sites)
    apply!(zero, lightcone)

    #zero2 = initialize_vac(N, sites)
    #apply!(zero2, lc_list2)

    fid = abs(dot(zero, mps_final))
    @assert isapprox(fid, 1)

    # save lightcone to file
    best_guess = Array(lightcone)
    jldsave(pathname*"$(N)_$(eps)_ansatz.jld2"; best_guess)

    # save important info to file
    params = Dict([("N", N), ("eps", eps), ("site1_empty", site1_empty), ("start_tau", tau)])
    jldsave(pathname*"$(N)_$(eps)_params.jld2"; params)

    # save original mps to file
    f = h5open(pathname*"$(N)_mps.h5","w")
    write(f,"psi",mps)
    close(f)
    
    return
end


function invertMPS2(pathname, N, eps, invertMethod)

    params = load_object(pathname*"$(N)_$(eps)_params.jld2")
    @show params
    best_guess = load_object(pathname*"$(N)_$(eps)_ansatz.jld2")
    best_guess = [Matrix{ComplexF64}(U) for U in best_guess]
    @show typeof(best_guess)
    f = h5open(pathname*"$(N)_mps.h5","r")
    mps = read(f,"psi",MPS)
    close(f)
    start_tau = params["start_tau"]

    results_final = invert(mps, invertMethod; 
                                nruns = 1, 
                                site1_empty = params["site1_empty"], 
                                eps = eps, 
                                start_tau = start_tau, 
                                init_array = best_guess)
    jldsave(pathname*"$(N)_$(eps)_$(start_tau)ST_result.jld2"; results_final)

    return
end

function invertMPSfinal(mps::MPS, invertMethod; eps = 1e-5, pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\", nthreads = 4)
    N = length(mps)
    invertMPS1(mps, invertMethod; eps = eps, pathname = pathname)
    invertMPS2(pathname, N, eps, invertMethod)
    return
end


function invertMPSculo(mps::MPS; kargs...)
    N = length(mps)
    mps_trunc = deepcopy(mps)
    for i in 1:N-1
        cut!(mps_trunc, i)
    end
    @show abs(dot(mps, mps_trunc))

    orthogonalize!(mps_trunc, N)
    sites = siteinds(mps_trunc)
    links = linkinds(mps_trunc)

    combiners1 = [combiner((sites[1], links[1]))]
    combiners = [combiner([sites[i]; links[i-1:i]]) for i in 2:N-1]
    combinersN = [combiner((sites[N], links[N-1]))]
    combiners = [combiners1; combiners; combinersN]

    combinedinds = [combinedind(comb) for comb in combiners]
    tensor_list = [combiners[i]*mps_trunc[i] for i in 1:N]

    Vlist = [Array(tensor_list[i], combinedinds[i]) for i in 1:N]

    Ulist = [iso_to_unitary(V) for V in Vlist]
    init_array = [kron(Ulist[2*i], Ulist[2*i-1]) for i in 1:div(N,2)]
    @show length(init_array)

    results = invert(mps, invertGlobalSweep; init_array = init_array, kargs...)

    return results
end



end