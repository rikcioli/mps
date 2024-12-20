"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertSweep(mpo::Vector{it.ITensor}, tau, input_inds::Vector{<:it.Index}; d = 2, output_inds = nothing, conv_err = 1E-6, n_sweeps = 1E6)
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


"Calls invertSweep for Vector{ITensor} mpo input with increasing inversion depth tau until it converges with fidelity F = 1-err_to_one"
function invertSweep(mpo::Vector{it.ITensor}, input_inds::Vector{<:it.Index}; err_to_one = 1E-6, start_tau = 1, n_runs = 10, kargs...)
    println("Tolerance $err_to_one, starting from depth $start_tau")
    tau = start_tau
    found = false

    local bw_best, sweep_best
    fid_best = 0
    while !found
        println("Attempting depth $tau with $n_runs runs...")
        fids = []
        for _ in 1:n_runs
            bw, fid, sweep = invertSweep(mpo, tau, input_inds; kargs...)
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

"Wrapper for ITensorsMPS.MPS input. Calls invertSweep by first conjugating mps (mps to invert must be above)"
function invertSweep(mps::itmps.MPS; tau = 0, kargs...)
    obj = typeof(mps)
    println("Attempting inversion of $obj")
    mps = conj(mps)
    siteinds = it.siteinds(mps)

    if iszero(tau)
        results = invertSweep(mps[1:end], siteinds; kargs...)
    else
        results = invertSweep(mps[1:end], tau, siteinds; kargs...)
    end
    return results
end

"Wrapper for ITensorsMPS.MPO input. Calls invertSweep by first conjugating and extracting upper and lower indices"
function invertSweep(mpo::itmps.MPO; tau = 0, kargs...)
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
        results = invertSweep(mpo[1:end], siteinds; output_inds = outinds, kargs...)
    else
        results = invertSweep(mpo[1:end], tau, siteinds; output_inds = outinds, kargs...)
    end
    return results
end

"Wrapper for ITensors.ITensor input. Calls invertSweep on isometry V by first promoting it to a unitary matrix U"
function invertSweep(V::it.ITensor, input_ind::it.Index; tau = 0, kargs...)
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
        results = invertSweep(Umpo, sites; output_inds = sites', kargs...)
    else
        results = invertSweep(Umpo, tau, sites; output_inds = sites', kargs...)
    end
    return results
end


"Cost function and gradient for invertGlobal optimization"
function _fgGlobal(U_array::Vector{<:Matrix}, lightcone, mpo, mpo_low)
    mpo = deepcopy(mpo)
    mpo_low = deepcopy(mpo_low)
    updateLightcone!(lightcone, U_array)
    len = length(lightcone.coords)

    # compute gradient by removing one unitary at a time and applying
    # all unitaries that come before to the virtual mpo_low
    # all unitaries that come after to the original mpo
    # the shift to the next unitary is made by applying U_k Udag_k+1
    grad::Vector{Matrix} = []
    for k in len:-1:2   # contract up to first unitary
        contract!(mpo, lightcone, k)
    end

    # evaluate gradient by sweeping over snake of unitaries
    d = lightcone.d
    for k in 1:len
        _, inds_k = lightcone.coords[k]
        ddUk = contract(mpo, mpo_low)
        ddUk = Array(ddUk, inds_k)
        ddUk = conj(reshape(ddUk, (d^2, d^2)))
        push!(grad, ddUk)
        if k < len
            contract!(mpo_low, lightcone, k)
            contract!(mpo, lightcone, k+1; dagger=true)
        end
    end
    # compute total overlap (which is a complex number)
    contract!(mpo_low, lightcone, len)
    overlap = Array(contract(mpo, mpo_low))[1]
    # we use the absolute value as a cost function
    abs_ov = abs(overlap)

    # correct gradient to account for a smooth cost function (the overlap squared)
    grad *= overlap/abs_ov
    riem_grad = project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -abs_ov
    riem_grad = -riem_grad

    return cost, riem_grad
end

# NEW VERSION USING OPT TECH
"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertGlobal(mpo::Union{Vector{it.ITensor}, itmps.MPS, itmps.MPO}, tau, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; maxiter = 1000, gradtol = 1E-8)
    mpo = deepcopy(mpo[1:end])
    N = length(mpo)
    siteinds = input_inds

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    lightcone = newLightcone(siteinds, tau; lightbounds = (false, false))

    if N == 2   #solution is immediate via SVD
        env = conj(mpo[1]*mpo[2])

        inds = siteinds
        U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
        u, v = it.commonind(U, S), it.commonind(Vdag, S)

        # evaluate fidelity
        newfid = real(tr(Array(S, (u, v))))
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        lightcone.circuit[1][1] = gate_ji_opt

        println("Matrix is 2-local, converged to fidelity $newfid immediately")
        return lightcone, -newfid, nothing
    end

    # we create a virtual lower mpo to temporarily contract left snakes with
    # it will just be an array of deltas that will be contracted at the end with the 
    # upper mpo (which is the original mpo) and the left snakes
    dim = output_inds[1].space
    new_outinds = it.siteinds(dim, N)
    mpo_low::Vector{it.ITensor} = []
    for i in 1:N
        it.replaceind!(mpo[i], output_inds[i], new_outinds[i])
        push!(mpo_low, it.delta(new_outinds[i], it.prime(siteinds[i], tau)))
    end

    # setup optimization stuff
    arrU0 = Array(lightcone)
    fg = arrU -> _fgGlobal(arrU, lightcone, mpo, mpo_low)

    # Quasi-Newton method
    m = 5
    algorithm = LBFGS(m;maxiter = maxiter, gradtol = gradtol, verbosity = 1)

    # optimize and store results
    # note that arrUmin is already stored in current lightcone, ready to be applied to mps
    arrUmin, neg_overlap, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport =true , inner = inner);

    return lightcone, neg_overlap, gradmin, numfg, normgradhistory

end


"Calls invertBW for Vector{ITensor} mpo input with increasing inversion depth tau until it converges to chosen 'overlap' up to error 'eps'"
function invertGlobal(mpo::Union{Vector{it.ITensor}, itmps.MPS, itmps.MPO}, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; overlap = 1, eps = 1E-6, start_tau = 2, kargs...)
    obj = typeof(mpo)
    println("Attempting inversion of $obj to overlap $overlap up to error $eps, starting from depth $start_tau")
    tau = start_tau
    found = false

    while !found
        println("Attempting depth $tau...")
        lc, neg_overlap, rest... = invertGlobal(mpo, tau, input_inds, output_inds; kargs...)
        err = abs(overlap + neg_overlap)
        if err < eps
            found = true
            println("Convergence within desired error achieved with depth $tau\n")
            return tau, lc, err, rest...
        end
        
        if tau > 15
            println("Attempt stopped at tau = $tau, ITensor cannot go above")
            break
        end

        tau += 1
    end
end


"Wrapper for ITensorsMPS.MPS input. Before calling invertGlobal, it conjugates mps (mps to invert must be above)
and prepares a layer of zero bras to construct the mpo |0><psi|. Then calls invertGlobal with overlap 1 and error eps"
function invertGlobal(mps::itmps.MPS; tau = 0, kargs...)
    N = length(mps)
    mps = conj(mps)
    siteinds = it.siteinds(mps)
    outinds = siteinds'

    for i in 1:N
        ind = siteinds[i]
        vec = [1; [0 for _ in 1:ind.space-1]]
        mps[i] *= it.ITensor(vec, ind')
    end


    if iszero(tau)
        tau, lc, err, rest... = invertGlobal(mps, siteinds, outinds; overlap=1, kargs...)
        return tau, lc, err, rest...
    else
        lc, neg_overlap, rest... = invertGlobal(mps, tau, siteinds, outinds; kargs...)
        return lc, abs(1+neg_overlap), rest...
    end

end


"Wrapper for ITensorsMPS.MPO input. Calls invertGlobal by first conjugating and extracting upper and lower indices"
function invertGlobal(mpo::itmps.MPO; tau = 0, kargs...)
    N = length(mpo)
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
        tau, lc, err, rest... = invertGlobal(mpo, siteinds, outinds; overlap = 2^N, kargs...)
        return tau, lc, err, rest...
    else
        lc, neg_overlap, rest... = invertGlobal(mpo, tau, siteinds, outinds; kargs...)
        return lc, abs(2^N+neg_overlap), rest...
    end

end