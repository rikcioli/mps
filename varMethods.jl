import Distributed

"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertSweepLC(mpo::Union{Vector{it.ITensor}, itmps.MPS, itmps.MPO}, tau, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; d = 2, conv_err = 1E-6, maxiter = 1E5, normalization = 1, init_array::Union{Nothing, Vector{<:Matrix}} = nothing)::Dict{String, Any}
    mpo = deepcopy(mpo[1:end])
    N = length(mpo)

    # noprime the input inds
    # change the output inds to a prime of the input inds to match the inds of the first layer of gates
    siteinds = it.noprime(input_inds)
    d = siteinds[1].space
    tempinds = it.siteinds(d, N)
    for i in 1:N
        it.replaceind!(mpo[i], input_inds[i], tempinds[i])
        it.replaceind!(mpo[i], output_inds[i], it.prime(siteinds[i], tau))
        it.replaceind!(mpo[i], tempinds[i], siteinds[i])
    end

    if N == 2   #solution is immediate via SVD
        env = conj(mpo[1]*mpo[2])

        inds = siteinds
        U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
        u, v = it.commonind(U, S), it.commonind(Vdag, S)

        # evaluate fidelity
        newfid = real(tr(Array(S, (u, v))))/normalization
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        lightcone = newLightcone(siteinds, 1; lightbounds = (false, false))
        lightcone.circuit[1] = it.replaceprime(gate_ji_opt, tau => 1)

        println("Matrix is 2-local, converged to fidelity $newfid immediately")
        return Dict([("lightcone", lightcone), ("overlap", newfid), ("niter", 1)])
    end

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    lightcone = newLightcone(siteinds, tau; U_array = init_array, lightbounds = (false, false))

    L_blocks::Vector{it.ITensor} = []
    R_blocks::Vector{it.ITensor} = []

    # construct L_1
    # first item is zero projector, then there are the gates in ascending order, then the mpo site
    leftmost_block = mpo[1]  # 2 indices
    push!(L_blocks, leftmost_block)

    # construct R_N
    rightmost_block = mpo[N]
    push!(R_blocks, rightmost_block)

    # contract everything on the right and save rightmost_block at each intermediate step
    # must be done only the first time, when j=2 (so contract up to j=3)
    for k in (N-1):-1:3
        # extract right gates associated with site k
        gates_k = lightcone.gates_by_site[k]
        right_gates_k = [lightcone.circuit[gate["pos"]] for gate in gates_k if gate["orientation"]=="R"]
        all_blocks = [right_gates_k; mpo[k]]

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

    while sweep < maxiter

        # extract all gates touching site j
        gates_j = lightcone.gates_by_site[j]
        tensors_j = [lightcone.circuit[gate["pos"]] for gate in gates_j]

        # optimize each gate
        for i in 1:tau
            gate_ji = gates_j[i]    # extract gate ji
            
            left_tensors = []
            right_tensors = []
            for l in [1:i-1; i+1:tau]
                gate_jl = gates_j[l]
                if gate_jl["orientation"] == "L"
                    push!(left_tensors, tensors_j[l])
                else
                    push!(right_tensors, tensors_j[l])
                end
            end

            contract_left = []
            contract_right = []
            if isodd(tau+j)
                push!(contract_left, mpo[j])
            else
                push!(contract_right, mpo[j])
            end
            contract_left = [left_tensors; contract_left]
            contract_right = [right_tensors; contract_right]
            
            env_left = leftmost_block
            for gate in contract_left
                env_left *= gate
            end
            env_right = rightmost_block
            for gate in contract_right
                env_right *= gate
            end

            env = conj(env_left*env_right)

            inds = gate_ji["inds"][1:2]
            U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
            u, v = it.commonind(U, S), it.commonind(Vdag, S)

            # evaluate fidelity as Tr(Env*Gate), i.e. the overlap (NOT SQUARED)
            newfid = real(tr(Array(S, (u, v))))/normalization

            #replace gate_ji with optimized one, both in gates_j (used in this loop) and in circuit
            gate_ji_opt = U * it.replaceind(Vdag, v, u)
            tensors_j[i] = gate_ji_opt
            lightcone.circuit[gate_ji["pos"]] = gate_ji_opt
        end

        # while loop end conditions
        if newfid >= 1
            newfid = 1
            break
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

        # if we arrived at the end of the sweep
        if (j==2 && first_sweep==false) || j==N-1
            # compare the relative increase in frobenius norm
            ratio = 1 - sqrt((1-newfid)/(1-fid))
            #convergence occurs when fidelity after new sweep doesn't change considerably
            if -conv_err < ratio < conv_err && first_sweep == false
                break   
            end
            if ratio >= 0   # only register if the ratio has increased, otherwise it's useless
                fid = newfid
            end
        end

        
        if N > 3
            # update L_blocks or R_blocks depending on sweep direction
            if rev == false
                gates_to_append = tensors_j[(1+mod(j,2)):2:end]     # follow fig. 6
                all_blocks = [gates_to_append; mpo[j]]
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
                gates_to_append = tensors_j[(2-mod(j,2)):2:end]
                all_blocks = [gates_to_append; mpo[j]]
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

    println("Converged to fidelity $newfid with $sweep sweeps\n")

    return Dict([("lightcone", lightcone), ("overlap", newfid), ("niter", sweep)])

end


"Cost function and gradient for invertGlobalSweep optimization"
function _fgGlobalSweep(U_array::Vector{<:Matrix}, lightcone, mpo; normalization = 1)
    updateLightcone!(lightcone, U_array)
    d = lightcone.d
    N = lightcone.size

    R_blocks::Vector{it.ITensor} = []

    # construct L_1
    # first item is the delta, i.e. the first site of mpo_low which is composed of only deltas, then the mpo site
    leftmost_block = mpo[1]  # 2 indices

    # construct R_N
    rightmost_block = mpo[N]

    # contract everything on the left and save rightmost_block at each intermediate step
    # must be done only the first time, when j=2 (so contract up to j=3)
    for k in N:-1:3
        # extract left gates associated with site k
        gates_k = lightcone.gates_by_site[k]
        tensors_left = [lightcone.circuit[gate["pos"]] for gate in gates_k if gate["orientation"]=="L"]

        all_blocks = (k==N ? tensors_left : [tensors_left; mpo[k]])
        for block in all_blocks
            rightmost_block *= block
        end
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # start the sweep
    grad = [Array{ComplexF64}(undef, 0, 0) for _ in 1:lightcone.n_unitaries]

    for j in 2:N
        # extract all gates on the left of site j
        gates_j = lightcone.gates_by_site[j]
        tensors_left = [lightcone.circuit[gate["pos"]] for gate in gates_j if gate["orientation"]=="L"]

        # evaluate gradient by removing each gate
        for l in 1:length(tensors_left)
            not_l_tensors = [tensors_left[1:l-1]; tensors_left[l+1:end]]
            contract_left = [mpo[j]; not_l_tensors]
            
            env_left = leftmost_block
            for gate in contract_left
                env_left *= gate
            end

            env = (j<N ? env_left*rightmost_block : env_left)

            gate_jl = filter(gate -> gate["orientation"] == "L", gates_j)[l]
            gate_jl_inds, gate_jl_pos = gate_jl["inds"], gate_jl["pos"]
            ddUjl = Array(env, gate_jl_inds)
            ddUjl = conj(reshape(ddUjl, (d^2, d^2)))
            grad[gate_jl_pos] = ddUjl
        end

        # update leftmost_block for next j and add it to L_blocks list
        all_blocks = [tensors_left; mpo[j]]
        for block in all_blocks
            leftmost_block *= block
        end

        # update rightmost_block for next j
        if j < N-1
            rightmost_block = R_blocks[j]       #R_blocks starts from site 3
        end
    end

    # compute environment now that we contracted all blocks, so that we are effectively computing the overlap
    overlap = Array(leftmost_block)[1]

    # we use the real part of the overlap as a cost function
    abs_ov = real(overlap)/normalization

    # correct gradient to account for the cost function being the absolute value of the overlap, not the abs squared
    #grad *= overlap/abs_ov
    grad = grad/normalization
    riem_grad = project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -abs_ov
    riem_grad = -riem_grad

    return cost, riem_grad

end

"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it"
function invertGlobalSweep(mpo::Union{Vector{it.ITensor}, itmps.MPS, itmps.MPO}, tau, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; lightbounds = (false, false), maxiter = 20000, gradtol = 1E-8, normalization = 1, init_array::Union{Nothing, Vector{<:Matrix}} = nothing)::Dict{String, Any}
    mpo = deepcopy(mpo[1:end])
    N = length(mpo)

    # noprime the input inds
    # change the output inds to a prime of the input inds to match the inds of the first layer of gates
    siteinds = it.noprime(input_inds)
    d = siteinds[1].space
    tempinds = it.siteinds(d, N)
    for i in 1:N
        it.replaceind!(mpo[i], input_inds[i], tempinds[i])
        it.replaceind!(mpo[i], output_inds[i], it.prime(siteinds[i], tau))
        it.replaceind!(mpo[i], tempinds[i], siteinds[i])
    end

    if N == 2   #solution is immediate via SVD
        env = conj(mpo[1]*mpo[2])

        inds = siteinds
        U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
        u, v = it.commonind(U, S), it.commonind(Vdag, S)

        # evaluate fidelity
        newfid = real(tr(Array(S, (u, v))))/normalization
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        lightcone = newLightcone(siteinds, 1; lightbounds = (false, false))
        lightcone.circuit[1] = it.replaceprime(gate_ji_opt, tau => 1)

        println("Matrix is 2-local, converged to fidelity $newfid immediately")
        return Dict([("lightcone", lightcone), ("overlap", newfid), ("niter", 1)])
    end

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    lightcone = newLightcone(siteinds, tau; U_array = init_array, lightbounds = lightbounds)


    # setup optimization stuff
    arrU0 = Array(lightcone)    # ALWAYS NEEDED, since if the init_array has lower number of unitaries the lightcone fills it with identities
    fg = arrU -> _fgGlobalSweep(arrU, lightcone, mpo; normalization = normalization)

    # Quasi-Newton method
    m = 5
    algorithm = LBFGS(m;maxiter = maxiter, gradtol = gradtol, verbosity = 1)

    # optimize and store results
    # note that arrUmin is already stored in current lightcone, ready to be applied to mps
    arrUmin, neg_overlap, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport = true, inner = inner);
    updateLightcone!(lightcone, arrUmin)

    return Dict([("lightcone", lightcone), ("overlap", -neg_overlap), ("gradmin", gradmin), ("niter", numfg), ("history", normgradhistory)])

end


"Calls invertSweepLC for Vector{ITensor} mpo input with increasing inversion depth tau until it converges to chosen 'overlap' up to error 'eps'"
function invert(mpo::Union{Vector{it.ITensor}, itmps.MPS, itmps.MPO}, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}, invertMethod; tau = 0, eps = 1e-12, nruns = 1, overlap = 1, start_tau = 1, return_all = false, kargs...)
    obj = typeof(mpo)
    print("Attempting inversion of $obj with the following parameters:\nInversion method: $invertMethod\nNumber of runs: $nruns\n")

    fixed_tau_mode = true
    if iszero(tau)
        print("Overlap (normalized): $overlap\nRelative error: $eps\nStarting depth: $start_tau\n")
        tau = start_tau
        fixed_tau_mode = false
    else
        print("Inversion circuit depth: $tau\n")

    end

    found = false
    best_guess = nothing
    while !found
        println("Attempting depth $tau...")
        # choose multiprocessing method
        results_array::Vector{Dict{String, Any}} = [Dict([("ready", 0)]) for _ in 1:nruns]
        Threads.@threads for i in 1:nruns
            res = invertMethod(mpo, tau, input_inds, output_inds; init_array = best_guess, kargs...)
            results_array[i] = res
        end

        # results_array = [invertMethod(mpo, tau, input_inds, output_inds; kargs...) for _ in 1:nruns]
        
        ## f = obj -> invertMethod(obj, tau, input_inds, output_inds; kargs...)
        ## results_array = Distributed.pmap(f, (mpo for _ in 1:nruns))
        
        errs = [abs(overlap - results["overlap"]) for results in results_array]
        err_min_pos = argmin(errs)
        err_min = errs[err_min_pos]
        if err_min < eps || tau > 24 || fixed_tau_mode
            found = true
            tau = results_array[err_min_pos]["lightcone"].depth
            for i in 1:nruns
                results_array[i]["err"] = errs[i]
                results_array[i]["tau"] = tau
            end
            resdict = results_array[err_min_pos]

            if err_min < eps
                println("Found inversion circuit up to required error with depth $tau\n")
            elseif tau > 24
                println("Inversion failed, algorithm stopped at tau = 25")
            end
            return resdict
        end
        best_guess = Array(results_array[err_min_pos]["lightcone"])
        tau += 1
    end
end


"Wrapper for ITensorsMPS.MPS input. Before calling invert, it conjugates mps (mps to invert must be above)
and prepares a layer of zero bras to construct the mpo |0><psi|. Then calls invertMethod with overlap 1 and error eps"
function invert(mps::itmps.MPS, invertMethod; kargs...)
    N = length(mps)
    mps = conj(mps)
    siteinds = it.siteinds(mps)
    outinds = siteinds'

    for i in 1:N
        ind = siteinds[i]
        vec = [1; [0 for _ in 1:ind.space-1]]
        mps[i] *= it.ITensor(vec, ind')
    end

    results = invert(mps, siteinds, outinds, invertMethod; overlap=1, normalization=1, kargs...)
    return results
end


"Wrapper for ITensorsMPS.MPO input. Calls invert by first taking the dagger and extracting upper and lower indices"
function invert(mpo::itmps.MPO, invertMethod; kargs...)
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

    results = invert(mpo, outinds, siteinds, invertMethod; overlap = 1, normalization = 2^N, kargs...) #DO NOT CHANGE NORMALIZATION, WE NEED FIDELITY TO BE NORM. TO 1 OR THE RATIO WON'T WORK
    return results
end

"Combines invertSweepLC with invertGlobalSweep for ITensorsMPS.MPS or ITensorsMPS.MPO objects"
function invertCombined(object::Union{itmps.MPS, itmps.MPO}; kargs...)
    results_sweep = invert(object, invertSweepLC; kargs...)
    lc = results_sweep["lightcone"]
    arrU0 = Array(lc)

    results_global = invert(object, invertGlobalSweep; init_array = arrU0, kargs...)
    return results_global
end

"Combines invertSweepLC with invertGlobalSweep for Vector{ITensor} objects"
function invertCombined(object::Vector{it.ITensor}, siteinds, outinds; kargs...)
    results_sweep = invert(object, input_inds, output_inds, invertSweepLC; kargs...)
    lc = results_sweep["lightcone"]
    arrU0 = Array(lc)

    results_global = invert(object, input_inds, output_inds, invertGlobalSweep; init_array = arrU0, kargs...)
    return results_global
end



"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertSweep(mpo::Vector{it.ITensor}, tau, input_inds::Vector{<:it.Index}; d = 2, output_inds = nothing, conv_err = 1E-6, maxiter = 1E6)
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

    while sweep < maxiter

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

        if j == 2 && first_sweep == false
            rev = false
            sweep += 1
        end

        if j == N-1
            first_sweep = false
            rev = true
            sweep += 1
        end

        # if we arrived at the end of the sweep
        if (j==2 && first_sweep==false) || j==N-1
            # compare the relative increase in frobenius norm
            ratio = 1 - sqrt((1-newfid)/(1-fid))
            #convergence occurs when fidelity after new sweep doesn't change considerably
            if -conv_err < ratio < conv_err && first_sweep == false
                break   
            end
            if ratio >= 0   # only register if the ratio has increased, otherwise it's useless
                fid = newfid
            end
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


"Calls invertSweep for Vector{ITensor} mpo input with increasing inversion depth tau until it converges with fidelity F = 1-eps"
function invertSweep(mpo::Vector{it.ITensor}, input_inds::Vector{<:it.Index}; eps = 1E-6, start_tau = 1, n_runs = 10, kargs...)
    println("Tolerance $eps, starting from depth $start_tau")
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

        if abs(1-fid_best) < eps
            found = true
            println("Convergence within desired error achieved with depth $tau\n")
            break
        end
        
        if tau > 25
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



function _fg_disentangle(U_array::Vector{<:Matrix}, lightcone, reduced_mps, counter)
    updateLightcone!(lightcone, U_array)
    d = lightcone.d
    N = length(reduced_mps)
    siteinds = it.siteinds(reduced_mps)

    grad_j = [Array{ComplexF64}(undef, 0, 0) for _ in 1:lightcone.n_unitaries]
    purity_k = [0.0 for _ in 1:N-1]
    purity_j = [0.0 for _ in 1:lightcone.n_unitaries] #needed for gradient, less elements than pur_k
    
    oddstep = (lightcone.siteinds[1] == it.noprime(siteinds[1]))
    for k in (isodd(counter[]) ? (N-1:-1:1) : (1:N-1))
        it.orthogonalize!(reduced_mps, k)
        twoblocks = reduced_mps[k:k+1]
        block = twoblocks[1]*twoblocks[2]
        
        tensor = block
        indsL = it.uniqueinds(twoblocks[1], twoblocks[2])
        indsR = it.uniqueinds(twoblocks[2], twoblocks[1])

        theres_a_gate = (oddstep && isodd(k)) || (!oddstep && iseven(k))    # condition for which there's a unitary between site k and k+1
        if theres_a_gate
            j = (!oddstep && iseven(N)) ? k-1 : k   #need a map between the sites of reduced_mps and the lightcone, which is N-to-(N-2) if N even and step even
            gate = lightcone.gates_by_site[j][1]
            tensor_gate = lightcone.circuit[gate["pos"]]
            tensor *= tensor_gate
            upinds = gate["inds"][1:2]
            indsL = it.replaceinds(indsL, upinds', upinds)
            indsR = it.replaceinds(indsR, upinds', upinds)
        end

        combL = it.combiner(indsL)
        combR = it.combiner(indsR)
        tensor2 = conj(combL*tensor)
        tensor3 = combL*tensor*combR
        tensor4 = conj(tensor*combR)
        ddUk = block*tensor2*tensor3*tensor4

        if theres_a_gate
            ddUk_mat = 4*Array(ddUk, gate["inds"])
            ddUk_mat = conj(reshape(ddUk_mat, (d^2, d^2)))
            grad_j[gate["pos"]] = ddUk_mat
            ddUk *= tensor_gate
        end

        purity = real(Array(ddUk)[1])
        if theres_a_gate
            purity_j[gate["pos"]] = purity
        end
        purity_k[k] = purity
    end

    cost = -(sum(log.(purity_k)))
    grad = -(grad_j ./ purity_j)

    riem_grad = project(U_array, grad)
    counter[] = counter[]+1
    return cost, riem_grad

end


function disentangle!(mps::itmps.MPS, maxtau = 10; maxiter = 10000, gradtol = 1e-8, support_list = nothing)
    if isnothing(support_list)
        support_list = [(1, length(mps))]    # region of the mps we want to act on with the brickwork (in this case set to all)
    end
    lc_list::Vector{Lightcone} = []
    costs = []
    
    for support in support_list
        cost_i = []
        lcs::Vector{Lightcone} = []
        local red_siteinds
        for i in 1:maxtau
            it.truncate!(mps, cutoff = 1e-16)
            it.orthogonalize!(mps, support[end]-1)
            reduced_mps = itmps.MPS(mps[support[1]:support[end]])

            N = length(reduced_mps)
            reduced_mps.llim = N-2
            reduced_mps.rlim = N
            red_siteinds = it.siteinds(reduced_mps)

            lc_support = (isodd(i) ? 1 : 2, iseven(i+N) ? N-1 : N)     # region the lightcone acts on - will be equal to support only if support is even and i is even, else it will be less

            lc_sites = red_siteinds[lc_support[1]:lc_support[end]]
            lightcone = newLightcone(lc_sites, 1; lightbounds = (false, false))
            
            for j in 1:N
                it.replaceind!(reduced_mps[j], red_siteinds[j], red_siteinds[j]')
            end

            # setup optimization stuff
            arrU0 = Array(lightcone)
            counter = Ref(1)
            fg = arrU -> _fg_disentangle(arrU, lightcone, reduced_mps, counter)

            # Quasi-Newton method
            m = 5
            algorithm = LBFGS(m;maxiter = maxiter, gradtol = gradtol, verbosity = 1)

            # optimize and store results
            # note that arrUmin is already stored in current lightcone, ready to be applied to mps
            arrUmin, cost, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport = true , inner = inner);
            updateLightcone!(lightcone, arrUmin)

            apply!(mps, lightcone)
            push!(cost_i, abs(cost))
            push!(lcs, lightcone)

            if -1e-15 < cost < 1e-15
                break
            end
        end
        layers = [Array(lc) for lc in lcs]
        layers = [isodd(i) ? layers[i] : reverse(layers[i]) for i in 1:length(layers)]
        lc_tot = newLightcone(red_siteinds, length(layers))
        push!(lc_list, lc_tot)
        push!(costs, cost_i)
    end

    return Dict([("lc_list", lc_list), ("costs", costs)])

end