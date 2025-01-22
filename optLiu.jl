include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots

using OptimKit, LaTeXStrings, LinearAlgebra, Statistics


"Cost function and gradient for invertGlobalSweep optimization"
function _fgLiu(U_array::Vector{<:Matrix}, lightcone, reduced_mps::Vector{it.ITensor})
    mt.updateLightcone!(lightcone, U_array)
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
    riem_grad = mt.project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -overlap_sq
    riem_grad = - riem_grad

    return cost, riem_grad

end


"Compute cost and riemannian gradient"
#old version where we contract upper mps with lower mps, causes variation to lose lock conditions
function fgLiuOld(U_array::Vector{<:Matrix}, lightcone, reduced_mps::Vector{it.ITensor})
    mt.updateLightcone!(lightcone, U_array)

    #reduced_mps = it.prime(reduced_mps, lightcone.depth)
    reduced_mps = [it.prime(tensor, lightcone.depth) for tensor in reduced_mps]
    conj_mps = deepcopy(reduced_mps)

    # apply each unitary to mps
    mt.contract!(conj_mps, lightcone)
    conj_mps = conj(conj_mps)

    # project A region onto |0><0| (using the range property of 
    # lightcone to determine which sites are involved)
    d = lightcone.d
    zero_vec = [1; [0 for _ in 1:d-1]]
    zero_mat = kron(zero_vec, zero_vec')
    for l in lightcone.range[end][1]:lightcone.range[end][2]
        ind = lightcone.sitesAB[l]
        zero_proj = it.ITensor(zero_mat, ind, ind')
        conj_mps[l] = it.replaceind(conj_mps[l]*zero_proj, ind', ind)
    end

    len = length(lightcone.coords)
    grad::Vector{Matrix} = []
    for k in 1:len
        mps_low = deepcopy(reduced_mps)
        mps_up = deepcopy(conj_mps)
        for l in 1:k-1
            mt.contract!(mps_low, lightcone, l)
        end
        for l in len:-1:k+1
            mt.contract!(mps_up, lightcone, l)
        end
        
        _, inds_k = lightcone.coords[k]
        ddUk = mt.contract(mps_low, mps_up)
        ddUk = Array(ddUk, inds_k)
        ddUk = 2*conj(reshape(ddUk, (d^2, d^2)))
        push!(grad, ddUk)
    end
    
    riem_grad = mt.project(U_array, grad)

    # # check that the gradient STAYS TANGENT
    # arrUinv = [U' for U in U_array]
    # grad_id = arrUinv .* riem_grad
    # non_skewness = norm(grad_id - skew.(grad_id))

    mt.contract!(reduced_mps, lightcone)
    # cost function is the fidelity, i.e. the square of the overlap
    fid = abs(Array(mt.contract(reduced_mps, conj_mps))[1])
    cost = -fid
    riem_grad = - riem_grad

    return cost, riem_grad
end

# old version 
function invertMPSLiu(mps::itmps.MPS, tau, sizeAB, spacing; d = 2, eps_trunc = 0.01)

    @assert tau > 0
    
    isodd(sizeAB) && throw(DomainError(sizeAB, "Choose an even number for sizeAB"))
    isodd(spacing) && throw(DomainError(spacing, "Choose an even number for the spacing between regions"))

    N = length(mps)
    isodd(N) && throw(DomainError(N, "Choose an even number for N"))
    mps = deepcopy(mps)
    siteinds = it.siteinds(mps)
    i = spacing+1
    initial_pos::Vector{Int64} = []
    while i+sizeAB-1 < N+1
        push!(initial_pos, i)
        i += sizeAB+spacing
    end
    rangesAB = [(i, min(i+sizeAB-1, N)) for i in initial_pos]
    @show rangesAB
    rangesA = []

    V_list = []
    lc_list::Vector{mt.Lightcone} = []
    err_list = []
    for i in initial_pos
        last_site = min(i+sizeAB-1, N)
        k_sites = siteinds[i:last_site]
        it.orthogonalize!(mps, div(i+last_site, 2))

        # extract reduced mps on k_sites and construct lightcone structure of depth tau
        reduced_mps = mps[i:last_site]
        # FOR NOW CAN ONLY DEAL WITH (TRUE, TRUE)
        #lightbounds != (true, true) && 
        #    throw(DomainError(lightbounds, "Try to choose spacing so that regions to invert are away from boundaries"))

        lightcone = mt.newLightcone(k_sites, tau; lightbounds = (true, true))
        rangeA = lightcone.range[end]
        push!(rangesA, (rangeA[1]+i-1, rangeA[2]+i-1))

        # setup optimization stuff
        arrU0 = Array(lightcone)
        fg = arrU -> _fgLiu(arrU, lightcone, reduced_mps)

        # Quasi-Newton method
        m = 5
        algorithm = LBFGS(m;maxiter = 10000, gradtol = 1E-8, verbosity = 1)

        # optimize and store results
        # note that arrUmin is already stored in current lightcone, ready to be applied to mps
        arrUmin, err, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport = true, inner = mt.inner);
        
        # reduced_mps = [it.prime(tensor, tau) for tensor in reduced_mps]
        # mt.contract!(reduced_mps, lightcone)
        # for l in i:last_site
        #     mps[l] = it.noprime(reduced_mps[l-i+1])
        # end
        mt.updateLightcone(lightcone, arrUmin)
        push!(lc_list, lightcone)
        push!(V_list, arrUmin)
        push!(err_list, 1+err)
    end

    it.prime!(mps, tau)
    mt.contract!(mps, lc_list, initial_pos)
    mps = it.noprime(mps)
    
    results_second_part = []
    mps_trunc = deepcopy(mps)

    @show rangesA
    boundaries = [0]
    for rangeA in rangesA
        if rangeA[1]>1
            mt.cut!(mps_trunc, rangeA[1]-1)
            push!(boundaries, rangeA[1]-1)
        end
        if rangeA[end]<N
            mt.cut!(mps_trunc, rangeA[end])
            push!(boundaries, rangeA[end])
        end
    end
    err_trunc = norm(mps - mps_trunc)
    @show err_trunc

    trunc_siteinds = it.siteinds(mps_trunc)
    trunc_linkinds = it.linkinds(mps_trunc)

    ranges = []
    for l in 1:length(boundaries)-1
        push!(ranges, (boundaries[l]+1, boundaries[l+1]))
    end
    push!(ranges, (boundaries[end], N))
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
        
        tau_inv, _, err_inv, _ = mt.invertGlobalSweep(reduced_mps; start_tau = (range in rangesA ? 1 : 2))
        push!(results_second_part, [tau_inv, err_inv])
    end
    
    return V_list, err_list, lc_list, mps_trunc, results_second_part

end


function invertMPSLiu(mps::itmps.MPS; d = 2, eps_trunc = 0.01, eps_inv = 0.01)

    N = length(mps)
    isodd(N) && throw(DomainError(N, "Choose an even number for N"))
    siteinds = it.siteinds(mps)
    #eps_liu = eps_trunc/N

    local mps_trunc, boundaries, rangesA, V_list, err_list, lc_list, err_trunc
    tau = 1
    while true
        mps_copy = deepcopy(mps)
        # first determine depth of input state, i.e. find lightcone that inverts to 0 up to error eps_liu
        # we do this by inverting an increasingly wider lightcone on sites 1:2j for j=1,2,3,...
        ###found = false
        ###right_endsite = 2
        ###tau = 1

        ###println("Finding depth of initial state up to error $eps_liu")
        ###while !found
        ###    reduced_mps = mps[1:right_endsite]
        ###    local_sites = siteinds[1:right_endsite]
        ###    lightcone = mt.newLightcone(local_sites, tau; lightbounds = (true, true))
        ###    # setup optimization stuff
        ###    arrU0 = Array(lightcone)
        ###    fg = arrU -> _fgLiu(arrU, lightcone, reduced_mps)
        ###    # Quasi-Newton method
        ###    m = 5
        ###    algorithm = LBFGS(m;maxiter = 10000, gradtol = 1E-8, verbosity = 1)
        ###    # optimize and store results
        ###    # note that arrUmin is already stored in current lightcone, ready to be applied to mps
        ###    arrUmin, negfid, _ = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport =true , inner = mt.inner);

        ###    err = 1+negfid
        ###    if err < eps_inv
        ###        found = true
        ###    else
        ###        tau += 1
        ###        right_endsite += 2 
        ###    end
        ###end

        # at this point we have tau, we already know that both the sizeAB and the spacing have to be chosen
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
        lc_list::Vector{mt.Lightcone} = []
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
            lightcone = mt.newLightcone(k_sites, tau; lightbounds = lightbounds)
            rangeA = lightcone.range[end]
            push!(rangesA, (rangeA[1]+i-1, rangeA[2]+i-1))

            # setup optimization stuff
            arrU0 = Array(lightcone)
            fg = arrU -> _fgLiu(arrU, lightcone, reduced_mps)
            # Quasi-Newton method
            m = 5
            algorithm = LBFGS(m;maxiter = 10000, gradtol = 1E-8, verbosity = 1)
            # optimize and store results
            # note that arrUmin is already stored in current lightcone, ready to be applied to mps
            arrUmin, err, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport =true , inner = mt.inner);
            
            push!(lc_list, lightcone)
            push!(V_list, arrUmin)
            push!(err_list, 1+err)
        end

        it.prime!(mps_copy, tau)
        mt.contract!(mps_copy, lc_list, initial_pos)
        mps_copy = it.noprime(mps_copy)
        mps_trunc = deepcopy(mps_copy)

        @show rangesA
        boundaries = [0]
        for rangeA in rangesA
            if rangeA[1]>1
                mt.cut!(mps_trunc, rangeA[1]-1)
                push!(boundaries, rangeA[1]-1)
            end
            if rangeA[end]<N
                mt.cut!(mps_trunc, rangeA[end])
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
    lc_list2::Vector{mt.Lightcone} = []
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
        
        tau2, lc2, err2, _ = mt.invertGlobalSweep(reduced_mps; start_tau = (range in rangesA ? 1 : 2), eps = eps_inv)
        push!(lc_list2, lc2)
        push!(tau_list2, tau2)
        push!(err_list2, err2)
    end


    # finally create a 0 state and apply all the V to recreate the original state for final total error
    mps_final = mt.initialize_vac(N, trunc_siteinds)
    mt.apply!(mps_final, lc_list2)
    err2tot = 1-abs(Array(mt.contract(mps_final, conj(mps_trunc)))[1])
    @show err2tot
    it.replace_siteinds!(mps_final, siteinds)
    mt.apply!(mps_final, lc_list, dagger=true)
    err_total = 1-abs(Array(mt.contract(mps_final, conj(mps)))[1])
    @show err_total
    
    return Dict([("lc1", lc_list), ("tau1", tau), ("err1", err_list), ("lc2", lc_list2), ("tau2", tau_list2), ("err2", err_list2), ("mps_final", mps_final), ("err_trunc", err_trunc), ("err_inv", err2tot), ("err_total", err_total)])

end


it.set_warn_order(28)

N = 16

mps = mt.randMPS(N, 2)
#mps = mt.initialize_fdqc(N, 3)
siteinds = it.siteinds(mps)


#results = invertMPSLiu(mps)
#mpsfinal, lclist, mps_trunc, second_part = results[3:6]

#@time results = mt.invertGlobalSweep(mps)
results2 = mt.invertMPSMalzGlobal(mps, eps_malz = 0.4)



# Plots.plot(results[5][:,2],label ="L-BFGS",yscale =:log10)
# Plots.plot!(xlabel = "iterations",ylabel = L"||\nabla{f}||")
# Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\gradhist.png");
# 
# 
# entropies = [mt.entropy(mpsfinal, i) for i in 1:N-1]
# norms = [norm(mpsfinal - mt.project_tozero(mpsfinal, [i])) for i in 1:N]
# 
# res = mt.invertBW(mps)


# reduced_mps = mps[2:9]
# reduced_inds = siteinds[2:9]
# 
# lightcone = mt.newLightcone(reduced_inds, 4)
# n_unitaries = length(lightcone.coords)
# arrU0 = Array(lightcone)
# fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)

#Gradient Descent algorithm 
# iter = 10000
# linesearch = HagerZhangLineSearch()
# algorithm = GradientDescent(;maxiter = iter,gradtol = eps(),linesearch, verbosity=2) 
# grad_norms = []
# function finalize!(x, f, g, numiter)
#     push!(grad_norms, inner(g, g))
#     return x,f,g
# end
# 
# optimize(fg, arrU0, algorithm; retract = retract, inner = inner, finalize! = finalize!);
# 
# Plots.plot(grad_norms,label ="GD",yscale =:log10)
# Plots.plot!(xlabel = "iterations",ylabel = L"||\nabla{f}||^2")


# Quasi-Newton method
# m = 5
# iter = 1000
# linesearch = HagerZhangLineSearch()
# algorithm = LBFGS(m;maxiter = iter, verbosity = 2)
# grad_norms = []
# function finalize!(x, f, g, numiter)
#     if f < eps()
#         f = 0
#         g *= eps()
#     end
#     push!(grad_norms, sqrt(inner(g, g)))
#     return x,f,g
# end
# 
# optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport =true , inner = inner, finalize! = finalize!);
# Plots.plot(grad_norms,label ="L-BFGS",yscale =:log10)
# Plots.plot!(xlabel = "iterations",ylabel = L"||\nabla{f}||")



# # testing riemannian gradient
# n_unitaries = length(lightcone.coords)
# # generate random point on M
# arrU0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
# arrU0dag = [U' for U in arrU0]
# # 
# # # generate random tangent Vector
# arrV = [mt.random_unitary(4) for _ in 1:n_unitaries]
# arrV = mt.skew.(arrV)
# arrV = arrU0 .* arrV
# arrV /= sqrt(mt.inner(arrV, arrV))
# 
# # compute f and gradf, check that gradf is in TxM and compute inner prod in x
# fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)
# #fg = arrU -> (real(tr(arrU'arrU)), project(arrU, 2*arrU))
# func, grad = fg(arrU0)
# # bring grad back to the tangent space to the identity and check it's skew hermitian
# arrX = arrU0dag .* grad     
# norm(arrX - skew.(arrX))
# prod = inner(grad, arrV)
# 
# # test retraction and geodesic distance
# t = 0.001
# dist_un(retract(arrU0, arrV, t)[1], arrU0)
# sqrt(inner(t*arrV, t*arrV))
# 
# # test derivative
# norm((retract(arrU0, arrV, t)[1] .- arrU0)/t - arrV)
# # it works, so the problem is in the grad i think
# DF = (fg(retract(arrU0, arrV, t)[1])[1] - func)/t
# prod
# norm(DF - prod)
# 
# 
# # compute E(t) for several values of t
# E = t -> abs(fg(mt.retract(arrU0, arrV, t)[1])[1] - func - t*prod)
# 
# tvals = exp10.(-8:0.1:0)
# 
# Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
# Plots.plot!(tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
# Plots.plot!(tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")