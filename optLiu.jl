include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots

using OptimKit, LaTeXStrings, LinearAlgebra, Statistics


"Compute cost and riemannian gradient"
function fgLiu(U_array::Vector{<:Matrix}, lightcone, reduced_mps::Vector{it.ITensor})
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
    fid = real(Array(mt.contract(reduced_mps, conj_mps))[1])
    cost = -fid
    riem_grad = - riem_grad

    return cost, riem_grad
end


function invertMPSLiu(mps::itmps.MPS, tau, sizeAB, spacing; d = 2)

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
        lightbounds = (i==1 ? false : true, last_site==N ? false : true)

        lightcone = mt.newLightcone(k_sites, tau; lightbounds = lightbounds)

        # setup optimization stuff
        arrU0 = Array(lightcone)
        fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)

        # Quasi-Newton method
        m = 5
        iter = 1000
        algorithm = LBFGS(m;maxiter = iter, gradtol = 1E-8, verbosity = 2)

        # optimize and store results
        # note that arrUmin is already stored in current lightcone, ready to be applied to mps
        arrUmin, err, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport =true , inner = mt.inner);
        
        # reduced_mps = [it.prime(tensor, tau) for tensor in reduced_mps]
        # mt.contract!(reduced_mps, lightcone)
        # for l in i:last_site
        #     mps[l] = it.noprime(reduced_mps[l-i+1])
        # end

        push!(lc_list, lightcone)
        push!(V_list, arrUmin)
        push!(err_list, 1+err)
    end

    it.prime!(mps, tau)
    mt.contract!(mps, lc_list, initial_pos)
    mps = it.noprime(mps)
    

    ncones = length(lc_list)
    rangesC = []
    for i in 1:ncones-1
        rel_range = lc_list[i].range[end][2]
        ninit = initial_pos[i] + rel_range
        nend = ninit + 2*(tau-1) + spacing - 1
        push!(rangesC, (ninit,nend))
    end
    
    @show rangesC
    results_second_part = []
    mps_trunc = deepcopy(mps)
    if length(rangesC) > 0
        # project to 0 the sites which are supposed to be already 0
        
        rangesA = [(1, rangesC[1][1]-1)]
        nC = length(rangesC)
        for l in 2:nC
            push!(rangesA, (rangesC[l-1][2]+1, rangesC[l][1]-1))
        end
        push!(rangesA, (rangesC[end][2]+1, N))
        @show rangesA

        for range in rangesA
            mt.project_tozero!(mps_trunc, [i for i in range[1]:range[2]])
        end
        siteinds = it.siteinds(mps_trunc)

        for range in rangesC
            # extract reduced mps and remove external linkind (which will be 1-dim)
            reduced_mps = mps_trunc[range[1]:range[2]]

            comb1 = it.combiner(it.linkinds(mps_trunc)[range[1]-1], siteinds[range[1]])
            reduced_mps[1] *= comb1

            comb2 = it.combiner(it.linkinds(mps_trunc)[range[2]], siteinds[range[2]])
            reduced_mps[end] *= comb2


            reduced_mps = itmps.MPS(reduced_mps)
            
            _, fid_best, _, tau_best = mt.invertBW(reduced_mps)
            push!(results_second_part, [fid_best, tau_best])
        end
    end

    return V_list, err_list, mps, lc_list, mps_trunc, results_second_part
end


function _fgGlobal(U_array::Vector{<:Matrix}, lightcone, mpo, mpo_low)
    mpo = deepcopy(mpo)
    mpo_low = deepcopy(mpo_low)
    mt.updateLightcone!(lightcone, U_array)
    len = length(lightcone.coords)
    # compute gradient by removing one unitary at a time and applying
    # all unitaries that come before to the virtual mpo_low
    # all unitaries that come after to the original mpo
    # the shift to the next unitary is made by applying U_k Udag_k+1
    grad::Vector{Matrix} = []
    for k in len:-1:2   # contract up to first unitary
        mt.contract!(mpo, lightcone, k)
    end

    # evaluate gradient by sweeping over snake of unitaries
    d = lightcone.d
    for k in 1:len
        _, inds_k = lightcone.coords[k]
        ddUk = mt.contract(mpo, mpo_low)
        ddUk = Array(ddUk, inds_k)
        ddUk = conj(reshape(ddUk, (d^2, d^2)))
        push!(grad, ddUk)
        if k < len
            mt.contract!(mpo_low, lightcone, k)
            mt.contract!(mpo, lightcone, k+1; dagger=true)
        end
    end
    # compute total overlap (which is a complex number)
    mt.contract!(mpo_low, lightcone, len)
    overlap = Array(mt.contract(mpo, mpo_low))[1]
    abs_ov = abs(overlap)

    # correct gradient to account for a smooth cost function (the overlap squared)
    grad *= overlap/abs_ov
    riem_grad = mt.project(U_array, grad)

    # put a - sign so that it minimizes
    cost = -abs_ov
    riem_grad = -riem_grad

    return cost, riem_grad
end

# NEW VERSION USING OPT TECH
"Given a Vector{ITensor} 'mpo', construct the depth-tau brickwork circuit of 2-qu(d)it unitaries that approximates it;
If no output_inds are given the object is assumed to be a state, and a projection onto |0> is inserted"
function invertGlobal(mpo::Vector{it.ITensor}, tau, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; d = 2, conv_err = 1E-6, n_sweeps = 1E6)
    mpo = deepcopy(mpo)
    N = length(mpo)
    siteinds = input_inds

    # create random brickwork circuit
    # circuit[i][j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    lightcone = mt.newLightcone(siteinds, tau; lightbounds = (false, false))

    if N == 2   #solution is immediate via SVD
        env = conj(mpo[1]*mpo[2])

        inds = siteinds
        U, S, Vdag = it.svd(env, inds, cutoff = 1E-15)
        u, v = it.commonind(U, S), it.commonind(Vdag, S)

        # evaluate fidelity
        newfid = real(tr(Array(S, (u, v))))
        gate_ji_opt = U * it.replaceind(Vdag, v, u)
        lightcone.circuit[1][1] = gate_ji_opt

        if mpo_mode
            newfid /= 2^N # normalize if mpo mode
        end

        println("Matrix is 2-local, converged to fidelity $newfid immediately")
        return lightcone, newfid
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


    # testing riemannian gradient
    n_unitaries = length(lightcone.coords)
    # generate random point on M
    arrU0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
    arrU0dag = [U' for U in arrU0]
    # 
    # # generate random tangent Vector
    arrV = [mt.random_unitary(4) for _ in 1:n_unitaries]
    arrV = mt.skew.(arrV)
    arrV = arrU0 .* arrV
    arrV /= sqrt(mt.inner(arrV, arrV))
    
    func, grad = fg(arrU0)
    prod = mt.inner(grad, arrV)
    
    # compute E(t) for several values of t
    E = t -> abs(fg(mt.retract(arrU0, arrV, t)[1])[1] - func - t*prod)
    
    tvals = exp10.(-8:0.1:0)
    
    Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\gradtest2.png");

    # Quasi-Newton method
    m = 5
    iter = 1000
    algorithm = LBFGS(m;maxiter = iter, gradtol = 1E-8, verbosity = 2)

    # optimize and store results
    # note that arrUmin is already stored in current lightcone, ready to be applied to mps
    arrUmin, cost, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport =true , inner = mt.inner);

    return lightcone, 1+cost, gradmin, numfg, normgradhistory

end

"Calls invertBW for Vector{ITensor} mpo input with increasing inversion depth tau until it converges with fidelity F = 1-err_to_one"
function invertGlobal(mpo::Vector{it.ITensor}, input_inds::Vector{<:it.Index}, output_inds::Vector{<:it.Index}; err_to_one = 1E-6, start_tau = 1, n_runs = 10, kargs...)
    println("Tolerance $err_to_one, starting from depth $start_tau")
    tau = start_tau
    found = false

    while !found
        println("Attempting depth $tau...")
        bw, err, sweep = invertGlobal(mpo, tau, input_inds, output_inds; kargs...)

        if abs(err) < err_to_one
            found = true
            println("Convergence within desired error achieved with depth $tau\n")
            return bw, err, sweep
        end
        
        #if tau > 9
        #    println("Attempt stopped at tau = $tau, ITensor cannot go above")
        #    break
        #end

        tau += 1
    end
    return bw_best, err_best, sweep_best, tau
end

"Wrapper for ITensorsMPS.MPS input. Calls invertBW by first conjugating mps (mps to invert must be above)
and then preparing a layer of zero bras to construct the mpo |0><psi|"
function invertGlobal(mps::itmps.MPS; tau = 0, kargs...)
    N = length(mps)
    obj = typeof(mps)
    println("Attempting inversion of $obj")
    mps = conj(mps)
    siteinds = it.siteinds(mps)
    outinds = siteinds'

    for i in 1:N
        ind = siteinds[i]
        vec = [1; [0 for _ in 1:ind.space-1]]
        mps[i] *= it.ITensor(vec, ind')
    end


    if iszero(tau)
        results = invertGlobal(mps[1:end], siteinds, outinds; kargs...)
    else
        results = invertGlobal(mps[1:end], tau, siteinds, outinds; kargs...)
    end
    return results
end


N = 15

mps = mt.randMPS(N, 2)
#mps = mt.initialize_fdqc(N, 3)
siteinds = it.siteinds(mps)

#results = invertMPSLiu(mps, 3, 8, 0)
#mpsfinal, lclist, mps_trunc, second_part = results[3:6]

results = invertGlobal(mps, tau=2)
#results2 = mt.invertMPSMalz(mps)

Plots.plot(results[5][:,2],label ="L-BFGS",yscale =:log10)
Plots.plot!(xlabel = "iterations",ylabel = L"||\nabla{f}||")
Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\gradhist.png");


entropies = [mt.entropy(mpsfinal, i) for i in 1:N-1]
norms = [norm(mpsfinal - mt.project_tozero(mpsfinal, [i])) for i in 1:N]

res = mt.invertBW(mps)


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