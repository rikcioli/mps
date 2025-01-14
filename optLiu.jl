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
    # cost function is the fidelity, i.e. the square of the overlap
    fid = abs(Array(mt.contract(reduced_mps, conj_mps))[1])
    cost = -fid
    riem_grad = - riem_grad

    return cost, riem_grad
end


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
        fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)

        # Quasi-Newton method
        m = 5
        iter = 1000
        algorithm = LBFGS(m;maxiter = iter, gradtol = 1E-8, verbosity = 1)

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
        end
        if range[end]<N
            comb2 = it.combiner(trunc_linkinds[range[2]], trunc_siteinds[range[2]])
            reduced_mps[end] *= comb2
        end

        reduced_mps = itmps.MPS(reduced_mps)
        
        tau_inv, _, err_inv, _ = mt.invertGlobal(reduced_mps; start_tau = (range in rangesA ? 1 : 2))
        push!(results_second_part, [tau_inv, err_inv])
    end
    
    return V_list, err_list, mps, lc_list, mps_trunc, results_second_part

end


function invertMPSLiu(mps::itmps.MPS; d = 2, eps_trunc = 0.01, eps_inv = 0.01)

    N = length(mps)
    isodd(N) && throw(DomainError(N, "Choose an even number for N"))
    mps = deepcopy(mps)
    siteinds = it.siteinds(mps)
    eps_liu = eps_trunc/N

    local mps_trunc, boundaries, rangesA
    while true
        # first determine depth of input state, i.e. find lightcone that inverts to 0 up to error eps_liu
        # we do this by inverting an increasingly wider lightcone on sites 1:2j for j=1,2,3,...
        it.orthogonalize!(mps, 2)
        found = false
        right_endsite = 2
        tau = 1

        println("Finding depth of initial state up to error $eps_liu")
        while !found
            reduced_mps = mps[1:right_endsite]
            local_sites = siteinds[1:right_endsite]
            lightcone = mt.newLightcone(local_sites, tau; lightbounds = (true, true))
            # setup optimization stuff
            arrU0 = Array(lightcone)
            fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)
            # Quasi-Newton method
            m = 5
            algorithm = LBFGS(m;maxiter = 10000, gradtol = 1E-8, verbosity = 1)
            # optimize and store results
            # note that arrUmin is already stored in current lightcone, ready to be applied to mps
            arrUmin, negfid, _ = optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport =true , inner = mt.inner);

            err = 1+negfid
            if err < eps_inv
                found = true
            else
                tau += 1
                right_endsite *= 2
            end
        end

        # at this point we have tau, we already know that both the sizeAB and the spacing have to be chosen
        # so that the final state is a tensor product of pure states
        sizeAB = 6*(tau-1)
        spacing = 2*(tau-1)
        println("Found depth tau = $tau, imposing sizeAB = $sizeAB and spacing = $spacing for factorization")

        @assert tau > 0
        isodd(sizeAB) && throw(DomainError(sizeAB, "Choose an even number for sizeAB"))
        isodd(spacing) && throw(DomainError(spacing, "Choose an even number for the spacing between regions"))
        
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
        println("Inverting reduced density matrices...")
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
            fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)
            # Quasi-Newton method
            m = 5
            algorithm = LBFGS(m;maxiter = 10000, gradtol = 1E-8, verbosity = 1)
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

        if err_trunc <= eps_trunc
            break
        else
            println("Convergence not found with initial depth tau = $tau, reducing error on initial inversion")
            eps_liu /= 2
        end
    end

    println("Truncation reached within requested eps_trunc, inverting local states...")
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
        end
        if range[end]<N
            comb2 = it.combiner(trunc_linkinds[range[2]], trunc_siteinds[range[2]])
            reduced_mps[end] *= comb2
        end

        reduced_mps = itmps.MPS(reduced_mps)
        
        tau_2, _, err_2, _ = mt.invertGlobalSweep(reduced_mps; start_tau = (range in rangesA ? 1 : 2), eps = eps_inv)
        push!(results_second_part, [tau_2, err_2])
    end
    
    return V_list, err_list, mps, lc_list, mps_trunc, results_second_part

end


it.set_warn_order(21)

N = 24


#mps = mt.randMPS(N, 2)
mps = mt.initialize_fdqc(N, 2)
siteinds = it.siteinds(mps)


results = invertMPSLiu(mps)
#mpsfinal, lclist, mps_trunc, second_part = results[3:6]

#@time results = mt.invertGlobalSweep(mps)
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