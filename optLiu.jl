include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using OptimKit, LaTeXStrings, LinearAlgebra


#this function fixes the phases; each column of a unitary has an arbitrary phase
# by choosing the phase ϕ_i to be the phase of the largest entry in the i'th column
function fix_phase(U)
    ϕs = []
    for v in eachcol(U)
        i = argmax(abs.(v))
        ϕ = atan(imag(v[i]),real(v[i]))
        push!(ϕs,ϕ)
    end
    return U*diagm(exp.(-1im*ϕs))
end

"Geodesic distance between two arrays of unitaries"
function dist_un(arrU1::Vector{<:AbstractMatrix}, arrU2::Vector{<:AbstractMatrix})
    distances_sq = []
    for k in length(arrU1)
        U1, U2 = arrU1[k], arrU2[k]
        V = eigvals(fix_phase(U1)'fix_phase(U2))
        dist_sq = sum(map(x->atan(imag(x),real(x))^2,V)) +eps()
        push!(distances_sq, dist_sq)
    end
    return sqrt(sum(distances_sq)) 
end

function skew(X)
    return (X - X')/2
end

function skew(arrX::Vector{<:AbstractMatrix})
    arrXinv = [X' for X in arrX]
    return (arrX .- arrXinv)/2
end

"Project arbitrary array of unitaries arrD onto the tangent space at arrU"
function project(arrU::Vector{<:AbstractMatrix}, arrD::Vector{<:AbstractMatrix})
    #return (arrD .- (arrU .* [D' for D in arrD] .* arrU))/2
    return arrU .* skew([U' for U in arrU] .* arrD)
end

function polar(M)
    U, S, V = svd(M)
    P = V*diagm(S)*V'
    W = U*V'
    return P, W
end

function polar(arrM::Vector{<:AbstractMatrix})
    arrU = [polar(M)[2] for M in arrM]
    return arrU
end


#move  U in the direction of X with step length t, 
#X is the gradient obtained using projection.
#return both the "retracted" unitary as well as the tangent vector at the retracted point
# always stabilize unitarity and skewness, it could leave tangent space due to numerical errors
function retract(arrU, arrX, t)

    # ensure unitarity of arrU
    arrU_polar = polar(arrU)
    # non_unitarity = norm(arrU - arrU_polar)/length(arrU)
    # @show non_unitarity
    arrU = arrU_polar

    arrUinv = [U' for U in arrU]
    arrX_id = arrUinv .* arrX

    # ensure skewness of arrX_id
    # non_skewness = norm(arrX_id - skew(arrX_id))/length(arrU)
    # @show non_skewness
    # if non_skewness > 1E-15 + eps()
    #     throw(DomainError(non_skewness, "arrX is not in the tangent space at arrU"))
    # end
    arrX_id = skew(arrX_id)

    # construct the geodesic at the tangent space at unity
    # then move it to the correct point by multiplying by arrU
    arrU_new = arrU .* exp.(t*arrX_id)

    # move arrX to the new tangent space arrU_new
    arrX_new = arrU_new .* arrX_id #move first to the tangent space at unity, then to the new point
    return arrU_new, arrX_new
end

function inner(arrU, arrX, arrY)
    return real(tr(arrX'*arrY))
end
function inner(arrX, arrY)
    return real(tr(arrX'*arrY))
end

#parallel transport
"""transport tangent vector ξ along the retraction of x in the direction η (same type as a gradient) 
with step length α, can be in place but the return value is used. 
Transport also receives x′ = retract(x, η, α)[1] as final argument, 
which has been computed before and can contain useful data that does not need to be recomputed"""
function transport!(ξ, arrU, η, α, arrU_new)
    arrUinv = [U' for U in arrU]
    ξ = arrU_new .* arrUinv .* ξ
    return ξ
end


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
    
    riem_grad = project(U_array, grad)

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

    mps = deepcopy(mps)
    N = length(mps)
    siteinds = it.siteinds(mps)
    i = 2
    initial_pos::Vector{Int64} = []
    while i < N
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
        lightbounds = (true, true)
        # FOR NOW CAN ONLY DEAL WITH (TRUE, TRUE) 
        # if i == 1
        #     lightbounds = (false, true)
        # elseif last_site == N
        #     lightbounds = (true, false)
        # end
        lightcone = mt.newLightcone(k_sites, tau; lightbounds = lightbounds)

        # setup optimization stuff
        arrU0 = Array(lightcone)
        fg = arrU -> fgLiu(arrU, lightcone, reduced_mps)

        # Quasi-Newton method
        m = 5
        iter = 1000
        algorithm = LBFGS(m;maxiter = iter, gradtol = 1E-8, verbosity = 2)
        grad_norms = []
        function finalize!(x, f, g, numiter)
            push!(grad_norms, sqrt(inner(g, g)))
            return x,f,g
        end

        # optimize and store results
        # note that arrUmin is already stored in current lightcone, ready to be applied to mps
        arrUmin, err, gradmin, numfg, normgradhistory = optimize(fg, arrU0, algorithm; retract = retract, transport! = transport!, isometrictransport =true , inner = inner, finalize! = finalize!);
        
        # reduced_mps = [it.prime(tensor, tau) for tensor in reduced_mps]
        # mt.contract!(reduced_mps, lightcone)
        # for l in i:last_site
        #     mps[l] = it.noprime(reduced_mps[l-i+1])
        # end

        push!(lc_list, lightcone)
        push!(V_list, arrUmin)
        push!(err_list, err)
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

    # project to 0 the sites which are supposed to be already 0
    mps_trunc = deepcopy(mps)
    
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

    results_second_part = []
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

    return V_list, err_list, mps, lc_list, mps_trunc, results_second_part
end


N = 31
#siteinds = it.siteinds("Qubit", N)
#mps = mt.randMPS(siteinds, 2)
mps = mt.initialize_fdqc(N, 2)
siteinds = it.siteinds(mps)

results = invertMPSLiu(mps, 2, 8, 4)
mpsfinal, lclist, mps_trunc, second_part = results[3:6]
entropies = [mt.entropy(mpsfinal, i) for i in 1:N-1]
norms = [norm(mpsfinal - mt.project_tozero(mpsfinal, [i])) for i in 1:N]




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
# 
# # generate random point on M
# arrU0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
# arrU0dag = [U' for U in arrU0]
# # 
# # # generate random tangent Vector
# arrV = [mt.random_unitary(4) for _ in 1:n_unitaries]
# arrV = skew.(arrV)
# arrV = arrU0 .* arrV
# arrV /= sqrt(inner(arrV, arrV))
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
# E = t -> abs(fg(retract(arrU0, arrV, t)[1])[1] - func - t*prod)
# 
# tvals = exp10.(-8:0.1:0)
# 
# Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
# Plots.plot!(tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
# Plots.plot!(tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")