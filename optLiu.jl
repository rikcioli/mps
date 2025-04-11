include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots

using OptimKit, LaTeXStrings, LinearAlgebra, Statistics, JET, Profile, DataFrames, CSV


it.set_warn_order(28)


function _fgEntanglement(U_array::Vector{<:Matrix}, lightcone, reduced_mps, counter)
    mt.updateLightcone!(lightcone, U_array)
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

    riem_grad = mt.project(U_array, grad)
    counter[] = counter[]+1
    return cost, riem_grad

end


function reduceEntanglement!(mps::itmps.MPS; maxtau = 10, maxiter = 10000, gradtol = 1e-8)
    cost_i = []
    bond_dim_i = []
    lcs::Vector{mt.Lightcone} = []
    support = (1, length(mps))    # region of the mps we want to act on with the brickwork (in this case set to all)

    for i in 1:maxtau
        it.truncate!(mps, cutoff = 1e-16)
        it.orthogonalize!(mps, support[end]-1)
        @show it.linkinds(mps)[div(support[end],2)]
        push!(bond_dim_i, it.linkinds(mps)[div(support[end],2)].space)
        reduced_mps = itmps.MPS(mps[support[1]:support[end]])

        N = length(reduced_mps)
        reduced_mps.llim = N-2
        reduced_mps.rlim = N
        red_siteinds = it.siteinds(reduced_mps)

        lc_support = (isodd(i) ? 1 : 2, iseven(i+N) ? N-1 : N)     # region the lightcone acts on - will be equal to support only if support is even and i is even, else it will be less

        lc_sites = red_siteinds[lc_support[1]:lc_support[end]]
        lightcone = mt.newLightcone(lc_sites, 1; lightbounds = (false, false))
        
        for j in 1:N
            it.replaceind!(reduced_mps[j], red_siteinds[j], red_siteinds[j]')
        end

        # setup optimization stuff
        arrU0 = Array(lightcone)
        counter = Ref(1)
        fg = arrU -> _fgEntanglement(arrU, lightcone, reduced_mps, counter)
        m = 5
        algorithm = LBFGS(m;maxiter = maxiter, gradtol = gradtol, verbosity = 1)
        arrUmin, cost, gradmin, numfg, normgradhistory = mt.optimize(fg, arrU0, algorithm; retract = mt.retract, transport! = mt.transport!, isometrictransport = true , inner = mt.inner);
        mt.updateLightcone!(lightcone, arrUmin)

        mt.apply!(mps, lightcone)
        push!(cost_i, abs(cost))
        push!(lcs, lightcone)

        if -1e-15 < cost < 1e-15
            break
        end
    end

    return Dict([("lightcones", lcs), ("costs", cost_i), ("bond_dim", bond_dim_i)])

end

function compare_disentangler(Nlist = [10,20,30], D=2, maxtau=20)

    cost_list = []
    bond_list = []
    for N in Nlist
        mps = it.random_mps(it.siteinds(2, N), linkdims = D)
        #mps = mt.initialize_ghz(N)

        results = reduceEntanglement!(mps, maxtau = maxtau)
        costs = results["costs"]
        bond_dim = results["bond_dim"]
        push!(cost_list, costs)
        push!(bond_list, bond_dim)
    end

    cost_list = cost_list ./ Nlist

    data = DataFrame(depths = 1:maxtau, purities30 = cost_list[1], chi30 = bond_list[1])
    CSV.write("D:\\Julia\\MyProject\\Data\\disentanglerD2_last.csv", data)

    #Plots.plot(title = L"D = 2", ylabel = L"-\log(\mathcal{P})", xlabel = L"\tau")
    #Plots.plot!(1:length(cost_list), costs, lc=:red, primary=false, legend=false)
    #Plots.plot!(1:length(costs), costs, seriestype=:scatter, mc=:red)
    #Plots.plot!(yscale=:log)
    #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\disentanglerGHZ.pdf")
#
    #Plots.plot(title = L"N=11, \ GHZ", ylabel = L"\chi", xlabel = L"\tau")
    #Plots.plot!(1:length(costs), bond_dim, lc=:red, primary=false, legend=false)
    #Plots.plot!(1:length(costs), bond_dim, seriestype=:scatter, mc=:red)
    ##Plots.plot!(yscale=:log)
    #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\disentanglerGHZbond.pdf")

end


#####data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\disentanglerD2_first2.csv"))
#####Plots.plot(title = L"D = 2", ylabel = L"-\log(\mathcal{P})/N", xlabel = L"\tau")
#####Plots.plot!(1:20, data.purities10, lc=:red, primary=false, legend=true)
#####Plots.plot!(1:20, data.purities10, seriestype=:scatter, mc=:red, label=L"N=10")
#####Plots.plot!(1:20, data.purities20, lc=:green, primary=false)
#####Plots.plot!(1:20, data.purities20, seriestype=:scatter, mc=:green, label=L"N=20")
######Plots.plot!(1:10, data.purities30, lc=:blue, primary=false)
######Plots.plot!(1:10, data.purities30, seriestype=:scatter, mc=:blue, label=L"N=30")
#####Plots.plot!(yscale=:log, xscale=:log)
#####Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\disentanglerD2_first2.pdf")
#####
#####Plots.plot(title = L"D = 2", ylabel = L"\chi", xlabel = L"\tau")
#####Plots.plot!(1:20, data.chi10, lc=:red, primary=false, legend=true)
#####Plots.plot!(1:20, data.chi10, seriestype=:scatter, mc=:red, label=L"N=10")
#####Plots.plot!(1:20, data.chi20, lc=:green, primary=false)
#####Plots.plot!(1:20, data.chi20, seriestype=:scatter, mc=:green, label=L"N=20")
######Plots.plot!(1:10, data.chi30, lc=:blue, primary=false)
######Plots.plot!(1:10, data.chi30, seriestype=:scatter, mc=:blue, label=L"N=30")
######Plots.plot!(yscale=:log)
#####Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\disentanglerD2chi_first2.pdf")

function execute()
    #energy, psi = mt.initialize_ising(50, 1000)
    Nqubit = 50
    psi = it.random_mps(it.siteinds("Qubit", Nqubit), linkdims = 2)
    psi_copy = deepcopy(psi)
    for b in 1:Nqubit-1
        mt.cut!(psi_copy, b)
    end
    @show abs(dot(psi, psi_copy))
    #results = mt.invertMPSLiu(psi, mt.invertGlobalSweep, eps = 1e-3)
    #results = mt.invert(psi, mt.invertGlobalSweep, eps = 1e-3)
    return psi, results
end

psi, results = execute();

mps = it.random_mps(it.siteinds("Qubit", 20); linkdims = 2)
@time results = mt.invert(mps, mt.invertGlobalSweep; eps = 1e-2, reuse_previous = false, start_tau = 5, nruns = 1)
@profview results = mt.invert(mps, mt.invertGlobalSweep; eps = 1e-2, reuse_previous = false, start_tau = 5, nruns = 1)
#results = invertMPSLiu(mps, mt.invertGlobalSweep)
#mpsfinal, lclist, mps_trunc, second_part = results[3:6]
#mps2 = mps[1:end]
#@report_opt mt._fgGlobalSweep(Uarr, lc, mps2)
#results2 = mt.invertMPSMalz(mps, mt.invertGlobalSweep; q=4, kargsV = (nruns = 10, ))









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