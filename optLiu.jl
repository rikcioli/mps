include("mpsMethods.jl")
import .MPSMethods as mt
import Plots

using ITensors, ITensorMPS
using OptimKit, LaTeXStrings, LinearAlgebra, Statistics, JET, Profile, DataFrames, CSV


ITensors.set_warn_order(28)



function execute()
    #energy, psi = mt.initialize_ising(50, 1000)
    Nqubit = 50
    psi = random_mps(siteinds("Qubit", Nqubit), linkdims = 2)
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

mps = random_mps(siteinds("Qubit", 10); linkdims = 2)
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