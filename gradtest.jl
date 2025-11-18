include("mpsMethods.jl")
import .MPSMethods as mt
import Plots
using ITensors, ITensorMPS
using LaTeXStrings, LinearAlgebra
using Zygote
using HDF5


#t1 = random_mps(siteinds("Qubit", N), linkdims = 2)
#t2 = random_mps(ComplexF64, siteinds("Qubit", N), linkdims = 2)
#
#pathname="D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mps1\\"
#f = h5open(pathname*"$(N)_mps.h5","r")
#psi0 = read(f,"psi",MPS)
#close(f)



N = 4
sites = siteinds("Qubit",N)
#mps = mt.random_mps(sites, linkdims=2)
#lc = mt.newLightcone(sites, 4)

test = [mt.random_unitary(2) for _ in 1:N]
mpo = MPO([ITensor(test[i], sites[i]', sites[i]) for i in 1:N])
mt.invertMPSLiu(mpo, mt.invertGlobalSweep)



### # testing riemannian gradient
### n_unitaries = lc.n_unitaries
### # generate random point on M
### arrU0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
### arrU0dag = [U' for U in arrU0]
### # 
### # # generate random tangent Vector
### arrV = [mt.random_unitary(4) for _ in 1:n_unitaries]
### arrV = mt.skew.(arrV)
### arrV = arrU0 .* arrV
### arrV /= sqrt(mt.inner(arrV, arrV))
### 
### # compute f and gradf, check that gradf is in TxM and compute inner prod in x
### fg = arrU -> mt._fgLiu(arrU, lc, mps[1:N])
### #fg = arrU -> (real(tr(arrU'arrU)), project(arrU, 2*arrU))
### func, grad = fg(arrU0)
### # bring grad back to the tangent space to the identity and check it's skew hermitian
### arrX = arrU0dag .* grad     
### norm(arrX - mt.skew.(arrX))
### prod = mt.inner(grad, arrV)
### 
### # test retraction and geodesic distance
### t = 0.001
### mt.dist_un(mt.retract(arrU0, arrV, t)[1], arrU0)
### sqrt(mt.inner(t*arrV, t*arrV))
### 
### # test derivative
### norm((mt.retract(arrU0, arrV, t)[1] .- arrU0)/t - arrV)
### # it works, so the problem is in the grad i think
### DF = (fg(mt.retract(arrU0, arrV, t)[1])[1] - func)/t
### prod
### norm(DF - prod)
### 
### 
### # compute E(t) for several values of t
### E = t -> abs(fg(mt.retract(arrU0, arrV, t)[1])[1] - func - t*prod)
### 
### tvals = exp10.(-8:0.1:0)
### 
### Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
### Plots.plot!(tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
### Plots.plot!(tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")

function genPoint()
    n_unitaries = lc.n_unitaries
    # generate random point on M
    U0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
    U0dag = [U' for U in U0]
    return U0, U0dag
end

function genTanVec(U)
    V = [randn(ComplexF64, 4, 4) for _ in 1:n_unitaries]
    V = mt.skew.(V)
    V = U .* V
    V /= sqrt(mt.inner(V, V))
end

function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
    U0, U0dag = genPoint()
    V = genTanVec(U0)
    func, grad = computeCostGrad(U0)
    gradV = inner(grad, V) 
    E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end

plot = testGrad(genPoint, genTanVec, arrU -> mt._fgLiu(arrU, lc, mps[1:N]), mt.inner, mt.retract)



# testing riemannian gradient
n_unitaries = lc.n_unitaries
# generate random point on M
arrU0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
arrU0dag = [U' for U in arrU0]
# 
# # generate random tangent Vector
arrV = [mt.random_unitary(4) for _ in 1:n_unitaries]
arrV = mt.skew.(arrV)
arrV = arrU0 .* arrV
arrV /= sqrt(mt.inner(arrV, arrV))

# compute f and gradf, check that gradf is in TxM and compute inner prod in x
f = arrU -> mt._fLiu(arrU, lc, mps[1:N])
f(arrU0)
g = arrU -> Zygote.gradient(f, arrU)[1]

fg_AD = arrU -> (f(arrU), mt.project(arrU, g(arrU)))
fg = arrU -> mt._fgLiu(arrU, lc, mps[1:N])

func, grad = fg(arrU0)

# bring grad back to the tangent space to the identity and check it's skew hermitian
arrX = arrU0dag .* grad     
norm(arrX - mt.skew.(arrX))
prod = mt.inner(grad, arrV)

# test retraction and geodesic distance
t = 0.001
mt.dist_un(mt.retract(arrU0, arrV, t)[1], arrU0)
sqrt(mt.inner(t*arrV, t*arrV))

# test derivative
DF = (fg(mt.retract(arrU0, arrV, t)[1])[1] - func)/t
prod
norm(DF - prod)


# compute E(t) for several values of t
E = t -> abs(fg(mt.retract(arrU0, arrV, t)[1])[1] - func - t*prod)

tvals = exp10.(-8:0.1:0)

Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
Plots.plot!(tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
Plots.plot!(tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")

