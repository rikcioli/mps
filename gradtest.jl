include("rrules.jl")
include("optFunctions.jl")
include("mpsMethods.jl")
import .MPSMethods as mt
import Plots
using ITensors, ITensorMPS
using LaTeXStrings, LinearAlgebra
using Zygote
using MatrixAlgebraKit



function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end 

function genPoint(n_unitaries)
    # generate random point on M
    U0 = [random_unitary(4) for _ in 1:n_unitaries]
    U0dag = [U' for U in U0]
    return U0, U0dag
end

function genTanVec(U, n_unitaries)
    V = [randn(ComplexF64, 4, 4) for _ in 1:n_unitaries]
    V = skew.(V)
    V = U .* V
    V /= sqrt(inner(V, V))
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



function prepareCostGradGlobalSweep(U_array, mps)
    N = length(mps)
    sites = siteinds(mps)
    lc = mt.newLightcone(sites, 2; U_array=U_array, lightbounds=(false,false));
    n = lc.n_unitaries
    
    mps = conj(mps)
    outinds = sites'
    
    for i in 1:N
        ind = sites[i]
        vec = [1; [0 for _ in 1:ind.space-1]]
        mps[i] *= ITensor(vec, ind')
    end
    
    mpo = deepcopy(mps[1:end])
    input_inds = sites
    output_inds = sites'
    tau = 2
    
    # noprime the input inds
    # change the output inds to a prime of the input inds to match the inds of the first layer of gates
    sites = noprime(input_inds)
    d = sites[1].space
    tempinds = siteinds(d, N)
    for i in 1:N
        replaceind!(mpo[i], input_inds[i], tempinds[i])
        replaceind!(mpo[i], output_inds[i], prime(sites[i], tau))
        replaceind!(mpo[i], tempinds[i], sites[i])
    end

    return (lc, mpo[:])
end

lc, mpo = prepareCostGradGlobalSweep()
plot = testGrad(() -> genPoint(n), U -> genTanVec(U, n), arrU -> mt._fgGlobalSweep(arrU, lc, mpo[1:N]), mt.inner, mt.retract)




# TEST OVERLAP
psi = randomMPS(ComplexF64, siteinds("Qubit", 30), 8)
U_array = [kron(random_unitary(2), random_unitary(2)) for _ in 1:29];
n = length(U_array)

f = arrU -> overlap(arrU, psi; trunc=(maxrank=2,))
@profview overlap(U_array, psi; trunc=(maxrank=2,))
gradient(f, U_array)

g = arrU -> project(arrU, gradient(f, arrU)[1])
plot = testGrad(() -> genPoint(n), U -> genTanVec(U, n), arrU -> (f(arrU), g(arrU)), inner, retract)

using Profile
let psi=psi
    func = psi -> real(ITensorMPS.inner(truncDM(psi, trunc=(maxrank=2,)), psi))
    @profview gradient(func, psi)
end


result_time = begin
    times1 = Float64[]
    times2 = Float64[]
    for N in 2:2:100
        tvals1 = Float64[]
        tvals2 = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 2)
            t1 = @elapsed truncDM(psi, trunc=(maxrank=1,))
            t2 = @elapsed truncate!(psi, maxdim=1)
            push!(tvals1, t1)
            push!(tvals2, t2)
        end
        push!(times1, sum(tvals1)/100)
        push!(times2, sum(tvals2)/100)
    end
    times1, times2
end
plot = Plots.plot(2:2:100, result_time[1], label="truncDM")
Plots.plot!(plot, 2:2:100, result_time[2], label="truncate")



result_trunc = begin
    overlapDM = Float64[]
    overlapIT = Float64[]
    for N in 2:2:50
        dm = Float64[]
        it = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 32)
            psit1 = truncDM(psi, trunc=(maxrank=2,))
            psit2 = truncate(psi, maxdim=2)
            push!(dm, real(inner(psit1, psi)))
            push!(it, real(inner(psit2, psi)))
        end
        push!(overlapDM, sum(dm)/100)
        push!(overlapIT, sum(it)/100)
    end
    overlapDM, overlapIT
end
plot = Plots.plot(2:2:50, result_trunc[1], label="truncDM", xlabel=L"$N$", ylabel=L"$\langle \psi |\psi_t\rangle$")
Plots.plot!(plot, 2:2:50, result_trunc[2], label="truncate", yscale=:log)


result_time_scaling = begin
    f = psi -> real(inner(truncDM(psi, trunc=(maxrank=2,)), psi))
    trunc_time = Float64[]
    trunc_grad_time = Float64[]
    for N in 2:2:50
        @show N
        times = Float64[]
        grad_times = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 8)
            t1 = @elapsed f(psi)
            t2 = @elapsed gradient(f, psi)
            push!(times, t1)
            push!(grad_times, t2)
        end
        push!(trunc_time, sum(times)/100)
        push!(trunc_grad_time, sum(grad_times)/100)
    end
    trunc_time, trunc_grad_time
end
plot = Plots.plot(2:2:50, result_time_scaling[1], label="truncDM_time", xlabel=L"$N$", ylabel=L"$\langle \psi |\psi_t\rangle$")
Plots.plot!(plot, 2:2:50, result_time_scaling[2], label="grad_truncDM_time")


overlap_time_scaling = begin
    func = (arrU, psi) -> overlap(arrU, psi; trunc=(maxrank=2,))
    fgGlobal = (arrU, psi) -> mt._fgGlobalSweep(arrU, prepareCostGradGlobalSweep(arrU, psi)...)
    times = Float64[]
    grad_times = Float64[]
    timesGlobalSweep = Float64[]
    grad_timesGlobalSweep = Float64[]
    for N in 2:2:50
        @show N
        times_N = Float64[]
        grad_times_N = Float64[]
        times_glob_N = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 8)
            arrU = [random_unitary(4) for _ in 1:(N-1)]
            f = arrU -> func(arrU, psi)
            fg = arrU -> fgGlobal(arrU, psi)
            t1 = @elapsed f(arrU)
            t2 = @elapsed gradient(f, arrU)
            t3 = @elapsed fg(arrU)
            push!(times_N, t1)
            push!(grad_times_N, t2)
            push!(times_glob_N, t3)
        end
        push!(times, sum(times_N)/100)
        push!(grad_times, sum(grad_times_N)/100)
        push!(timesGlobalSweep, sum(times_glob_N)/100)
    end
    times, grad_times, timesGlobalSweep
end
plot = Plots.plot(2:2:50, overlap_time_scaling[1], label="overlap_time", xlabel=L"$N$", ylabel=L"$t \ (s)$")
Plots.plot!(plot, 2:2:50, overlap_time_scaling[2], label="grad_overlap_time")
Plots.plot!(plot, 2:2:50, overlap_time_scaling[3], label="global_sweep_time")

test
### mpo = MPO([ITensor((1.0+0.0im)*mt.Id, sites[i]', sites[i]) for i in 1:N])
### mpo = MPO(lc)
### 
### mpo, outinds, ininds = mt.time_evolution_MPO(N, 0.1, 0.01)
### for i in eachindex(mpo)
###     mpo[i] = replaceind(mpo[i], outinds[i], ininds[i]')
### end
### 
### #mt.invert(mpo, mt.invertGlobalSweep; eps=0.5)
### 
### mt.invertMPSLiu(mpo, 3)
### 
### sites = reduce(vcat, siteinds(mpo; plev=0))
### is_mpo = isa(mpo, MPO)
### if is_mpo
###     outinds = uniqueinds(reduce(vcat, siteinds(mpo)), sites)
###     contrinds = prime(sites, -1)
###     for i in 1:N
###         replaceind!(mpo[i], sites[i], contrinds[i])
###         replaceind!(mpo[i], outinds[i], sites[i])
###     end
### end
### 
### plot = testGrad(() -> genPoint(n), U -> genTanVec(U, n), arrU -> mt._fgLiu(arrU, lc, mpo[1:N]; is_mpo=true), mt.inner, mt.retract)










#t1 = random_mps(siteinds("Qubit", N), linkdims = 2)
#t2 = random_mps(ComplexF64, siteinds("Qubit", N), linkdims = 2)
#
#pathname="D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mps1\\"
#f = h5open(pathname*"$(N)_mps.h5","r")
#psi0 = read(f,"psi",MPS)
#close(f)







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
### f = arrU -> mt._fLiu(arrU, lc, mps[1:N])
### f(arrU0)
### g = arrU -> Zygote.gradient(f, arrU)[1]
### 
### fg_AD = arrU -> (f(arrU), mt.project(arrU, g(arrU)))
### fg = arrU -> mt._fgLiu(arrU, lc, mps[1:N])
### 
### func, grad = fg(arrU0)
### 
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