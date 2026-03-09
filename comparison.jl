include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
#using DataFrames, StatsPlots
using JLD2, HDF5

#Strided.disable_threads()
#@show ITensors.Strided.get_num_threads()
#BLAS.set_num_threads(56)
#@show ITensors.blas_get_num_threads()



function testRandom(Nrange, eps_array, pathname)
    for N in Nrange
        psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 2)
        mt.invertMPS1(psi0, pathname)
    end
    mt.invertMPS2(pathname, Nrange, eps_array)
end

function testXY(Nrange, eps_array, pathname)
    for N in Nrange
        sites = siteinds("S=1/2", N)
        Hamiltonian = mt.H_XY(sites, 0.7, 0.1)
        energy, psi0 = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,80,60,40,30,30,20])
        mt.invertMPS1(psi0, pathname)
    end
    mt.invertMPS2(pathname, Nrange, eps_array)
end

function testIsing(Nrange, eps_array, pathname)
    for N in Nrange
        sites = siteinds("S=1/2", N)
        Hamiltonian = mt.H_ising(sites, -1., 0.5, 0.05)
        energy, psi0 = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,80,60,40,30,30,20])
        mt.invertMPS1(psi0, pathname)
    end
    mt.invertMPS2(pathname, Nrange, eps_array)
end

function testXXZ(Nrange, eps_array, pathname)
    for N in Nrange
        sites = siteinds("S=1/2", N)
        Hamiltonian = mt.H_heisenberg(sites, -1., -1., -0.5, -0.1, -0.1)
        energy, psi0 = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,80,60,40,30,30,20])
        mt.invertMPS1(psi0, pathname)
    end
    mt.invertMPS2(pathname, Nrange, eps_array)
end


function invertExisting(Nrange, eps_array, pathname)
    for N in Nrange
        f = h5open(pathname*"$(N)_mps.h5","r")
        psi0 = read(f,"psi",MPS)
        close(f)

        dt1 = @elapsed mt.invertMPS1(psi0, pathname; invertMethod = mt.invertSweepLC)
        jldsave(pathname*"time_invert1_$(N)_0.5.jld2"; dt1)
    end
    mt.invertMPS2(pathname, Nrange, eps_array; invertMethod = mt.invertSweepLC)
end


# RUN ON CLUSTER
# let
#     folder = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/randMPS/mpstest/"
#     N = 200
#     D = 16
#     psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = D)
#     f = h5open(folder*"sweep/$(N)_mps.h5","w")
#     write(f,"psi",psi0)
#     close(f)
# 
#     f = h5open(folder*"global/$(N)_mps.h5","w")
#     write(f,"psi",psi0)
#     close(f)
# 
#     # f = h5open(folder*"sweep/$(N)_mps.h5","r")
#     # psi0 = read(f,"psi",MPS)
#     # close(f)
# 
#     mt.invertMPSLiu(psi0, 6; invertMethod = mt.invertSweepLC, folder = folder*"sweep/")
#     mt.invertMPS2(folder*"sweep/", [N], [0,2,3,4,5,6], 0.01; invertMethod = mt.invertSweepLC)
# 
#     mt.invertMPSLiu(psi0, 6; invertMethod = mt.invertGlobalSweep, folder = folder*"global/")
#     mt.invertMPS2(folder*"global/", [N], [0,2,3,4,5,6], 0.01; invertMethod = mt.invertGlobalSweep)
# end






#folder = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"

function test_contraction(N::Integer, tau::Integer; U_array = nothing)

    sites = siteinds("Qubit", N)
    lc = mt.newLightcone(sites, tau, lightbounds=(false,false), U_array = U_array)

    right_inds = reverse([prime(lc.siteinds[end-2], i) for i in 0:tau])

    right_vec = [ITensor([1; 0], right_inds[1])]
    for i in 2:2:tau
        U, S, Vd = svd(delta(right_inds[i], right_inds[i+1]), right_inds[i])
        push!(right_vec, U)
        push!(right_vec, S*Vd)
    end
    right_mps = MPS(right_vec)

    ent = Float64[]
    max_chi = Int64[]
    norms = Float64[]
    for j in N-3:-1:4
        @show j
        if iseven(j)
            orthogonalize!(right_mps, 1)
            right_mps[1] = right_mps[1]*ITensor([1 0; 0 0], prime(sites[j:j+1], tau))
        end

        gates = lc.gates_by_site[j]
        for gate in gates
            if gate[:orientation] == "R"
                mt.contract!(right_mps, lc.circuit[gate[:pos]], gate[:depth]; )
            end
        end

        if isodd(j)
            orthogonalize!(right_mps, tau+1)
            right_mps[end] = right_mps[end]*ITensor([1 0; 0 0], sites[j:j+1])
        end

        orthogonalize!(right_mps, div(tau,2))
        norm_j = norm(right_mps)
        push!(norms, norm_j)
        normalize!(right_mps)

        if iseven(j)
            push!(ent, mt.entropy(right_mps, div(tau,2))/log(2))
            push!(max_chi, maximum(linkdims(right_mps)))
        end
    end

    return ent, max_chi, norms
end


ansatz = load_object("D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mps1\\100_ansatz.jld2")

results = [test_contraction(100, tau; U_array = [mt.SWAP for _ in 1:(49*tau)]) for tau in [8, 12, 16, 20]]
#results = [test_contraction(100, tau; U_array = [ansatz; ansatz; ansatz; ansatz][1:49*tau]) for tau in [8, 12, 16, 20]]

plot = Plots.plot(title=L"N = 100", ylabel=L"S_{max}/log(2)", xlabel="l/2")
Plots.plot!(plot, 1:length(results[1][1]), results[1][1], label=L"\tau=8")
Plots.plot!(plot, 1:length(results[2][1]), results[2][1], label=L"\tau=12")
Plots.plot!(plot, 1:length(results[3][1]), results[3][1], label=L"\tau=16")
Plots.plot!(plot, 1:length(results[4][1]), results[4][1], label=L"\tau=20")
Plots.savefig(plot, "swap.png")

result = mt.invert(psi0, mt.invertSweepLC; tau=4, maxiter = 5000)
lc = result["lightcone"]

tentropies = result["tentropy"]
tentropies = [tentropies[i][1:end] for i in eachindex(tentropies)]
avg_tents = [mean(tentropies[i])./log(2) for i in eachindex(tentropies)]

max_ent = [maximum(maximum(tentropies[i]))/log(2) for i in eachindex(tentropies)]

Plots.plot(1:length(max_ent), max_ent, title=L"\tau = 10", ylabel=L"S_{max}/log(2)", xlabel="sweep")
Plots.savefig("tau10")






# f = h5open(folder*"$(N)_mps.h5","w")
# write(f,"psi",psi0)
# close(f)
# eps_list = [0.999, 0.99, 0.9, 0.09]



# check at what N the invertTrunc breaks
N = 1000
D = 2
#psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = D)
## 
folder = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
f = h5open(folder*"$(N)_mps.h5","r")
psi0 = read(f,"psi",MPS)
close(f)

psi0 = let N = N
    sites = siteinds("S=1/2", N)
    Hamiltonian = mt.H_heisenberg(sites, -1., -1., -0.5, -0.1, -0.1)
    energy, psi0 = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff=1e-15)
    psi0
end

results = let N = N, psi0 = psi0
    res = mt.invertMPSTrunc(psi0, 0.1; start_tau = 1, invertMethod = mt.invertSweepLC)
    res
end


results
## ## 
## resultsLiu = let N = N, psi0 = psi0, folder = folder
##     mt.invertMPSLiu(psi0, 3; invertMethod = mt.invertSweepLC, folder = folder)
## end
## ## 
## 
## let
##     mt.invertMPS2(folder, [N], [0,2,3], 0.1; invertMethod = mt.invertSweepLC)
## end
## 
## res = load_object(folder*"test\\30_3_params.jld2")
## res


# check role of start tau
## N = 60
## pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
## 
## let N = N, psi0 = psi0
##     mt.invertMPSTrunc(pathname, [N], [0.1]; start_tau = 4)
## end
## 
## res = load_object(pathname*"trunc_60_0.1_result.jld2")
## 
## #res1 = load_object(pathname*"trunc_20_0.03_result.jld2")
## #res2 = load_object(pathname*"trunc_20_0.03_result.jld2")
## res3 = #load_object(pathname*"trunc_20_0.03_result.jld2")
## res4 = load_object(pathname*"trunc_20_0.03_result.jld2")
## 
## plt = Plots.plot(1:20001, 1 .+ res1["history"][:,1], yscale=:log10, xscale=:log10)
## pltgrad = Plots.plot(1:20001, res1["history"][:,2], yscale=:log10, scale=:log10)



# check role of region factor
# N = 60
# psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 2)
# 
# reslist = let N = N, psi0 = psi0
#     reslist = []
#     for i in 0:3
#         results = mt.invertMPSLiu(psi0, 0.3; region_factor=i)
#         #fidlast = results[results.depth .== 3, :fid]
#         push!(reslist, results)
#     end
#     reslist
# end


# compare times of invertSweepLC and invertGlobalSweep
## N = 30
## psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 4)
## 
## time1_global, time1_sweep, time2_global, time2_sweep = let N=N, psi0 = psi0
##     eps = 0.2
##     #pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
##     pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/test/"
##     
##     time1_global = @elapsed mt.invertMPS1(psi0, pathname; invertMethod = mt.invertGlobalSweep, maxiter = 10000)
##     time1_sweep = @elapsed mt.invertMPS1(psi0, pathname; invertMethod = mt.invertSweepLC, maxiter = 10000)
##     time2_global = @elapsed mt.invertMPS2(pathname, N, eps; invertMethod = mt.invertGlobalSweep, maxiter = 10000)
##     time2_sweep = @elapsed mt.invertMPS2(pathname, N, eps; invertMethod = mt.invertSweepLC, maxiter = 10000)
## 
## 
##     #pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/xxz/Jz2.5/"
##     #Nrange = [60:40:300]
## 
##     # for N in Nrange
##     #     sites = siteinds("S=1/2", N)
##     #     Hamiltonian = mt.H_heisenberg(sites, 1., 1., 2.5, 0., 0.)
##     #     energy, psi = mt.initialize_gs(Hamiltonian, sites; nsweeps = 20, maxdim = [10,50,200])
##     #     f = h5open(pathname*"$(N)_mps.h5","w")
##     #     write(f,"psi",psi)
##     #     close(f)
##     # end
## 
##     # for N in Nrange
##     #     f = h5open(pathname*"$(N)_mps.h5","r")
##     #     psi = read(f,"psi",MPS)
##     #     close(f)
##     #     mt.invertMPSLiu(psi, 6; folder=pathname)
##     # end
##     time1_global, time1_sweep, time2_global, time2_sweep
## end
## 
## @show time1_global, time1_sweep, time2_global, time2_sweep



## N = 20
## folder = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
## 
## psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 2)
## f = h5open(folder*"$(N)_mps.h5","w")
## write(f,"psi",psi0)
## close(f)
## eps_list = [0.999, 0.99, 0.9, 0.09]
## 
## let N = N, psi0 = psi0, eps_list = eps_list, folder = folder
##     eps_final = 0.009
##     for eps in eps_list
##         dt = @elapsed mt.invertMPSLiu(psi0, eps; folder = folder)
##         jldsave(folder*"time_invert1_$(N)_$(eps).jld2"; dt)
##     end
## 
##     Threads.@threads for eps in eps_list
##         _params = load_object(folder*"$(N)_$(eps)_params.jld2")
##         _best_guess = load_object(folder*"$(N)_$(eps)_ansatz.jld2")
##         _best_guess = [Matrix{ComplexF64}(U) for U in _best_guess]
## 
##          _dt2 = @elapsed _results = mt.invert(psi0, mt.invertGlobalSweep; 
##                                 nruns = 1, 
##                                 reuse_previous = true,
##                                 site1_empty = _params["site1_empty"], 
##                                 eps = eps_final, 
##                                 start_tau = _params["start_tau"], 
##                                 init_array = _best_guess)
##         _taufinal = _results["tau"]
##         jldsave(folder*"invert_$(N)_$(eps).jld2"; _taufinal)
##         jldsave(folder*"time_invert2_$(N)_$(eps).jld2"; _dt2)
##     end
## 
##     mt.invertMPSTrunc(folder, [N], [eps_final])
## end
## 
## dt1s = []
## dt2s = []
## for eps in eps_list
##     dt1 = load_object(folder*"time_invert1_$(N)_$(eps).jld2")
##     push!(dt1s, dt1)
##     dt2 = load_object(folder*"time_invert2_$(N)_$(eps).jld2")
##     push!(dt2s, dt2)
## end


#N = 20
#psi1 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 2)
#
#psi_sweep, psi_global = let N=N, psi0=psi0
#
#    sites = siteinds(psi0)
#    res1 = mt.invert(psi0, mt.invertSweepLC; eps=0.01)
#    res2 = mt.invert(psi0, mt.invertGlobalSweep; eps=0.01)
#
#    lc1 = res1["lightcone"]
#    ovlp1 = res1["overlap"]
#
#    lc2 = res2["lightcone"]
#    ovlp2 = res2["overlap"]
#
#    psirec1 = mt.initialize_vac(N, sites)
#    psirec2 = mt.initialize_vac(N, sites)
#
#    mt.apply!(psirec1, lc1)
#    mt.apply!(psirec2, lc2)
#
#    @show ovlp1
#    @show dot(psirec1, psi0)
#    @show ovlp2
#    @show dot(psirec2, psi0)
#
#    psirec1, psirec2
#end