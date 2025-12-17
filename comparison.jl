include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
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
let
    folder = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/randMPS/mpstest/"
    N = 200
    D = 16
    psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = D)
    f = h5open(folder*"sweep/$(N)_mps.h5","w")
    write(f,"psi",psi0)
    close(f)

    f = h5open(folder*"global/$(N)_mps.h5","w")
    write(f,"psi",psi0)
    close(f)

    # f = h5open(folder*"sweep/$(N)_mps.h5","r")
    # psi0 = read(f,"psi",MPS)
    # close(f)

    mt.invertMPSLiu(psi0, 6; invertMethod = mt.invertSweepLC, folder = folder*"sweep/")
    mt.invertMPS2(folder*"sweep/", [N], [0,2,3,4,5,6], 0.01; invertMethod = mt.invertSweepLC)

    mt.invertMPSLiu(psi0, 6; invertMethod = mt.invertGlobalSweep, folder = folder*"global/")
    mt.invertMPS2(folder*"global/", [N], [0,2,3,4,5,6], 0.01; invertMethod = mt.invertGlobalSweep)
end




# check at what N the invertTrunc breaks
## N = 30
## D = 2
## #psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = D)
## ## 
## folder = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
## f = h5open(folder*"$(N)_mps.h5","r")
## psi0 = read(f,"psi",MPS)
## close(f)
## ## 
## ## # results = let N = N, psi0 = psi0
## ## #     res = mt.invertMPSTrunc(psi0, 0.5; start_tau = 1, invertMethod = mt.invertSweepLC, folder = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\")
## ## #     res
## ## # end
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