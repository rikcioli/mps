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


N = 30
psi0 = random_mps(ComplexF64, siteinds("Qubit", N); linkdims = 4)

time1_global, time1_sweep, time2_global, time2_sweep = let N=N, psi0 = psi0
    eps = 0.2
    #pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\mpstest\\"
    pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/test/"
    
    time1_global = @elapsed mt.invertMPS1(psi0, pathname; invertMethod = mt.invertGlobalSweep, maxiter = 10000)
    time1_sweep = @elapsed mt.invertMPS1(psi0, pathname; invertMethod = mt.invertSweepLC, maxiter = 10000)
    time2_global = @elapsed mt.invertMPS2(pathname, N, eps; invertMethod = mt.invertGlobalSweep, maxiter = 10000)
    time2_sweep = @elapsed mt.invertMPS2(pathname, N, eps; invertMethod = mt.invertSweepLC, maxiter = 10000)


    #pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/xxz/Jz2.5/"
    #Nrange = [60:40:300]

    # for N in Nrange
    #     sites = siteinds("S=1/2", N)
    #     Hamiltonian = mt.H_heisenberg(sites, 1., 1., 2.5, 0., 0.)
    #     energy, psi = mt.initialize_gs(Hamiltonian, sites; nsweeps = 20, maxdim = [10,50,200])
    #     f = h5open(pathname*"$(N)_mps.h5","w")
    #     write(f,"psi",psi)
    #     close(f)
    # end

    # for N in Nrange
    #     f = h5open(pathname*"$(N)_mps.h5","r")
    #     psi = read(f,"psi",MPS)
    #     close(f)
    #     mt.invertMPSLiu(psi, 6; folder=pathname)
    # end
    time1_global, time1_sweep, time2_global, time2_sweep
end

@show time1_global, time1_sweep, time2_global, time2_sweep


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