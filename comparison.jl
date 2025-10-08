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
        psi0 = random_mps(siteinds("Qubit", N), linkdims = 2)
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


let
    pathname = "D:\\Julia\\MyProject\\Data\\xxz\\"
    Nrange = [40]

    for N in Nrange
        sites = siteinds("S=1/2", N)
        Hamiltonian = mt.H_heisenberg(sites, 1., 1., 2.5, 0., 0.)
        energy, psi = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,100,100,100,100,100,100])
        mt.invertMPSLiu(psi, 6; folder=pathname)
    end
    #psi0 = random_mps(siteinds("Qubit", N), linkdims = 4)
    #mt.invertMPS1(psi0, pathname; invertMethod = mt.invertSweepLC)

    #for N in [40]
    #    sites = siteinds("S=1/2", N)
    #    ##Hamiltonian = mt.H_ising(sites, -1., 0.5, 0.05)
    #    Hamiltonian = mt.H_XY(sites, 0, 0)
    #    ##Hamiltonian = mt.H_heisenberg(sites, -1., -0.5, 0.1, 0.1)
    #    energy, psi = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,80,60,40,30,30,20])
    #    energy2, psi2 = mt.initialize_gs(Hamiltonian, sites; nsweeps = 10, maxdim = [10,50,100,100,100,100,100,100,100,100])
    #    @show dot(psi, psi2)
    #end
    #psi = initialize_ghz(N)
    #psi = random_mps(siteinds("Qubit", N), linkdims = 2)
    #mt.invertMPS1(psi, mt.invertGlobalSweep; eps = eps, pathname = pathname, ansatz_eps = 0.5)
    #mt.invertMPS2(pathname, N, eps, mt.invertGlobalSweep; maxiter = 20000)
end

