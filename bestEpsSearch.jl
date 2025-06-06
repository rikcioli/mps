include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV, HDF5, JLD2
using JET
#using DataFrames, StatsPlots

#Strided.disable_threads()
#@show ITensors.Strided.get_num_threads()
#BLAS.set_num_threads(56)
#@show ITensors.blas_get_num_threads()

#pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\bestEps\\"
#results = [load_object(pathname*"$(100)Q_$(ansatz_eps).jld2") for ansatz_eps in [1., 0.5]]
#taus = [res["tau"] for res in results]

#pathname = "D:\\Julia\\MyProject\\Data\\ising\\"
#params = load_object(pathname*"1000_0.001_params.jld2")


let
    pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\bestEps\\"
    N = 50
    psi = random_mps(siteinds("Qubit", N), linkdims = 2)
    #f = h5open(pathname*"$(N)_mps.h5","r")
    #psi = read(f,"psi",MPS)
    #close(f)
    eps = 1e-2
    ansatz_eps_list = [0.5]

    for ansatz_eps in ansatz_eps_list
        mt.invertMPS1(psi, mt.invertGlobalSweep; eps = eps, pathname = pathname, ansatz_eps = ansatz_eps)
        results = mt.invertMPS2(pathname, N, eps, mt.invertGlobalSweep)
        jldsave(pathname*"$(N)Q_$(ansatz_eps).jld2"; results)
    end
    
end

