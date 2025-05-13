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
#results = [load_object(pathname*"$(ansatz_eps).jld2") for ansatz_eps in [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]]
#taus = [res["tau"] for res in results]


let
    pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\bestEps\\"
    N = 100
    psi = random_mps(siteinds("Qubit", N), linkdims = 2)
    eps = 1e-2
    ansatz_eps_list = [4, 2, 1, 0.5, 0.1]

    for ansatz_eps in ansatz_eps_list
        mt.invertMPS1(psi, mt.invertGlobalSweep; eps = eps, pathname = pathname, ansatz_eps = ansatz_eps)
        results = mt.invertMPS2(pathname, N, eps, mt.invertGlobalSweep)
        jldsave(pathname*"$(ansatz_eps).jld2"; results)
    end
    
end

