include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
using DataFrames, StatsPlots
using JLD2, HDF5


# Create an empty DataFrame
df = DataFrame(N=Int[], eps=Float64[], depth=Int[], nmps=Int[])
pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\"
Nrange = [20, 40, 60, 80, 100]
eps_array = [0.1, 0.02, 0.004, 0.0008]

for N in Nrange
    for eps in eps_array
        # For each file, push a new row
        depth = load_object(pathname*"invert_$(N)_$(eps).jld2")
        push!(df, (N=N, eps=eps, depth=depth, nmps=1))
    end
end
CSV.write(pathname*"result1.csv", df)

filter(row -> row.N == 10 && row.eps == 0.1, df)