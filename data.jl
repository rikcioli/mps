include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
using DataFrames, StatsPlots
using JLD2, HDF5

df_list = []
pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\invertFinal\\"

for i in 1:5
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], nmps=Int[])
    folder = pathname*"mps$i\\"
    Nrange = [20, 40, 60, 80, 100]
    eps_array = [0.1, 0.02, 0.004, 0.0008]

    for N in Nrange
        for eps in eps_array
            # For each file, push a new row
            depth = load_object(folder*"invert_$(N)_$(eps).jld2")
            push!(df, (N=N, eps=eps, depth=depth, nmps=i))
        end
    end
    CSV.write(folder*"result.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(pathname*"result.csv", df_final)
