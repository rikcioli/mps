include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles

function execute(Nrange, eps_list, gate)
    tau_depth_eps = []
    
    for N in Nrange
        @show N
        proj = [div(N,2)]
        tau_eps = []
    
        for eps in eps_list
            psi = mt.initialize_fdqc(N, div(N,2), gate, gate)
            mt.project_tozero!(psi, proj)
            tau, lc, err, rest... = mt.invertGlobalSweep(psi, eps = eps, maxiter = 20000, start_tau=1)
            push!(tau_eps, tau)
        end
    
        push!(tau_depth_eps, tau_eps)
    end
    return tau_depth_eps
end

eps_list = [0.1*2.0^(-j) for j in 1:5]
Nrange = 2:2:10
gate = mt.random_unitary(4)
tau_depth_eps = execute(Nrange, eps_list, gate)

depths = div.(Vector(Nrange), 2)
tau_eps_depth = [getindex.(tau_depth_eps,i) for i=1:length(tau_depth_eps[1])]
Plots.plot(ylabel = L"\tau", xlabel = L"\mathrm{true \ depth}")
Plots.plot!(depths, tau_eps_depth[1], lc=:green, primary=false)
Plots.plot!(depths, tau_eps_depth[1], seriestype=:scatter, mc=:green, markersize=:3, primary=true, label=L"\epsilon=0.1")
Plots.plot!(depths, tau_eps_depth[2], lc=:blue, primary=false)
Plots.plot!(depths, tau_eps_depth[2], seriestype=:scatter, mc=:blue, markersize=:3, primary=true, label=L"\epsilon=0.05")
Plots.plot!(depths, tau_eps_depth[3], lc=:red, primary=false)
Plots.plot!(depths, tau_eps_depth[3], seriestype=:scatter, mc=:red, markersize=:3, primary=true, label=L"\epsilon=0.025")
Plots.plot!(depths, tau_eps_depth[4], lc=:yellow, primary=false)
Plots.plot!(depths, tau_eps_depth[4], seriestype=:scatter, mc=:yellow, markersize=:3, primary=true, label=L"\epsilon=0.0125")
#Plots.plot!(1:5, taus_pereps[2], lc=:blue, primary=false)
#Plots.plot!(1:5, taus_pereps[2], seriestype=:scatter, mc=:blue, markersize=:3, primary=true, label=L"\epsilon=0.05")
#Plots.plot!(1:5, taus_pereps[3], lc=:red, primary=false)
#Plots.plot!(1:5, taus_pereps[3], seriestype=:scatter, mc=:red, markersize=:3, primary=true, label=L"\epsilon=0.025")

### # Prepare initial state
### 
### N = 12
### c = div(N,2)
### proj = [c]   # choose which site to measure
### eps_list = [0.1, 0.05, 0.025]
### 
### taus_perdepth_pereps = []
### 
### for depth in 1:5
###     tau_pereps = []
###     # Random FDQC of depth tau
###     mps = mt.initialize_fdqc(N, depth)
###     mps = mt.project_tozero(mps, proj)
###     
###     for eps in eps_list
###         taus = []
###         for run in 1:1000
###             _, _, _, inverted_tau = mt.invertBW(mps; err_to_one = eps, start_tau = depth)
###             push!(taus, inverted_tau)
###         end
###         inverted_tau = median(taus)
###         push!(tau_pereps, inverted_tau)
###     end
###     push!(taus_perdepth_pereps, tau_pereps)
### end
### 
### taus_pereps = [getindex.(taus_perdepth_pereps,i) for i=1:length(taus_perdepth_pereps[1])]
### 
### 
### Plots.plot(title = L"N=12", ylabel = L"\tau", xlabel = L"\mathrm{true \ depth}")
### Plots.plot!(1:5, taus_pereps[1], lc=:green, primary=false)
### Plots.plot!(1:5, taus_pereps[1], seriestype=:scatter, mc=:green, markersize=:3, primary=true, label=L"\epsilon=0.1")
### Plots.plot!(1:5, taus_pereps[2], lc=:blue, primary=false)
### Plots.plot!(1:5, taus_pereps[2], seriestype=:scatter, mc=:blue, markersize=:3, primary=true, label=L"\epsilon=0.05")
### Plots.plot!(1:5, taus_pereps[3], lc=:red, primary=false)
### Plots.plot!(1:5, taus_pereps[3], seriestype=:scatter, mc=:red, markersize=:3, primary=true, label=L"\epsilon=0.025")




#unitaries = [reshape(Array(circ[i,j], it.inds(circ[i,j])), (4, 4)) for i in 1:tau, j in 1:div(N,2)]

## function execute(N::Int64; state_depth = 2, n_sweeps = 100, n_runs = 1000)
## 
##     fid1_depth_run::Vector{Vector{Float64}} = []
##     fid2_depth_run::Vector{Vector{Float64}} = []
##     avg_fid1::Vector{Float64} = []
##     avg_fid2::Vector{Float64} = []
## 
##     for depth in 1:5
##         println("Running depth $depth")
##         fid1_run::Vector{Float64} = []
##         fid2_run::Vector{Float64} = []
## 
##         for run in 1:n_runs
##             if mod(run, 100) == 0
##                 println("Run $run of n_runs")
##             end
##             # create random FDQC of depth 2
##             testMPS = initialize_vac(N)
##             siteinds = it.siteinds(testMPS)
##             testMPS = brickwork(testMPS, state_depth)
## 
##             # try to invert it
##             fid1, sweep1 = invertMPS(testMPS, depth, n_sweeps)
##             err1 = 1-sqrt(fid1)
##             push!(fid1_run, err1)
## 
##             # measure middle qubit
##             pos = div(N,2)
##             zero_projs::Vector{it.ITensor} = []
##             for ind in siteinds
##                 vec = [1; [0 for _ in 1:ind.space-1]]
##                 push!(zero_projs, it.ITensor(kron(vec, vec'), ind, ind'))
##             end
##             proj = zero_projs[pos]
##             testMPS[pos] = it.noprime(testMPS[pos]*proj)
##             it.normalize!(testMPS)
## 
##             # try to invert measured one
##             fid2, sweep2 = invertMPS(testMPS, depth, 100)
##             err2 = 1-sqrt(fid2)
##             push!(fid2_run, err2)
##         end
##         push!(fid1_depth_run, fid1_run)
##         push!(fid2_depth_run, fid2_run)
##         push!(avg_fid1, mean(fid1_run))
##         push!(avg_fid2, mean(fid2_run))
##     end
## 
##     return fid1_depth_run, fid2_depth_run, avg_fid1, avg_fid2
## 
## end
## 
## nruns = 10
## err1_all, err2_all, err1, err2 = execute(21, state_depth = 3, n_runs = nruns)
## 
## swapped_results1 = [getindex.(err1_all,i) for i=1:length(err1_all[1])]
## swapped_results2 = [getindex.(err2_all,i) for i=1:length(err2_all[1])]
## 
## Plots.plot(1:5, swapped_results1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err1, lc=:green, primary=false)
## Plots.plot!(1:5, err1, seriestype=:scatter, mc=:green, legend=true, label="Depth-3")
## Plots.plot!(yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"\epsilon = 1-|\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## 
## Plots.plot!(1:5, swapped_results2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err2, lc=:red, primary=false)
## Plots.plot!(1:5, err2, seriestype=:scatter, mc=:red, label="Depth-3 + measure", legend=:bottomright)
## #Plots.plot!(title = L"N="*string(N), ylabel = L"\epsilon / M", xlabel = L"\tau")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\err_depth3.pdf");
## 
## 
## fid1, fid2 = 1 .- err1, 1 .- err2
## 
## swapped_fid1 = [1 .- swapped_results1[i, 1] for i in 1:nruns]
## swapped_fid2 = [1 .- swapped_results2[i, 1] for i in 1:nruns]
## 
## Plots.plot(1:5, swapped_fid1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, swapped_fid2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, fid1, lc=:green, primary=false)
## Plots.plot!(1:5, fid1, seriestype=:scatter, mc=:green, label="Depth-3", legend=:bottomright)
## Plots.plot!(ylims = (1E-1, 1.2), yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"F = |\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## Plots.plot!(1:5, fid2, lc=:red, primary=false)
## Plots.plot!(1:5, fid2, seriestype=:scatter, mc=:red, label="Depth-3 + measure")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\fid_depth3.pdf");