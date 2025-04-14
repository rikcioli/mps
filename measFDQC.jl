include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
#import Plots
#using LaTeXStrings
using LinearAlgebra, Statistics
using CSV
using DataFrames
#using Colors

function execute(Nrange, eps_list, gate)
    tau_depth_eps = []
    
    for N in Nrange
        @show N
        proj = [div(N,2)]
        tau_eps = []
    
        for eps in eps_list
            psi = mt.initialize_fdqc(N, div(N,2), gate, gate)
            mt.project_tozero!(psi, proj)
            start_tau = 1
            if length(tau_eps) > 0
                start_tau = tau_eps[end]
            end
            results = mt.invert(psi, mt.invertGlobalSweep; eps = eps, maxiter = 10000, nruns=12, start_tau=start_tau)
            tau = results["tau"]
            push!(tau_eps, tau)
        end
    
        push!(tau_depth_eps, tau_eps)
    end
    return tau_depth_eps
end


function meas_invert(Nrange, eps_list)

    for N in Nrange
        @show N
        proj = [div(N,2)]
        eps = eps_list[end]
        err_hist = []
        tau_hist = []
        for t in 6:20
            @show t
            psi = mt.initialize_fdqc(N, t)
            mt.project_tozero!(psi, proj)

            start_tau = t==6 ? 8 : max(1, t-2)
            for i in eachindex(err_hist)
                if err_hist[i] <= eps_list[1]
                    start_tau = max(1, tau_hist[i]-1)
                    break
                end
            end

            results = mt.invert(psi, mt.invertGlobalSweep; eps = eps, maxiter = t*10000, nruns = 12, start_tau = start_tau, reuse_previous = false)
            err_hist = results["err_history"]
            tau_hist = [i for i in start_tau:results["tau"]]

            data = DataFrame(depths = [i+start_tau-1 for i in eachindex(err_hist)], errs = err_hist)
            CSV.write("D:\\Julia\\MyProject\\Data\\measFDQC\\$(N)Q_$(t).csv", data)
        end
    end
end


let
    eps_list = [0.02*2.0^(-j) for j in 1:3]
    Nrange = 10:2:10
    meas_invert(Nrange, eps_list)
end

## for N in 4:2:6
##     for eps in [0.02*2.0^(-j) for j in 1:3]
##         tauvst = []
##         for t in 1:20
##             data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\measFDQC\\reuse_false\\$(N)Q_$(eps)_$t.csv"))
##             tau = data.tau[1]
##             push!(tauvst, tau)
##         end
##         data = DataFrame(depths = tauvst)
##         CSV.write("D:\\Julia\\MyProject\\Data\\measFDQC\\reuse_false\\$(N)Q_$(eps).csv", data)
##     end
## end
## 
## N = 8
## plt = Plots.plot(title = L"N = "*"$(N)", ylabel = L"\mathrm{inverted \ depth}", xlabel = L"\mathrm{true \ depth}")
## for eps in [0.02*2.0^(-j) for j in 1:3]
##     data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\measFDQC\\reuse_false\\$(N)Q_$(eps).csv"))
##     depths = data.depths
##     @show depths
##     Plots.plot!(plt, [i for i in 1:20], depths, marker = (:circle,5), primary=true, label=L"\epsilon="*"$(eps)")
## end
## plt
## Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\measFDQC\\smallN_largeT_$(N)Q.pdf")
## 
## 
## 
## Plots.plot!(depths, data.e2, marker = (:circle,5), primary=true, label=L"\epsilon=0.2")
## Plots.plot!(depths, data.e1, marker = (:circle,5), primary=true, label=L"\epsilon=0.1")
## Plots.plot!(depths, data.e05, marker = (:circle,5), primary=true, label=L"\epsilon=0.05")
## Plots.plot!(depths, data.e025, marker = (:circle,5), primary=true, label=L"\epsilon=0.025")
## Plots.plot!(depths, data.e0125, marker = (:circle,5), primary=true, label=L"\epsilon=0.0125")
## Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\measFDQC.pdf")


#####let Nrange, gate, eps_list
#####
#####    eps_list = [0.0125*2.0^(-j) for j in 1:3]
#####    Nrange = 2:2:12
#####    gate = mt.random_unitary(4)
#####
#####    tau_depth_eps = execute(Nrange, eps_list, gate)
#####
#####    depths = div.(Vector(Nrange), 2)
#####    tau_eps_depth = [getindex.(tau_depth_eps,i) for i=1:length(tau_depth_eps[1])]
#####
#####    data = DataFrame(Error_required = eps_list, Depth = tau_eps_depth)
#####    CSV.write("D:\\Julia\\MyProject\\Data\\MalzVSLiu\\measFD2.csv", data)
#####
#####    #Plots.plot(ylabel = L"\tau", xlabel = L"\mathrm{true \ depth}")
#####    #Plots.plot!(depths, tau_eps_depth[1], lc=:green, primary=false)
#####    #Plots.plot!(depths, tau_eps_depth[1], seriestype=:scatter, mc=:green, markersize=:3, primary=true, label=L"\epsilon=0.1")
#####    #Plots.plot!(depths, tau_eps_depth[2], lc=:blue, primary=false)
#####    #Plots.plot!(depths, tau_eps_depth[2], seriestype=:scatter, mc=:blue, markersize=:3, primary=true, label=L"\epsilon=0.05")
#####    #Plots.plot!(depths, tau_eps_depth[3], lc=:red, primary=false)
#####    #Plots.plot!(depths, tau_eps_depth[3], seriestype=:scatter, mc=:red, markersize=:3, primary=true, label=L"\epsilon=0.025")
#####    #Plots.plot!(depths, tau_eps_depth[4], lc=:yellow, primary=false)
#####    #Plots.plot!(depths, tau_eps_depth[4], seriestype=:scatter, mc=:yellow, markersize=:3, primary=true, label=L"\epsilon=0.0125")
#####    #Plots.plot!(1:5, taus_pereps[2], lc=:blue, primary=false)
#####    #Plots.plot!(1:5, taus_pereps[2], seriestype=:scatter, mc=:blue, markersize=:3, primary=true, label=L"\epsilon=0.05")
#####    #Plots.plot!(1:5, taus_pereps[3], lc=:red, primary=false)
#####    #Plots.plot!(1:5, taus_pereps[3], seriestype=:scatter, mc=:red, markersize=:3, primary=true, label=L"\epsilon=0.025")
#####end
#####
#####data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\MalzVSLiu\\measFD2.csv"))
#####Plots.plot(ylabel = L"\mathrm{inverted \ depth}", xlabel = L"\mathrm{true \ depth}")
#####depths = data.True_depth
#####
#####Plots.plot!(depths, data.e2, marker = (:circle,5), primary=true, label=L"\epsilon=0.2")
#####Plots.plot!(depths, data.e1, marker = (:circle,5), primary=true, label=L"\epsilon=0.1")
#####Plots.plot!(depths, data.e05, marker = (:circle,5), primary=true, label=L"\epsilon=0.05")
#####Plots.plot!(depths, data.e025, marker = (:circle,5), primary=true, label=L"\epsilon=0.025")
#####Plots.plot!(depths, data.e0125, marker = (:circle,5), primary=true, label=L"\epsilon=0.0125")
#####Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\measFDQC.pdf")



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