include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
using DataFrames, StatsPlots
using DelimitedFiles

it.set_warn_order(28)

@show Threads.nthreads()

# Prepare initial state
eps_array = [0.2*2.0^(-i) for i in 0:4]
#eps_array = [0.01]

N = 30

function execute(command, N, eps_array; D = 2, tau = 3)
# Choose object to invert
    if command == "randMPS"
        test = mt.randMPS(N, D)
    elseif command == "FDQC"
        test = mt.initialize_fdqc(N, tau)
    elseif command == "FDMPO"
        siteinds = it.siteinds("Qubit", N)
        test = mt.MPO(mt.newLightcone(siteinds, tau))
    end

    tau_sweep = []
    tau_global = []
    err_sweep = []
    err_global = []
    niter_sweep = []
    niter_global = []


    for eps in eps_array
        println("\nAssuming error threshold eps = $eps\n")

        # check that latest err is greater than the required one, otherwise we already know what tau is and we can move to next error
        if length(err_sweep) > 0
            if err_sweep[end] < eps
                push!(tau_sweep, tau_sweep[end])
                push!(err_sweep, err_sweep[end])
                push!(niter_sweep, niter_sweep[end])
                continue
            end
        end

        results = mt.invert(test, mt.invertSweepLC; nruns = 1, maxiter = 100000, eps = eps, start_tau = (eps == eps_array[1] ? 1 : tau_sweep[end]+1))
        err1, tau1, niter1 = results["err"], results["tau"], results["niter"]
        #err_sweep = [result["err"] for result in res_array]
        #niter_sweep = [result["niter"] for result in res_array]
        push!(tau_sweep, tau1)
        push!(err_sweep, err1)
        push!(niter_sweep, niter1)




        println("\nAssuming error threshold eps = $eps\n")

        # check that latest err is greater than the required one, otherwise we already know what tau is and we can move to next error
        if length(err_global) > 0
            if err_global[end] < eps
                push!(tau_global, tau_global[end])
                push!(err_global, err_global[end])
                push!(niter_global, niter_global[end])
                continue
            end
        end

        #results2 = mt.invert(test, mt.invertGlobalSweep; nruns = 1, eps = eps, gradtol = 1e-8, maxiter = 100000, start_tau = (eps == eps_array[1] ? 1 : tau_global[end]+1))
        results2 = mt.invertSweep(test; n_runs=1, maxiter = 100000, eps=eps, start_tau = (eps == eps_array[1] ? 1 : tau_global[end]+1))
        #res2_array = results2["All"]
        err2, tau2, niter2 = results2["err"], results2["tau"], results2["niter"]
        push!(tau_global, tau2)
        push!(err_global, err2)
        push!(niter_global, niter2)
        #err_global = [result["err"] for result in res2_array]
        #niter_global = [result["niter"] for result in res2_array]
    end

    

    #data = [eps_array, tau_sweep, tau_global]
    #writedlm( "D:\\Julia\\MyProject\\Data\\SweepVsGlobal\\randmps.csv", data, ',')
    data_sweep = DataFrame(Error = eps_array, Depth1 = tau_sweep, Iterations1 = niter_sweep)
    data_global = DataFrame(Depth2 = tau_global, Iterations2 = niter_global)
    data = hcat(data_sweep, data_global)

    data = DataFrame(Err_sweep = err_sweep, Niter_sweep = niter_sweep, Err_global = err_global, Niter_global = niter_global)
    #CSV.write("D:\\Julia\\MyProject\\Data\\SweepVsGlobal\\"*command*"_30Q_10R.csv", data)

    Plots.plot(title = L"N=30, \ D=2", ylabel = L"\tau", xlabel = L"\epsilon", xflip = true)
    Plots.plot!(eps_array, tau_sweep, lc=:green, primary=false)
    Plots.plot!(eps_array, tau_sweep, seriestype=:scatter, mc=:green, legend=true, label="invertSweep")

    Plots.plot!(eps_array, tau_global, lc=:red, primary=false)
    Plots.plot!(eps_array, tau_global, seriestype=:scatter, mc=:red, label="invertGlobalSweep", legend=:bottomright)
    Plots.plot!(xscale=:log)
    #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\SweepVSGlobal"*command*"_30Q_10R.pdf")
    return data
end

data = execute("randMPS", N, eps_array)

#hist = @df data histogram(cols(1:4); layout=4, bins = 100)
#Plots.histogram!(hist, title = L"N = 30, \tau = 3")
#Plots.histogram!(hist[1], xlabel = L"\epsilon")
#Plots.histogram!(hist[2], xlabel = L"N_{iter}")
#Plots.histogram!(hist[3], xlabel = L"\epsilon")
#Plots.histogram!(hist[4], xlabel = L"N_{iter}")
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_SweepVSGlobal_hist.pdf")

#Plots.histogram2d(data.Err_sweep, data.Niter_sweep, xlabel = L"\mathrm{Error}", ylabel = L"N_{iter}", title = L"\mathrm{invertSweepLC}, N=30, \tau = 3, N_{run}=100", bins = (30,20))
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_Sweep_2dhist.pdf")
#Plots.histogram2d(data.Err_global, data.Niter_global, xlabel = L"\mathrm{Error}", ylabel = L"N_{iter}", title = L"\mathrm{invertGlobalSweep}, N=30, \tau = 3, N_{run}=100", bins = (30,20))
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_Global_2dhist.pdf")


#data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\SweepVsGlobal\\randmps.csv"))#

#Plots.plot(title = L"N=30, \ D=2", ylabel = L"N_{iter}", xlabel = L"\epsilon", xflip = true)
#Plots.plot!(eps_array, data.Iterations1, lc=:green, primary=false)
#Plots.plot!(eps_array, data.Iterations1, seriestype=:scatter, mc=:green, legend=true, label="invertSweep")#

#Plots.plot!(eps_array, data.Iterations2, lc=:red, primary=false)
#Plots.plot!(eps_array, data.Iterations2, seriestype=:scatter, mc=:red, label="invertGlobalSweep", legend=:bottomright)
#Plots.plot!(xscale=:log, yscale=:log)
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\SweepVSGlobal_randMPS_niter.pdf")














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