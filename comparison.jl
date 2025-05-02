include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
using JET
#using DataFrames, StatsPlots

ITensors.set_warn_order(28)


#sites = siteinds("Qubit", 10)
#mpo = random_mpo(sites)
#
#input_inds = sites
#output_inds = sites'
#tau = 2
#N = length(mpo)
#
## noprime the input inds
## change the output inds to a prime of the input inds to match the inds of the first layer of gates
#sites = noprime(input_inds)
#d = sites[1].space
#tempinds = siteinds(d, N)
#for i in 1:N
#    replaceind!(mpo[i], input_inds[i], tempinds[i])
#    replaceind!(mpo[i], output_inds[i], prime(sites[i], tau))
#    replaceind!(mpo[i], tempinds[i], sites[i])
#end
#
#lc = mt.newLightcone(sites, tau)
#U_array = Array(lc)
#@report_opt mt.updateLightcone!(lc, U_array)




function execute(command, N, eps_array; D = 2, tau = 3)
# Choose object to invert
    if command == "randMPS"
        test = random_mps(siteinds(2, N), linkdims = D)
        orthogonalize!(test, 3)
    elseif command == "FDQC"
        test = mt.initialize_fdqc(N, tau)
    end

    tau_malz = []
    err_malz = []
    tau_liu = []
    err_liu = []

    #for eps in eps_array
    #    println("\nAssuming error threshold eps = $eps\n")
#
    #    results = mt.invertMPSMalz(test, mt.invertGlobalSweep; eps = eps, kargsV = (nruns = 20,))
    #    err_total = results["err_total"]
    #    tau_total = maximum(results["V_tau"]) + maximum(results["W_tau"])
    #    push!(tau_malz, tau_total)
    #    push!(err_malz, err_total)
    #end

    #mt.disentangle!(test, 3)

    start_tau = 1
    for eps in eps_array
        println("\nAssuming error threshold eps = $eps\n")

        if eps != eps_array[1]
            if err_liu[end] < eps
                push!(err_liu, err_liu[end])
                push!(tau_liu, tau_liu[end])
                continue
            end
        end

        factor = 1
        results = mt.invertMPSLiu(test, mt.invertGlobalSweep; start_tau = start_tau, eps = eps, kargs_inv = (nruns = 8,))
        err_total = results["err_total"]
        tau_total = maximum(results["tau2"]) + results["tau1"]
        push!(tau_liu, tau_total)
        push!(err_liu, err_total)
        start_tau = results["tau1"]
    end

    data = DataFrame(Error = eps_array, tau_Liu = tau_liu, err_Liu = err_liu)
    CSV.write("D:\\Julia\\MyProject\\Data\\MalzVSLiu\\"*command*"_DisenLiu.csv", data)

    Plots.plot(title = L"N="*string(N), ylabel = L"\tau", xlabel = L"\epsilon", xflip = true)
    #Plots.plot!(eps_array, tau_malz, lc=:green, primary=false)
    #Plots.plot!(eps_array, tau_malz, seriestype=:scatter, mc=:green, legend=true, label="Malz")

    Plots.plot!(eps_array, tau_liu, lc=:red, primary=false)
    Plots.plot!(eps_array, tau_liu, seriestype=:scatter, mc=:red, label="Liu", legend=:bottomright)
    Plots.plot!(xscale=:log)

    Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\MalzVSLiu"*command*"DisenLiu.pdf")

    return data
end

function testLiu(psi, eps)
    results = mt.invertMPSLiu(psi, mt.invertGlobalSweep, eps = eps)
    err_total = results["err_total"]
    tau_total = maximum(results["tau2"]) + results["tau1"]
    return err_total, tau_total
end

function liu_tau_vs_N(Nrange, eps_array)
    for N in Nrange
        @show N
        energy, psi = mt.initialize_ising(N, 2)
        #psi = random_mps(siteinds("Qubit", N), linkdims = 2)
        tau_liu = []
        err_liu = []

        for eps in eps_array
            println("\nAssuming error threshold eps = $eps\n")
    
            if !isempty(err_liu)
                if err_liu[end] < eps
                    push!(err_liu, err_liu[end])
                    push!(tau_liu, tau_liu[end])
                    continue
                end
            end
    
            results = mt.invertMPSLiu(psi, mt.invertGlobalSweep; eps = eps, kargs_inv = (nruns = 8,))
            err_total = results["err_total"]
            tau_total = maximum(results["tau2"]) + results["tau1"]
            push!(tau_liu, tau_total)
            push!(err_liu, err_total)
        end
    
        data = DataFrame(Error = eps_array, tau_Liu = tau_liu, err_Liu = err_liu)
        CSV.write("D:\\Julia\\MyProject\\Data\\randMPS\\$(N)Q_final_EpsVsTau.csv", data)
    end
end





function create_invert(N, tau_list)
    # Choose object to invert
    tau_liu = []
    err_liu = []
    start_tau = 2
    for tau in tau_list
        test = mt.initialize_fdqc(N, tau)
    
        
        results = mt.invertMPSLiu(test, mt.invertGlobalSweep; start_tau = start_tau, eps = 1e-5, kargs_inv = (nruns = 10, maxiter = 10000))

        err_total = results["err_total"]
        tau_total = maximum(results["tau2"]) + results["tau1"]
        push!(tau_liu, tau_total)
        push!(err_liu, err_total)
        start_tau = results["tau1"]
    
    end
    
    data = DataFrame(True_depth = tau_list, Inverted_depth = tau_liu, Error = err_liu)
    CSV.write("D:\\Julia\\MyProject\\Data\\MalzVSLiu\\LiuD3QC.csv", data)
    
    Plots.plot(title = L"N="*string(N), ylabel = L"\tilde{\tau}_{\mathrm{tot}}", xlabel = L"\tau")
    #Plots.plot!(eps_array, tau_malz, lc=:green, primary=false)
    #Plots.plot!(eps_array, tau_malz, seriestype=:scatter, mc=:green, legend=true, label="Malz")
    
    Plots.plot!(tau_list, tau_liu, lc=:red, primary=false)
    Plots.plot!(tau_list, tau_liu, seriestype=:scatter, mc=:red, legend=false)
    #Plots.plot!(xscale=:log)

    Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\LiuD3QCdepth.pdf")
    
    return data
end


function final_tau_vs_N(Nrange, eps_array)
    for N in Nrange
        @show N
        #energy, psi = mt.initialize_ising(N, 2)
        psi = random_mps(siteinds("Qubit", N), linkdims = 2)
        tau_liu = []
        err_liu = []

        for eps in eps_array
            println("\nAssuming error threshold eps = $eps\n")
    
            if !isempty(err_liu)
                if err_liu[end] < eps
                    push!(err_liu, err_liu[end])
                    push!(tau_liu, tau_liu[end])
                    continue
                end
            end
    
            results, results_final = mt.invertMPSfinal(psi, mt.invertGlobalSweep; eps = eps)
            err1 = results["err_total"]
            tau1 = maximum(results["tau2"]) + results["tau1"]
            err_total = results_final["err"]
            tau_total = results_final["tau"]
            @show err1, tau1, err_total, tau_total
            push!(tau_liu, tau_total)
            push!(err_liu, err_total)
        end
    
        data = DataFrame(Error = eps_array, tau_Liu = tau_liu, err_Liu = err_liu)
        CSV.write("D:\\Julia\\MyProject\\Data\\randMPS\\$(N)Q_final_EpsVsTau.csv", data)
    end
end


let
    N = 10
    eps = 1e-1
    psi = random_mps(siteinds("Qubit", N), linkdims = 2)
    mt.invertMPS1(psi, mt.invertGlobalSweep; eps = eps, pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\")
    mt.invertMPS2("D:\\Julia\\MyProject\\Data\\randMPS\\", N, eps; nthreads = 1)
end


#@profview datas = create_invert(20, [2])


#test = mt.randMPS(N, 2)
#results = [mt.invertMPSMalz(test, mt.invertGlobalSweep; q=i, kargsV = (nruns = 10, )) for i in 2:5]


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