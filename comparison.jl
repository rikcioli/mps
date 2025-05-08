include("mpsMethods.jl")
import .MPSMethods as mt
using ITensors, ITensorMPS
#import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using CSV
using JET
#using DataFrames, StatsPlots

#Strided.disable_threads()
#@show ITensors.Strided.get_num_threads()
#BLAS.set_num_threads(56)
#@show ITensors.blas_get_num_threads()

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
    N = 1000
    eps = 1e-3
    psi = random_mps(siteinds("Qubit", N), linkdims = 2)
    mt.invertMPS1(psi, mt.invertGlobalSweep; eps = eps, pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\")
    #mt.invertMPS2("D:\\Julia\\MyProject\\Data\\randMPS\\", N, eps, mt.invertGlobalSweep)
end

