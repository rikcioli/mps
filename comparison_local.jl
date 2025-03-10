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
#eps_array = [0.2*2.0^(-i) for i in 0:4]
eps_array = [0.01]

#N = 30

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


    for tau in 1:6
        #println("\nAssuming error threshold eps = $eps\n")
        println("Inverting tau = $tau\n")

        # check that latest err is greater than the required one, otherwise we already know what tau is and we can move to next error
        #if length(err_sweep) > 0
        #    if err_sweep[end] < eps
        #        push!(tau_sweep, tau_sweep[end])
        #        push!(err_sweep, err_sweep[end])
        #        push!(niter_sweep, niter_sweep[end])
        #        continue
        #    end
        #end

        #results = mt.invert(test, mt.invertSweepLC; return_all=true, nruns = 10, maxiter = 20000, eps = eps, tau = (eps == eps_array[1] ? 3 : tau_sweep[end]+1))
        results = mt.invert(test, mt.invertSweepLC; nruns = 10, maxiter = 1000, tau = tau)
        #res_array = results["All"]
        err1, tau1, niter1 = results["err"], results["tau"], results["niter"]
        #err_sweep = [result["err"] for result in res_array]
        #niter_sweep = [result["niter"] for result in res_array]
        push!(tau_sweep, tau1)
        push!(err_sweep, err1)
        push!(niter_sweep, niter1)




        #println("\nAssuming error threshold eps = $eps\n")

        # check that latest err is greater than the required one, otherwise we already know what tau is and we can move to next error
        #if length(err_global) > 0
        #    if err_global[end] < eps
        #        push!(tau_global, tau_global[end])
        #        push!(err_global, err_global[end])
        #        push!(niter_global, niter_global[end])
        #        continue
        #    end
        #end

        #results2 = mt.invert(test, mt.invertGlobalSweep; return_all=true, nruns = 100, eps = eps, gradtol = 1e-8, maxiter = 20000, start_tau = (eps == eps_array[1] ? 3 : tau_global[end]+1))
        results2 = mt.invert(test, mt.invertGlobalSweep; nruns = 10, gradtol = 1e-8, maxiter = 1000, tau = tau)
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
    ###data_sweep = DataFrame(Error = eps_array, Depth1 = tau_sweep, Iterations1 = niter_sweep)
    ###data_global = DataFrame(Depth2 = tau_global, Iterations2 = niter_global)
    ###data = hcat(data_sweep, data_global)

    data = DataFrame(Tau = tau_sweep, Err_sweep = err_sweep, Niter_sweep = niter_sweep, Err_global = err_global, Niter_global = niter_global)
    CSV.write("D:\\Julia\\MyProject\\Data\\SweepVsGlobal\\"*command*"_30Q_2D_10R_6DM_capE3.csv", data)

    ###Plots.plot(title = L"N=30, \ D=2", ylabel = L"\tau", xlabel = L"\epsilon", xflip = true)
    ###Plots.plot!(eps_array, tau_sweep, lc=:green, primary=false)
    ###Plots.plot!(eps_array, tau_sweep, seriestype=:scatter, mc=:green, legend=true, label="invertSweep")
###
    ###Plots.plot!(eps_array, tau_global, lc=:red, primary=false)
    ###Plots.plot!(eps_array, tau_global, seriestype=:scatter, mc=:red, label="invertGlobalSweep", legend=:bottomright)
    ###Plots.plot!(xscale=:log)
    #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\SweepVSGlobal"*command*"_30Q_10R.pdf")
    return data
end


function logbins(data_array; nbins = 50)
    bounds = extrema(data_array)
    roundbounds = round.(bounds, sigdigits=1)
    intbounds = (Int(floor(log10(bounds[1]))), Int(ceil(log10(bounds[2]))))
    bins = 10.0 .^ range(intbounds[1], stop = intbounds[2], length = nbins)
    return bins, roundbounds, intbounds
end


"Constructs a random MPS of N qubits and bond dimension D and inverts up to maxtau, saving errors at each step"
function test(N::Integer, D::Integer, maxtau::Integer)
    #mps = mt.initialize_fdqc(N, D)
    mps = it.random_mps(it.siteinds("Qubit", N); linkdims = D)
    reslist = []
    best_guess = nothing
    for tau in 1:maxtau
        results = mt.invert(mps, mt.invertGlobalSweep; tau = tau, nruns = 8, gradtol = 1e-8, maxiter = 100000, init_array=best_guess)
        push!(reslist, results)
        best_guess = Array(results["lightcone"])
        data = DataFrame(err = results["err"])
        CSV.write("D:\\Julia\\MyProject\\Data\\randMPS\\$(N)Q_$(D)D_$(tau)T.csv", data)
    end
    data = DataFrame(Depths = 1:maxtau, Errors = [res["err"] for res in reslist])
    CSV.write("D:\\Julia\\MyProject\\Data\\randMPS\\$(N)Q_$(D)D_.csv", data)
end

function test4(range, D::Integer, maxtau::Integer)
    for N in range
        test(N, D, maxtau)
    end
end


"Constructs a random MPS of N qubits and bond dimension D and checks how many steps to invert exactly,
saving the half-chain entanglement entropy at each step"
function test2(N::Integer, D::Integer)
    N2 = div(N,2)
    #mps = mt.initialize_fdqc(N, N2+mod(N2+1,2))
    mps = it.random_mps(it.siteinds("Qubit", N); linkdims = D)
    maxtau = N+1
    tau_list = 1:maxtau
    reslist = []
    for tau in tau_list
        results = mt.invert(mps, mt.invertGlobalSweep; nruns = 8, gradtol = 1e-8, maxiter = 100000, tau = tau)
        push!(reslist, results)
        if results["err"] < 1e-10
            maxtau = tau
            break
        end
    end
    
    entropy_tau = []
    for tau in 1:maxtau
        zero = mt.initialize_vac(N, it.siteinds(mps))
        entropy_arr = []
        mt.apply!(zero, reslist[tau]["lightcone"]; entropy_arr = entropy_arr)
        push!(entropy_tau, entropy_arr)
    end
    entropy_mps = mt.entropy(mps, div(N,2))
    
    return mps, reslist, entropy_mps, entropy_tau
end

"Applies test2 for N in [Nmin, Nmax]"
function test3(Nmin::Integer, Nmax::Integer, D::Integer)
    Nlist = Nmin:2:Nmax
    true_ent = []
    reconstr_entr = []
    reconstr_errors = []
    for N in Nlist
        _, reslist, entropy_mps, entropy_tau = test2(N, D)
        push!(true_ent, entropy_mps)
        push!(reconstr_errors, [res["err"] for res in reslist])

        completed_entropy_tau = []
        for i in 1:length(entropy_tau)
            array = entropy_tau[i]
            temp = iseven(div(N,2)) ? [0.0] : []
            for item in array
                push!(temp, item)
                push!(temp, item)
            end
            temp = temp[1:i]
            push!(completed_entropy_tau, temp)
        end

        push!(reconstr_entr, completed_entropy_tau)

        data = DataFrame(Errors = reconstr_errors[end], Entropy = reconstr_entr[end], TrueEnt = entropy_mps)
        N2 = div(N,2)
        CSV.write("D:\\Julia\\MyProject\\Data\\$(N)Q_$(D)D_randMPS_test3.csv", data)
    end

    return true_ent, reconstr_entr, reconstr_errors
end



#mps, reslist, entropy_mps, entropy_tau = test2(8,2)
#true_ent, reconstr_entr, reconstr_errors = test3(10, 10, 4)
test4(10:10:50, 8, 5)

#######plt = Plots.plot(title = L"D = 2", ylabel = L"\epsilon", xlabel = L"\tau")
#######
#######for N in 4:2:14
#######    data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\$(N)Q_2D_randMPS_test3.csv"))
#######    errs = data.Errors
#######    @show errs
#######    Plots.plot!(plt, 1:length(errs), errs, marker = (:circle,5), primary=true, label="N = $N")
#######end
#######
#######plt
#######Plots.plot!(plt, yscale=:log)
#######Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\randMPS_exact_inv.pdf")


#hist = @df data histogram(cols(1:4); layout=4, bins = 100)


#hist = Plots.histogram(title = L"N = 16, \tau = 3", layout=4)
#bins1, rbounds1, ibounds1 = logbins(data.Err_sweep)
#Plots.histogram!(hist[1], data.Err_sweep, bins = bins1, label="Err_sweep", legend=:top, xlabel = L"\epsilon", xscale=:log10, xlims = (rbounds1[1]/2, rbounds1[2]*2), xticks = (10.0 .^ (ibounds1[1]:ibounds1[2])))
#Plots.histogram!(hist[2], data.Niter_sweep, bins=50, label="Niter_sweep", xlabel = L"N_{iter}")
#bins2, rbounds2, ibounds2 = logbins(data.Err_global)
#Plots.histogram!(hist[3], data.Err_global, bins = bins2, label="Err_global", legend=:top, xscale=:log10, xlims = (rbounds2[1]/2, rbounds2[2]*2), xlabel = L"\epsilon", xticks = (10.0 .^ (ibounds2[1]:ibounds2[2])))
#Plots.histogram!(hist[4], data.Niter_global, bins=50, label="Niter_global", xlabel = L"N_{iter}")
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_16Q_SweepVSGlobal_hist.pdf")
#
#Plots.histogram2d(data.Err_sweep, data.Niter_sweep, bins = (bins1, 50), xscale=:log10, xticks = (10.0 .^ (ibounds1[1]:ibounds1[2])))
#Plots.histogram2d!(xlabel = L"\epsilon", ylabel = L"N_{iter}", title = L"\mathrm{invertSweepLC}, N=16, \tau = 3, N_{run}=100")
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_16Q_Sweep_2dhist.pdf")
#
#Plots.histogram2d(data.Err_global, data.Niter_global, bins = (bins2, 50), xscale=:log10, xticks = (10.0 .^ (ibounds2[1]:ibounds2[2])))
#Plots.histogram2d!(xlabel = L"\epsilon", ylabel = L"N_{iter}", title = L"\mathrm{invertGlobalSweep}, N=16, \tau = 3, N_{run}=100")
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\FDQC_16Q_Global_2dhist.pdf")




#data = DataFrame(CSV.File("D:\\Julia\\MyProject\\Data\\SweepVsGlobal\\randMPS_30Q_4D_10R_7DM.csv"))

#Plots.plot(title = L"N=30, \ D=2", ylabel = L"\epsilon", xlabel = L"\tau")
#Plots.plot!(data.Tau, data.Err_sweep, lc=:green, primary=false)
#Plots.plot!(data.Tau, data.Err_sweep, seriestype=:scatter, mc=:green, legend=true, label="invertSweepLC")#

#Plots.plot!(data.Tau, data.Err_global, lc=:red, primary=false)
#Plots.plot!(data.Tau, data.Err_global, seriestype=:scatter, mc=:red, label="invertGlobalSweep", legend=:topright)
#Plots.plot!(yscale=:log)
#Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\randMPS_2D_errVStau_capE3.pdf")

