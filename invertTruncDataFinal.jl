#using MKL
#include("rrules.jl")
#include("optFunctions.jl")
include("RGmethods.jl")
using ITensors, ITensorMPS
using LinearAlgebra
using JLD2, HDF5
using DataFrames, CSV
using LaTeXStrings
using Plots
using ColorSchemes

home = "D:\\Julia\\MyProject\\InversionData\\testdata\\"



Base.@kwdef mutable struct InversionInstructions
    maxrank::Union{Nothing, Int} = nothing
    maxerror::Union{Nothing, Float64} = nothing
    atol::Float64 = 1e-10
    maxiter::Int = 1000000
    gradtol::Float64 = 1e-8
    n_checkpoint::Int = 1000
    skip_outer::Bool = false
    m::Int = 5                                      
    Σε_max::Float64 = 1e-2
    c_ref::Float64 = Inf                             
    ρ0::Float64 = 1.0
    ρ::Float64 = 1.0
    λ0::Float64 = 1.0
    λ::Float64 = 1.0                                
    outer_iters::Int = 25
    inner_maxiter::Int = 10000
    inner_gradtol::Float64 = 1e-5
    ρ_growth::Float64 = 2.0
end

# copy for a mutable struct (field-by-field)
Base.copy(x::InversionInstructions) = InversionInstructions(
    (getfield(x, f) for f in fieldnames(InversionInstructions))...)


function find_max_finished_tau(folder::String, Nvals::Vector{Int}; include_checkpoints = false)
    tauNvals = Tuple{Int,Int}[]

    for N in Nvals
        pattern = Regex("N$(N)_T(\\d+)\\.jld2")
        taus = [parse(Int, m.captures[1]) for f in readdir(folder)
                for m in [match(pattern, f)] if !isnothing(m)]
        isempty(taus) && continue
        sort!(taus, rev=true)

        if include_checkpoints
            push!(tauNvals, (taus[1], N))
            continue
        else
            for tau in taus
                result = load_object(folder*"N$(N)_T$(tau).jld2")
                if get(result, :finished, true)
                    push!(tauNvals, (tau, N))
                    break
                end
            end
        end
    end

    return tauNvals
end


function extract_data(folder::String, Nvals::Vector{Int}; include_checkpoints = false)

    tauNvals = find_max_finished_tau(folder, Nvals, include_checkpoints=include_checkpoints)

    Nout = Int[]
    tauout = Int[]
    err_vs_tau = Float64[]
    costratio_vs_tau = Float64[]
    time_vs_tau = Float64[]
    niter_vs_tau = Int64[]
    time1iter_vs_tau = Float64[]

    for (maxtau, N) in tauNvals
        #f = h5open(folder*"$(N)_mps.h5","r")
        #psi_og = read(f,"psi",MPS)
        #close(f)

        for tau in 1:maxtau
            results = load_object(folder*"N$(N)_T$(tau).jld2")
            @show N, tau
            err = results[:err]
            time = results[:time]
            niter = size(results[:normgradhistory], 1)
            cost = results[:overlap_cost]

            push!(Nout, N)
            push!(tauout, tau)
            push!(err_vs_tau, err)
            push!(time_vs_tau, time)
            push!(niter_vs_tau, niter)
            push!(time1iter_vs_tau, time/niter)
            push!(costratio_vs_tau, abs(cost[2])/cost[1])
        end
    end

    df = DataFrame(N=Nout, tau=tauout, err=err_vs_tau, time=time_vs_tau,
                   niter=niter_vs_tau, time1iter=time1iter_vs_tau, costratio=costratio_vs_tau)
    return df
end

function extract_plots(data::DataFrame, title::String)
    Nvals = sort(unique(data.N))
    colors_err = [get(ColorSchemes.inferno, 0.5)]
    colors_time = [get(ColorSchemes.bluegreenyellow, 0.5)]
    if length(Nvals) > 1
        colors_err = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=length(Nvals))]
        colors_time = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=length(Nvals))]
    end
    plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon",
                xflip=true, title=title, legend=:topleft, dpi=400)
    plt_time_err = plot(xscale=:log, ylabel=L"t \ (s)",
                    xlabel=L"\epsilon", xflip=true, title=title,
                    legend=:topleft, dpi=400)
    plt_time_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_niter_tau = plot(ylabel=L"\mathrm{niter}", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_time1iter_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_costratio_tau = plot(ylabel="cost[2]/cost[1]", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    for (pos, N) in enumerate(Nvals)
        subdata = sort(data[data.N .== N, :], :tau)

        plot!(plt_err, subdata.err, subdata.tau, label=L"N="*"$(N)", color=colors_err[pos])
        plot!(plt_time_err, subdata.err, subdata.time, label=L"N="*"$(N)", color=colors_time[pos])
        plot!(plt_time_tau, subdata.tau, subdata.time, label=L"N="*"$(N)", color=colors_time[pos])
        plot!(plt_niter_tau, subdata.tau, subdata.niter, label=L"N="*"$(N)", color=colors_time[pos])
        plot!(plt_time1iter_tau, subdata.tau[2:end], subdata.time1iter[2:end], label=L"N="*"$(N)", color=colors_time[pos])
        plot!(plt_costratio_tau, subdata.tau, subdata.costratio, label=L"N="*"$(N)", color=colors_err[pos], yscale=:log10)
    end

    return [plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau]
end


function extract_plots_xi(data::DataFrame, title::String)
    N = only(unique(data.N))
    xis = sort(unique(data.xi))
    colors_err = [get(ColorSchemes.inferno, 0.5)]
    colors_time = [get(ColorSchemes.bluegreenyellow, 0.5)]
    if length(xis) > 1
        colors_err = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=length(xis))]
        colors_time = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=length(xis))]
    end
    plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon",
                xflip=true, title=title, legend=:topleft, dpi=400)
    plt_err_resc = plot(xscale=:log, ylabel=L"\tau/\xi", xlabel=L"\xi \epsilon",
                xflip=true, title=title, legend=:topleft, dpi=400)
    plt_time_err = plot(xscale=:log, ylabel=L"t \ (s)",
                    xlabel=L"\epsilon", xflip=true, title=title,
                    legend=:topleft, dpi=400)
    plt_time_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_niter_tau = plot(ylabel=L"\mathrm{niter}", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_time1iter_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_costratio_tau = plot(ylabel="cost[2]/cost[1]", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    for (pos, xi) in enumerate(xis)
        subdata = sort(data[data.xi .== xi, :], :tau)

        plot!(plt_err, subdata.err, subdata.tau, label=L"\xi="*"$(xi)", color=colors_err[pos])
        plot!(plt_err_resc, xi .* subdata.err, subdata.tau ./ xi, label=L"\xi="*"$(xi)", color=colors_err[pos])
        plot!(plt_time_err, subdata.err, subdata.time, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_time_tau, subdata.tau, subdata.time, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_niter_tau, subdata.tau, subdata.niter, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_time1iter_tau, subdata.tau[2:end], subdata.time1iter[2:end], label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_costratio_tau, subdata.tau, subdata.costratio, label=L"\xi="*"$(xi)", color=colors_err[pos], yscale=:log10)
    end

    return [plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau, plt_err_resc]
end




function extract_tau_vs_xi_fixederr(data::DataFrame; err_ref = 1e-2)
    Nvals = sort(unique(data.N))
    xis = sort(unique(data.xi))

    function first_tau(subdata, err_ref)
        isempty(subdata) && return missing
        subdata = sort(subdata, :tau)
        idx = findfirst(subdata.err .< err_ref)
        return isnothing(idx) ? missing : subdata.tau[idx]
    end

    colors_N = [get(ColorSchemes.inferno, 0.5)]
    if length(Nvals) > 1
        colors_N = [get(ColorSchemes.inferno, t) for t in range(0.1, stop=0.9, length=length(Nvals))]
    end

    plt_xi_N = plot(xlabel = L"\xi", ylabel = L"\tau", title = L"\epsilon="*"$(err_ref)", legend = :topleft, dpi = 400)
    for (pos, N) in enumerate(Nvals)
        tau_vals = [first_tau(data[(data.N .== N) .& (data.xi .== xi), :], err_ref) for xi in xis]
        plot!(plt_xi_N, xis, tau_vals, marker = :circle, label = "N=$(N)", color = colors_N[pos])
    end

    return plt_xi_N
end


function extract_tau_vs_xi_fixedN(data::DataFrame; err_refs = [1e-2, 1e-3], N_fixed = 100)
    xis = sort(unique(data.xi))

    # Allow either a single value or a collection of values
    err_refs = err_refs isa Number ? [err_ref] : collect(err_refs)

    function first_tau(subdata, err_ref)
        isempty(subdata) && return missing
        subdata = sort(subdata, :tau)
        idx = findfirst(subdata.err .< err_ref)
        return isnothing(idx) ? missing : subdata.tau[idx]
    end

    colors_err = [get(ColorSchemes.bluegreenyellow, 0.5)]
    if length(err_refs) > 1
        colors_err = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.1, stop=0.9, length=length(err_refs))]
    end

    plt_xi_err = plot(xlabel = L"\xi", ylabel = L"\tau", title = L"N="*"$(N_fixed)", legend = :topleft, dpi = 400)
    for (pos, err_ref) in enumerate(err_refs)
        tau_vals = [first_tau(data[(data.N .== N_fixed) .& (data.xi .== xi), :], err_ref) for xi in xis]
        plot!(plt_xi_err, xis, tau_vals, marker = :circle, label = L"\epsilon="*"$(err_ref)", color = colors_err[pos])
    end

    return plt_xi_err
end


function extract_tau_vs_N_fixedfid(data::DataFrame, fid_refs, title::String)
    Nvals = sort(unique(data.N))
    @show Nvals
    fid_refs = fid_refs isa Number ? [fid_refs] : collect(fid_refs)

    function first_tau(subdata, fid_refs)
        isempty(subdata) && return missing
        subdata = sort(subdata, :tau)
        idx = findfirst(subdata.err .< fid_refs)
        return isnothing(idx) ? missing : subdata.tau[idx]
    end

    colors = [get(ColorSchemes.inferno, 0.5)]
    if length(fid_refs) > 1
        colors = [get(ColorSchemes.inferno, t) for t in range(0.1, stop=0.9, length=length(fid_refs))]
    end

    plt_tau_N = plot(xlabel = L"N", ylabel = L"\tau", title = title, legend = :topleft, dpi = 400)
    for (pos, fid) in enumerate(fid_refs)
        err = 1-sqrt(fid)
        tau_vals = [first_tau(data[(data.N .== N), :], err) for N in Nvals]
        @show err, tau_vals
        plot!(plt_tau_N, Nvals, tau_vals, marker = :circle, label = L"F="*"$(fid)", color = colors[pos])
    end

    return plt_tau_N
end



# Overlay data from several folders (e.g. different bond dims) on the same plots,
# using a different linestyle per folder and a different color per N.
# `dfs_labels` is a Vector of (df, label) tuples, one per folder, in the order
# they should be styled (solid, dash, dot, dashdot, ...).
function extract_plots_compare(dfs_labels::Vector{<:Tuple{DataFrame,String}}, Nvals::Vector{Int}, title::String;
                                linestyles = [:solid, :dash, :dot, :dashdot])

    colors_err = length(Nvals) > 1 ?
        [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=length(Nvals))] :
        [get(ColorSchemes.inferno, 0.5)]
    colors_time = length(Nvals) > 1 ?
        [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=length(Nvals))] :
        [get(ColorSchemes.bluegreenyellow, 0.5)]

    plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon",
                xflip=true, title=title, legend=:topleft, dpi=400)
    plt_time_err = plot(xscale=:log, ylabel=L"t \ (s)",
                    xlabel=L"\epsilon", xflip=true, title=title,
                    legend=:topleft, dpi=400)
    plt_time_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_niter_tau = plot(ylabel=L"\mathrm{niter}", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_niter_tau_cumulative = plot(ylabel=L"\mathrm{niter}", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_time1iter_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)
    plt_costratio_tau = plot(ylabel="cost[2]/cost[1]", xlabel=L"\tau",
                    title=title, legend=:topleft, dpi=400)

    for (fidx, (data, flabel)) in enumerate(dfs_labels)
        ls = linestyles[mod1(fidx, length(linestyles))]
        for (pos, N) in enumerate(Nvals)
            subdata = sort(data[data.N .== N, :], :tau)
            isempty(subdata) && continue
            lbl = L"N="*"$(N), "*flabel

            plot!(plt_err, subdata.err, subdata.tau, label=lbl, color=colors_err[pos], line=ls)
            plot!(plt_time_err, subdata.err, subdata.time, label=lbl, color=colors_time[pos], line=ls)
            plot!(plt_time_tau, subdata.tau, subdata.time, label=lbl, color=colors_time[pos], line=ls)
            plot!(plt_niter_tau, subdata.tau, subdata.niter, label=lbl, color=colors_time[pos], line=ls)
            plot!(plt_time1iter_tau, subdata.tau[2:end], subdata.time1iter[2:end], label=lbl, color=colors_time[pos], line=ls)
            plot!(plt_costratio_tau, subdata.tau, subdata.costratio, label=lbl, color=colors_err[pos], yscale=:log10, line=ls)
        end
    end

    return [plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau]
end


# Error and gradient norm of every depth after M iterations, for several M: the
# depth-vs-error curve each maxiter = M would have given, read from the per-iteration
# history (normgradhistory row M+1 = (cost f, |grad|) after M iterations, err = 1 - exp(-f)).
# A depth that converged before M keeps its final values; a depth that has not run
# M iterations (yet) gets NaN. include_checkpoints = false reads converged depths only;
# true also reads unfinished ones (checkpoints of runs still going or interrupted).
# Only meaningful for parallel runs (folder = <folder>/parallel/): in a sequential run
# depth tau starts from where depth tau-1 ended, so stopping earlier would have changed
# the start of every later depth.
function extract_data_iters(folder::String, Nvals::Vector{Int}, iters; include_checkpoints = false)
    iters = sort(Int.(collect(iters)))
    Nout = Int[]
    tauout = Int[]
    Mout = Int[]
    err_vs_M = Float64[]
    gradnorm_vs_M = Float64[]
    converged_vs_M = Bool[]

    for N in Nvals
        pattern = Regex("^N$(N)_T(\\d+)\\.jld2\$")
        taus = sort([parse(Int, m.captures[1]) for f in readdir(folder)
                     for m in [match(pattern, f)] if !isnothing(m)])
        for tau in taus
            results = load_object(folder*"N$(N)_T$(tau).jld2")
            converged = get(results, :converged, get(results, :finished, true))   # old files have no :converged
            (converged || include_checkpoints) || continue
            h = results[:normgradhistory]
            for M in iters
                row = M + 1 <= size(h, 1) ? M + 1 :
                      (converged && size(h, 1) > 0) ? size(h, 1) : 0
                push!(Nout, N)
                push!(tauout, tau)
                push!(Mout, M)
                push!(err_vs_M, row > 0 ? -expm1(-h[row, 1]) : NaN)
                push!(gradnorm_vs_M, row > 0 ? h[row, 2] : NaN)
                push!(converged_vs_M, converged)
            end
        end
    end

    df = DataFrame(N=Nout, tau=tauout, M=Mout, err=err_vs_M, gradnorm=gradnorm_vs_M, converged=converged_vs_M)
    return df
end

# Depth vs error and depth vs gradient norm of one N, one curve per iteration count M
# (data from extract_data_iters).
function plot_err_vs_iters(data::DataFrame, N::Int, title::String)
    data = sort(data[data.N .== N, :], [:M, :tau])
    iters = sort(unique(data.M))
    colors = length(iters) > 1 ?
        [get(ColorSchemes.viridis, t) for t in range(0.9, stop=0.1, length=length(iters))] :
        [get(ColorSchemes.viridis, 0.5)]

    plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon",
                xflip=true, title=title, legend=:topleft, dpi=400)
    plt_grad = plot(yscale=:log, xlabel=L"\tau", ylabel=L"||\nabla f||",
                title=title, legend=:topleft, dpi=400)
    for (pos, M) in enumerate(iters)
        subdata = data[data.M .== M, :]
        plot!(plt_err, subdata.err, subdata.tau, label=L"M="*"$(M)", color=colors[pos], marker=:circle, markersize=2)
        plot!(plt_grad, subdata.tau, subdata.gradnorm, label=L"M="*"$(M)", color=colors[pos], marker=:circle, markersize=2)
    end
    plot!(plt_grad, 1:maximum(data.tau), [1e-8 for _ in 1:maximum(data.tau)], label="gradtol", color=:black, linestyle=:dash)

    return [plt_err, plt_grad]
end

# Where each depth would have stopped with the windowed value criterion of invert_maxrank,
# replayed on the per-iteration history of a run made without it (err_reltol = 0, gradtol only).
# As in invert_maxrank: every n_conv iterations the error is compared with the one n_conv
# iterations earlier, and the run stops if it changed by less than 1e-13 or decreased by less
# than err_reltol relative to its current value; it also stops as soon as |grad| < gradtol.
# The replay assumes one uninterrupted run (a resumed run restarts the window, so its checks
# can sit a few iterations off) and reads row M+1 of normgradhistory as iteration M.
# One row per (N, tau, err_reltol): stop_iter / err_stop / gradnorm_stop where it would have
# stopped, stopped_by = "value", "gradtol" or "not yet" (the history ends before any stop:
# err_stop is then the error at the last iteration available, a lower bound on the iterations).
function extract_reltol_stops(folder::String, Nvals::Vector{Int}, reltols; n_conv = 1000, gradtol = 1e-8,
                              err_floor_atol = 1e-13)
    rows = NamedTuple[]
    for N in Nvals
        pattern = Regex("^N$(N)_T(\\d+)\\.jld2\$")
        taus = sort([parse(Int, m.captures[1]) for f in readdir(folder) for m in [match(pattern, f)] if !isnothing(m)])
        for tau in taus
            results = load_object(folder*"N$(N)_T$(tau).jld2")
            time = results[:time]
            h = results[:normgradhistory]
            n = size(h, 1) - 1                                   # iterations recorded
            n < 0 && continue
            err(M) = -expm1(-h[M+1, 1])
            g = findfirst(<(gradtol), h[:, 2])                    # first row with |grad| < gradtol
            M_grad = g === nothing ? typemax(Int) : g - 1
            for reltol in reltols
                M_value = typemax(Int)
                for M in 2n_conv:n_conv:n
                    Δ = err(M - n_conv) - err(M)                  # > 0: the error decreased
                    if abs(Δ) < err_floor_atol || (Δ >= 0 && Δ / err(M) < reltol)
                        M_value = M; break
                    end
                end
                M_stop = min(M_value, M_grad)
                by = M_stop == typemax(Int) ? "not yet" : (M_value <= M_grad ? "value" : "gradtol")
                M_use = by == "not yet" ? n : M_stop
                time1iter = time/n
                avg_time = time1iter*M_use
                push!(rows, (N = N, tau = tau, reltol = reltol, stop_iter = M_use, err_stop = err(M_use),
                             gradnorm_stop = h[M_use+1, 2], stopped_by = by, niter_total = n, time1iter = time1iter, avg_time = avg_time, costratio=NaN,
                             err_final = err(n), converged_final = get(results, :converged, false)))
            end
        end
    end
    return DataFrame(rows)
end

# Depth vs error at the stopping point of each err_reltol, for one N (data from extract_reltol_stops),
# with the error at the end of the actual run. Depths that would not have stopped yet are drawn hollow.
function plot_reltol_stops(data::DataFrame, N::Int, title::String)
    data = sort(data[data.N .== N, :], [:reltol, :tau])
    reltols = sort(unique(data.reltol), rev = true)
    colors = [get(ColorSchemes.viridis, t) for t in range(0.9, stop=0.1, length=length(reltols))]
    plt = plot(xscale=:log10, ylabel=L"\tau", xlabel=L"\epsilon", xflip=true, title=title, legend=:topleft, dpi=400)
    final = unique(data[:, [:tau, :err_final]])
    plot!(plt, final.err_final, final.tau, label="end of the run", color=:black, linestyle=:dash)
    for (pos, r) in enumerate(reltols)
        d = data[data.reltol .== r, :]
        plot!(plt, d.err_stop, d.tau, label="err_reltol = $(r)", color=colors[pos])
        stopped = d.stopped_by .!= "not yet"
        scatter!(plt, d.err_stop[stopped], d.tau[stopped], color=colors[pos], markersize=3, label="")
        scatter!(plt, d.err_stop[.!stopped], d.tau[.!stopped], color=:white, markerstrokecolor=colors[pos], markersize=3, label="")
    end
    return plt
end

# Iterations at which each depth would have stopped, per err_reltol (data from extract_reltol_stops),
# with the length of the actual run. Depths that would not have stopped yet are drawn hollow at the
# end of their history (a lower bound).
function plot_reltol_iters(data::DataFrame, N::Int, title::String)
    data = sort(data[data.N .== N, :], [:reltol, :tau])
    reltols = sort(unique(data.reltol), rev = true)
    colors = [get(ColorSchemes.viridis, t) for t in range(0.9, stop=0.1, length=length(reltols))]
    plt = plot(yscale=:log10, xlabel=L"\tau", ylabel="iterations", title=title, legend=:bottomright, dpi=400)
    final = unique(data[:, [:tau, :niter_total]])
    plot!(plt, final.tau, max.(final.niter_total, 1), label="end of the run", color=:black, linestyle=:dash)
    for (pos, r) in enumerate(reltols)
        d = data[data.reltol .== r, :]
        plot!(plt, d.tau, max.(d.stop_iter, 1), label="err_reltol = $(r)", color=colors[pos])
        stopped = d.stopped_by .!= "not yet"
        scatter!(plt, d.tau[stopped], max.(d.stop_iter[stopped], 1), color=colors[pos], markersize=3, label="")
        scatter!(plt, d.tau[.!stopped], max.(d.stop_iter[.!stopped], 1), color=:white, markerstrokecolor=colors[pos], markersize=3, label="")
    end
    return plt
end

# Power law fit tau = A * err^alpha (linear fit of log(tau) vs log(err)) at fixed N and M,
# using only depths tau >= taumin (data from extract_data_iters).
function fit_tau_vs_err(data::DataFrame, N::Int, M::Int; taumin::Int = 1)
    d = data[(data.N .== N) .& (data.M .== M) .& .!isnan.(data.err) .& (data.err .> 0) .& (data.tau .>= taumin), :]
    x = log.(d.err)
    y = log.(Float64.(d.tau))
    A = hcat(ones(length(x)), x)
    c = A \ y
    r = y - A*c
    s2 = sum(abs2, r) / (length(x) - 2)
    se = sqrt.(s2 .* diag(inv(A'A)))
    R2 = 1 - sum(abs2, r) / sum(abs2, y .- sum(y)/length(y))
    println("N=$N, M=$M, taumin=$taumin, npts=$(length(x)): tau = $(exp(c[1])) * err^($(c[2]) ± $(se[2])), R2=$R2  =>  err ~ tau^($(1/c[2]))")
    return (prefactor = exp(c[1]), alpha = c[2], alpha_err = se[2], R2 = R2, npts = length(x))
end




### ISING ###

## SEQUENTIAL ##

home = "D:\\Julia\\MyProject\\InversionData\\testdata\\"
folder = home*"ising\\g0.75\\m5_long\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals; include_checkpoints=false)
CSV.write(folder*"df.csv", df)          # read by plots.ipynb
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "Ising g=0.75, m=5, chi=10")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

pltfolder = home*"ising\\plots\\g0.75\\"
savefig(plt_err, pltfolder*"tau_vs_eps.png")
savefig(plt_time_err, pltfolder*"time_vs_eps.png")
savefig(plt_time_tau, pltfolder*"time_vs_tau.png")
savefig(plt_niter_tau, pltfolder*"niter_vs_tau.png")
savefig(plt_time1iter_tau, pltfolder*"time1iter_vs_tau.png")
savefig(plt_costratio_tau, pltfolder*"costratio_vs_tau.png")


sel   = df[df.tau .>= 3, :]
X     = [log.(1 ./ sel.err) log.(sel.N) ones(nrow(sel))]
a, b, c = X \ Float64.(sel.tau)                  # check b ≈ a
A, c0 = [log.(sel.N ./ sel.err) ones(nrow(sel))] \ Float64.(sel.tau)

plt_collapse = plot(xscale=:log10, xflip=true, xlabel=L"\epsilon/N", ylabel=L"\tau",
                    title="Ising g=0.75", legend=:topleft, dpi=400)
Ns = sort(unique(df.N))
cols = [get(ColorSchemes.inferno, t) for t in range(0.7, 0.1, length=length(Ns))]
for (k, N) in enumerate(Ns)
    sub = sort(df[df.N .== N, :], :tau)
    plot!(plt_collapse, sub.err ./ N, sub.tau, marker=:circle, ms=3, color=cols[k], label="N=$N")
end
x = exp10.(range(log10(minimum(df.err ./ df.N)), log10(maximum(sel.err ./ sel.N)), length=100))
plot!(plt_collapse, x, A .* log.(1 ./ x) .+ c0, color=:black, ls=:dash,
      label=latexstring("\\tau = $(round(A; digits=2))\\,\\log(N/\\epsilon) $(round(c0; digits=2))"))



plt = extract_tau_vs_N_fixederr(df, [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2], "testing")
plot!(plt, xscale=:log10)

qs = Float64.(qlist[2:end])
y = log.(errs_per_block[2:end])
M = [ones(length(qs)) qs]
c = M\y                 # c = [log(A), -κ]
A, κ = exp(c[1]), -c[2]

γ = 2.0                    # fix by convention
ξ = γ / κ
println("A = $A, κ = γ/ξ = $κ, ξ = $ξ (with γ = $γ)")
plot!(qlist, A .* exp.(-κ*qlist), label=L"A e^{-q/ξ}", color=:red, linestyle=:dash)


## PARALLEL ##

folder = home*"ising\\g0.75\\m5_long\\parallel\\"
Nvals = Vector(100:100:300)
df = extract_data(folder, Nvals; include_checkpoints=true)
CSV.write(folder*"df.csv", df)          # read by plots.ipynb
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "Ising g=0.75, m=5, chi=20")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau





# XXZ

## SEQUENTIAL ##

home = "D:\\Julia\\MyProject\\InversionData\\testdata\\"
folder = home*"xxz\\Jz2.5\\m20_driver\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
CSV.write(folder*"df.csv", df)          # read by plots.ipynb
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

pltfolder = folder*"xxz\\plots\\m20_driver\\"
savefig(plt_err, pltfolder*"tau_vs_eps.png")
savefig(plt_time_err, pltfolder*"time_vs_eps.png")
savefig(plt_time_tau, pltfolder*"time_vs_tau.png")
savefig(plt_niter_tau, pltfolder*"niter_vs_tau.png")
savefig(plt_time1iter_tau, pltfolder*"time1iter_vs_tau.png")
savefig(plt_costratio_tau, pltfolder*"costratio_vs_tau.png")

plt = extract_tau_vs_N_fixederr(df, [4e-3, 8e-3, 1e-2], "testing")
plot!(plt, xscale=:log10)




folder = home*"xxz\\Jz2.5\\m20_long\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = home*"xxz\\Jz2.5\\m20_chi36\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=36")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



## PARALLEL ##

folder = home*"xxz\\Jz2.5\\sweeps\\chi24\\parallel\\"
Nvals = Vector(100:100:300)
df = extract_data(folder, Nvals; include_checkpoints=true)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = home*"xxz\\Jz2.5\\sweeps\\chi36\\parallel\\"
Nvals = Vector(100:100:300)
df = extract_data(folder, Nvals; include_checkpoints=true)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=36")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = home*"xxz\\Jz2.5\\sweeps\\chi48\\parallel\\"
Nvals = Vector(100:100:300)
df = extract_data(folder, Nvals; include_checkpoints=true)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ Jz=2.5, m=20, chi=48")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = home*"xxz\\Jz2.5\\sweeps\\chi24\\parallel\\"
df_iters = extract_data_iters(folder, [100], [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]; include_checkpoints=true)
N = 100
plt_err_iters, plt_grad_iters = plot_err_vs_iters(df_iters, N, "XXZ, N=$(N)")
plt_err_iters
plt_grad_iters

Nvals = [100]
df = extract_data(home*"xxz\\Jz2.5\\m20_driver\\", Nvals)
plot!(plt_err_iters, df.err, 1:length(df.err), color=:black, linestyle=:dash, label="sequential")

savefig(plt_err_iters, home*"corrmps\\plots\\niter_xxz_chi24.png")

plot!(plt_err_iters, yscale=:log10)

for taumin in [1, 2, 3], N in [100]
    fit_tau_vs_err(df_iters, N, 100_000; taumin=taumin)
end

fit = fit_tau_vs_err(df_iters, N, 100_000; taumin=4)
errs_fit = df_iters[(df_iters.N .== N) .& (df_iters.M .== 100_000) .& .!isnan.(df_iters.err), :err]
errs_fit = exp10.(range(log10(minimum(errs_fit)), log10(maximum(errs_fit)), length=100))
plot!(plt_err_iters, errs_fit, fit.prefactor .* errs_fit .^ fit.alpha,
      label=L"\tau \propto \epsilon^{%$(round(fit.alpha, digits=3))}", color=:red, linestyle=:dash)

savefig(plt_err_iters, home*"corrmps\\plots\\niter_xxz_chi24_log.png")

df_stops = extract_reltol_stops(folder, [100, 200, 300], [1e-1, 1e-2, 1e-3, 1e-4])
plt_err = plot_reltol_stops(df_stops, 100, "test")
plt_iter = plot_reltol_iters(df_stops, 100, "test")




folder = home*"xxz\\Jz2.5\\sweeps\\chi36\\parallel\\"
df_iters = extract_data_iters(folder, [100], [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]; include_checkpoints=true)
N = 100
plt_err_iters, plt_grad_iters = plot_err_vs_iters(df_iters, N, "XXZ, N=$(N)")
plt_err_iters
plt_grad_iters

plot!(plt_err_iters, yscale=:log10)

for taumin in [1, 2, 3], N in [100]
    fit_tau_vs_err(df_iters, N, 100_000; taumin=taumin)
end

fit = fit_tau_vs_err(df_iters, N, 500_000; taumin=4)
errs_fit = df_iters[(df_iters.N .== N) .& (df_iters.M .== 100_000) .& .!isnan.(df_iters.err), :err]
errs_fit = exp10.(range(log10(minimum(errs_fit)), log10(maximum(errs_fit)), length=100))
plot!(plt_err_iters, errs_fit, fit.prefactor .* errs_fit .^ fit.alpha,
      label=L"\tau \propto \epsilon^{%$(round(fit.alpha, digits=3))}", color=:red, linestyle=:dash)



folder = home*"xxz\\Jz2.5\\sweeps\\chi48\\parallel\\"
df_iters = extract_data_iters(folder, [100], [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]; include_checkpoints=false)
N = 100
plt_err_iters, plt_grad_iters = plot_err_vs_iters(df_iters, N, "XXZ, N=$(N)")
plt_err_iters
plt_grad_iters

plot!(plt_err_iters, yscale=:log10)

for taumin in [1, 2, 3], N in [100]
    fit_tau_vs_err(df_iters, N, 100_000; taumin=taumin)
end

fit = fit_tau_vs_err(df_iters, N, 500_000; taumin=4)
errs_fit = df_iters[(df_iters.N .== N) .& (df_iters.M .== 100_000) .& .!isnan.(df_iters.err), :err]
errs_fit = exp10.(range(log10(minimum(errs_fit)), log10(maximum(errs_fit)), length=100))
plot!(plt_err_iters, errs_fit, fit.prefactor .* errs_fit .^ fit.alpha,
      label=L"\tau \propto \epsilon^{%$(round(fit.alpha, digits=3))}", color=:red, linestyle=:dash)





df_chi24 = extract_data(home*"xxz\\Jz2.5\\sweeps\\chi24\\parallel\\", [100, 200, 300]; include_checkpoints=false)
df_chi36 = extract_data(home*"xxz\\Jz2.5\\sweeps\\chi36\\parallel\\", [100, 200, 300]; include_checkpoints=false)
df_chi48 = extract_data(home*"xxz\\Jz2.5\\sweeps\\chi48\\parallel\\", [100, 200, 300]; include_checkpoints=false)

Nvals_compare = [100, 200, 300]
dfs_labels = [(df_chi24, "chi=24 parallel"), (df_chi36, "chi=36 parallel"), (df_chi48, "chi=48 parallel")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



df_chi2 = extract_data(home*"xxz\\Jz2.5\\sweeps\\chi24\\parallel\\", [100]; include_checkpoints=true)
df_chi10 = extract_data(home*"xxz\\Jz10\\chi20\\parallel\\", [100]; include_checkpoints=true)
df_chi20 = extract_data(home*"xxz\\Jz20\\chi20\\parallel\\", [100]; include_checkpoints=false)

Nvals_compare = [100]
dfs_labels = [(df_chi2, "Jz=2.5 parallel"), (df_chi10, "Jz=10 parallel"), (df_chi20, "Jz=20 parallel")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau





### CORRMPS ###

### N FIXED 

## SEQUENTIAL ##
# Actual paper results
xilist = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
dfs = [extract_data(home*"corrmps\\chi20_long\\xi$(xi)\\", Vector(100:100:300)) for xi in xilist]
combined_df = reduce(vcat, [insertcols!(copy(df), 1, :xi => xi) for (df, xi) in zip(dfs, xilist)])
CSV.write(home*"corrmps\\chi20_long\\combined_df.csv", combined_df)          # read by plots.ipynb
N = 100
df_N = combined_df[combined_df.N .== N, :]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau, plt_err_resc =
    extract_plots_xi(df_N, "N=$(N) comparison")

plt_err
plot!(plt_err, xlim=(1e-12, 1.2), ylim=(0.5,30), legend=:topright)
plt_err_resc
plot!(plt_err_resc, xlim=(1e-8, 60.2), ylim=(-0.2, 10.), legend=:bottomright)
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

savefig(plt_err, "testdata\\corrmps\\plots\\tau_vs_eps.png")
savefig(plt_err_resc, folder*"\\corrmps\\plots\\tau_resc_vs_eps.png")

plt_xi_N = extract_tau_vs_xi_fixederr(combined_df; err_ref=2e-1)
plt_xi_err = extract_tau_vs_xi_fixedN(combined_df; err_refs=[1e-1, 5e-2, 1e-2], N_fixed=100)
savefig(plt_xi_N, "testdata\\corrmps\\plots\\tau_vs_xi_fixederr.png")
savefig(plt_xi_err, "testdata\\corrmps\\plots\\tau_vs_xi_fixedN.png")



## PARALLEL ##

xilist = [0.5, 1.0, 2.0]
dfs = [extract_data(home*"corrmps\\chi20\\xi$(xi)\\parallel\\", Vector(100:100:100); include_checkpoints=true) for xi in xilist]
combined_df = reduce(vcat, [insertcols!(copy(df), 1, :xi => xi) for (df, xi) in zip(dfs, xilist)])
CSV.write(folder*"corrmps\\chi20\\combined_df_parallel.csv", combined_df)          # read by plots.ipynb
N = 100
df_N = combined_df[combined_df.N .== N, :]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau, plt_err_resc =
    extract_plots_xi(df_N, "N=$(N) comparison")

plt_err
plot!(plt_err, xlim=(1e-12, 1.2), ylim=(0.5,30), legend=:topright)


xi = 0.5
folder = home*"corrmps\\chi20\\xi$(xi)\\parallel\\"
df_iters = extract_data_iters(folder, [100], [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000, 300_000]; include_checkpoints=false)
N = 100
plt_err_iters, plt_grad_iters = plot_err_vs_iters(df_iters, N, "xi=$(xi), N=$(N)")
plt_err_iters
plt_grad_iters

plot!(plt_err_iters, yscale=:log10)

df_stops = extract_reltol_stops(folder, [100], [1e-1, 1e-2, 1e-3, 1e-4])
plt_err = plot_reltol_stops(df_stops, 100, "test")
plt_iter = plot_reltol_iters(df_stops, 100, "test")


df_seq = extract_data(home*"corrmps\\chi20_long\\xi$(xi)\\", [100]; include_checkpoints=true)
for label in [:time, :niter]
    df_seq[:, label] = cumsum(df_seq[:, label])
end
reltol_list = [1e-6]
df_par = extract_reltol_stops(folder, [100], reltol_list)
df_reltol_list = []
for reltol in reltol_list
    df_reltol = df_par[df_par.reltol .== reltol, :]
    rename!(df_reltol, :stop_iter => :niter)
    rename!(df_reltol, :err_stop => :err)
    rename!(df_reltol, :avg_time => :time)
    push!(df_reltol_list, df_reltol)
end

Nvals_compare = [100]
dfs_labels = vcat([(df_seq, "xi=$(xi) sequential")], [(df_reltol_list[j], "xi=$(xi) parallel $(reltol_list[j])") for j in eachindex(reltol_list)])

plt_err, plt_time_err, plt_time_tau, 
plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau


### XI FIXED

folder = home*"corrmps\\chi20_long\\xi4.0\\"
Nvals = [20, 50, 100, 200, 500, 1000]
df = extract_data(folder, Nvals; include_checkpoints=true)
CSV.write(folder*"df.csv", df)          # read by plots.ipynb
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "xi=4, chi=20")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

xilist = [4.0]
combined_df = reduce(vcat, [insertcols!(copy(df), 1, :xi => xi) for (df, xi) in zip([df], xilist)])
fidelity = 0.90
plt = extract_tau_vs_N_fixedfid(df, fidelity, "test")
plot!(plt, xscale=:log10)

#plt_xi_err = extract_tau_vs_xi_fixedN(combined_df; err_refs=[1e-1, 5e-2, 1e-2], N_fixed=100)


# CNOT ESTIMATE

g_from_xi(ξ::Real) = tanh(1 / (2ξ))
xi_from_g(g::Real) = 1 / abs(log((1 - g) / (1 + g)))

function correlated_mps(sites::Vector{<:Index}, g::Number;
                        L = Matrix(1.0I, 2, 2), R = Matrix(1.0I, 2, 2),
                        normalize::Bool = true)
    N = length(sites)
    N >= 2 || error("Need N ≥ 2")
    all(dim.(sites) .== 2) || error("All site indices must have dimension 2")
    D = 2
    T = promote_type(typeof(g), eltype(L), eltype(R), Float64)

    A = zeros(T, D, 2, D)                # A[a, i, b] = (A^{i-1})_{ab}
    A[:, 1, :] = [0 0; 1 1]              # A^0
    A[:, 2, :] = [1 g; 0 0]              # A^1

    links = [Index(D, "Link,l=$j") for j in 1:(N - 1)]
    psi = MPS(N)
    psi[1] = ITensor(T.(L), sites[1], links[1])
    for j in 2:(N - 1)
        psi[j] = ITensor(A, links[j - 1], sites[j], links[j])
    end
    psi[N] = ITensor(T.(R), links[N - 1], sites[N])

    normalize && normalize!(psi)
    return psi
end

correlated_mps_xi(sites, ξ::Real; kwargs...) = correlated_mps(sites, g_from_xi(ξ); kwargs...)

N = 200
psi = correlated_mps_xi(siteinds("S=1/2", N), 4.0)
qlist = [2, 4, 8, 16, 32]
nblocks = [mod(N, q) == 0 ? N÷q : N÷q+1 for q in qlist]
errs = blocking_scaling(psi, qlist)
errs_per_block = errs ./ nblocks
plot(qlist, errs_per_block, yscale=:log10, xlabel=L"q", ylabel=L"\epsilon/M", title="Blocking scaling for correlated MPS with ξ=4.0")

qs = Float64.(qlist[2:end])
y = log.(errs_per_block[2:end])
M = [ones(length(qs)) qs]
c = M\y                 # c = [log(A), -κ]
A, κ = exp(c[1]), -c[2]

γ = 2.0                    # fix by convention
ξ = γ / κ
println("A = $A, κ = γ/ξ = $κ, ξ = $ξ (with γ = $γ)")
plot!(qlist, A .* exp.(-κ*qlist), label=L"A e^{-q/ξ}", color=:red, linestyle=:dash)


function CNOT_estimate(q::Int)
    nswaps = max(0,q-2)
    cnot_swaps = 3*nswaps
    niso = max(0,q-2)
    cnot_iso = 10*niso
    return cnot_swaps + cnot_iso
end

function CNOT_estimate(F::Real, ξ::Real, Nvals::Vector{Int})
    err = 1 - sqrt(F)
    estimates = Int64[]
    q = 2
    for N in Nvals
        psi = correlated_mps_xi(siteinds("S=1/2", N), ξ)
        q = blocking_scaling_err(psi, err; qstart=q)
        push!(estimates, CNOT_estimate(q))
    end
    return estimates
end

Nvals = unique(round.(Int, exp10.(range(1, 3, length = 60))))     # log-spaced 10 ... 1000
estimates = CNOT_estimate(0.9, 4.0, Nvals)
plot(Nvals, estimates, xscale=:log10)
CSV.write("D:\\Julia\\MyProject\\InversionData\\testdata\\corrmps\\chi20_long\\xi4.0\\cnot_malz.csv",
          DataFrame(N = Nvals, T_cnot = estimates))          # read by plots.ipynb



### CSV exports read by plots.ipynb (single-panel figures 1-5) ###
its = [1000, 2000, 5000, 10_000, 20_000, 30_000, 50_000, 60_000, 70_000, 80_000, 100_000, 150_000, 200_000, 300_000, 500_000]
for chi in (24, 36, 48)          # XXZ Jz=2.5 parallel: error after M iterations (figs 1, 3)
    f = home*"xxz/Jz2.5/sweeps/chi$(chi)/parallel/"
    CSV.write(f*"df_iters.csv", extract_data_iters(f, [100, 200, 300], its; include_checkpoints=true))
    CSV.write(f*"df_reltol_stops.csv", extract_reltol_stops(f, [100, 200, 300], [1e-1, 1e-2, 1e-3, 1e-4]))
end

for xi in [0.5, 1.0, 2.0]
    f = home*"corrmps/chi20/xi$(xi)/parallel/"               # figs 4, 5: corrmps xi=1 parallel
    CSV.write(f*"df_iters.csv", extract_data_iters(f, [100], its; include_checkpoints=true))
    CSV.write(f*"df_reltol_stops.csv", extract_reltol_stops(f, [100], [1e-1, 1e-2, 1e-3, 1e-4]))
    f = home*"corrmps/chi20_long/xi$(xi)/"                    # fig 5: sequential (niter is cumulated in the notebook)
    CSV.write(f*"df_N100.csv", extract_data(f, [100]; include_checkpoints=true))
end

### Truncation of the ground states and Ising parallel runs, read by plots.ipynb (tau vs chi_trunc^2) ###
# eps = 1 - |<psi|psi_chi>| after an SVD truncation of the ground state to bond dimension chi
function truncation_eps(folder::String, Nvals; chis = nothing)
    rows = NamedTuple[]
    for N in Nvals
        isfile(folder*"$(N)_mps.h5") || continue
        psi = h5open(h -> read(h, "psi", MPS), folder*"$(N)_mps.h5", "r")
        psi = psi / norm(psi)
        for chi in (chis === nothing ? (1:maxlinkdim(psi)) : chis)
            t = truncate(psi; maxdim = chi, cutoff = 0.0)
            push!(rows, (N = N, chi = chi, eps = 1 - abs(inner(psi, t)) / norm(t)))
        end
    end
    return DataFrame(rows)
end
CSV.write(home*"xxz/Jz2.5/truncation_eps.csv", truncation_eps(home*"xxz/Jz2.5/", [60, 100, 140, 180, 200, 220, 260, 300]; chis = 1:24))
CSV.write(home*"ising/g0.75/m5_long/truncation_eps.csv", truncation_eps(home*"ising/g0.75/m5_long/", [60, 100, 140, 180, 200, 220, 260, 300]))
f = home*"ising/g0.75/m5_long/parallel/"
CSV.write(f*"df_iters.csv", extract_data_iters(f, [100, 200, 300],
          [1000, 2000, 5000, 10_000, 20_000, 30_000, 50_000, 60_000, 70_000, 100_000, 200_000, 300_000, 500_000]; include_checkpoints=true))
