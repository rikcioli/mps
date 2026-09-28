#using MKL
#include("rrules.jl")
#include("optFunctions.jl")
using ITensors, ITensorMPS
using LinearAlgebra
using JLD2, HDF5
using DataFrames, CSV
using LaTeXStrings
using Plots
using ColorSchemes


Base.@kwdef mutable struct InversionInstructions
    maxrank::Union{Nothing, Int} = nothing
    maxerror::Union{Nothing, Float64} = nothing
    atol::Float64 = 1e-8
    maxiter::Int = 1000000
    gradtol::Float64 = 1e-8
    n_checkpoint::Int = 5000
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


function entropy!(psi::MPS, b::Integer)  
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log2(p)
    end
    return SvN
end

function spectrum(psi::MPS, b::Integer)
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)

    spec = diag(Matrix{Float64}(S, inds(S)))
    return spec
end


function dagger(Uarray::Vector{<:AbstractMatrix})
    Udagger = adjoint.(reverse(Uarray))
    return Udagger
end

function H_spin(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hy::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jx,"Sx",j,"Sx",j+1
        os += Jy,"Sy",j,"Sy",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hx,"Sx",j
        os += hy,"Sy",j
        os += hz,"Sz",j
    end
    os += hx,"Sx",N
    os += hy,"Sy",N
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_spin(sites, Jxy::Real, Jz::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jxy/2,"S+",j,"S-",j+1
        os += Jxy/2,"S-",j,"S+",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hz,"Sz",j
    end
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_XY(sites, g::Real, hx::Real)
    return H_spin(sites, -(1+g), -(1-g), 0., hx, 0., 0.) 
end

function H_heisenberg(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hz::Real)
    return H_spin(sites, Jx, Jy, Jz, hx, 0., hz)
end

function H_heisenberg(sites, Jxy::Real, Jz::Real, hz::Real)
    return H_spin(sites, Jxy, Jz, hz)
end

function initialize_gs(H::MPO, sites; nsweeps = 5, maxdim = [10,20,100,100,200], cutoff = 1e-15, linkdims=2, kwargs...)
    psi0 = random_mps(ComplexF64, sites; linkdims=linkdims)
    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff,kwargs...)
    return energy, psi
end

function XXZ(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_heisenberg(sites, -1., -1., -0.5, -0.1, -0.1)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function XY(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_XY(sites, 0.0, 0.5)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function ising(N::Int)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5/2)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end

function transfer_matrix(mps::MPS)
    sites = siteinds(mps)
    Tlist = [conj(mps[i])' * delta(sites[i], sites[i]') * mps[i] for i in eachindex(mps)]
    return Tlist
end

function corr_length(ψ::MPS, b::Int)
    N = length(ψ)
    orthogonalize!(ψ, N)
    nn = b
    Tten = transfer_matrix(ψ)[nn]
    links = linkinds(ψ)
    cL = combiner(links[nn-1], links[nn-1]')
    cR = combiner(links[nn], links[nn]')
    Tten *= cL
    Tten *= cR
    Tmat = Matrix(Tten, combinedind(cL), combinedind(cR))
    Svals = real(eigvals(Tmat))
    ξ = -1/log(Svals[end-1])
    return ξ
end


function check_xi()
    corrs::Vector{Vector{Float64}} = []
    for Jz in 0.1:0.1:2.5
        Hamiltonian = H_heisenberg(sites, 1, 1, Jz, 0, 0)
        corr_J = Float64[]
        for chi in [10, 20, 30, 40, 50]
            psi0 = random_mps(ComplexF64, sites; linkdims=2)
            energy, psi = dmrg(Hamiltonian,psi0;nsweeps=10,maxdim=20)
            ξ = corr_length(psi, 20)
            push!(corr_J, ξ)
        end
        push!(corrs, corr_J)
    end
    return corrs
end


function find_max_finished_tau(folder::String, Nvals::Vector{Int})
    tauNvals = Tuple{Int,Int}[]

    for N in Nvals
        pattern = Regex("N$(N)_T(\\d+)\\.jld2")
        taus = [parse(Int, m.captures[1]) for f in readdir(folder)
                for m in [match(pattern, f)] if !isnothing(m)]
        isempty(taus) && continue

        for tau in sort(taus, rev=true)
            result = load_object(folder*"N$(N)_T$(tau).jld2")
            if get(result, :finished, true)
                push!(tauNvals, (tau, N))
                break
            end
        end
    end

    return tauNvals
end


function extract_data(folder::String, Nvals::Vector{Int})

    tauNvals = find_max_finished_tau(folder, Nvals)

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
        plot!(plt_time_err, subdata.err, subdata.time, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_time_tau, subdata.tau, subdata.time, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_niter_tau, subdata.tau, subdata.niter, label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_time1iter_tau, subdata.tau[2:end], subdata.time1iter[2:end], label=L"\xi="*"$(xi)", color=colors_time[pos])
        plot!(plt_costratio_tau, subdata.tau, subdata.costratio, label=L"\xi="*"$(xi)", color=colors_err[pos], yscale=:log10)
    end

    return [plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau]
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


# ERROR VS DEPTH FOR EACH N AND DIFFERENT MAXERRORS
N = 100
maxtau = 7
maxerrors = [1e-5, 1e-6, 1e-7, 1e-8]

f = h5open("testdata\\XY\\$(N)_mps.h5","r")
psi_og = read(f,"psi",MPS)
close(f)
sites = siteinds(psi_og)

zerostate = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zerostate, 1)

plterr = plot(yscale=:log, xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|", title="N=$(N)")
for maxerror in maxerrors
    err_vs_tau = Float64[]
    for tau in 1:maxtau
        results = load_object("testdata\\XY\\N$(N)_T$(tau)_E$(maxerror).jld2")
        err = results[:err]

        arrU = results[:arrU]
        psi = apply_brickwork(arrU, zerostate; 
                        normalize_after_layer = false,
                        trunc=(maxerror=1e-15,))
        reconstr_err = 1-abs(dot(psi, psi_og))
        @show abs(reconstr_err-err)
        push!(err_vs_tau, reconstr_err)
    end
    plot!(plterr, 1:maxtau, err_vs_tau, label=L"\mathrm{maxerror}="*"$(maxerror)")
end
plterr
savefig(plterr, "testdata\\XY\\plots\\N$(N)_error.png")


# ERROR VS DEPTH FOR DIFFERENT N, FIXED MAXERROR
maxtau = 8
maxerror = 1e-5

plt = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon", xflip=true, title="E=$(maxerror)", legend=:bottomright)
for N in [20, 40, 60, 80, 100]
    f = h5open("testdata\\XY\\$(N)_mps.h5","r")
    psi_og = read(f,"psi",MPS)
    close(f)
    sites = siteinds(psi_og)

    zerostate = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zerostate, 1)

    err_vs_tau = Float64[]
    for tau in 1:maxtau
        results = load_object("testdata\\XY\\N$(N)_T$(tau)_E$(maxerror).jld2")
        err = results[:err]

        arrU = results[:arrU]
        psi = apply_brickwork(arrU, zerostate; 
                        normalize_after_layer = false,
                        trunc=(maxerror=1e-15,))
        reconstr_err = 1-abs(dot(psi, psi_og))
        @show abs(reconstr_err-err)
        push!(err_vs_tau, reconstr_err)
    end
    plot!(plt, err_vs_tau, 1:maxtau, label=L"N="*"$(N)")
end
plt
savefig(plt, "testdata\\XY\\plots\\E$(maxerror)_error.png")


# ERROR VS N FOR DIFFERENT N, FIXED MAXERROR
maxtau = 8
maxerror = 1e-5
errvals = [0.1, 0.02, 0.004, 0.0008]

vals = []
for N in [20, 40, 60, 80, 100]
    f = h5open("testdata\\XY\\$(N)_mps.h5","r")
    psi_og = read(f,"psi",MPS)
    close(f)
    sites = siteinds(psi_og)

    zerostate = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zerostate, 1)

    err_vs_tau = Float64[]
    time_vs_tau = Float64[]
    for tau in 1:maxtau
        results = load_object("testdata\\XY\\N$(N)_T$(tau)_E$(maxerror).jld2")
        err = results[:err]
        time = results[:time]

        arrU = results[:arrU]
        psi = apply_brickwork(arrU, zerostate; 
                        normalize = false,
                        trunc=(maxerror=1e-12,))
        reconstr_err = 1-abs(dot(psi, psi_og))
        @show abs(reconstr_err-err)
        push!(err_vs_tau, reconstr_err)
        push!(time_vs_tau, time)
    end

    for err in errvals
        tau_err = findfirst(x -> x < err, err_vs_tau)
        push!(vals, (N, err, tau_err, time_vs_tau[tau_err]))
    end
end


colors = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=4)]
plt = plot(ylabel=L"\tau", xlabel=L"N", title="E=$(maxerror)", legend=:topleft, dpi=400)
for (pos, err) in enumerate(errvals)
    taus = [val[3] for val in vals if val[2] == err]
    plot!(plt, 20:20:100, taus, label=L"\epsilon="*"$(err)", color=colors[pos])
end
plt
savefig(plt, "testdata\\XY\\plots\\E$(maxerror)_tau_vs_N.png")


# TIME FOR SAME THING
colors = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=4)]
plt = plot(ylabel=L"t \ (s)", yscale = :log10, xlabel=L"N", title="E=$(maxerror)", legend=:topleft, dpi=400)
for (pos, err) in enumerate(errvals)
    times = [val[4] for val in vals if val[2] == err]
    plot!(plt, 20:20:100, times, label=L"\epsilon="*"$(err)", color=colors[pos])
end
plt
savefig(plt, "testdata\\XY\\plots\\E$(maxerror)_time_vs_N.png")






colors_Upsi = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=4)]
colors_zerostate = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.9, stop=0.1, length=4)]

N = 100
maxtau = 7
maxerrors = [1e-5, 1e-6, 1e-7, 1e-8]

f = h5open("testdata\\XY\\$(N)_mps.h5","r")
psi_og = read(f,"psi",MPS)
close(f)
sites = siteinds(psi_og)
chimax = maxlinkdim(psi_og)

zerostate = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zerostate, 1)

pltchi_psi = plot(1:maxtau, [chimax for _ in 1:maxtau], line=:dash, 
                            xlabel=L"\tau", ylabel=L"\chi", 
                            label=L"\psi", title="N=$(N)", dpi=400)
spectra_Upsi_maxerr = []
spectra_zerostate_maxerr = []
for (pos, maxerror) in enumerate(maxerrors)
    maxchis_Upsi = []
    maxchis_zerostate = []
    spectra_Upsi = []
    spectra_zerostate = []
    for i in 1:maxtau
        result = load_object("testdata\\XY\\N$(N)_T$(i)_E$(maxerror).jld2")
        arrU = result[:arrU]
        arrdg = dagger(arrU)

        orthogonalize!(psi_og, iseven(i) ? 1 : N)
        orthogonalize!(zerostate, iseven(i) ? 1 : N)

        state = apply_brickwork(arrdg, psi_og; 
                        shift = mod(i-1,2), 
                        to_right = iseven(i), 
                        normalize = true,
                        trunc=(maxerror=maxerror,))
        maxchi = maxlinkdim(state)
        push!(maxchis_Upsi, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_Upsi, spec)

        state = apply_brickwork(arrU, zerostate; 
                        normalize = true,
                        trunc=(maxerror=maxerror,))
        maxchi = maxlinkdim(state)
        push!(maxchis_zerostate, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_zerostate, spec)
    end
    push!(spectra_Upsi_maxerr, spectra_Upsi)
    push!(spectra_zerostate_maxerr, spectra_zerostate)
    plot!(pltchi_psi, 1:maxtau, maxchis_Upsi, color = colors_Upsi[pos], label=L"U^{\dagger}|\psi\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
    plot!(pltchi_psi, 1:maxtau, maxchis_zerostate, color = colors_zerostate[pos], label=L"U|0\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
end
pltchi_psi
savefig(pltchi_psi, "testdata\\XY\\plots\\bonddim_maxerr_trunc.png")


pltchi_psi = plot(1:maxtau, [chimax for _ in 1:maxtau], line=:dash, 
                            xlabel=L"\tau", ylabel=L"\chi", 
                            label=L"\psi", title="N=$(N)", dpi=400)
spectra_Upsi_maxerr = []
spectra_zerostate_maxerr = []
for (pos, maxerror) in enumerate(maxerrors)
    maxchis_Upsi = []
    maxchis_zerostate = []
    spectra_Upsi = []
    spectra_zerostate = []
    for i in 1:maxtau
        result = load_object("testdata\\XY\\N$(N)_T$(i)_E$(maxerror).jld2")
        arrU = result[:arrU]
        arrdg = dagger(arrU)

        orthogonalize!(psi_og, iseven(i) ? 1 : N)
        orthogonalize!(zerostate, iseven(i) ? 1 : N)

        state = apply_brickwork(arrdg, psi_og; 
                        shift = mod(i-1,2), 
                        to_right = iseven(i), 
                        normalize = true,
                        trunc=(maxerror=1e-12,))
        maxchi = maxlinkdim(state)
        push!(maxchis_Upsi, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_Upsi, spec)

        state = apply_brickwork(arrU, zerostate; 
                        normalize = true,
                        trunc=(maxerror=1e-12,))
        maxchi = maxlinkdim(state)
        push!(maxchis_zerostate, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_zerostate, spec)
    end
    push!(spectra_Upsi_maxerr, spectra_Upsi)
    push!(spectra_zerostate_maxerr, spectra_zerostate)
    plot!(pltchi_psi, 1:maxtau, maxchis_Upsi, color = colors_Upsi[pos], label=L"U^{\dagger}|\psi\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
    plot!(pltchi_psi, 1:maxtau, maxchis_zerostate, color = colors_zerostate[pos], label=L"U|0\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
end
pltchi_psi
savefig(pltchi_psi, "testdata\\XY\\plots\\bonddim_1e-12_trunc.png")








using ColorSchemes
colors = [get(ColorSchemes.inferno, t) for t in range(0.9, stop=0.1, length=8)]

specs = spectra_err[1]

spec_psi = spectrum(psi, div(N,2))
pltspec = plot(1:length(spec_psi), spec_psi, line=:dash, 
            xlabel=L"j", ylabel=L"\lambda_j", yscale=:log10, 
            ylimits = (4e-5,1.1), xlimits = (0, 15),
            dpi=400, palette=:inferno, label=L"\psi")
for tau in 1:8
    plot!(pltspec, 1:length(specs[tau]), specs[tau], color=colors[tau], label=L"\tau="*"$(tau)")
end
pltspec
savefig(pltspec, "testdata\\spectra_zero.png")


plt = plot(1:8, [entL for _ in 1:8], label=L"ψ", line=:dash, xlabel=L"\mathrm{Layer}", ylabel=L"S")
for (i, vals) in enumerate(results[:entsL])
    if isodd(i)
        plot!(plt, 1:2:length(vals), vals[1:2:end], label=L"\tau="*"$i", m=:circle)
    end
end
plt


plt_compare = plot(yscale=:log10, 
        ylabel=ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|",
        xlabel=L"\tau")
for maxerror in [1e-4, 1e-5, 1e-6, 1e-7]
    results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
    states = [apply_brickwork(circuit, zerostate) for circuit in results[:arrUs]]
    errs_true = [1 - abs(dot(state, psi)) for state in states]
    plot!(plt_compare, 1:8, results[:errs], line=:dash, label=L"\mathrm{maxerror}="*"$(maxerror)")
    plot!(plt_compare, errs_true, label=L"\mathrm{maxerror}="*"$(maxerror)")
end
plt_compare



### RANDMPS ###


# ERROR VS DEPTH FOR CHOSEN N, INVERT2
N = 80
maxtau = 20
folder = "mps1"

err_vs_tau = Float64[]
time_vs_tau = Float64[]
for tau in 1:maxtau
    results = load_object("testdata\\rand\\$(folder)\\N$(N)_T$(tau).jld2")
    push!(err_vs_tau, results[:err])
    push!(time_vs_tau, results[:time])
end

plterr = plot(yscale=:log, xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|", 
                title=L"\chi = 2, \ N="*"$(N)", dpi=400);
plot!(plterr, 1:maxtau, err_vs_tau)
#savefig(plterr, "testdata\\rand\\plots\\$(folder)\\eps_vs_tau.png")
plttime = plot(xlabel=L"\tau", ylabel=L"t \ (s)", title=L"\chi=2, \ N="*"$(N)", dpi=400);
plot!(plttime, 1:maxtau, time_vs_tau)
#savefig(plttime, "testdata\\ising\\plots\\$(folder)\\N$(N)_g1.5_time.png")

tau = 7
normgradplt = plot(dpi=400, title="mps1, N=$(N)", yscale=:log10);
ress = load_object("testdata\\rand\\$(folder)\\N$(N)_T$(tau).jld2");
plot!(normgradplt, ress.normgradhistory[:,2], label=L"$||\nabla f||, \tau = $"*"$(tau)")
plot!(normgradplt, ress.normgradhistory[:,1], label=L"$f, \tau = $"*"$(tau)")
#savefig(normgradplt, "testdata\\rand\\plots\\normgradhist.png")
using CSV
using DataFrames
df = DataFrame(ress.normgradhistory, :auto)
CSV.write("testdata\\rand\\$(folder)\\N$(N)_T$(tau).csv", df)


normgradplt = plot(dpi=400, title="mps1, N=$(N)", yscale=:log10);
niters = Int[]
for tau in 2:maxtau
    ress = load_object("testdata\\rand\\$(folder)\\N$(N)_T$(tau).jld2");
    push!(niters, length(ress.normgradhistory[:,2]))
    plot!(normgradplt, ress.normgradhistory[:,2], label=L"$||\nabla f||, \tau = $"*"$(tau)")
    #plot!(normgradplt, ress.normgradhistory[:,1], label=L"$f$, \tau = $"*"$(tau)")
end
normgradplt

iterplt = plot(dpi=400, title="mps1, N=$(N)", yscale=:log10, legend=:topleft);
plot!(iterplt, niters, xscale=:log10)
#plot!(iterplt, 1:8, 1e2*(1:8) .^2, label=L"\tau^2")
plot!(iterplt, 1:8, 1e2*(1:8) .^3, label=L"\tau^3")
plot!(iterplt, 1:8, 1e2*(1:8) .^4, label=L"\tau^4")




### ISING ###

# ERROR VS DEPTH FOR CHOSEN N, INVERT2
N = 128
sites = siteinds("Qubit", N)
zerostate = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zerostate, 1)

results = load_object("testdata\\N60_T7.jld2")

maxtau = 8

err_vs_tau = Float64[]
time_vs_tau = Float64[]
for tau in 1:maxtau
    results = load_object("testdata\\ising\\g1.5\\N$(N)_T$(tau).jld2")
    push!(err_vs_tau, results[:err])
    push!(time_vs_tau, results[:time])
end
plterr = plot(yscale=:log, xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|", 
                title=L"g=1.5, \ N=128", dpi=400)
plot!(plterr, 1:maxtau, err_vs_tau)
savefig(plterr, "testdata\\ising\\plots\\N$(N)_g1.5_error.png")
plttime = plot(xlabel=L"\tau", ylabel=L"t \ (s)", title=L"g=1.5, \ N=128", dpi=400)
plot!(plttime, 1:maxtau, time_vs_tau)
savefig(plttime, "testdata\\ising\\plots\\N$(N)_g1.5_time.png")


maxtau = 5

err_vs_tau = Float64[]
time_vs_tau = Float64[]
psi = load_object("testdata\\ising\\ising_L128_g1.0.jld2")
for tau in 1:maxtau
    results = load_object("testdata\\ising\\g1.0\\N$(N)_T$(tau).jld2")
    push!(err_vs_tau, results[:err])
    push!(time_vs_tau, results[:time])
end
plterr = plot(xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|", 
                title=L"g=1.0, \ N=128", yscale=:log, dpi=400)
plot!(plterr, 1:maxtau, err_vs_tau, ylim=(1e-1,1))
savefig(plterr, "testdata\\ising\\plots\\N$(N)_g1.0_error.png")
plttime = plot(xlabel=L"\tau", ylabel=L"t \ (s)", title=L"g=1.0, \ N=128", dpi=400)
plot!(plttime, 1:maxtau, time_vs_tau)
savefig(plttime, "testdata\\ising\\plots\\N$(N)_g1.0_time.png")



N = 128
maxtau = 8
maxerrors = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
colors_Upsi = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=length(maxerrors))]
colors_zerostate = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.9, stop=0.1, length=length(maxerrors))]


psi_og = load_object("testdata\\ising\\ising_L$(N)_g1.5.jld2")
sites = siteinds(psi_og)
chimax = maxlinkdim(psi_og)

zerostate = MPS(sites, ["0" for _ in 1:N])
orthogonalize!(zerostate, 1)

pltchi_psi = plot(1:maxtau, [chimax for _ in 1:maxtau], line=:dash, 
                            xlabel=L"\tau", ylabel=L"\chi", 
                            label=L"\psi", title=L"g=1.5, N=128", dpi=400)
spectra_Upsi_maxerr = []
spectra_zerostate_maxerr = []
for (pos, maxerror) in enumerate(maxerrors)
    maxchis_Upsi = []
    maxchis_zerostate = []
    spectra_Upsi = []
    spectra_zerostate = []
    for i in 1:maxtau
        result = load_object("testdata\\ising\\g1.5\\N$(N)_T$(i).jld2")
        arrU = result[:arrU]
        arrdg = dagger(arrU)

        orthogonalize!(psi_og, iseven(i) ? 1 : N)
        orthogonalize!(zerostate, iseven(i) ? 1 : N)

        state = apply_brickwork(arrdg, psi_og; 
                        shift = mod(i-1,2), 
                        to_right = iseven(i), 
                        normalize = false,
                        trunc=(maxerror=maxerror,))
        maxchi = maxlinkdim(state)
        push!(maxchis_Upsi, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_Upsi, spec)

        state = apply_brickwork(arrU, zerostate; 
                        normalize = false,
                        trunc=(maxerror=maxerror,))
        maxchi = maxlinkdim(state)
        push!(maxchis_zerostate, maxchi)
        spec = spectrum(state, div(N,2))
        push!(spectra_zerostate, spec)
    end
    push!(spectra_Upsi_maxerr, spectra_Upsi)
    push!(spectra_zerostate_maxerr, spectra_zerostate)
    plot!(pltchi_psi, 1:maxtau, maxchis_Upsi, color = colors_Upsi[pos], label=L"U^{\dagger}|\psi\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
    plot!(pltchi_psi, 1:maxtau, maxchis_zerostate, color = colors_zerostate[pos], label=L"U|0\rangle, \ \mathrm{maxerror}="*"$(maxerror)")
end
pltchi_psi
savefig(pltchi_psi, "testdata\\ising\\plots\\bonddim_g1.5.png")


# ERROR VS DEPTH FOR DIFFERENT N

folder = "testdata\\ising\\g1.5\\maxiter\\"

results = load_object(folder*"N60_T15.jld2")

for tau in 3:9
    results = load_object(folder*"N60_T$tau.jld2")
    CSV.write("$(folder)\\N60_T$tau.csv", DataFrame(results.normgradhistory, :auto))
end
instr = load_object(folder*"N60_T1_instructions.jld2")
tes = load_object("testdata\\ising\\ising_L128_g1.5.jld2")

colors_err = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=length(tauNvals))]
colors_time = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=length(tauNvals))]

plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon", 
                xflip=true, title="Ising g=3", legend=:bottomright, dpi=400)
plt_time_err = plot(xscale=:log, ylabel=L"t \ (s)", 
                xlabel=L"\epsilon", xflip=true, title="Ising g=3", 
                legend=:bottomright, dpi=400)
plt_time_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau", 
                title="Ising g=3", legend=:bottomright, dpi=400)
plt_E_err = plot(xscale=:log, ylabel=L"\langle H \rangle", xlabel=L"\epsilon", 
                xflip=true, title="Ising g=3", legend=:bottomright, dpi=400)
plt_E2_err = plot(xscale=:log, yscale=:log, ylabel=L"\langle H^2 \rangle - E^2", xlabel=L"\epsilon", 
                xflip=true, title="Ising g=3", legend=:topright, dpi=400)
plt_E2_tau = plot(yscale=:log, ylabel=L"\langle H^2 \rangle - E^2", xlabel=L"\tau", 
                title="Ising g=3", legend=:topright, dpi=400)
E_N_tau::Vector{Vector{Float64}} = []
E2_N_tau::Vector{Vector{Float64}} = []
for (pos, (maxtau, N)) in enumerate(tauNvals)
    f = h5open(folder*"$(N)_mps.h5","r")
    psi_og = read(f,"psi",MPS)
    close(f)
    psi_og = dense(psi_og)
    sites = siteinds(psi_og)

    zerostate = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zerostate, 1)

    err_vs_tau = Float64[]
    time_vs_tau = Float64[]

    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5)
    Etrue = real(ITensorMPS.inner(psi_og',H,psi_og))
    E2true = real(ITensorMPS.inner(H, psi_og, H, psi_og)) - Etrue^2
    E_vs_tau = Float64[]
    E2_vs_tau = Float64[]

    for tau in 1:maxtau
        results = load_object(folder*"N$(N)_T$(tau).jld2")
        @show N, tau
        err = results[:err]

        arrU = results[:arrU]
        psi, _ = apply_brickwork(arrU, zerostate; 
                        trunc=(maxrank=20,))
        E = real(ITensorMPS.inner(psi',H,psi))
        E2 = real(ITensorMPS.inner(H, psi, H, psi)) - E^2

        #reconstr_err = 1-abs(dot(psi, psi_og))
        #@show abs(reconstr_err-err)
        push!(err_vs_tau, err)
        push!(time_vs_tau, results[:time])
        push!(E_vs_tau, E)
        push!(E2_vs_tau, E2)
    end
    plot!(plt_err, err_vs_tau, 1:maxtau, label=L"N="*"$(N)", color=colors_err[pos])
    plot!(plt_time_err, err_vs_tau, time_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    plot!(plt_time_tau, 1:maxtau, time_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    plot!(plt_E_err, err_vs_tau, E_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    plot!(plt_E_err, [1e-7, 1], [Etrue, Etrue], line=:dash, color=colors_time[pos], label=L"E_{\mathrm{true}}")
    plot!(plt_E2_err, err_vs_tau, E2_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    plot!(plt_E2_err, [1e-7, 1], [E2true, E2true], line=:dash, color=colors_time[pos])
    plot!(plt_E2_tau, 1:maxtau, E2_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    push!(E_N_tau, E_vs_tau)
    push!(E2_N_tau, E2_vs_tau)
end
plt_E_err
plt_E2_err
plt_E2_tau
plt_err
plt_time_err
plt_time_tau
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")

err_range = [1/(10^i) for i in 1:6]
tau_range = err_range .^ (-1/4)
plot!(plt_err, err_range, tau_range, line=:dash, 
            yscale=:log10, label=L"\tau = \epsilon^{-1/4}")
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps_log.png")

err_range = [1/(10^i) for i in 1:6]
time_range = 1e2 .* err_range .^ (-1/4)
plot!(plt_time_err, err_range, time_range, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4}")
time_range_log = 4e-1 .* time_range .* (-log.(err_range))
plot!(plt_time_err, err_range, time_range_log, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4} \log(1/\epsilon)")

savefig(plt_time_err, "testdata\\xxz\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\xxz\\plots\\time_vs_tau.png")

results = [load_object(folder*"N60_T$(tau).jld2") for tau in 1:9]
gradnorms = [res[:gradnorm] for res in results]
resc_norms = [res[:gradnorm]/sqrt(length(res[:arrU])) for res in results]
converged = [res[:converged] for res in results]
costs = [res[:overlap_cost] for res in results]
plot(resc_norms)



folder = "testdata\\ising\\g1.5\\maxiter\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising, maxiter=20000, chi=10")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\ising\\g1.5\\maxiter_chi10\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising, maxiter=20000, chi=10")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\ising\\g1.5\\maxiter_chi30\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising, maxiter=20000, chi=30")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\ising\\g1.5\\nolim\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising nolim")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau

res = load_object(folder*"N60_T6.jld2")



folder = "testdata\\ising\\g1.5\\m50_nolim\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising m=50 nolim")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\ising\\g1.5\\"
Nvals = Vector(20:40:20)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "Ising test")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\ising\\g1.5\\m5\\"
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "Ising g=1.5, m=5, chi=10")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

savefig(plt_err, "testdata\\ising\\plots\\tau_vs_eps.png")
savefig(plt_time_err, "testdata\\ising\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\ising\\plots\\time_vs_tau.png")
savefig(plt_niter_tau, "testdata\\ising\\plots\\niter_vs_tau.png")
savefig(plt_time1iter_tau, "testdata\\ising\\plots\\time1iter_vs_tau.png")
savefig(plt_costratio_tau, "testdata\\ising\\plots\\costratio_vs_tau.png")


folder = "testdata\\ising\\g1.5\\m5_driver\\"
Nvals = Vector(60:40:300)
df2 = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df2, "Ising g=1.5, m=5, chi=10")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



df_m5      = extract_data("testdata\\ising\\g1.5\\m5_driver\\", Vector(60:40:300))
df_m5_fix = extract_data("testdata\\ising\\g1.5\\m5_e-15_driver\\", Vector(60:40:300))

Nvals_compare = [60, 100, 140, 180, 220, 260, 300]
dfs_labels = [(df_m5, "atol 1e-8"), (df_m5_fix, "atol 1e-15")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "Ising g=1.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau


### XY ###


# ERROR VS DEPTH FOR DIFFERENT N

tauNvals = [(25,20), (19,40), (15,60), (14,80), (13,100)]

colors_err = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=5)]
colors_time = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=5)]

plt_err = plot(xscale=:log, ylabel=L"\tau", xlabel=L"\epsilon", 
                xflip=true, title="XY", legend=:bottomright, dpi=400)
plt_time_err = plot(xscale=:log, ylabel=L"t \ (s)", 
                xlabel=L"\epsilon", xflip=true, title="XY", 
                legend=:bottomright, dpi=400)
plt_time_tau = plot(ylabel=L"t \ (s)", xlabel=L"\tau", 
                title="XY", legend=:bottomright, dpi=400)
for (pos, (maxtau, N)) in enumerate(tauNvals)
    f = h5open("testdata\\XY\\$(N)_mps.h5","r")
    psi_og = read(f,"psi",MPS)
    close(f)
    sites = siteinds(psi_og)

    zerostate = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zerostate, 1)

    err_vs_tau = Float64[]
    time_vs_tau = Float64[]

    for tau in 1:maxtau
        results = load_object("testdata\\XY\\N$(N)_T$(tau).jld2")
        cost = results[:cost]
        err = 1-exp(-sum(cost))

        #arrU = results[:arrU]
        #psi = apply_brickwork(arrU, zerostate; 
        #                normalize = false,
        #                trunc=(maxerror=1e-15,))
        #reconstr_err = 1-abs(dot(psi, psi_og))
        #@show abs(reconstr_err-err)
        push!(err_vs_tau, err)
        push!(time_vs_tau, results[:time])
    end
    plot!(plt_err, err_vs_tau, 1:maxtau, label=L"N="*"$(N)", color=colors_err[pos])
    plot!(plt_time_err, err_vs_tau, time_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
    plot!(plt_time_tau, 1:maxtau, time_vs_tau, label=L"N="*"$(N)", color=colors_time[pos])
end
plt_err
plt_time_err
plt_time_tau
savefig(plt_err, "testdata\\XY\\plots\\tau_vs_eps.png")

err_range = [1/(10^i) for i in 1:6]
tau_range = err_range .^ (-1/4)
plot!(plt_err, err_range, tau_range, line=:dash, 
            yscale=:log10, label=L"\tau = \epsilon^{-1/4}")
savefig(plt_err, "testdata\\XY\\plots\\tau_vs_eps_log.png")

err_range = [1/(10^i) for i in 1:6]
time_range = 1e2 .* err_range .^ (-1/4)
plot!(plt_time_err, err_range, time_range, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4}")
time_range_log = 4e-1 .* time_range .* (-log.(err_range))
plot!(plt_time_err, err_range, time_range_log, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4} \log(1/\epsilon)")

savefig(plt_time_err, "testdata\\XY\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\XY\\plots\\time_vs_tau.png")


ress = load_object("testdata\\XY\\N100_T12.jld2")
normgradplt = plot(ress.normgradhistory[:,2], label=L"$||\nabla f||$", dpi=400)
plot!(normgradplt, ress.normgradhistory[:,1], label=L"$f$", yscale=:log10, title="XY, N=100, depth=12")
savefig(normgradplt, "testdata\\XY\\plots\\normgradhist.png")

ress = load_object("testdata\\ising\\g1.5\\N128_T8.jld2")
normgradplt = plot(ress.normgradhistory[:,2], label=L"$||\nabla f||$", dpi=400)
plot!(normgradplt, ress.normgradhistory[:,1], label=L"$f$", yscale=:log10, title="Ising, N=128, depth=8")
savefig(normgradplt, "testdata\\ising\\plots\\normgradhist.png")

# ERROR VS N FOR DIFFERENT N, FIXED MAXERROR
errvals = [0.1, 0.02, 0.004, 0.0008]

vals = []
for (maxtau, N) in tauNvals
    f = h5open("testdata\\XY\\$(N)_mps.h5","r")
    psi_og = read(f,"psi",MPS)
    close(f)
    sites = siteinds(psi_og)

    zerostate = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zerostate, 1)

    err_vs_tau = Float64[]
    time_vs_tau = Float64[]
    for tau in 1:maxtau
        results = load_object("testdata\\XY\\N$(N)_T$(tau).jld2")
        err = results[:err]
        time = results[:time]

        # arrU = results[:arrU]
        # psi = apply_brickwork(arrU, zerostate; 
        #                 normalize = false,
        #                 trunc=(maxerror=1e-12,))
        # reconstr_err = 1-abs(dot(psi, psi_og))
        # @show abs(reconstr_err-err)
        push!(err_vs_tau, err)
        push!(time_vs_tau, time)
    end

    for err in errvals
        tau_err = findfirst(x -> x < err, err_vs_tau)
        push!(vals, (N, err, tau_err, time_vs_tau[tau_err]))
    end
end


colors = [get(ColorSchemes.inferno, t) for t in range(0.7, stop=0.1, length=4)]
plt = plot(ylabel=L"\tau", xlabel=L"N", title="XY", legend=:topleft, dpi=400)
for (pos, err) in enumerate(errvals)
    taus = [val[3] for val in vals if val[2] == err]
    plot!(plt, 20:20:100, taus, label=L"\epsilon="*"$(err)", color=colors[pos])
end
plt
savefig(plt, "testdata\\XY\\plots\\tau_vs_N.png")


# TIME FOR SAME THING
colors = [get(ColorSchemes.bluegreenyellow, t) for t in range(0.7, stop=0.1, length=4)]
plt = plot(ylabel=L"t \ (s)", xlabel=L"N", legend=:topleft, dpi=400)
for (pos, err) in enumerate(errvals)
    times = [val[4] for val in vals if val[2] == err]
    plot!(plt, 20:20:100, times, label=L"\epsilon="*"$(err)", color=colors[pos])
end
plt
savefig(plt, "testdata\\XY\\plots\\time_vs_N.png")






results = check_xi()

resh_res = [[res[i] for res in results] for i in 1:5]

plt = plot(title="XXZ, N=40", legend=:topleft, dpi=400)
for i in 1:5
    plot!(plt, resh_res[i], xlabel=L"Jz", ylabel=L"\xi", label=L"\chi="*"$(i*10)")
end
plt



# XXZ

# ERROR VS DEPTH FOR DIFFERENT N

folder = "testdata\\xxz\\Jz2.5\\maxiter\\"
res = load_object(folder*"N60_T14.jld2");
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "XXZ, maxiter=20000")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")

err_range = [1/(10^i) for i in 1:6]
tau_range = err_range .^ (-1/4)
plot!(plt_err, err_range, tau_range, line=:dash, 
            yscale=:log10, label=L"\tau = \epsilon^{-1/4}")
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps_log.png")

err_range = [1/(10^i) for i in 1:6]
time_range = 1e2 .* err_range .^ (-1/4)
plot!(plt_time_err, err_range, time_range, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4}")
time_range_log = 4e-1 .* time_range .* (-log.(err_range))
plot!(plt_time_err, err_range, time_range_log, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4} \log(1/\epsilon)")

savefig(plt_time_err, "testdata\\xxz\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\xxz\\plots\\time_vs_tau.png")



folder = "testdata\\xxz\\Jz2.5\\nolim\\"
res = load_object(folder*"N60_T7.jld2");
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "XXZ nolim")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")

err_range = [1/(10^i) for i in 1:6]
tau_range = err_range .^ (-1/4)
plot!(plt_err, err_range, tau_range, line=:dash, 
            yscale=:log10, label=L"\tau = \epsilon^{-1/4}")
savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps_log.png")

err_range = [1/(10^i) for i in 1:6]
time_range = 1e2 .* err_range .^ (-1/4)
plot!(plt_time_err, err_range, time_range, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4}")
time_range_log = 4e-1 .* time_range .* (-log.(err_range))
plot!(plt_time_err, err_range, time_range_log, line=:dash, 
            yscale=:log10, label=L"t = \epsilon^{-1/4} \log(1/\epsilon)")




folder = "testdata\\xxz\\Jz2.5\\m50_nolim_rescaled\\"
res = load_object(folder*"N60_T7.jld2");
Nvals = Vector(60:40:300)
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "XXZ m=50 nolim")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau

res = load_object(folder*"N60_T7.jld2")



folder = "testdata\\xxz\\Jz2.5\\PreAnalyticSVDPullback\\m50_nolim\\"
res = load_object(folder*"N60_T6.jld2");
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau = extract_plots(df, "XXZ m=50 nolim")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau



folder = "testdata\\xxz\\Jz2.5\\m50\\"
Nvals = Vector(60:40:300)
res = load_object(folder*"N60_T12.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ m=50")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\xxz\\Jz2.5\\m50_chi30\\"
Nvals = Vector(60:40:300)
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ m=50, chi=30")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\xxz\\Jz2.5\\m50_chi40\\"
Nvals = Vector(60:40:300)
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ m=50, chi=40")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



# COMPARE m50, m50_chi30, m50_chi40 ON THE SAME PLOTS, FOR A FEW SELECTED N

df_m50      = extract_data("testdata\\xxz\\Jz2.5\\m50\\", Vector(60:40:300))
df_m50_chi30 = extract_data("testdata\\xxz\\Jz2.5\\m50_chi30\\", Vector(60:40:300))
df_m50_chi40 = extract_data("testdata\\xxz\\Jz2.5\\m50_chi40\\", Vector(60:40:300))

Nvals_compare = [60, 100, 140]
dfs_labels = [(df_m50, "chi=24"), (df_m50_chi30, "chi=30"), (df_m50_chi40, "chi=40")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")
savefig(plt_time_err, "testdata\\xxz\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\xxz\\plots\\time_vs_tau.png")
savefig(plt_niter_tau, "testdata\\xxz\\plots\\niter_vs_tau.png")
savefig(plt_time1iter_tau, "testdata\\xxz\\plots\\time1iter_vs_tau.png")
savefig(plt_costratio_tau, "testdata\\xxz\\plots\\costratio_vs_tau.png")


df_m50      = extract_data("testdata\\xxz\\Jz2.5\\m50\\", Vector(60:40:300))
df_m20_7 = extract_data("testdata\\xxz\\Jz2.5\\m20_7\\", Vector(60:40:300))
df_m20_dist = extract_data("testdata\\xxz\\Jz2.5\\m20_dist\\", Vector(60:40:300))

Nvals_compare = [60, 100, 140]
dfs_labels = [(df_m50, "m=50, chi=24"), (df_m20_7, "m=20 7 threads, chi=24"), (df_m20_dist, "m=20 dist, chi=24")]
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\xxz\\Jz2.5\\m20_og_168\\"
Nvals = Vector(60:40:300)
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df_m20 = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df_m20, "XXZ Jz=2.5, chi=24, m=20")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")
savefig(plt_time_err, "testdata\\xxz\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\xxz\\plots\\time_vs_tau.png")
savefig(plt_niter_tau, "testdata\\xxz\\plots\\niter_vs_tau.png")
savefig(plt_time1iter_tau, "testdata\\xxz\\plots\\time1iter_vs_tau.png")
savefig(plt_costratio_tau, "testdata\\xxz\\plots\\costratio_vs_tau.png")




folder = "testdata\\xxz\\Jz2.5\\m20_7\\"
Nvals = Vector(60:40:300)
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df_m20 = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df_m20, "XXZ Jz=2.5, chi=24, m=20")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\xxz\\Jz2.5\\m20_dist\\"
Nvals = Vector(60:40:300)
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "XXZ m=20 dist, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau




df_m50      = extract_data("testdata\\xxz\\Jz2.5\\m50\\", Vector(60:40:300))
df_m20_og_168 = extract_data("testdata\\xxz\\Jz2.5\\m20_og_168\\", Vector(60:40:300))

Nvals_compare = [60, 100, 140, 180, 220, 260, 300]
dfs_labels = [(df_m50, "m=50, chi=24"), (df_m20_og_168, "m=20, chi=24")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau
plot!(plt_costratio_tau, legend=:bottomright)

savefig(plt_err, "testdata\\xxz\\plots\\tau_vs_eps.png")
savefig(plt_time_err, "testdata\\xxz\\plots\\time_vs_eps.png")
savefig(plt_time_tau, "testdata\\xxz\\plots\\time_vs_tau.png")
savefig(plt_niter_tau, "testdata\\xxz\\plots\\niter_vs_tau.png")
savefig(plt_time1iter_tau, "testdata\\xxz\\plots\\time1iter_vs_tau.png")
savefig(plt_costratio_tau, "testdata\\xxz\\plots\\costratio_vs_tau.png")


df_m50      = extract_data("testdata\\xxz\\Jz2.5\\m50\\", Vector(60:40:300))
df_m20_gc1 = extract_data("testdata\\xxz\\Jz2.5\\m20_gc1\\", Vector(60:40:300))
df_m20_gc4 = extract_data("testdata\\xxz\\Jz2.5\\m20_gc4\\", Vector(60:40:300))

Nvals_compare = [60, 100, 140]
dfs_labels = [(df_m50, "m=50, chi=24"), (df_m20_gc1, "m=20 gc1, chi=24"), (df_m20_gc4, "m=20 gc4, chi=24")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



df_m50      = extract_data("testdata\\xxz\\Jz2.5\\m50\\", Vector(60:40:300))
df_m20_7 = extract_data("testdata\\xxz\\Jz2.5\\m20_7\\", Vector(60:40:300))
df_m20_dr = extract_data("testdata\\xxz\\Jz2.5\\m20_driver\\", Vector(60:40:300))

Nvals_compare = [180, 220, 260]
dfs_labels = [(df_m50, "m=50, chi=24"), (df_m20_7, "m=20_7, chi=24"), (df_m20_dr, "m=20 driver, chi=24")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "XXZ Jz=2.5 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

savefig(plt_time1iter_tau, "testdata\\xxz\\plots\\driver_comparison.png")


### CORRMPS

folder = "testdata\\corrmps\\xi10\\"
Nvals = [300]
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "xi=10, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\corrmps\\run1\\xi10\\"
Nvals = [300]
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "xi=10, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



folder = "testdata\\corrmps\\chi10\\xi10.0\\"
Nvals = [100, 200, 300]
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "xi=10, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau



df_chi10_xi10 = extract_data("testdata\\corrmps\\chi10\\xi10.0\\", Vector(100:100:300))
df_chi20_xi10 = extract_data("testdata\\corrmps\\chi20\\xi10.0\\", Vector(100:100:300))
Nvals_compare = [100, 200, 300]
dfs_labels = [(df_chi10_xi10, "chi=10"), (df_chi20_xi10, "chi=20")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "xi=10 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

CSV.write("testdata\\corrmps\\xi10_chi10.csv", df_chi10_xi10)
CSV.write("testdata\\corrmps\\xi10_chi20.csv", df_chi20_xi10)



df = extract_data("testdata\\corrmps\\chi20\\xi4.0\\", [50, 100, 200])
df_fix = extract_data("testdata\\corrmps\\chi20_fixatol\\xi4.0\\", [50, 100, 200])
Nvals_compare = [50, 100, 200]
dfs_labels = [(df, "1e-8"), (df_fix, "1e-15")]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_compare(dfs_labels, Nvals_compare, "driver comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau


N=100
xilist = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
dfs = [extract_data("testdata\\corrmps\\chi20\\xi$(xi)\\", Vector(100:100:300)) for xi in xilist]
combined_df = reduce(vcat, [insertcols!(copy(df), 1, :xi => xi) for (df, xi) in zip(dfs, xilist)])
df_N = combined_df[combined_df.N .== N, :]

plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau =
    extract_plots_xi(df_N, "N=100 comparison")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau

plot!(plt_err, xlim=(1e-12, 1.2), ylim=(0.5,20), legend=:topright)
savefig(plt_err, "testdata\\corrmps\\plots\\tau_vs_eps.png")

plt_xi_N = extract_tau_vs_xi_fixederr(combined_df; err_ref=1e-1)
plt_xi_err = extract_tau_vs_xi_fixedN(combined_df; err_refs=[1e-1, 5e-2, 1e-2], N_fixed=100)
savefig(plt_xi_N, "testdata\\corrmps\\plots\\tau_vs_xi_fixederr.png")
savefig(plt_xi_err, "testdata\\corrmps\\plots\\tau_vs_xi_fixedN.png")



folder = "testdata\\corrmps\\chi10\\xi4.0\\"
Nvals = [10, 20, 50, 100, 200, 500, 1000]
#res = load_object(folder*"N60_T9.jld2");
#CSV.write("$(folder)\\N60_T10.csv", DataFrame(res.normgradhistory, :auto))
df = extract_data(folder, Nvals)
plt_err, plt_time_err, plt_time_tau, plt_niter_tau, plt_time1iter_tau, plt_costratio_tau = extract_plots(df, "xi=10, chi=24")

plt_err
plt_time_err
plt_time_tau
plt_niter_tau
plt_time1iter_tau
plt_costratio_tau






###



ress = load_object("testdata\\xxz\\Jz2.5\\N300_T11.jld2")
normgradplt = plot(ress.normgradhistory[:,2], label=L"$||\nabla f||$", dpi=400)
plot!(normgradplt, ress.normgradhistory[:,1], label=L"$f$", yscale=:log10, title="XXZ, N=300, depth=11")
savefig(normgradplt, "testdata\\xxz\\plots\\normgradhist.png")



folder = "D:\\Julia\\MyProject\\Data\\xxz"
df = CSV.read(folder*"\\df_all.csv", DataFrame)

Ns = sort(unique(df.N))

plt_depth_fid = plot(ylabel=L"\tau", xlabel=L"\epsilon", 
                xflip=true, xscale=:log, title="XXZ", legend=:bottomright, dpi=400)
for N in Ns
    sub = sort(df[df.N .== N, :], :depth)
    plot!(plt_depth_fid, 1 .-sqrt.(sub.fid), sub.depth, marker=:circle, label="N=$N")
end
plt_depth_fid
#savefig(plt_depth_fid, folder*"\\plots\\depth_vs_fid.png")

plt_depth_fid = plot(ylabel=L"\tau", xlabel=L"\epsilon", 
                xflip=true, xscale=:log, yscale=:log, title="XXZ", legend=:bottomright, dpi=400)
for N in Ns
    sub = sort(df[df.N .== N .&& df.depth .!= 0, :], :depth)
    plot!(plt_depth_fid, 1 .-sqrt.(sub.fid), sub.depth, marker=:circle, label="N=$N")
end
plt_depth_fid
err_range = [1/(10^i) for i in 1:6]
tau_range = err_range .^ (-1/4)
plot!(plt_depth_fid, err_range, tau_range, line=:dash, 
            yscale=:log10, label=L"\tau = \epsilon^{-1/4}")


plt_time_fid = plot(xlabel="time", ylabel="fidelity", title="XXZ", xscale=:log10, dpi=400)
for N in Ns
    sub = sort(df[df.N .== N, :], :time)
    plot!(plt_time_fid, sub.time, sub.fid, marker=:circle, label="N=$N")
end
savefig(plt_time_fid, folder*"\\plots\\time_vs_fid.png")




function load_last_tau_results(folder::String, Nvals::Vector{Int})
    results = []
    for N in Nvals
        pattern = Regex("^N$(N)_T(\\d+)\\_checkpoint_instructions.jld2\$")
        taus = [parse(Int, m.captures[1]) for f in readdir(folder)
                for m in [match(pattern, f)] if !isnothing(m)]
        isempty(taus) && continue

        last_tau = maximum(taus)
        push!(results, load_object(folder*"N$(N)_T$(last_tau).jld2"))
    end
    return results
end

folder = "D:\\Julia\\MyProject\\mps\\testdata\\xxz\\Jz2.5\\cursed\\"
Nvals = Vector(60:40:300)  # fill in the N values to load
results = load_last_tau_results(folder, Nvals)

plot(results[7].normgradhistory, yscale=:log10)