#using MKL
include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using OptimKit
using Zygote
using LinearAlgebra
using JLD2
#using LaTeXStrings
#using Plots

using Logging
Logging.disable_logging(Logging.Warn)



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

function H_spin(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hy::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jx,"Sx",j,"Sz",j+1
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

function H_XY(sites, g::Real, hx::Real)
    return H_spin(sites, -(1+g), -(1-g), 0., hx, 0., 0.) 
end

function H_heisenberg(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hz::Real)
    return H_spin(sites, Jx, Jy, Jz, hx, 0., hz)
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

"Extends m x n isometry to M x M unitary, where M is the power of 2 which bounds max(m, n) from above"
function iso_to_unitary(V::AbstractArray)
    nrows, ncols = size(V, 1), size(V, 2)
    dagger = false
    if ncols > nrows
        V = V'
        nrows, ncols = ncols, nrows
        dagger = true
    end
    V = V[:, vec(sum(abs.(V), dims=1) .> 1E-10)]
    nrows, ncols = size(V, 1), size(V, 2)

    bitlength = length(digits(nrows-1, base=2))     # find number of sites of dim 2
    D = 2^bitlength

    U = zeros(ComplexF64, (D, D))
    U[1:nrows, 1:ncols] = V
    kerU = nullspace(U')
    U[:, ncols+1:D] = kerU

    if dagger
        U = copy(U')
    end

    return U
end

"Extract isometries from bond dimension-1 MPS and convert them to depth-1 brickwork circuit"
function to_layer(ψ::MPS)
    N = length(ψ)
    @assert maxlinkdim(ψ)==1
    orthogonalize!(ψ, N)
    sites = siteinds(ψ)
    links = linkinds(ψ)

    combiners1 = [combiner((sites[1], links[1]))]
    combiners = [combiner([sites[j]; links[j-1:j]]) for j in 2:N-1]
    combinersN = [combiner((sites[N], links[N-1]))]
    combiners = [combiners1; combiners; combinersN]

    combinedinds = [combinedind(comb) for comb in combiners]
    tensor_list = [combiners[j]*ψ[j] for j in 1:N]

    Vlist = [Array{ComplexF64}(tensor_list[j], combinedinds[j]) for j in 1:N]

    Ulist = [iso_to_unitary(V) for V in Vlist]
    layer = [kron(Ulist[2*j], Ulist[2*j-1]) for j in 1:div(N,2)]
    return layer
end




function invert(ψ::MPS, maxtau::Int; pathname = "", trunc=NamedTuple(), reuse_previous=false, warm_start::Union{Vector{<:AbstractMatrix}, Nothing} = nothing)
    N = length(ψ)
    sites = siteinds(ψ)
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    # Captures ψ, zeromps and kwargs
    cost_function = (arrU) -> begin
        ϕ = apply_brickwork(arrU, zeromps; trunc=trunc)
        return -real(sproduct(ψ, ϕ))
    end

    # Combines function and projected gradient
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end


    arrUs = [[zeros(ComplexF64, 4, 4) for _ in 1:n_unitaries(N, tau)] for tau in 1:maxtau]
    errs = zeros(maxtau)
    times = zeros(maxtau)
    bonddims = zeros(Int64, maxtau)
    entsL = [zeros(tau) for tau in 1:maxtau]
    entsR = [zeros(tau) for tau in 1:maxtau]
    arrU0 = isnothing(warm_start) ? Matrix{ComplexF64}[] : warm_start

    for tau in 1:maxtau
        @show tau
        nU_tau = n_unitaries_layer(N, tau)
        if reuse_previous 
            if tau>1 || isnothing(warm_start)  # add a layer of unitaries close to identity to increase the circuit depth
                Vs = skew([randn(ComplexF64, 4, 4) for _ in 1:nU_tau])
                newU = [retract(Matrix{ComplexF64}(I, (4,4)), V, 0.01)[1] for V in Vs]
                arrU0 = vcat(arrU0, newU)
            end
        else
            arrU0 = [random_unitary(4) for _ in 1:n_unitaries(N, tau)]
        end

        m = 5
        algorithm = LBFGS(m;maxiter = 10000, gradtol = 1e-8, verbosity = 2)

        # optimize and store results
        time_tau = @elapsed arrUmin, neg_overlap, gradmin, numfg, normgradhistory = 
                                                            optimize(fg, arrU0, algorithm; 
                                                                retract = retract, 
                                                                transport! = transport!, 
                                                                isometrictransport = true, 
                                                                inner = inner)
        
        if reuse_previous
            arrU0 = arrUmin
        end

        currentU = 1
        entsL_tau = zeros(tau)
        entsR_tau = zeros(tau)
        state = zeromps
        for i in 1:tau
            nUlayer = n_unitaries_layer(N, i)
            state = apply_brickwork(arrUmin[currentU : currentU+nUlayer-1], state; 
                                                    to_right=isodd(i), 
                                                    trunc=trunc)

            state_copy = copy(state)
            entL = entropy!(state_copy, div(N,2))
            entR = entropy!(state_copy, div(N,2)+1)
            @show i, entL, entR
            entsL_tau[i] += entL
            entsR_tau[i] += entR
            currentU += nUlayer
        end
        err_tau = 1+neg_overlap
        bonddim_tau = maximum(linkdims(state))
        result_tau = (N=N, tau=tau, trunc=trunc, time=time_tau, 
                        err=err_tau, bonddim=bonddim_tau, arrU=arrUmin, 
                        entsL=entsL_tau, entsR=entsR_tau,
                        gradmin=gradmin, niter=numfg, normgradhistory=normgradhistory)

        maxerror = trunc[:maxerror]
        save_object(pathname*"N$(N)_T$(tau)_E$(maxerror).jld2", result_tau)

        arrUs[tau] .+= arrUmin
        entsL[tau] .+= entsL_tau
        entsR[tau] .+= entsR_tau
        errs[tau] = err_tau
        bonddims[tau] = bonddim_tau
        times[tau] = time_tau
    end

    result = (N=N, maxtau=maxtau, trunc=trunc, times=times,
    errs=errs, bonddims=bonddims, arrUs=arrUs, entsL=entsL, entsR=entsR)

    return result
end

function runinversion(N, maxtau, maxerror, pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/xy/g0h0.5/")
    #pathname = "testdata/"
    f = h5open(pathname*"$(N)_mps.h5","r")
    psi = read(f,"psi",MPS)
    close(f)
    psi = XY(N)[2]
    orthogonalize!(psi, 1)
    psi_cut = move_center(psi, N; trunc=(maxrank=1,), normalize=true)
    @show maxlinkdim(psi)
    @show norm(psi_cut)
    @show dot(psi_cut, psi)
    @show maxlinkdim(psi_cut)

    # Extract the U_start from the truncated mps
    U_start = to_layer(psi_cut)

    # We add some random noise to help escaping the saddle point
    Vs = skew([randn(ComplexF64, 4, 4) for _ in eachindex(U_start)])
    newU = [retract(Matrix{ComplexF64}(I, (4,4)), V, 0.01)[1] for V in Vs]
    U_start = newU .* U_start

    result = invert(psi, maxtau; pathname=pathname, trunc=(maxerror=maxerror,), warm_start=U_start, reuse_previous=true)
    save_object(pathname*"N$(N)_E$(maxerror).jld2", result)
end

function dagger(Uarray::Vector{<:AbstractMatrix})
    Udagger = adjoint.(reverse(Uarray))
    return Udagger
end


let
    Nlist = [80, 100]
    maxerrorlist = [1e-5, 1e-6, 1e-7, 1e-8]
    pairs = collect(Iterators.product(Nlist, maxerrorlist))
    for (N, maxerror) in pairs
        runinversion(N, 12, maxerror)
    end
end



# N = 50
# ### #sites = siteinds("Qubit", N)
# ### #psi = random_mps(ComplexF64, sites; linkdims=2)
# energy, psi = XXZ(N)
# entL = entropy!(psi, div(N,2))
# entR = entropy!(psi, div(N,2)+1)
# chimax = maxlinkdim(psi)
# 
# 
# zerostate = MPS(siteinds(psi), ["0" for _ in 1:N])
# orthogonalize!(zerostate, 1)
# 
# plterr = plot(yscale=:log10, xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|")
# for maxerror in [1e-4, 1e-5, 1e-6, 1e-7]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     plot!(plterr, 1:8, results[:errs], label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# plterr
# savefig(plterr, "testdata\\error.png")
# 
# pltchi = plot(1:8, [chimax for _ in 1:8], line=:dash, xlabel=L"\tau", ylabel=L"\chi", label=L"\psi")
# spectra_err = []
# for maxerror in [1e-4]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     arrUs = results[:arrUs]
#     spectra = []
#     for i in 1:8
#         state = apply_brickwork(arrUs[i], zerostate; 
#                         normalize_after_layer = false,
#                         trunc=(maxerror=maxerror,))
#         reconstr_err = 1-abs(dot(state, zerostate))
#         inv_err = results[:errs][i]
#         abs_diff = abs(reconstr_err - inv_err)
#         rel_diff = abs_diff/inv_err
# 
#         spec = spectrum(state, div(N,2))
#         push!(spectra, spec)
#     end
#     push!(spectra_err, spectra)
#     plot!(pltchi, 1:8, results[:bonddims], label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# pltchi
# savefig(pltchi, "testdata\\bonddim.png")
# 
# plttime = plot(xlabel=L"\tau", ylabel=L"t \ (s)")
# for maxerror in [1e-4, 1e-5, 1e-6, 1e-7]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     plot!(plttime, 1:8, results[:times], label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# plttime
# savefig(plttime, "testdata\\times.png")
# 
# 
# 
# 
# pltchi_psi = plot(1:8, [chimax for _ in 1:8], line=:dash, xlabel=L"\tau", ylabel=L"\chi", label=L"\psi")
# spectra_err = []
# for maxerror in [1e-4]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     arrUs = results[:arrUs]
#     maxchis = []
#     spectra = []
#     for i in 1:8
#         orthogonalize!(psi, iseven(i) ? 1 : N)
#         arrdg = dagger(arrUs[i])
#         @show maxlinkdim(psi)
#         state = apply_brickwork(arrdg, psi; 
#                         shift = mod(i-1,2), 
#                         to_right = iseven(i), 
#                         normalize_after_layer = false,
#                         trunc=(maxerror=maxerror,))
#         reconstr_err = 1-abs(dot(state, zerostate))
#         inv_err = results[:errs][i]
#         abs_diff = abs(reconstr_err - inv_err)
#         rel_diff = abs_diff/inv_err
# 
#         spec = spectrum(state, div(N,2))
#         push!(spectra, spec)
#         @show abs_diff
#         @show rel_diff
#         @show norm(state)
#         @show maxlinkdim(state)
#         maxchi = maxlinkdim(state)
#         push!(maxchis, maxchi)
#     end
#     push!(spectra_err, spectra)
#     plot!(pltchi_psi, 1:8, maxchis, label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# pltchi_psi
# savefig(pltchi_psi, "testdata\\bonddim_Upsi.png")
# 
# 
# pltchi_psi = plot(1:8, [chimax for _ in 1:8], line=:dash, xlabel=L"\tau", ylabel=L"\chi", label=L"\psi")
# spectra_err = []
# for maxerror in [1e-4]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     arrUs = results[:arrUs]
#     maxchis = []
#     spectra = []
#     for i in 1:8
#         orthogonalize!(psi, iseven(i) ? 1 : N)
#         arrdg = dagger(arrUs[i])
#         @show maxlinkdim(psi)
#         state = apply_brickwork(arrdg, psi; 
#                         shift = mod(i-1,2), 
#                         to_right = iseven(i), 
#                         normalize_after_layer = false,
#                         trunc=(maxerror=maxerror,))
#         reconstr_err = 1-abs(dot(state, zerostate))
#         inv_err = results[:errs][i]
#         abs_diff = abs(reconstr_err - inv_err)
#         rel_diff = abs_diff/inv_err
# 
#         spec = spectrum(state, div(N,2))
#         push!(spectra, spec)
#         @show abs_diff
#         @show rel_diff
#         @show norm(state)
#         @show maxlinkdim(state)
#         maxchi = maxlinkdim(state)
#         push!(maxchis, maxchi)
#     end
#     push!(spectra_err, spectra)
#     plot!(pltchi_psi, 1:8, maxchis, label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# pltchi_psi
# savefig(pltchi_psi, "testdata\\bonddim_Upsi.png")
# 
# 
# using ColorSchemes
# colors = [get(ColorSchemes.inferno, t) for t in range(0.9, stop=0.1, length=8)]
# 
# specs = spectra_err[1]
# 
# spec_psi = spectrum(psi, div(N,2))
# pltspec = plot(1:length(spec_psi), spec_psi, line=:dash, 
#             xlabel=L"j", ylabel=L"\lambda_j", yscale=:log10, 
#             ylimits = (4e-5,1.1), xlimits = (0, 15),
#             dpi=400, palette=:inferno, label=L"\psi")
# for tau in 1:8
#     plot!(pltspec, 1:length(specs[tau]), specs[tau], color=colors[tau], label=L"\tau="*"$(tau)")
# end
# pltspec
# savefig(pltspec, "testdata\\spectra_zero.png")
# 
# 
# plt = plot(1:8, [entL for _ in 1:8], label=L"ψ", line=:dash, xlabel=L"\mathrm{Layer}", ylabel=L"S")
# for (i, vals) in enumerate(results[:entsL])
#     if isodd(i)
#         plot!(plt, 1:2:length(vals), vals[1:2:end], label=L"\tau="*"$i", m=:circle)
#     end
# end
# plt
# 
# 
# plt_compare = plot(yscale=:log10, 
#         ylabel=ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|",
#         xlabel=L"\tau")
# for maxerror in [1e-4, 1e-5, 1e-6, 1e-7]
#     results = load_object("testdata\\heisenberg_N$(N)_T8_E$(maxerror).jld2")
#     states = [apply_brickwork(circuit, zerostate) for circuit in results[:arrUs]]
#     errs_true = [1 - abs(dot(state, psi)) for state in states]
#     plot!(plt_compare, 1:8, results[:errs], line=:dash, label=L"\mathrm{maxerror}="*"$(maxerror)")
#     plot!(plt_compare, errs_true, label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# plt_compare
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# N = 10
# energy, psi = XXZ(N)
# entL = entropy!(psi, div(N,2))
# entR = entropy!(psi, div(N,2)+1)
# chimax = maxlinkdim(psi)
# 
# zerostate = MPS(siteinds(psi), ["0" for _ in 1:N])
# orthogonalize!(zerostate, 1)
# 
# maxerror = 1e-4
# maxdepth = 20
# results = [load_object("testdata\\heisenberg_N$(N)_T$(tau)_E$(maxerror).jld2") for tau in 1:20]
# 
# plterr = plot(yscale=:log10, xlabel=L"\tau", ylabel=L"1-|\langle\psi|U^{(\tau)}|0\rangle|")
# plot!(plterr, 1:maxdepth, [res[:err] for res in results], label=L"\mathrm{maxerror}="*"$(maxerror)")
# savefig(plterr, "testdata\\N10_error.png")
# 
# 
# plttime = plot(xlabel=L"\tau", ylabel=L"t \ (s)")
# plot!(plttime, 1:maxdepth, [res[:time] for res in results], label=L"\mathrm{maxerror}="*"$(maxerror)")
# plttime
# savefig(plttime, "testdata\\N10_times.png")
# 
# pltchi = plot(1:maxdepth, [chimax for _ in 1:maxdepth], line=:dash, xlabel=L"\tau", ylabel=L"\chi", label=L"\psi")
# spectra_err = []
# for maxerror in [1e-4]
#     arrUs = [res[:arrU] for res in results]
#     spectra = []
#     for i in 1:maxdepth
#         state = apply_brickwork(arrUs[i], zerostate; 
#                         normalize_after_layer = false,
#                         trunc=(maxerror=maxerror,))
#         reconstr_err = 1-abs(dot(state, zerostate))
#         inv_err = results[i][:err]
#         abs_diff = abs(reconstr_err - inv_err)
#         rel_diff = abs_diff/inv_err
# 
#         spec = spectrum(state, div(N,2))
#         push!(spectra, spec)
#     end
#     push!(spectra_err, spectra)
#     plot!(pltchi, 1:maxdepth, [res[:bonddim] for res in results], label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# pltchi
# savefig(pltchi, "testdata\\N10_bonddim.png")
# 
# 
# 
# pltchi_psi = plot(1:maxdepth, [chimax for _ in 1:maxdepth], line=:dash, xlabel=L"\tau", ylabel=L"\chi", label=L"\psi")
# spectra_err = []
# for maxerror in [1e-4]
#     arrUs = [res[:arrU] for res in results]
#     maxchis = []
#     spectra = []
#     for i in 1:maxdepth
#         orthogonalize!(psi, iseven(i) ? 1 : N)
#         arrdg = dagger(arrUs[i])
#         @show maxlinkdim(psi)
#         state = apply_brickwork(arrdg, psi; 
#                         shift = mod(i-1,2), 
#                         to_right = iseven(i), 
#                         normalize_after_layer = false,
#                         trunc=(maxerror=maxerror,))
#         reconstr_err = 1-abs(dot(state, zerostate))
#         inv_err = results[i][:err]
#         abs_diff = abs(reconstr_err - inv_err)
#         rel_diff = abs_diff/inv_err
# 
#         spec = spectrum(state, div(N,2))
#         push!(spectra, spec)
#         @show abs_diff
#         @show rel_diff
#         @show norm(state)
#         @show maxlinkdim(state)
#         maxchi = maxlinkdim(state)
#         push!(maxchis, maxchi)
#     end
#     push!(spectra_err, spectra)
#     plot!(pltchi_psi, 1:maxdepth, maxchis, label=L"\mathrm{maxerror}="*"$(maxerror)")
# end
# pltchi_psi
# savefig(pltchi_psi, "testdata\\N10_bonddim_Upsi.png")
# 
# 
# 
# using ColorSchemes
# colors = [get(ColorSchemes.inferno, t) for t in range(0.9, stop=0.1, length=maxdepth)]
# 
# specs = spectra_err[1]
# 
# spec_psi = spectrum(psi, div(N,2))
# pltspec = plot(1:length(spec_psi), spec_psi, line=:dash, 
#             xlabel=L"j", ylabel=L"\lambda_j", yscale=:log10, 
#             ylimits = (4e-5,1.1), xlimits = (0, 15),
#             dpi=400, palette=:inferno, label=L"\psi")
# for tau in 1:maxdepth
#     plot!(pltspec, 1:length(specs[tau]), specs[tau], color=colors[tau], label=L"\tau="*"$(tau)")
# end
# pltspec
# savefig(pltspec, "testdata\\N10_spectra_Upsi.png")

