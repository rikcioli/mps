#using MKL
include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using OptimKit
using Zygote
using LinearAlgebra
using JLD2
using HDF5
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


function invert2(ψ::MPS, maxtau::Int; mintau::Int = 1, maxerror = 1e-8, maxrank = nothing, pathname = "", reuse_previous=false, warm_start::Union{Vector{<:AbstractMatrix}, Nothing} = nothing)
    N = length(ψ)
    sites = siteinds(ψ)
    chimax = maxlinkdim(ψ)
    maxrank = isnothing(maxrank) ? chimax : maxrank
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    arrU0 = isnothing(warm_start) ? random_circuit(N, mintau) : warm_start

    for tau in mintau:maxtau
        @show tau

        trunc = (2^tau <= chimax ? (maxerror=maxerror,) : (maxrank=maxrank,))
        # Captures ψ, zeromps and kwargs
        cost_function = (arrU) -> begin
            ϕ, logs = apply_brickwork_normalize(arrU, zeromps; trunc=trunc)
            return -log(abs(sproduct(ψ, ϕ)))-logs
        end

        # Combines function and projected gradient
        fg = arrU -> begin
            func, grad = withgradient(cost_function, arrU)
            grad = project(arrU, grad[1])
            return func, grad
        end

        cost_split_func = (arrU) -> begin
            ϕ, logs = apply_brickwork_normalize(arrU, zeromps; trunc=trunc)
            return (-log(abs(sproduct(ψ, ϕ))), -logs)
        end

        m = 5
        algorithm = LBFGS(m; maxiter = 10000, gradtol = 1e-8, verbosity = 2)

        # optimize and store results
        time_tau = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory = 
                                                            optimize(fg, arrU0, algorithm; 
                                                                retract = retract, 
                                                                transport! = transport!, 
                                                                isometrictransport = true, 
                                                                inner = inner)
        
        cost = cost_split_func(arrUmin)
        @show cost

        result_tau = (N=N, tau=tau, trunc=trunc, cost=cost,
                        time=time_tau, err=1-exp(-fmin), arrU=arrUmin, 
                        gradmin=gradmin, niter=numfg, normgradhistory=normgradhistory)

        save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

        if reuse_previous
            arrU0 = add_layer(arrUmin, N, tau)
        else
            arrU0 = random_circuit(N, tau+1)
        end
    end

    return
end

function things_to_put_somewhere()
    bonddims = zeros(Int64, maxtau)
    entsL = [zeros(tau) for tau in 1:maxtau]
    entsR = [zeros(tau) for tau in 1:maxtau]

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
    bonddim_tau = maximum(linkdims(state))

    entsL[tau] .+= entsL_tau
    entsR[tau] .+= entsR_tau
    bonddims[tau] = bonddim_tau
end


function runinversion2(psi::MPS, maxtau; maxerror=1e-8, maxrank=nothing, pathname = "testdata\\XY\\")
    N = length(psi)
    orthogonalize!(psi, 1)

    psi_cut = move_center(psi, N; trunc=(maxrank=1,), normalize=true)

    # Extract the U_start from the truncated mps
    U_start = to_layer(psi_cut)

    # We add some random noise to help escaping the saddle point
    Vs = skew([randn(ComplexF64, 4, 4) for _ in eachindex(U_start)])
    newU = [retract(Matrix{ComplexF64}(I, (4,4)), V, 0.01)[1] for V in Vs]
    U_start = newU .* U_start

    invert2(psi, maxtau; maxerror=maxerror, maxrank=maxrank, pathname=pathname, warm_start=U_start, reuse_previous=true)
end

function continue_inversion(psi::MPS, maxtau::Int; maxerror=1e-8, maxrank=nothing, pathname = "testdata\\XY\\")
    N = length(psi)
    pattern = Regex("N$(N)_T(\\d+)\\.jld2")
    taus = [parse(Int, m.captures[1]) for f in readdir(pathname)
            for m in [match(pattern, f)] if !isnothing(m)]
    isempty(taus) && error("No saved checkpoint files found in $pathname for N=$N")
    last_tau_saved = maximum(taus)
    @info "Resuming from depth = $last_tau_saved"
    result = load_object(pathname*"N$(N)_T$(last_tau_saved).jld2")
    arrU = result.arrU
    warmU = add_layer(arrU, N, last_tau_saved)

    invert2(psi, maxtau; maxerror=maxerror, maxrank=maxrank, mintau=last_tau_saved+1, pathname=pathname, warm_start=warmU, reuse_previous=true)
end




if false
    let
        Nlist = [20]
        for N in Nlist
            runinversion2(N, 12; pathname = "testdata\\rand\\")
        end
    end
end


if false
    let
        pathname = "testdata/XY/"
        Nlist = [40]
        psis = MPS[]
        for N in Nlist
            f = h5open(pathname*"$(N)_mps.h5","r")
            psi = read(f,"psi",MPS)
            close(f)
            push!(psis, psi)
        end

        for psi in psis
            runinversion2(psi, 30; maxerror=1e-6, pathname = pathname)
        end
    end
end


if false
    let
        pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/ising/test/"
        glist = [1.0, 1.5]
        psis = MPS[]
        for g in glist
            psi = load_object(pathname*"ising_L128_g$(g).jld2")
            psi = dense(psi)
            push!(psis, psi)
        end

        pairs = collect(Iterators.product(glist, psis))
        Threads.@threads for (g, psi) in pairs
            runinversion2(psi, 30; pathname = pathname*"g$(g)new/")
        end
    end
end

if false
    let
        N=60
        pathname = "testdata\\rand\\mps1\\"
        f = h5open(pathname*"$(N)_mps.h5","r")
        psi = read(f,"psi",MPS)
        close(f)

        continue_inversion(psi, 3; maxrank=8, pathname = pathname*"test\\")
    end
end


if false
    let
        pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/xy/g0h0.5/"
        Nlist = [20, 40, 60, 80, 100]
        psis = MPS[]
        for N in Nlist
            f = h5open(pathname*"$(N)_mps.h5","r")
            psi = read(f,"psi",MPS)
            close(f)
            push!(psis, psi)
        end

        Threads.@threads for psi in psis
            runinversion2(psi, 30; pathname = pathname*"trunc/")
        end
    end
end


if true
    let
        pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/randMPS/mps1/"
        Nlist = 20:20:100
        psis = MPS[]
        for N in Nlist
            f = h5open(pathname*"$(N)_mps.h5","r")
            psi = read(f,"psi",MPS)
            close(f)
            push!(psis, psi)
        end

        Threads.@threads for psi in psis
            continue_inversion(psi, 30; maxrank=8, pathname = pathname*"trunc8/")
        end
    end
end
