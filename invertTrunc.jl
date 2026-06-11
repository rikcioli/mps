using MKL
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
        ϕ = apply_brickwork(arrU, zeromps; trunc=trunc, normalize=true)
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
        
        err_tau = 1+neg_overlap

        result_tau = (N=N, tau=tau, trunc=trunc,
                        time=time_tau, err=err_tau, arrU=arrUmin, 
                        gradmin=gradmin, niter=numfg, normgradhistory=normgradhistory)
        maxerror = trunc[:maxerror]
        save_object(pathname*"N$(N)_T$(tau)_E$(maxerror).jld2", result_tau)

        arrUs[tau] .+= arrUmin
        errs[tau] = err_tau
        times[tau] = time_tau

        if reuse_previous
            arrU0 = arrUmin
        end
    end

    result = (N=N, maxtau=maxtau, trunc=trunc, times=times,
    errs=errs, arrUs=arrUs)

    return result
end


function invert2(ψ::MPS, maxtau::Int; pathname = "", reuse_previous=false, warm_start::Union{Vector{<:AbstractMatrix}, Nothing} = nothing)
    N = length(ψ)
    sites = siteinds(ψ)
    chimax = maxlinkdim(ψ)
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    # Captures ψ, zeromps and kwargs
    cost_function = (arrU) -> begin
        ϕ, logs = apply_brickwork_normalize(arrU, zeromps; trunc=(maxrank=chimax, atol=eps()))
        return -log(real(sproduct(ψ, ϕ)))-logs
    end

    err_func = (arrU) -> begin
        ϕ = apply_brickwork(arrU, zeromps; trunc=(maxerror=1e-12,), normalize=true)
        return 1-real(sproduct(ψ, ϕ))
    end

    cost_split_func = (arrU) -> begin
        ϕ, logs = apply_brickwork_normalize(arrU, zeromps; trunc=(maxrank=chimax, atol=eps()))
        return (-log(real(sproduct(ψ, ϕ))), -logs)
    end

    # Combines function and projected gradient
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end
    
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
        time_tau = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory = 
                                                            optimize(fg, arrU0, algorithm; 
                                                                retract = retract, 
                                                                transport! = transport!, 
                                                                isometrictransport = true, 
                                                                inner = inner)
        
        cost = cost_split_func(arrUmin)
        @show cost
        err_tau = err_func(arrUmin)
        @show err_tau
        @show (1-exp(-fmin) - err_tau)/err_tau

        result_tau = (N=N, tau=tau, trunc=trunc, cost=cost,
                        time=time_tau, err=err_tau, arrU=arrUmin, 
                        gradmin=gradmin, niter=numfg, normgradhistory=normgradhistory)

        save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

        if reuse_previous
            arrU0 = arrUmin
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


function runinversion(N, maxtau, maxerror, pathname = "testdata\\ising\\")
    #f = h5open(pathname*"$(N)_mps.h5","r")
    #psi = read(f,"psi",MPS)
    #close(f)
    #psi = XY(N)[2]
    psi = load_object(pathname*"ising_L128_g1.5.jld2")
    psi = dense(psi)
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


function runinversion2(psi::MPS, maxtau; pathname = "testdata\\XY\\")
    #f = h5open(pathname*"$(N)_mps.h5","r")
    #psi = read(f,"psi",MPS)
    #close(f)
    #psi = random_mps(ComplexF64, siteinds("Qubit", N); linkdims=2)
    #save_object(pathname*"$(N)_mps.jld2", psi)
    #psi = load_object(pathname*"$(N)_mps.jld2")
    #psi = dense(psi)
    N = length(psi)

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

    invert2(psi, maxtau; pathname=pathname, warm_start=U_start, reuse_previous=true)
end


if false
    let
        Nlist = [20]
        maxerrorlist = [1e-2]
        pairs = collect(Iterators.product(Nlist, maxerrorlist))
        for (N, maxerror) in pairs
            runinversion(N, 12, maxerror)
        end
    end
end


if false
    let
        Nlist = [20]
        for N in Nlist
            runinversion2(N, 12; pathname = "testdata\\rand\\")
        end
    end
end


if true
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



