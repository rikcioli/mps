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


#using Logging
#Logging.disable_logging(Logging.Warn)

Base.@kwdef mutable struct InversionInstructions
    maxrank::Union{Nothing, Int} = nothing
    maxerror::Union{Nothing, Float64} = nothing
    atol::Float64 = 1e-8
    maxiter::Int = 1000000
    gradtol::Float64 = 1e-8
    N_checkpoint::Int = 5000
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

function XY(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_XY(sites, 0.0, 0.5)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function ising(N::Int)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end

function H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.0, hz=0.0)
    N = length(sites)
    os = OpSum()
    for j in 1:N-1
        os += Jxy/2, "S+", j, "S-", j+1
        os += Jxy/2, "S-", j, "S+", j+1     # same coefficient => Hermitian
        os += Jz,    "Sz", j, "Sz", j+1
    end
    for j in 1:N
        os += -hs*(-1)^j, "Sz", j           # STAGGERED — this is what was missing
        os += hz,         "Sz", j
    end
    return MPO(os, sites)
end


function XXZ(N::Int)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    H = H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.05)
    psi0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n=1:N])

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end



function invert_maxerr(ψ::MPS, tau::Int, pathname::String; resuming = false)

    N = length(ψ)
    instrpath = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" : pathname*"N$(N)_T$(tau)_instructions.jld2"
    instr = load_object(instrpath)

    sites = siteinds(ψ)
    maxerror = instr.maxerror
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    trunc = (maxerror=maxerror, atol=instr.atol)
    nU = n_unitaries(N, tau)
    N_checkpoint = instr.N_checkpoint

    savefile = load_object(pathname*"N$(N)_T$(tau).jld2")
    arrU0 = savefile.arrU
    if isnothing(arrU0)
        arrU0 = random_circuit(N, tau)
    end
    arrU0 = Vector{Matrix{ComplexF64}}(arrU0) 


    # --- overlap-only objective (no penalty) ---
    overlap_only = (lognorm_factors, ϕ) -> -log(abs(sproduct(ψ, ϕ))) - sum(lognorm_factors)

    cost_function = arrU -> begin
        ϕ, lognorm_factors = apply_brickwork_normalize(arrU, zeromps; trunc=trunc)
        return overlap_only(lognorm_factors, ϕ)
    end

    # initialize outputs so the save block is always well-defined,
    # even if outer_iters == 0 or the first optimize call fails to assign.
    arrUmin = arrU0
    fmin = NaN
    gradmin = nothing
    total_nghist::Matrix{Float64} = savefile.normgradhistory
    total_time_prior::Float64 = get(savefile, :time, 0.0)
    # this will be reshaped before final save


    m = instr.m
    maxiter = instr.maxiter
    gradtol = instr.gradtol
        
    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = 2)
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    normgradvec = Float64[]
    t_start = Base.time()

    function checkpoint_finalize!(x, f, g, numiter)
        gnorm = sqrt(inner(x, g, g))
        push!(normgradvec, f)
        push!(normgradvec, gnorm)

        if numiter % N_checkpoint == 0
            # compute lightweight diagnostics at this point
            ϕ, lognorm_f = apply_brickwork_normalize(x, zeromps; trunc=trunc)
            overlap_cost = (-log(abs(sproduct(ψ, ϕ))), -sum(lognorm_f))
            err          = 1 - exp(-sum(overlap_cost))
            gnorm        = sqrt(inner(x, g, g))

            n = length(normgradvec) ÷ 2
            vc = copy(normgradvec)
            ckpt_normgradhistory = permutedims(reshape(vc, 2, n))
            cum_nghist = vcat(total_nghist, ckpt_normgradhistory)
            cum_time   = total_time_prior + (Base.time() - t_start)

            ckpt = (N=N, tau=tau, arrU=x, gradmin=g, gradnorm=gnorm, normgradhistory=cum_nghist,  # current arrU and gradient at arrU
                    cost=f, overlap_cost=overlap_cost, err=err, time=cum_time,       # current function values, err and time
                    converged=false, finished=false)                                 # mid-run ⇒ not converged
            save_object(pathname*"N$(N)_T$(tau).jld2", ckpt)

            ckpt_instr = copy(instr)
            ckpt_instr.maxiter = max(instr.maxiter - numiter, 1)
            save_object(pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2", ckpt_instr)
            @info "checkpoint: tau=$tau iter=$numiter gradnorm=$gnorm err=$err"
        end
        return x, f, g
    end

    @show tau

    elapsed = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory =
        optimize(fg, arrUmin, algorithm;
                retract = retract, transport! = transport!,
                isometrictransport = true, inner = inner,
                finalize! = checkpoint_finalize!)

    cum_nghist = vcat(total_nghist, normgradhistory)
    cum_time = total_time_prior + elapsed


    # --- final diagnostics (overlap term reported separately from penalty) ---
    ϕf, lognorm_f = apply_brickwork_normalize(arrUmin, zeromps; trunc=trunc)
    overlap_cost  = (-log(abs(sproduct(ψ, ϕf))), -sum(lognorm_f))
    err = 1 - exp(-sum(overlap_cost))
    final_gnorm = isempty(normgradhistory) ? sqrt(inner(arrUmin, gradmin, gradmin)) : normgradhistory[end, 2]
    converged = final_gnorm <= gradtol      # did the final LBFGS actually converge?
    finished = true
    @show err, converged, finished

        
    result_tau = (N=N, tau=tau, arrU=arrUmin, gradmin=gradmin, # current arrU and gradient at arrU
                  gradnorm=final_gnorm, numfg=numfg, normgradhistory=cum_nghist,      # OptimKit's other returns
                  cost=fmin, overlap_cost=overlap_cost, err=err,            # current function values
                  converged=converged, finished=finished, time=cum_time)   # mid-run ⇒ not converged
    save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

    new_instr = copy(instr)
    save_object(pathname*"N$(N)_T$(tau+1)_instructions.jld2", new_instr)

    return
end



function invert_maxrank(ψ::MPS, tau::Int, pathname::String; resuming = false)

    N = length(ψ)
    instrpath = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" : pathname*"N$(N)_T$(tau)_instructions.jld2"
    instr = load_object(instrpath)

    sites = siteinds(ψ)
    chimax = maxlinkdim(ψ)
    maxrank = instr.maxrank
    if isnothing(maxrank)
        maxrank = chimax
        instr.maxrank = maxrank
    end
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    trunc = (maxrank=maxrank, atol=instr.atol)
    nU = n_unitaries(N, tau)
    N_checkpoint = instr.N_checkpoint

    savefile = load_object(pathname*"N$(N)_T$(tau).jld2")
    arrU0 = savefile.arrU
    if isnothing(arrU0)
        arrU0 = random_circuit(N, tau)
    end
    arrU0 = Vector{Matrix{ComplexF64}}(arrU0)

    overlap_only = (Snorms, ϕ) -> -log(abs(sproduct(ψ, ϕ))) - sum(log.(Snorms))

    cost_function = arrU -> begin
        ϕ, Snorms = apply_brickwork(arrU, zeromps; trunc=trunc)
        return overlap_only(Snorms, ϕ)
    end

    arrUmin = arrU0
    fmin = NaN
    gradmin = nothing
    total_nghist::Matrix{Float64} = savefile.normgradhistory
    total_time_prior::Float64 = get(savefile, :time, 0.0)

    m = instr.m
    maxiter = instr.maxiter
    gradtol = instr.gradtol

    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = 2)
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    # === value-based (windowed) convergence settings =======================
    n_check        = 500       # check the infidelity every n_check iterations
    err_reltol     = 1e-3      # stop when the relative decrease over the window < this
    err_floor_atol = 1e-13     # also stop if the absolute change is below the arithmetic
                               # floor — REPLACE with your measured ε_C (Double64 test)
    converged_by_value = Ref(false)
    last_ckpt_err      = Ref(NaN)
    # =======================================================================

    normgradvec = Float64[]
    n_stall = 100
    stall_atol = 1e-14
    t_start = Base.time()
    function checkpoint_finalize!(x, f, g, numiter)
        gnorm = sqrt(inner(x, g, g))
        push!(normgradvec, f)
        push!(normgradvec, gnorm)

        # --- windowed value-based convergence check --------------------------
        # f == sum(overlap_cost), so the infidelity is err = -expm1(-f);
        # expm1 avoids cancellation when f is exponentially small.
        # MUST run before the stall block so a value-converged run isn't flagged as stuck.
        if numiter % n_check == 0
            err_now = -expm1(-f)
            if !isnan(last_ckpt_err[])
                Δ = last_ckpt_err[] - err_now                       # > 0 means error decreased
                if abs(Δ) < err_floor_atol                          # within the noise floor ⇒ converged
                    converged_by_value[] = true
                elseif Δ ≥ 0 && Δ / abs(err_now) < err_reltol       # decreasing, but slowly ⇒ converged
                    converged_by_value[] = true
                end
                # Δ < 0 and above the floor ⇒ error rose meaningfully ⇒ keep going
            end
            last_ckpt_err[] = err_now
        end

        if numiter % n_stall == 0
            niters_recorded = length(normgradvec) ÷ 2
            if niters_recorded >= n_stall
                recent_gnorms = @view normgradvec[end - 2*n_stall + 2 : 2 : end]
                spread = maximum(recent_gnorms) - minimum(recent_gnorms)
                # A frozen gradient is the EXPECTED converged state now, so a value-
                # converged run must not be flagged as stuck.
                converged_ok = (gnorm <= gradtol) || converged_by_value[]
                if spread <= stall_atol && !converged_ok
                    errorfile = (N=N, tau=tau, niter=numiter, gradnorm=gnorm, arrU=x, cost=f,
                                spread=spread, n_stall=n_stall, stall_atol=stall_atol)
                    save_object(pathname*"N$(N)_T$(tau)_gradbreak.jld2", errorfile)
                    error("Gradient norm frozen (spread=$spread) over last $n_stall iterations at " *
                        "tau=$tau, iter=$numiter, gnorm=$gnorm, but NOT converged. Likely an exact " *
                        "spectral degeneracy at the truncation cut made the SVD-adjoint gradient singular.")
                end
            end
        end

        if numiter % N_checkpoint == 0
            ϕ, Snorms = apply_brickwork(x, zeromps; trunc=trunc)
            overlap_cost = (-log(abs(sproduct(ψ, ϕ))), -sum(log.(Snorms)))
            err          = -expm1(-sum(overlap_cost))
            gnorm        = sqrt(inner(x, g, g))

            n = length(normgradvec) ÷ 2
            vc = copy(normgradvec)
            ckpt_normgradhistory = permutedims(reshape(vc, 2, n))
            cum_nghist = vcat(total_nghist, ckpt_normgradhistory)
            cum_time   = total_time_prior + (Base.time() - t_start)

            ckpt = (N=N, tau=tau, arrU=x, gradmin=g, gradnorm=gnorm, normgradhistory=cum_nghist,
                    cost=f, overlap_cost=overlap_cost, err=err, time=cum_time,
                    converged=false, finished=false)
            save_object(pathname*"N$(N)_T$(tau).jld2", ckpt)

            ckpt_instr = copy(instr)
            ckpt_instr.maxiter = max(instr.maxiter - numiter, 1)
            save_object(pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2", ckpt_instr)
            @info "checkpoint: tau=$tau iter=$numiter gradnorm=$gnorm err=$err"
        end
        return x, f, g
    end

    # Converged if the windowed infidelity criterion fired (gradtol kept as a
    # harmless fallback that, in practice, essentially never triggers here).
    hasconverged = (x, f, g, normg) -> (converged_by_value[] || normg < gradtol)

    @show tau

    elapsed = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory =
        optimize(fg, arrUmin, algorithm;
                retract = retract, transport! = transport!,
                isometrictransport = true, inner = inner,
                finalize! = checkpoint_finalize!,
                hasconverged = hasconverged)

    cum_nghist = vcat(total_nghist, normgradhistory)
    cum_time = total_time_prior + elapsed

    # --- final diagnostics (overlap term reported separately from penalty) ---
    ϕf, Snormsf = apply_brickwork(arrUmin, zeromps; trunc=trunc)
    overlap_cost  = (-log(abs(sproduct(ψ, ϕf))), -sum(log.(Snormsf)))
    err = -expm1(-sum(overlap_cost))
    final_gnorm = isempty(normgradhistory) ? sqrt(inner(arrUmin, gradmin, gradmin)) : normgradhistory[end, 2]
    converged = converged_by_value[] || (final_gnorm <= gradtol)   # value criterion counts
    finished = true
    @show err, converged, finished

    result_tau = (N=N, tau=tau, arrU=arrUmin, gradmin=gradmin,
                  gradnorm=final_gnorm, numfg=numfg, normgradhistory=cum_nghist,
                  cost=fmin, overlap_cost=overlap_cost, err=err,
                  converged=converged, finished=finished, time=cum_time)
    save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

    new_instr = copy(instr)
    save_object(pathname*"N$(N)_T$(tau+1)_instructions.jld2", new_instr)

    return
end


function invert_maxrank_variational(ψ::MPS, tau::Int, pathname::String; resuming = false)

    N = length(ψ)
    instrpath = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" : pathname*"N$(N)_T$(tau)_instructions.jld2"
    instr = load_object(instrpath)

    sites = siteinds(ψ)
    chimax = maxlinkdim(ψ)
    maxrank = instr.maxrank
    if isnothing(maxrank)
        maxrank = chimax
        instr.maxrank = maxrank
    end
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    nU = n_unitaries(N, tau)
    N_checkpoint = instr.N_checkpoint

    savefile = load_object(pathname*"N$(N)_T$(tau).jld2")
    arrU0 = savefile.arrU
    if isnothing(arrU0)
        arrU0 = random_circuit(N, tau)
    end
    arrU0 = Vector{Matrix{ComplexF64}}(arrU0) 


    # --- overlap-only objective (no penalty) ---
    overlap_only = (lognorm_factors, ϕ) -> -log(abs(sproduct(ψ, ϕ))) - sum(lognorm_factors)

    cost_function = arrU -> begin
        ϕ, lognorm_factors = apply_brickwork_variational(arrU, zeromps, maxrank)
        return overlap_only(lognorm_factors, ϕ)
    end

    # initialize outputs so the save block is always well-defined,
    # even if outer_iters == 0 or the first optimize call fails to assign.
    arrUmin = arrU0
    fmin = NaN
    gradmin = nothing
    total_nghist::Matrix{Float64} = savefile.normgradhistory
    total_time_prior::Float64 = get(savefile, :time, 0.0)   # cumulative seconds from earlier runs
    # this will be reshaped before final save


    m = instr.m
    maxiter = instr.maxiter
    gradtol = instr.gradtol
        
    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = 2)
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    normgradvec = Float64[]
    n_stall = 100          # window to check for a frozen gradient
    stall_atol = 1e-14      # essentially bit-frozen
    t_start = Base.time()
    function checkpoint_finalize!(x, f, g, numiter)
        gnorm = sqrt(inner(x, g, g))
        push!(normgradvec, f)
        push!(normgradvec, gnorm)

        if numiter % n_stall == 0
            # --- breakage detection: has the gradient norm been frozen for n_stall iters? ---
            # normgradvec stores [f, gnorm] pairs, so gnorm entries are at even indices.
            niters_recorded = length(normgradvec) ÷ 2
            if niters_recorded >= n_stall
                recent_gnorms = @view normgradvec[end - 2*n_stall + 2 : 2 : end]   # last n_stall gnorms
                spread = maximum(recent_gnorms) - minimum(recent_gnorms)
                # Only a problem if frozen AND not legitimately converged
                converged_ok = gnorm <= gradtol
                if spread <= stall_atol && !converged_ok
                    errorfile = (N=N, tau=tau, niter=numiter, gradnorm=gnorm, arrU=x, cost=f, 
                                spread=spread, n_stall=n_stall, stall_atol=stall_atol)
                    save_object(pathname*"N$(N)_T$(tau)_gradbreak.jld2", errorfile)
                    error("Gradient norm frozen (spread=$spread) over last $n_stall iterations at " *
                        "tau=$tau, iter=$numiter, gnorm=$gnorm, but NOT converged " *
                        "(gradtol=$(gradtol)). Likely an exact spectral degeneracy at the " *
                        "truncation cut made the SVD-adjoint gradient singular. The optimizer is stuck.")
                end
            end
        end

        if numiter % N_checkpoint == 0
            # compute lightweight diagnostics at this point
            ϕ, lognorm_f = apply_brickwork_variational(x, zeromps, maxrank)
            overlap_cost = (-log(abs(sproduct(ψ, ϕ))), -sum(lognorm_f))
            err          = 1 - exp(-sum(overlap_cost))
            gnorm        = sqrt(inner(x, g, g))

            n = length(normgradvec) ÷ 2
            vc = copy(normgradvec)
            ckpt_normgradhistory = permutedims(reshape(vc, 2, n))
            cum_nghist = vcat(total_nghist, ckpt_normgradhistory)
            cum_time   = total_time_prior + (Base.time() - t_start)

            ckpt = (N=N, tau=tau, arrU=x, gradmin=g, gradnorm=gnorm, normgradhistory=cum_nghist,  # current arrU and gradient at arrU
                    cost=f, overlap_cost=overlap_cost, err=err, time=cum_time, # current function values
                    converged=false, finished=false)                                 # mid-run ⇒ not converged
            save_object(pathname*"N$(N)_T$(tau).jld2", ckpt)

            ckpt_instr = copy(instr)
            ckpt_instr.maxiter = max(instr.maxiter - numiter, 1)
            save_object(pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2", ckpt_instr)
            @info "checkpoint: tau=$tau iter=$numiter gradnorm=$gnorm err=$err"
        end
        return x, f, g
    end

    @show tau

    elapsed = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory =
        optimize(fg, arrUmin, algorithm;
                retract = retract, transport! = transport!,
                isometrictransport = true, inner = inner,
                finalize! = checkpoint_finalize!)

    cum_nghist = vcat(total_nghist, normgradhistory)
    cum_time = total_time_prior + elapsed


    # --- final diagnostics (overlap term reported separately from penalty) ---
    ϕf, lognorm_f = apply_brickwork_variational(arrUmin, zeromps, maxrank)
    overlap_cost  = (-log(abs(sproduct(ψ, ϕf))), -sum(lognorm_f))
    err = 1 - exp(-sum(overlap_cost))
    final_gnorm = isempty(normgradhistory) ? sqrt(inner(arrUmin, gradmin, gradmin)) : normgradhistory[end, 2]
    converged = final_gnorm <= gradtol      # did the final LBFGS actually converge?
    finished = true
    @show err, converged, finished

        
    result_tau = (N=N, tau=tau, arrU=arrUmin, gradmin=gradmin, # current arrU and gradient at arrU
                  gradnorm=final_gnorm, numfg=numfg, normgradhistory=cum_nghist,      # OptimKit's other returns
                  cost=fmin, overlap_cost=overlap_cost, err=err,            # current function values
                  converged=converged, finished=finished, time=cum_time)   # mid-run ⇒ not converged
    save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

    new_instr = copy(instr)
    save_object(pathname*"N$(N)_T$(tau+1)_instructions.jld2", new_instr)

    return
end



function invert3(ψ::MPS, tau::Int, pathname::String; resuming = false)

    N = length(ψ)
    instrpath = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" : pathname*"N$(N)_T$(tau)_instructions.jld2"
    instr = load_object(instrpath)

    sites = siteinds(ψ)
    chimax = maxlinkdim(ψ)
    maxrank = instr.maxrank
    if isnothing(maxrank)
        maxrank = chimax
        instr.maxrank = maxrank
    end
    zeromps = MPS(sites, ["0" for _ in 1:N])
    orthogonalize!(zeromps, 1)

    # initial values for the multipliers
    λ = instr.λ
    ρ = instr.ρ
    c_ref = instr.c_ref
    Σε_max = instr.Σε_max
    ρ_growth = instr.ρ_growth
    outer_iters = instr.outer_iters
    
    skip_outer = instr.skip_outer
    trunc = (maxrank=maxrank, atol=instr.atol)
    nU = n_unitaries(N, tau)
    N_checkpoint = instr.N_checkpoint

    savefile = load_object(pathname*"N$(N)_T$(tau).jld2")
    arrU0 = savefile.arrU
    if isnothing(arrU0)
        arrU0 = random_circuit(N, tau)
    end
    arrU0 = Vector{Matrix{ComplexF64}}(arrU0) 

    # --- discarded weight  Σ_ℓ ε_ℓ  as a function of the lognorm factors ---
    total_discarded = (lognorm_factors) -> begin
        perlayer_persite = [lognorm_factors[(N-1)*(i-1)+1 : min((N-1)*i, length(lognorm_factors))] for i in 1:tau]
        perlayer_logs = [sum(arr) for arr in perlayer_persite]
        return sum(1 .- exp.(2 .* perlayer_logs))
    end

    # --- overlap-only objective (no penalty) ---
    overlap_only = (lognorm_factors, ϕ) -> -log(abs(sproduct(ψ, ϕ))) - sum(lognorm_factors)

    # ===== Augmented Lagrangian outer loop =====
    aug_lagrangian = (arrU, λ, ρ) -> begin
        ϕ, lognorm_factors = apply_brickwork_normalize(arrU, zeromps; trunc=trunc)
        ov = overlap_only(lognorm_factors, ϕ)
        c  = total_discarded(lognorm_factors) - Σε_max
        λineq = max(λ + ρ*c, 0.0)
        return ov + (λineq^2 - λ^2)/(2ρ)
    end

    # initialize outputs so the save block is always well-defined,
    # even if outer_iters == 0 or the first optimize call fails to assign.
    arrUmin = arrU0
    fmin = NaN
    gradmin = nothing
    total_nghist::Matrix{Float64} = savefile.normgradhistory
    total_time_prior::Float64 = get(savefile, :time, 0.0)

    @show tau

    if !skip_outer

        m = instr.m
        inner_maxiter = instr.inner_maxiter
        inner_gradtol = instr.inner_gradtol
        algorithm = LBFGS(m; maxiter = inner_maxiter, gradtol = inner_gradtol, verbosity = 1)

        iter_no = 0
        Σε = 0.
        c = 0.

        time_outer = @elapsed begin
            for outer_no in 1:outer_iters
                iter_no = outer_no
                # Cost for THIS outer iteration: fixed (λ, ρ), constraint c = Σε - ε_max
                λ_cur, ρ_cur = λ, ρ
                cost_function = arrU -> aug_lagrangian(arrU, λ_cur, ρ_cur)
                fg = arrU -> begin
                    func, grad = withgradient(cost_function, arrU)
                    grad = project(arrU, grad[1])
                    return func, grad
                end

                arrUmin, fmin, gradmin, numfg, normgradhistory =
                    optimize(fg, arrUmin, algorithm;
                            retract = retract, transport! = transport!,
                            isometrictransport = true, inner = inner)

                # --- evaluate the constraint at the new point ---
                ϕ, lognorm_factors = apply_brickwork_normalize(arrUmin, zeromps; trunc=trunc)
                Σε = total_discarded(lognorm_factors)
                c  = Σε - Σε_max
                @show outer_no, λ, ρ, Σε, c

                # --- constraint violation (inequality: only positive c matters) ---
                viol = max(c, 0.0)

                # --- multiplier update (dual ascent) ---
                if viol <= 0.9 * c_ref
                    # constraint improved enough SINCE THE LAST λ-UPDATE: take the dual step
                    λ = max(λ + ρ*c, 0.0)
                    c_ref = viol          # reset reference to the new level
                    # ρ stays fixed — the multiplier is doing the work
                else
                    if ρ < 1e5      # cutoff for rho growth, as above this it makes the landscape too stiff
                        # not enough cumulative progress: strengthen ρ, leave λ alone
                        ρ *= ρ_growth
                    end
                    λ = max(λ + ρ*c, 0.0)
                    c_ref = min(c_ref, viol)
                end

                # --- convergence: constraint satisfied ---
                if c <= 0
                    @info "AL converged at tau=$tau, outer=$outer_no: Σε=$Σε within tol of Σε_max=$Σε_max"
                    break
                end

                # --- infeasibility warning: λ or ρ blowing up ⇒ depth/rank too tight ---
                if λ > 1e8 || ρ > 1e10
                    @warn "AL multiplier/penalty diverging at tau=$tau (λ=$λ, ρ=$ρ). " *
                        "Likely INFEASIBLE: depth $tau / rank $maxrank cannot represent ψ " *
                        "with discarded weight ≤ ε_max=$Σε_max. Consider more depth."
                    break
                end
            end
        end
        outer_info = (time=time_outer, iter_no=iter_no, discarded=Σε, λ=λ, ρ=ρ, c=c, Σε_max=Σε_max)
        save_object(pathname*"N$(N)_T$(tau)_outer_info.jld2", outer_info)
    end

    # after outer loop ends we run the full optimization with the found λ and ρ
    m = instr.m
    maxiter = instr.maxiter
    gradtol = instr.gradtol
        
    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = 2)
    cost_function = arrU -> aug_lagrangian(arrU, λ, ρ)
    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    n_stall = 100
    stall_atol = 1e-14
    normgradvec = Float64[]
    t_start = Base.time()
    function checkpoint_finalize!(x, f, g, numiter)
        gnorm = sqrt(inner(x, g, g))
        push!(normgradvec, f)
        push!(normgradvec, gnorm)

        if numiter % n_stall == 0
            # --- breakage detection: has the gradient norm been frozen for n_stall iters? ---
            # normgradvec stores [f, gnorm] pairs, so gnorm entries are at even indices.
            niters_recorded = length(normgradvec) ÷ 2
            if niters_recorded >= n_stall
                recent_gnorms = @view normgradvec[end - 2*n_stall + 2 : 2 : end]   # last n_stall gnorms
                spread = maximum(recent_gnorms) - minimum(recent_gnorms)
                # Only a problem if frozen AND not legitimately converged
                converged_ok = gnorm <= gradtol
                if spread <= stall_atol && !converged_ok
                    errorfile = (N=N, tau=tau, niter=numiter, gradnorm=gnorm, arrU=x, cost=f, 
                                spread=spread, n_stall=n_stall, stall_atol=stall_atol)
                    save_object(pathname*"N$(N)_T$(tau)_gradbreak.jld2", errorfile)
                    error("Gradient norm frozen (spread=$spread) over last $n_stall iterations at " *
                        "tau=$tau, iter=$numiter, gnorm=$gnorm, but NOT converged " *
                        "(gradtol=$(gradtol)). Likely an exact spectral degeneracy at the " *
                        "truncation cut made the SVD-adjoint gradient singular. The optimizer is stuck.")
                end
            end
        end

        if numiter % N_checkpoint == 0
            # compute lightweight diagnostics at this point
            ϕ, lognorm_f = apply_brickwork_normalize(x, zeromps; trunc=trunc)
            overlap_cost = (-log(abs(sproduct(ψ, ϕ))), -sum(lognorm_f))
            err          = 1 - exp(-sum(overlap_cost))
            Σε           = total_discarded(lognorm_f)
            gnorm        = sqrt(inner(x, g, g))

            n = length(normgradvec) ÷ 2
            vc = copy(normgradvec)
            ckpt_normgradhistory = permutedims(reshape(vc, 2, n))
            cum_nghist = vcat(total_nghist, ckpt_normgradhistory)
            cum_time   = total_time_prior + (Base.time() - t_start)

            ckpt = (N=N, tau=tau, arrU=x, gradmin=g, gradnorm=gnorm, normgradhistory=cum_nghist,  # current arrU and gradient at arrU
                    aug_cost=f, overlap_cost=overlap_cost, penalty_cost=Σε, err=err, time=cum_time, # current function values
                    converged=false, finished=false)                                 # mid-run ⇒ not converged
            save_object(pathname*"N$(N)_T$(tau).jld2", ckpt)

            ckpt_instr = copy(instr)
            ckpt_instr.maxiter = max(instr.maxiter - numiter, 1)
            ckpt_instr.skip_outer = true
            ckpt_instr.λ = λ
            ckpt_instr.ρ = ρ
            save_object(pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2", ckpt_instr)
            @info "checkpoint: tau=$tau iter=$numiter gradnorm=$gnorm err=$err"
        end
        return x, f, g
    end

    elapsed = @elapsed arrUmin, fmin, gradmin, numfg, normgradhistory =
        optimize(fg, arrUmin, algorithm;
                retract = retract, transport! = transport!,
                isometrictransport = true, inner = inner,
                finalize! = checkpoint_finalize!)

    cum_nghist = vcat(total_nghist, normgradhistory)
    cum_time = total_time_prior + elapsed


    # --- final diagnostics (overlap term reported separately from penalty) ---
    ϕf, lognorm_f = apply_brickwork_normalize(arrUmin, zeromps; trunc=trunc)
    overlap_cost  = (-log(abs(sproduct(ψ, ϕf))), -sum(lognorm_f))
    err = 1 - exp(-sum(overlap_cost))
    Σε_final = total_discarded(lognorm_f)
    final_gnorm = isempty(normgradhistory) ? sqrt(inner(arrUmin, gradmin, gradmin)) : normgradhistory[end, 2]
    converged = final_gnorm <= gradtol      # did the final LBFGS actually converge?
    finished = true
    @show err, Σε_final, λ, converged, finished

        
    result_tau = (N=N, tau=tau, arrU=arrUmin, gradmin=gradmin, # current arrU and gradient at arrU
                  gradnorm=final_gnorm, numfg=numfg, normgradhistory=cum_nghist,      # OptimKit's other returns
                  aug_cost=fmin, overlap_cost=overlap_cost, penalty_cost=Σε_final, err=err,            # current function values
                  λ=λ, ρ=ρ, Σε_max=Σε_max,       # outer loop parameters
                  converged=converged, finished=finished, time=cum_time)   # mid-run ⇒ not converged
    save_object(pathname*"N$(N)_T$(tau).jld2", result_tau)

    new_instr = copy(instr)
    new_instr.skip_outer = false
    new_instr.λ = λ
    new_instr.ρ = 1.0
    new_instr.Σε_max = err           # discarded-weight tolerance
    save_object(pathname*"N$(N)_T$(tau+1)_instructions.jld2", new_instr)

    return
end

function prepare_start(psi::MPS, pathname::String; kwargs...)
    # Prepare warm start
    N = length(psi)
    orthogonalize!(psi, 1)
    psi_cut = move_center(psi, N; trunc=(maxrank=1,), normalize=true)
    # Extract the U_start from the truncated mps
    U_start = to_layer(psi_cut)

    # We add some random noise to help escaping the saddle point
    Vs = skew([randn(ComplexF64, 4, 4) for _ in eachindex(U_start)])
    newU = [retract(Matrix{ComplexF64}(I, (4,4)), V, 0.01)[1] for V in Vs]
    U_start = newU .* U_start

    instructions = InversionInstructions(; kwargs...)
    save_object(pathname*"N$(N)_T1_instructions.jld2", instructions)
    

    savefile = (N=N, tau=1, arrU=U_start, gradnorm=Inf, numfg=0, 
                normgradhistory=Matrix{Float64}(undef, 0, 2), time=0.0,
                converged=false, finished=false)
    save_object(pathname*"N$(N)_T1.jld2", savefile)
end


function continue_inversion(psi::MPS, maxtau::Int, pathname::String, invertFunction::Function)
    N = length(psi)
    pattern = Regex("N$(N)_T(\\d+)\\.jld2")
    taus = [parse(Int, m.captures[1]) for f in readdir(pathname)
            for m in [match(pattern, f)] if !isnothing(m)]
    isempty(taus) && error("No saved checkpoint files found in $pathname for N=$N")
    last_tau = maximum(taus)

    result = load_object(pathname*"N$(N)_T$(last_tau).jld2")

    if get(result, :finished, true)  # default true for old files w/o the field
        if last_tau < maxtau
            newtau = last_tau+1
            @info "Depth $last_tau finished (converged=$(get(result,:converged,true))). " *
                "Adding a layer, continuing at depth $(newtau)."
            warmU = add_layer(result.arrU, N, last_tau)
            savefile = (N=N, tau=newtau, arrU=warmU, gradnorm=Inf, numfg=0,
                        normgradhistory=Matrix{Float64}(undef, 0, 2), time=0.0,
                        converged=false, finished=false)
            save_object(pathname*"N$(N)_T$(newtau).jld2", savefile)
            invertFunction(psi, newtau, pathname; resuming = false)
        else
            @info "Required maxtau already reached for this state"
            return :done
        end
    else
        if isinf(result.gradnorm)
            invertFunction(psi, last_tau, pathname; resuming = false)
        else
            @info "Depth $last_tau interrupted mid-solve (gradnorm=$(result.gradnorm)). Resuming."
            invertFunction(psi, last_tau, pathname; resuming = true)
        end
    end
    return :continue
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





if true
    let
        pathname = "testdata\\ising\\g1.5\\"
        Nlist = [20]
        psis = MPS[]
        for N in Nlist
            #E, psi = ising(N)
            #f = h5open(pathname*"$(N)_mps.h5","w")
            #write(f, "psi", psi)
            #close(f)
            f = h5open(pathname*"$(N)_mps.h5","r")
            psi = read(f,"psi",MPS)
            close(f)
            push!(psis, dense(psi))
        end

        for psi in psis
            prepare_start(psi, pathname; maxiter=10000000, maxrank=20)

            while true
                status = continue_inversion(psi, 30, pathname, invert_maxrank)
                status == :done && break
            end
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
        pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/xxz/Jz2.5/"
        Nlist = [60,100,140,180,220,260,300]
        psis = MPS[]
        for N in Nlist
            E, psi = XXZ(N)
            f = h5open(pathname*"$(N)_mps.h5","w")
            write(f, "psi", psi)
            close(f)
            # f = h5open(pathname*"$(N)_mps.h5","r")
            # psi = read(f,"psi",MPS)
            # close(f)
            push!(psis, psi)
        end

        Threads.@threads for psi in psis
            prepare_start(psi, pathname*"trunc/"; maxiter=20000)

            while true
                status = continue_inversion(psi, 30, pathname*"trunc/", invert_maxrank)
                status == :done && break
            end
        end
    end
end

if false
    let
        pathname = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/ising/g1.5/"
        Nlist = [60,100,140,180,220,260,300]
        psis = MPS[]
        for N in Nlist
            E, psi = ising(N)
            f = h5open(pathname*"$(N)_mps.h5","w")
            write(f, "psi", psi)
            close(f)
            # f = h5open(pathname*"$(N)_mps.h5","r")
            # psi = read(f,"psi",MPS)
            # close(f)
            push!(psis, psi)
        end

        Threads.@threads for psi in psis
            prepare_start(psi, pathname*"trunc/"; maxiter=20000)

            while true
                status = continue_inversion(psi, 30, pathname*"trunc/", invert_maxrank)
                status == :done && break
            end
        end
    end
end


