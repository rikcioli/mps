include("rrules.jl")
include("optFunctions.jl")
using ITensors, ITensorMPS
using OptimKit
using Zygote
using LinearAlgebra
using JLD2
#using LaTeXStrings
#using Plots

const IOLOCK = ReentrantLock()
save_locked(path, obj)  = lock(IOLOCK) do; save_object(path, obj); end
load_locked(path)       = lock(IOLOCK) do; upgrade(load_object(path)); end

#using Logging
#Logging.disable_logging(Logging.Warn)

Base.@kwdef mutable struct InversionInstructions
    maxrank::Union{Nothing, Int} = nothing
    maxerror::Union{Nothing, Float64} = nothing
    atol::Float64 = 1e-8
    maxiter::Int = 1000000
    gradtol::Float64 = 1e-8
    n_checkpoint::Int = 500
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
    # never cut inside a near-degenerate cluster of negligible singular values (consecutive
    # values within degen_rtol of each other, weight s²/Σs² < degen_wmax): the cut moves
    # down to keep the whole cluster, up to degen_maxextra values beyond maxrank (see
    # truncdegen_keep). Clusters of larger weight keep the plain cut, so a rank that is
    # too small still shows up as a GradientFreeze. degen_rtol = 0: always the plain cut.
    degen_rtol::Float64 = 1e-3
    degen_maxextra::Int = 4
    degen_wmax::Float64 = 1e-9
end

# copy for a mutable struct (field-by-field)
Base.copy(x::InversionInstructions) = InversionInstructions(
    (getfield(x, f) for f in fieldnames(InversionInstructions))...)

# Instruction files written before a field existed come back from JLD2 as a
# ReconstructedMutable: rebuild a real InversionInstructions from them. A missing
# degen_rtol means the run was started with the plain cut, and it stays that way.
upgrade(x) = x
function upgrade(x::JLD2.ReconstructedMutable{:InversionInstructions})
    kw = Dict{Symbol,Any}(k => getproperty(x, k) for k in propertynames(x)
                          if hasfield(InversionInstructions, k))
    get!(kw, :degen_rtol, 0.0)
    return InversionInstructions(; kw...)
end


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


"""
Thrown by `invert_maxrank` when the gradient norm stops changing without the run
having converged. In practice this is the optimizer pinned on a truncation cut
whose last kept and first discarded singular values have become degenerate
(see diagnose_cut.jl): the cost jumps there and the SVD-adjoint gradient is
meaningless. `continue_inversion(...; adapt)` catches it and raises `maxrank`.
"""
struct GradientFreeze <: Exception
    N::Int
    tau::Int
    niter::Int
    gradnorm::Float64
    spread::Float64
    n_stall::Int
    maxrank::Int
end

Base.showerror(io::IO, e::GradientFreeze) = print(io,
    "GradientFreeze: gradient norm frozen (spread=$(e.spread)) over the last $(e.n_stall) iterations ",
    "at N=$(e.N), tau=$(e.tau), iter=$(e.niter), gnorm=$(e.gradnorm), maxrank=$(e.maxrank), but NOT converged. ",
    "Likely an exact spectral degeneracy at the truncation cut made the SVD-adjoint gradient singular.")


const DEPTHLOG_HEADER = "timestamp,N,tau,event,maxrank,atol,niter,err,gradnorm,time_s,note"

"""
Append one row to `<pathname>N<N>_depths.csv`: a record of the parameters each
depth actually ran with, and of every freeze and the maxrank change it caused.
One file per N, so workers sharing a directory never write the same file.
"""
function log_depth_event(pathname, N; tau, event, maxrank, atol, niter = "", err = "",
                         gradnorm = "", time = "", note = "")
    path = pathname * "N$(N)_depths.csv"
    fmt(x) = x isa AbstractFloat ? string(round(x; sigdigits = 6)) : string(x)
    row = join(fmt.([Libc.strftime("%Y-%m-%d %H:%M:%S", Base.time()), N, tau, event,
                     something(maxrank, ""), atol, niter, err, gradnorm, time, note]), ",")
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, DEPTHLOG_HEADER)
        println(io, row)
    end
end


function invert_maxrank(ψ::MPS, tau::Int, pathname::String; resuming = false, err_reltol = 1e-3)

    N = length(ψ)
    instrpath = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" : pathname*"N$(N)_T$(tau)_instructions.jld2"
    instr = load_locked(instrpath)

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
    # every cut keeps a relative gap > degen_rtol (see InversionInstructions)
    instr.degen_rtol > 0 && (trunc = truncdegen_keep(TruncationStrategy(; trunc...);
                                                     rtol = instr.degen_rtol,
                                                     maxextra = instr.degen_maxextra,
                                                     wmax = instr.degen_wmax))
    nU = n_unitaries(N, tau)
    n_checkpoint = instr.n_checkpoint

    savefile = load_locked(pathname*"N$(N)_T$(tau).jld2")
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

    fg = arrU -> begin
        func, grad = withgradient(cost_function, arrU)
        grad = project(arrU, grad[1])
        return func, grad
    end

    # === value-based (windowed) convergence settings =======================
    n_conv        = 500       # check the infidelity every n_conv iterations
    # err_reltol (kwarg, default 1e-3): stop when the relative decrease over the window < this
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
        if numiter % n_conv == 0
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
                    save_locked(pathname*"N$(N)_T$(tau)_gradbreak.jld2", errorfile)
                    throw(GradientFreeze(N, tau, numiter, gnorm, spread, n_stall, maxrank))
                end
            end
        end

        if numiter % n_checkpoint == 0
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
            save_locked(pathname*"N$(N)_T$(tau).jld2", ckpt)

            ckpt_instr = copy(instr)
            ckpt_instr.maxiter = max(instr.maxiter - numiter, 1)
            save_locked(pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2", ckpt_instr)
            @info "checkpoint: tau=$tau iter=$numiter gradnorm=$gnorm err=$err"
        end
        return x, f, g
    end

    # Converged if the windowed infidelity criterion fired (gradtol kept as a
    # harmless fallback that, in practice, essentially never triggers here).
    hasconverged = (x, f, g, normg) -> (converged_by_value[] || normg < gradtol)

    @show tau

    algorithm = LBFGS(m; maxiter = maxiter, gradtol = gradtol, verbosity = 2)
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
    save_locked(pathname*"N$(N)_T$(tau).jld2", result_tau)

    new_instr = copy(instr)
    save_locked(pathname*"N$(N)_T$(tau+1)_instructions.jld2", new_instr)

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
    save_locked(pathname*"N$(N)_T1_instructions.jld2", instructions)
    

    savefile = (N=N, tau=1, arrU=U_start, gradnorm=Inf, numfg=0, 
                normgradhistory=Matrix{Float64}(undef, 0, 2), time=0.0,
                converged=false, finished=false)
    save_locked(pathname*"N$(N)_T1.jld2", savefile)
end


"""
    continue_inversion(psi, maxtau, pathname, invertFunction; adapt = nothing)

Run the next piece of work (start the next depth, or resume the current one).
Every finished depth is appended to `N<N>_depths.csv`.

`adapt = (step = 5, max = 40)` turns a `GradientFreeze` into a maxrank
increase: the gradbreak point becomes a checkpoint, maxrank grows by `step`
(for this depth and, through the copied instructions, all later ones) and
the next call resumes from there. Past `max` the freeze is rethrown.
"""
function continue_inversion(psi::MPS, maxtau::Int, pathname::String, invertFunction::Function;
                            adapt = nothing)
    N = length(psi)
    pattern = Regex("N$(N)_T(\\d+)\\.jld2")
    taus = [parse(Int, m.captures[1]) for f in readdir(pathname)
            for m in [match(pattern, f)] if !isnothing(m)]
    isempty(taus) && error("No saved checkpoint files found in $pathname for N=$N")
    last_tau = maximum(taus)

    result = load_locked(pathname*"N$(N)_T$(last_tau).jld2")

    if get(result, :finished, true)  # default true for old files w/o the field
        if last_tau < maxtau
            tau = last_tau+1
            @info "Depth $last_tau finished (converged=$(get(result,:converged,true))). " *
                "Adding a layer, continuing at depth $(tau)."
            warmU = add_layer(result.arrU, N, last_tau)
            savefile = (N=N, tau=tau, arrU=warmU, gradnorm=Inf, numfg=0,
                        normgradhistory=Matrix{Float64}(undef, 0, 2), time=0.0,
                        converged=false, finished=false)
            save_locked(pathname*"N$(N)_T$(tau).jld2", savefile)
            resuming = false
        else
            @info "Required maxtau already reached for this state"
            return :done
        end
    else
        tau = last_tau
        resuming = !isinf(result.gradnorm)
        resuming && @info "Depth $tau interrupted mid-solve (gradnorm=$(result.gradnorm)). Resuming."
    end

    try
        invertFunction(psi, tau, pathname; resuming = resuming)
    catch e
        (e isa GradientFreeze && adapt !== nothing) || rethrow()
        raise_maxrank_after_freeze!(pathname, e, adapt) || rethrow()
        return :continue
    end

    done  = load_locked(pathname*"N$(N)_T$(tau).jld2")
    instr = load_locked(pathname*"N$(N)_T$(tau+1)_instructions.jld2")   # a copy of what ran
    log_depth_event(pathname, N; tau, event = "finished", maxrank = instr.maxrank, atol = instr.atol,
                    niter = size(done.normgradhistory, 1), err = get(done, :err, ""),
                    gradnorm = done.gradnorm, time = get(done, :time, ""),
                    note = "converged=$(get(done, :converged, "")) degen_rtol=$(instr.degen_rtol) degen_wmax=$(instr.degen_wmax)")
    return :continue
end


"""
Turn the point where `invert_maxrank` froze into a checkpoint with a larger
maxrank, so that the next `continue_inversion` resumes from it. Returns false
(leaving everything untouched) if the new maxrank would exceed `adapt.max`.

Resuming from the frozen point rather than the last checkpoint keeps all the
progress: there the degenerate pair is split by the cut, and with the larger
maxrank both values are kept, so the cost is smooth again.
"""
function raise_maxrank_after_freeze!(pathname, e::GradientFreeze, adapt)
    N, tau, old = e.N, e.tau, e.maxrank
    new = old + adapt.step
    base  = pathname * "N$(N)_T$(tau)"
    gb    = load_locked(base * "_gradbreak.jld2")
    prev  = load_locked(base * ".jld2")
    plain = load_locked(base * "_instructions.jld2")
    err  = -expm1(-gb.cost)
    info = (; tau, maxrank = old, atol = plain.atol, niter = e.niter, err,
              gradnorm = e.gradnorm, time = get(prev, :time, 0.0))

    if new > adapt.max
        log_depth_event(pathname, N; info..., event = "freeze",
                        note = "maxrank $old: limit $(adapt.max) reached - giving up")
        return false
    end

    # keep the evidence; a later freeze at the same depth must not overwrite it
    mv(base * "_gradbreak.jld2", base * "_gradbreak_chi$(old).jld2"; force = true)

    # The resume reads the checkpoint instructions (it may not exist yet, if the
    # freeze came before the first checkpoint). The plain file is raised too, so
    # that restart_from_depth and later depths never fall back to the old value.
    ckpt_path = base * "_checkpoint_instructions.jld2"
    ckpt = copy(isfile(ckpt_path) && !isinf(prev.gradnorm) ? load_locked(ckpt_path) : plain)
    ckpt.maxrank  = new
    plain = copy(plain)
    plain.maxrank = new
    save_locked(base * "_instructions.jld2", plain)
    save_locked(ckpt_path, ckpt)

    # gradnorm finite => continue_inversion takes the resume path. The history
    # between the last checkpoint and the freeze is lost; the time is too.
    save_locked(base * ".jld2",
                (N = N, tau = tau, arrU = gb.arrU, gradnorm = gb.gradnorm,
                 normgradhistory = prev.normgradhistory, cost = gb.cost, err = err,
                 time = get(prev, :time, 0.0), converged = false, finished = false))

    log_depth_event(pathname, N; info..., event = "freeze", note = "maxrank $old -> $new")
    @warn "N=$N tau=$tau: gradient froze at maxrank=$old (iter $(e.niter)); resuming with maxrank=$new"
    return true
end


"""
    restart_from_depth(psi, depth_start, depth_max, pathname, invertFunction; kwargs...)

Discard every saved depth ≥ `depth_start`, build a fresh instruction set for
`depth_start` from the one used at `depth_start - 1` with `kwargs` overridden,
and run up to `depth_max`.

    restart_from_depth(psi, 7, 12, pathname, invert_maxrank; n_checkpoint=500)
"""
function restart_from_depth(psi::MPS, depth_start::Int, depth_max::Int,
                            pathname::String, invertFunction::Function; kwargs...)
    N = length(psi)
    depth_start >= 2 || error("depth_start must be ≥ 2; use prepare_start for depth 1")

    prev = load_locked(pathname*"N$(N)_T$(depth_start-1).jld2")
    get(prev, :finished, true) ||
        error("depth $(depth_start-1) is not marked finished; finish or resume it first")

    # 1. delete results, instructions, checkpoints and gradbreaks for tau ≥ depth_start
    pattern = Regex("^N$(N)_T(\\d+)(_instructions|_checkpoint_instructions|_gradbreak(_chi\\d+)?)?\\.jld2\$")
    for f in readdir(pathname)
        m = match(pattern, f)
        isnothing(m) && continue
        parse(Int, m.captures[1]) >= depth_start && rm(pathname*f)
    end

    # 2. seed depth_start from the *plain* instructions of depth_start-1
    #    (the plain file always holds the full maxiter, never a resume remainder)
    instr = copy(load_locked(pathname*"N$(N)_T$(depth_start-1)_instructions.jld2"))
    for (k, v) in kwargs
        hasfield(InversionInstructions, k) || error("InversionInstructions has no field :$k")
        setproperty!(instr, k, v)   # setproperty!, not setfield!, so values get converted
    end
    save_locked(pathname*"N$(N)_T$(depth_start)_instructions.jld2", instr)
    @info "restarting at depth $depth_start with $((; kwargs...))"

    # 3. run
    while continue_inversion(psi, depth_max, pathname, invertFunction) != :done
    end
    return
end

"""
    edit_ongoing_simulation!(pathname, N; kwargs...)

Change parameters of a depth that was interrupted mid-solve. Kill the
run first, then call this, then resume with `continue_inversion`.

Picks whichever instructions file the resume will actually read: the
checkpoint one if a checkpoint was written before the kill, the plain
one if the run died before reaching its first checkpoint.

    edit_ongoing_simulation!(pathname, 20; n_checkpoint=500)
"""
function edit_ongoing_simulation!(pathname::String, N::Int; kwargs...)
    isempty(kwargs) && error("nothing to change")

    pattern = Regex("^N$(N)_T(\\d+)\\.jld2\$")
    taus = [parse(Int, m.captures[1]) for f in readdir(pathname)
            for m in [match(pattern, f)] if !isnothing(m)]
    isempty(taus) && error("no saved results in $pathname for N=$N")
    tau = maximum(taus)

    result = load_locked(pathname*"N$(N)_T$(tau).jld2")
    get(result, :finished, true) &&
        error("depth $tau is finished, nothing is mid-solve — " *
              "use restart_from_depth to redo it with different parameters")

    # exactly the choice continue_inversion / invert_maxrank make on resume
    resuming = !isinf(result.gradnorm)
    path = resuming ? pathname*"N$(N)_T$(tau)_checkpoint_instructions.jld2" :
                      pathname*"N$(N)_T$(tau)_instructions.jld2"

    instr = load_locked(path)
    for (k, v) in kwargs
        hasfield(InversionInstructions, k) || error("InversionInstructions has no field :$k")
        setproperty!(instr, k, v)   # setproperty!, not setfield!, so values get converted
    end
    save_locked(path, instr)
    @info "updated $((; kwargs...)) in $path (tau=$tau, resuming=$resuming)"
    return instr
end
