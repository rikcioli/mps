# =====================================================================
#  driver.jl — one pinned worker process per task, sized to whatever
#  node/allocation the job landed on (see cluster_setup.jl)
#      julia --project=$HOME/MyProject driver.jl xxz
#      julia --project=$HOME/MyProject driver.jl corr
#      julia --project=$HOME/MyProject driver.jl xxz,corr    # both, one pool
#      julia --project=$HOME/MyProject driver.jl nnn,isinghl,isingD,xxzD
#      julia --project=$HOME/MyProject driver.jl nnn,isinghl -smt   # also use the hyperthreads
#  jobs: xxz, ising, corr, xi4, nnn, isinghl, isingD, xxzD (see JOBS below)
#  On the cluster: sbatch run_xxz.sh corr
# =====================================================================

const DATA     = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/"
const LOCAL    = "/local/cioli/data/"   # states synced by sync_states.sh (nnn, isinghl jobs)
const MAXTAU   = 30
const PREPARE  = :auto         # :auto = only tasks whose outdir has no N$(N)_T*.jld2 yet
                               # true   = always (overwrites T1 but NOT deeper files!)
                               # false  = never, resume only
const ADAPT    = (step = 5, max = 40)   # on a gradient freeze raise maxrank by `step`,
                                        # up to `max`; `nothing` = fail as before
const BLAS_NT  = 1             # BLAS threads per worker; :auto = share out spare cores
const USE_MKL  = true          # false = OpenBLAS (try it if the node is AMD)

# ---------------------------------------------------------------- task lists
# One NamedTuple per inversion:
#   label    for the log
#   N        chain length
#   outdir   where the inversion files go (they are N-keyed, so several N
#            can share a directory)
#   h5       the MPS: read if the file exists, otherwise built from `make`
#            and written there, so a resumed run inverts the very same state
#   make     nothing (h5 must already exist) or (:corr, xi)
#   prepare  kwargs for prepare_start
#   adapt    (optional) overrides ADAPT for this task

function xxz_tasks()
    path = DATA * "xxz/Jz2.5/"
    [(label = "xxz N=$N", N = N, outdir = path * "m20_driver/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (m = 20, maxiter = 10_000_000))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

function ising_tasks()
    path = DATA * "ising/g1.5/"
    [(label = "ising N=$N", N = N, outdir = path * "m5_driver/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (maxrank = 10, maxiter = 10_000_000))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

function corr_tasks()
    path = DATA * "corrMPS/chi20/"
    [(label = "corr xi=$xi N=$N", N = N, outdir = path * "xi$(xi)/", h5 = path * "xi$(xi)/$(N)_mps.h5",
      make = (:corr, xi), prepare = (maxrank = 20, maxiter = 10_000_000))
     for xi in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] for N in [100, 200, 300]]
end

function xi4_tasks()
    path = DATA * "corrMPS/chi20/"
    [(label = "corr xi=$xi N=$N", N = N, outdir = path * "xi$(xi)/", h5 = path * "xi$(xi)/$(N)_mps.h5",
      make = (:corr, xi), prepare = (maxrank = 20, maxiter = 10_000_000))
      for xi in [4.0] for N in [20, 50, 100, 200, 500, 1000]]
end

# settings of the sequential Mac runs (data/nnn_Jz2.5_D2_*_seq/run.jl): stop each depth when the infidelity
# improves by less than err_reltol over 1000 iterations; rank raises at degenerate cuts at their defaults
const SEQ_PREPARE = (atol = 1e-10, maxiter = 10_000_000, gradtol = 1e-8, err_reltol = 1e-3)

# NNN XXZ, Δ = 2.5, Δ₂ = 1.0, staggered field hs = 0.05 (make_states_nnn_isinghl.jl)
function nnn_tasks()
    path = LOCAL * "nnn/delta2_1.0/"
    [(label = "nnn Δ2=1.0 N=$N", N = N, outdir = path * "seq_chi48/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (; maxrank = 48, m=20, SEQ_PREPARE...), adapt = (step = 1, max = 100))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

# Ising H = −Σ SˣSˣ − 0.75 Σ Sᶻ − 0.81 Σ Sˣ (spin-1/2 operators; make_states_nnn_isinghl.jl)
function isinghl_tasks()
    path = LOCAL * "ising/g0.75_hL0.81/"
    [(label = "ising g=0.75 hL=0.81 N=$N", N = N, outdir = path * "seq_chi10/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (; maxrank = 10, SEQ_PREPARE...), adapt = (step = 1, max = 100))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

# N = 100 targets truncated to bond dimension D (step by step D_max → … → D, make_truncated_states.jl),
# inverted at the maxrank of the full-D runs: Ising g = 0.75 (D_max = 9) at 10, XXZ Jz = 2.5 (D_max = 24) at 24
function isingD_tasks()
    [(label = "ising g=0.75 D=$D N=100", N = 100, outdir = LOCAL * "ising/g0.75_D$D/seq_chi10/",
      h5 = LOCAL * "ising/g0.75_D$D/100_mps.h5", make = nothing,
      prepare = (; maxrank = 10, SEQ_PREPARE...), adapt = (step = 1, max = 100))
     for D in [2, 3, 4]]
end

function xxzD_tasks()
    [(label = "xxz Jz=2.5 D=$D N=100", N = 100, outdir = LOCAL * "xxz/Jz2.5_D$D/seq_chi24/",
      h5 = LOCAL * "xxz/Jz2.5_D$D/100_mps.h5", make = nothing,
      prepare = (; maxrank = 24, m=20, SEQ_PREPARE...), adapt = (step = 1, max = 100))
     for D in [2, 3, 4, 8, 16]]
end

const JOBS = Dict("xxz" => xxz_tasks, "ising" => ising_tasks, "corr" => corr_tasks, "xi4" => xi4_tasks,
                  "nnn" => nnn_tasks, "isinghl" => isinghl_tasks, "isingD" => isingD_tasks, "xxzD" => xxzD_tasks)

# ---------------------------------------------------------------- setup

# Resolved HERE, on the master. @__DIR__ inside an @everywhere block would be
# expanded on each worker, where there is no source file, so it would fall back
# to the worker's pwd() — the submit directory, not necessarily this script's.
const LIBPATH   = joinpath(@__DIR__, "inversion_lib.jl")
const SETUPPATH = joinpath(@__DIR__, "cluster_setup.jl")

# -smt (anywhere on the command line): also put workers on the hyperthreads. Physical cores are
# filled first, then their sibling threads (see smt_order in cluster_setup.jl).
const SMT = "-smt" in ARGS
jobargs = filter(a -> !startswith(a, "-"), ARGS)
jobnames = split(get(jobargs, 1, "xxz"), ',')
for j in jobnames
    haskey(JOBS, j) || error("unknown job \"$j\", choose from $(join(keys(JOBS), ", "))")
end
# longest first, so the schedule packs well when tasks outnumber workers
tasks = sort(reduce(vcat, [JOBS[j]() for j in jobnames]); by = t -> t.N, rev = true)

# Created here, once: several workers racing on mkpath for a shared
# directory can fail with EEXIST.
foreach(t -> mkpath(t.outdir), tasks)

include(SETUPPATH)

# Load the stack on the master FIRST. Many workers hitting an unbuilt
# precompile cache at once serialises badly and can fail outright.
USE_MKL && using MKL
using LinearAlgebra, OptimKit, Zygote, JLD2, HDF5

node = detect_node()
plan = plan_workers(node, length(tasks); blas_threads = BLAS_NT, use_mkl = USE_MKL, use_smt = SMT)
print_report(node, plan)
@info "jobs $(join(jobnames, ",")): $(length(tasks)) tasks, smt = $SMT"
start_workers(plan)

@everywhere begin
    $USE_MKL && using MKL
    using LinearAlgebra, OptimKit, Zygote, JLD2, HDF5
    include($SETUPPATH)
    include($LIBPATH)
    configure_worker($(plan.blas_threads))
end
report_workers()

# ---------------------------------------------------------------- work

@everywhere function load_or_make_psi(t)
    if isfile(t.h5)
        psi = h5open(f -> read(f, "psi", MPS), t.h5, "r")
    elseif t.make === nothing
        error("$(t.h5) does not exist and task \"$(t.label)\" has no recipe to build it")
    elseif t.make[1] === :corr
        psi = correlated_mps_xi(siteinds("Qubit", t.N), t.make[2])
        h5open(f -> write(f, "psi", psi), t.h5, "w")
    else
        error("unknown recipe $(t.make)")
    end
    length(psi) == t.N || error("$(t.h5) holds an MPS of length $(length(psi)), expected $(t.N)")
    return dense(psi)
end

@everywhere has_checkpoints(t) =
    any(f -> occursin(Regex("^N$(t.N)_T\\d+\\.jld2\$"), f), readdir(t.outdir))

@everywhere function run_task(t, maxtau::Int, prepare, adapt)
    psi = load_or_make_psi(t)
    if prepare === true || (prepare === :auto && !has_checkpoints(t))
        prepare_start(psi, t.outdir; t.prepare...)
    end
    adapt = haskey(t, :adapt) ? t.adapt : adapt
    s = @elapsed while continue_inversion(psi, maxtau, t.outdir, invert_maxrank; adapt) != :done
    end
    return s
end

# Captured as locals, so the closure ships their values: a closure naming the
# globals makes Distributed re-create them on the workers, which on Julia 1.13
# evaluates PREPARE = :auto as `auto` and fails with UndefVarError.
results = let maxtau = MAXTAU, prepare = PREPARE, adapt = ADAPT
    pmap(tasks; on_error = identity) do t
        run_task(t, maxtau, prepare, adapt)
    end
end

for (t, r) in zip(tasks, results)
    r isa Exception ? @error("$(t.label) failed", exception = r) :
                      @info "$(t.label) finished in $(round(r/3600, digits=2)) h"
end
