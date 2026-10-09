# =====================================================================
#  driver_parallel.jl — invert every depth of every state at once, one pinned
#  worker per (state, depth) pair (same node/worker setup as driver.jl)
#
#      julia --project=$HOME/MyProject driver_parallel.jl corr                    # start / resume / add depths 1:MAXDEPTH
#      julia --project=$HOME/MyProject driver_parallel.jl corr status             # depth vs error table, no workers
#      julia --project=$HOME/MyProject driver_parallel.jl corr continue 10-14,18 maxiter=50000 err_reltol=1e-4
#      julia --project=$HOME/MyProject driver_parallel.jl corr continue 12 match="xi=1.0 N=100"
#
#  continue: the depths to give more iterations (e.g. "10-14,18"), then optional
#  key=value overrides of InversionInstructions fields for this continuation, and
#  match=<text> to restrict to the tasks whose label contains <text>.
#  Several jobs in one pool: driver_parallel.jl xxz,corr ...
# =====================================================================

const DATA     = "/home/PERSONALE/riccardo.cioli3/MyProject/Data/"
const MAXDEPTH = 30
const ADAPT    = (step = 5, max = 40)   # on a gradient freeze raise maxrank by `step`, up to `max`;
                                        # `nothing` = the depth fails instead
const BLAS_NT  = 1             # BLAS threads per worker; :auto = share out spare cores
const USE_MKL  = true          # false = OpenBLAS (try it if the node is AMD)

# ---------------------------------------------------------------- task lists
# As in driver.jl, but every outdir is a parallel/ subfolder: a parallel run must
# not share its directory with a depth-by-depth one (same file names). The states
# (h5) are the same files, so both kinds of run invert the very same state.

function xxz_tasks()
    path = DATA * "xxz/Jz2.5/"
    [(label = "xxz N=$N", N = N, outdir = path * "m20_driver/parallel/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (m = 20, maxiter = 10_000_000))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

function ising_tasks()
    path = DATA * "ising/g1.5/"
    [(label = "ising N=$N", N = N, outdir = path * "m5_driver/parallel/", h5 = path * "$(N)_mps.h5",
      make = nothing, prepare = (maxrank = 10, maxiter = 10_000_000))
     for N in [60, 100, 140, 180, 220, 260, 300]]
end

function corr_tasks()
    path = DATA * "corrMPS/chi20/"
    [(label = "corr xi=$xi N=$N", N = N, outdir = path * "xi$(xi)/parallel/", h5 = path * "xi$(xi)/$(N)_mps.h5",
      make = (:corr, xi), prepare = (maxrank = 20, maxiter = 10_000_000))
     for xi in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] for N in [100, 200, 300]]
end

function xi4_tasks()
    path = DATA * "corrMPS/chi20/"
    [(label = "corr xi=$xi N=$N", N = N, outdir = path * "xi$(xi)/parallel/", h5 = path * "xi$(xi)/$(N)_mps.h5",
      make = (:corr, xi), prepare = (maxrank = 20, maxiter = 10_000_000))
      for xi in [4.0] for N in [20, 50, 100, 200, 500, 1000]]
end

const JOBS = Dict("xxz" => xxz_tasks, "ising" => ising_tasks, "corr" => corr_tasks, "xi4" => xi4_tasks)

# ---------------------------------------------------------------- arguments

const LIBPATH   = joinpath(@__DIR__, "inversion_lib.jl")
const SETUPPATH = joinpath(@__DIR__, "cluster_setup.jl")

jobnames = split(get(ARGS, 1, "corr"), ',')
for j in jobnames
    haskey(JOBS, j) || error("unknown job \"$j\", choose from $(join(keys(JOBS), ", "))")
end
mode = get(ARGS, 2, "start")
mode in ("start", "status", "continue") || error("mode must be start, status or continue, not \"$mode\"")

"\"10-14,18\" -> [10, 11, 12, 13, 14, 18]"
parse_depths(s) = sort(unique(reduce(vcat, [occursin('-', p) ? collect(range(parse.(Int, split(p, '-'))...)) :
                                            [parse(Int, p)] for p in split(s, ',')])))
parse_value(v) = v == "nothing" ? nothing : something(tryparse(Int, v), tryparse(Float64, v), tryparse(Bool, v), v)

tasks = reduce(vcat, [JOBS[j]() for j in jobnames])
depths = 1:MAXDEPTH
overrides = Pair{Symbol,Any}[]
if mode == "continue"
    length(ARGS) >= 3 || error("continue needs the depths, e.g. 10-14,18")
    depths = parse_depths(ARGS[3])
    for a in ARGS[4:end]
        k, v = split(a, '='; limit = 2)
        if k == "match"
            global tasks = filter(t -> occursin(v, t.label), tasks)
        else
            push!(overrides, Symbol(k) => parse_value(v))
        end
    end
    isempty(tasks) && error("no task label matches")
end

# Created here, once: several workers racing on mkpath for a shared directory can fail.
foreach(t -> mkpath(t.outdir), tasks)

# ---------------------------------------------------------------- status (no workers)

USE_MKL && using MKL
using LinearAlgebra, OptimKit, Zygote, JLD2, HDF5

if mode == "status"
    include(LIBPATH)
    for t in tasks
        println("\n==== ", t.label, "   ", t.outdir)
        any(f -> occursin(Regex("^N$(t.N)_T\\d+\\.jld2\$"), f), readdir(t.outdir)) ?
            parallel_status(t.outdir, t.N) : println("  (nothing yet)")
    end
    exit()
end

# ---------------------------------------------------------------- workers

include(SETUPPATH)
npairs = length(tasks) * length(depths)
node = detect_node()
plan = plan_workers(node, npairs; blas_threads = BLAS_NT, use_mkl = USE_MKL)
print_report(node, plan)
@info "jobs $(join(jobnames, ",")), mode $mode: $(length(tasks)) states × $(length(depths)) depths = $npairs runs"
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

t0 = time()
results = mode == "start" ? parallel_inversion(tasks, depths; adapt = ADAPT) :
                            continue_parallel(tasks, depths; adapt = ADAPT, overrides...)
nfail = count(r -> r.result isa Exception, results)
@info "done in $(round((time() - t0) / 3600; digits = 2)) h: $(length(results) - nfail) runs ok, $nfail failed"
for t in tasks
    println("\n==== ", t.label)
    parallel_status(t.outdir, t.N)
end
