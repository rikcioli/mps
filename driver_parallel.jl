# =====================================================================
#  driver_parallel.jl — invert every depth of every state at once, one pinned
#  worker per (state, depth) pair (same node/worker setup as driver.jl)
#
#      julia --project=$HOME/MyProject driver_parallel.jl [global flags] folder [flags] [folder [flags] ...]
#
#  Every <N>_mps.h5 in a folder is a state to invert; results go to <folder>/parallel/
#  (never mixed with depth-by-depth runs, which use the same file names).
#  Flags before the first folder apply to every folder; flags after a folder apply
#  to that folder only, overriding the global ones.
#
#  Per folder (or global):
#    -T 1_28,30        depths (ranges with _), required unless -status
#    -N 100,200        only these states (default: all <N>_mps.h5 in the folder)
#    -continue         continue the depths instead of starting them
#    -<field> <value>  any InversionInstructions field: -maxrank 20 -maxiter 10000000
#                      -m 20 -err_reltol 0.001 -gradtol 0 -n_snapshot 10000 -atol 1e-10 ...
#                      start: the instructions of the depths this call prepares
#                      (existing depths keep theirs); continue: overrides for this
#                      continuation (-maxiter = iterations of the continuation)
#  Global only:
#    -status           print the depth vs error table of every state, start no workers
#    -at 1e4,1e5       with -status: error after that many iterations (one column each)
#                      (-status and -at may appear anywhere)
#    -adapt 5,40       on a gradient freeze raise maxrank by 5, up to 40 (default); -adapt none
#    -blas 1           BLAS threads per worker (default 1; "auto" shares out spare cores)
#                      (-adapt and -blas go before the first folder)
#
#  Examples
#    driver_parallel.jl /data/xi1.0 -T 1_28,30 -maxrank 20 -maxiter 10000000 -m 20 -err_reltol 0.001 -n_snapshot 10000
#    driver_parallel.jl /data/xi1.0 -continue -T 12_14 -maxiter 50000 -err_reltol 1e-4
#    driver_parallel.jl -T 1_30 -maxiter 10000000  /data/xi1.0 -maxrank 20  /data/xi2.0 -maxrank 30 -T 1_20  /data/xi5.0 -continue -T 12
#    driver_parallel.jl /data/xi1.0 /data/xi2.0 -status -at 10000,100000
# =====================================================================

const USE_MKL = true            # false = OpenBLAS (try it if the node is AMD)
const LIBPATH   = joinpath(@__DIR__, "inversion_lib.jl")
const SETUPPATH = joinpath(@__DIR__, "cluster_setup.jl")

# Load the stack on the master first (many workers hitting an unbuilt precompile
# cache at once serialises badly), and the library: it is needed here to check
# the flags, find and prepare the states, and print the status.
USE_MKL && using MKL
using LinearAlgebra, OptimKit, Zygote, JLD2, HDF5, Distributed
include(LIBPATH)
include(SETUPPATH)

# ---------------------------------------------------------------- arguments

const NOVALUE     = ("continue", "status")                 # flags without a value
const GLOBAL_ONLY = ("status", "at", "adapt", "blas")
const FOLDER_KEYS = ("T", "N", "continue")

"\"1_28,30\" -> [1, 2, …, 28, 30]"
parse_depths(s) = sort(unique(reduce(vcat, [occursin('_', p) ? collect(range(parse.(Int, split(p, '_'))...)) :
                                            [parse(Int, p)] for p in split(s, ',')])))
parse_list(s) = [round(Int, parse(Float64, x)) for x in split(s, ',')]
parse_value(v) = v == "nothing" ? nothing : something(tryparse(Int, v), tryparse(Float64, v), tryparse(Bool, v), v)

function parse_args(args)
    isempty(args) && error("usage: driver_parallel.jl [global flags] folder [flags] [folder [flags] ...] (see the header)")
    global_opts = Dict{String,String}()
    folders = Tuple{String,Dict{String,String}}[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "-")
            key = a[2:end]
            known = key in NOVALUE || key in GLOBAL_ONLY || key in FOLDER_KEYS || hasfield(InversionInstructions, Symbol(key))
            known || error("unknown flag $a (not a driver flag nor an InversionInstructions field)")
            # -status and -at are global wherever they appear; -adapt and -blas must come first
            anywhere = key in ("status", "at")
            key in GLOBAL_ONLY && !anywhere && !isempty(folders) &&
                error("$a applies to the whole call: put it before the first folder")
            opts = (anywhere || isempty(folders)) ? global_opts : folders[end][2]
            if key in NOVALUE
                opts[key] = "true"; i += 1
            else
                i < length(args) || error("$a needs a value")
                opts[key] = args[i+1]; i += 2
            end
        else
            isdir(a) || error("$a is not a folder")
            push!(folders, (a, Dict{String,String}())); i += 1
        end
    end
    isempty(folders) && error("no folder given")
    return global_opts, folders
end

global_opts, folders = parse_args(ARGS)
status = haskey(global_opts, "status")
at     = haskey(global_opts, "at") ? parse_list(global_opts["at"]) : nothing
adapt  = get(global_opts, "adapt", "5,40") == "none" ? nothing :
         (s = parse_list(get(global_opts, "adapt", "5,40")); (step = s[1], max = s[2]))
blas   = get(global_opts, "blas", "1") == "auto" ? :auto : parse(Int, get(global_opts, "blas", "1"))

tasks = NamedTuple[]
for (folder, own) in folders
    o = merge(global_opts, own)                    # folder flags override global ones
    dir = rstrip(folder, '/')
    Ns = sort([parse(Int, m[1]) for f in readdir(dir) for m in [match(r"^(\d+)_mps\.h5$", f)] if m !== nothing])
    haskey(o, "N") && (Ns = filter(in(parse_list(o["N"])), Ns))
    isempty(Ns) && error("$folder: no <N>_mps.h5 state", haskey(o, "N") ? " with N in $(o["N"])" : "")
    status || haskey(o, "T") || error("$folder: no depths, give -T (e.g. -T 1_30)")
    depths = haskey(o, "T") ? parse_depths(o["T"]) : nothing
    settings = (; (Symbol(k) => parse_value(v) for (k, v) in o
                   if !(k in FOLDER_KEYS || k in GLOBAL_ONLY))...)
    for N in Ns
        push!(tasks, (label = "$dir N=$N", N = N, outdir = dir * "/parallel/", h5 = dir * "/$(N)_mps.h5",
                      make = nothing, depths = depths, mode = haskey(o, "continue") ? :continue : :start,
                      settings = settings))
    end
end

# ---------------------------------------------------------------- status (no workers)

if status
    for t in tasks
        println("\n==== ", t.label, "   ", t.outdir)
        isdir(t.outdir) && any(f -> occursin(Regex("^N$(t.N)_T\\d+\\.jld2\$"), f), readdir(t.outdir)) ?
            parallel_status(t.outdir, t.N; at, depths = t.depths) : println("  (nothing yet)")
    end
    exit()
end

for t in tasks
    @info "$(t.label): $(t.mode) depths $(t.depths) with $(t.settings)"
end
foreach(t -> mkpath(t.outdir), tasks)   # here, once: workers racing on mkpath can fail

# ---------------------------------------------------------------- workers

npairs = sum(length(t.depths) for t in tasks)
node = detect_node()
plan = plan_workers(node, npairs; blas_threads = blas, use_mkl = USE_MKL)
print_report(node, plan)
@info "$(length(tasks)) states, $npairs (state, depth) runs, adapt = $adapt"
start_workers(plan)

@everywhere workers() begin
    $USE_MKL && using MKL
    using LinearAlgebra, OptimKit, Zygote, JLD2, HDF5
    include($SETUPPATH)
    include($LIBPATH)
    configure_worker($(plan.blas_threads))
end
report_workers()

# ---------------------------------------------------------------- work

t0 = time()
results = run_parallel_tasks(tasks; adapt)
nfail = count(r -> r.result isa Exception, results)
@info "done in $(round((time() - t0) / 3600; digits = 2)) h: $(length(results) - nfail) runs ok, $nfail failed"
for t in tasks
    println("\n==== ", t.label)
    parallel_status(t.outdir, t.N; depths = t.depths)
end
