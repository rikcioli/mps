# =====================================================================
#  cluster_setup.jl — look at the node this job landed on, then start
#  one isolated, pinned Distributed worker per task, sized to it.
#
#      include("cluster_setup.jl")
#      node = detect_node()
#      plan = plan_workers(node, ntasks; blas_threads = 1)
#      print_report(node, plan)
#      start_workers(plan)
#
#  Each worker is a separate process: its own GC, heap and BLAS thread
#  team, pinned with taskset to its own physical cores (never straddling
#  a NUMA node). Linux only for the detection/pinning; elsewhere it falls
#  back to unpinned workers so the same driver still runs on a laptop.
# =====================================================================

using Distributed, LinearAlgebra

struct LogicalCPU
    id::Int          # OS cpu number, what taskset -c takes
    package::Int     # socket
    core::Int        # core_id, unique within a package
    numa::Int
end

struct NodeInfo
    hostname::String
    model::String
    vendor::String
    flags::Set{String}
    cpus::Vector{LogicalCPU}     # only the CPUs this job is allowed to use
    mem_bytes::Int               # memory this job may use
    mem_source::String
end

struct WorkerSlot
    cpus::Vector{Int}            # logical cpus this worker is pinned to
    numa::Int
end

struct Plan
    slots::Vector{WorkerSlot}
    blas_threads::Int
    heap_hint_bytes::Int
    use_mkl::Bool
    pin::Bool
    notes::Vector{String}
end

# ---------------------------------------------------------------- detection

tryread(path) = try read(path, String) catch; nothing end
readint(path, default) = something(tryparse(Int, strip(something(tryread(path), ""))), default)

"\"0-3,8,10-11\" -> [0,1,2,3,8,10,11]"
function parse_cpulist(s::AbstractString)
    out = Int[]
    for part in split(strip(s), ','; keepempty = false)
        if occursin('-', part)
            a, b = parse.(Int, split(part, '-'))
            append!(out, a:b)
        else
            push!(out, parse(Int, part))
        end
    end
    return out
end

# The affinity mask is what SLURM's cgroup actually lets us run on;
# SLURM_CPUS_PER_TASK and Sys.CPU_THREADS can both disagree with it.
function allowed_cpus()
    if Sys.islinux()
        for line in eachline("/proc/self/status")
            startswith(line, "Cpus_allowed_list:") && return parse_cpulist(split(line, ':')[2])
        end
    end
    return collect(0:Sys.CPU_THREADS-1)
end

function numa_of(cpu)
    d = "/sys/devices/system/cpu/cpu$cpu"
    isdir(d) || return 0
    for f in readdir(d)
        m = match(r"^node(\d+)$", f)
        m === nothing || return parse(Int, m[1])
    end
    return 0
end

# Off Linux the sysfs reads fail and every logical cpu counts as a core.
function cpu_topology(ids)
    map(ids) do c
        t = "/sys/devices/system/cpu/cpu$c/topology/"
        LogicalCPU(c, readint(t * "physical_package_id", 0), readint(t * "core_id", c), numa_of(c))
    end
end

function cpu_identity()
    model  = strip(Sys.cpu_info()[1].model)
    vendor = occursin("AMD", model) ? "AuthenticAMD" : occursin("Intel", model) ? "GenuineIntel" : "unknown"
    flags  = Set{String}()
    if Sys.islinux()
        for line in eachline("/proc/cpuinfo")
            occursin(':', line) || continue
            k, v = strip.(split(line, ':'; limit = 2))
            k == "vendor_id" && (vendor = v)
            k == "flags" && (flags = Set(split(v)); break)    # first cpu is enough
        end
    end
    return model, vendor, flags
end

# SLURM enforces --mem through a cgroup. The limit sits on the job cgroup
# while the step below it often says "max", so walk up and keep the tightest.
function cgroup_mem_limit()
    Sys.islinux() || return nothing
    lines = try readlines("/proc/self/cgroup") catch; return nothing end
    best = nothing
    for line in lines
        parts = split(line, ':'; limit = 3)
        length(parts) == 3 || continue
        _, ctrls, path = parts
        if isempty(ctrls)                                    # cgroup v2
            base, file = "/sys/fs/cgroup", "memory.max"
        elseif "memory" in split(ctrls, ',')                 # cgroup v1
            base, file = "/sys/fs/cgroup/memory", "memory.limit_in_bytes"
        else
            continue
        end
        p = String(path)
        while true
            v = tryparse(Int, strip(something(tryread(joinpath(base, lstrip(p, '/'), file)), "")))
            v === nothing || (best = best === nothing ? v : min(best, v))
            p == "/" && break
            p = dirname(p)
        end
    end
    return best
end

function slurm_mem_limit()
    mb(k) = tryparse(Int, get(ENV, k, ""))
    pernode = mb("SLURM_MEM_PER_NODE")
    pernode === nothing || return pernode * 2^20
    percpu, ncpu = mb("SLURM_MEM_PER_CPU"), mb("SLURM_CPUS_ON_NODE")
    (percpu === nothing || ncpu === nothing) || return percpu * ncpu * 2^20
    return nothing
end

function detect_node()
    model, vendor, flags = cpu_identity()
    cands = [("physical RAM", Int(Sys.total_memory())),
             ("cgroup", cgroup_mem_limit()),
             ("SLURM env", slurm_mem_limit())]
    filter!(c -> c[2] !== nothing, cands)
    src, mem = argmin(last, cands)
    return NodeInfo(gethostname(), model, vendor, flags,
                    cpu_topology(allowed_cpus()), mem, src)
end

# ----------------------------------------------------------------- planning

# One representative logical cpu per physical core (the lowest id); the
# hyperthread siblings stay idle, since two BLAS threads on one core gain
# next to nothing and fight over its caches.
function physical_cores(cpus)
    rep = Dict{Tuple{Int,Int}, LogicalCPU}()
    for c in cpus
        k = (c.package, c.core)
        (!haskey(rep, k) || c.id < rep[k].id) && (rep[k] = c)
    end
    return sort!(collect(values(rep)); by = c -> (c.numa, c.package, c.core))
end

# Cut each NUMA node into blocks of b cores, so no worker straddles sockets,
# then interleave the nodes, so a job with few workers still uses every
# memory controller and L3 instead of piling onto socket 0.
function core_blocks(cores, b)
    pernode = [collect(Iterators.partition([c for c in cores if c.numa == n], b))
               for n in sort(unique(c.numa for c in cores))]
    foreach(bl -> filter!(x -> length(x) == b, bl), pernode)
    blocks = Vector{Vector{LogicalCPU}}()
    for i in 1:maximum(length, pernode; init = 0), bl in pernode
        i <= length(bl) && push!(blocks, collect(bl[i]))
    end
    return blocks
end

"""
    plan_workers(node, ntasks; blas_threads = 1, use_mkl = true, use_smt = false,
                 master_mem = 2^31, mem_reserve = 0.10)

`blas_threads` is an Int, or `:auto` to hand the cores left over after one
per task to the tasks' BLAS teams. `use_smt = true` counts hyperthreads as
cores. `master_mem` is kept back for the driver process, and a further
`mem_reserve` fraction for OS/page cache/overshoot; the rest is split into
per-worker `--heap-size-hint`s.
"""
function plan_workers(node::NodeInfo, ntasks::Int; blas_threads = 1, use_mkl = true,
                      use_smt = false, master_mem = 2 * 2^30, mem_reserve = 0.10)
    notes = String[]
    cores = use_smt ? sort(node.cpus; by = c -> (c.numa, c.package, c.core, c.id)) :
                      physical_cores(node.cpus)
    ncores = length(cores)

    if blas_threads === :auto
        # largest b that still gives every task its own block
        b = max(1, ncores ÷ max(ntasks, 1))
        while b > 1 && length(core_blocks(cores, b)) < ntasks
            b -= 1
        end
    else
        b = clamp(blas_threads, 1, ncores)
    end
    blocks = core_blocks(cores, b)
    nw = min(ntasks, length(blocks))
    slots = [WorkerSlot([c.id for c in blocks[i]], blocks[i][1].numa) for i in 1:nw]

    heap = floor(Int, (node.mem_bytes - master_mem) * (1 - mem_reserve) / max(nw, 1))
    pin  = Sys.islinux() && Sys.which("taskset") !== nothing

    idle = ncores - nw * b
    idle > 0 && push!(notes, "$idle of $ncores cores left idle")
    ntasks > nw && push!(notes, "$ntasks tasks on $nw workers: pmap will queue the rest")
    heap < 2 * 2^30 && push!(notes, "only $(fmtbytes(heap)) heap per worker, consider fewer workers")
    Sys.islinux() && !pin && push!(notes, "taskset not found: workers will NOT be pinned")
    use_mkl && node.vendor == "AuthenticAMD" &&
        push!(notes, "AMD CPU with MKL: MKL may be slow here, benchmark with use_mkl = false (OpenBLAS)")
    tgt = get(ENV, "JULIA_CPU_TARGET", "")
    if "avx512f" in node.flags && !isempty(tgt) &&
       !occursin(r"avx512|skylake-avx512|cascadelake|icelake|sapphirerapids|znver4", tgt)
        push!(notes, "CPU has AVX-512 but JULIA_CPU_TARGET does not target it (MKL itself is unaffected)")
    end
    return Plan(slots, b, heap, use_mkl, pin, notes)
end

# ------------------------------------------------------------------ report

fmtbytes(x) = x >= 2^30 ? "$(round(x / 2^30; digits = 1)) GiB" : "$(round(x / 2^20; digits = 1)) MiB"

function print_report(node::NodeInfo, plan::Plan)
    cores  = physical_cores(node.cpus)
    numas  = sort(unique(c.numa for c in node.cpus))
    socks  = sort(unique(c.package for c in node.cpus))
    simd   = join(filter(in(node.flags), ["avx2", "fma", "avx512f"]), " ")
    io = IOBuffer()
    println(io, "── node ─────────────────────────────────────────")
    println(io, "host        ", node.hostname)
    println(io, "cpu         ", node.model, "  [", node.vendor, "]  ", simd)
    println(io, "allowed     ", length(node.cpus), " logical cpus = ", length(cores),
                " physical cores on ", length(socks), " socket(s), ", length(numas), " NUMA node(s)")
    for n in numas
        println(io, "  numa $n    ", count(c -> c.numa == n, cores), " cores")
    end
    println(io, "memory      ", fmtbytes(node.mem_bytes), " (from ", node.mem_source, ")")
    println(io, "── plan ─────────────────────────────────────────")
    println(io, "workers     ", length(plan.slots), " × ", plan.blas_threads, " BLAS thread(s), ",
                plan.use_mkl ? "MKL" : "OpenBLAS", plan.pin ? ", pinned" : ", NOT pinned")
    println(io, "heap hint   ", fmtbytes(plan.heap_hint_bytes), " per worker")
    for (i, s) in enumerate(plan.slots)
        println(io, "  worker $i  numa ", s.numa, "  cpus ", join(s.cpus, ","))
    end
    foreach(n -> println(io, "note        ", n), plan.notes)
    @info String(take!(io))
end

# ------------------------------------------------------------------ workers

"""
    start_workers(plan; project = Base.active_project(), exeflags = String[]) -> worker ids

One `addprocs` per slot, each launched under its own `taskset` mask so
every thread the worker ever creates (GC, libuv, BLAS) inherits the pinning.
"""
function start_workers(plan::Plan; project = Base.active_project(), exeflags = String[])
    b     = string(plan.blas_threads)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    env   = ["MKL_NUM_THREADS" => b, "OPENBLAS_NUM_THREADS" => b, "OMP_NUM_THREADS" => b]
    flags = ["--threads=1", "--project=$project",
             "--heap-size-hint=$(plan.heap_hint_bytes ÷ 2^20)M",
             # GC marking can use this worker's own cores: they sit idle
             # during a collection anyway, and belong to nobody else
             "--gcthreads=$b",
             exeflags...]
    ids = Int[]
    for s in plan.slots
        exe = plan.pin ? `taskset -c $(join(s.cpus, ',')) $julia` : julia
        append!(ids, addprocs(1; exename = exe, exeflags = flags, env = env))
    end
    return ids
end

# Call on every worker after the BLAS backend is loaded (i.e. after `using MKL`).
configure_worker(blas_threads) = BLAS.set_num_threads(blas_threads)

function worker_report()
    aff = "?"
    if Sys.islinux()
        for line in eachline("/proc/self/status")
            startswith(line, "Cpus_allowed_list:") && (aff = strip(split(line, ':')[2]))
        end
    end
    lib = basename(first(BLAS.get_config().loaded_libs).libname)
    return (id = myid(), aff = aff, blas = BLAS.get_num_threads(), lib = lib,
            gc = Threads.ngcthreads(), jt = Threads.nthreads())
end

# Checks that what the workers actually got matches the plan.
function report_workers(ids = workers())
    io = IOBuffer()
    println(io, "── workers as started ───────────────────────────")
    for r in [remotecall_fetch(worker_report, w) for w in ids]
        println(io, "worker ", lpad(r.id, 3), "  cpus ", rpad(r.aff, 12), " BLAS ", r.blas,
                    " (", r.lib, ")  julia threads ", r.jt, "  gc threads ", r.gc)
    end
    @info String(take!(io))
end
