include("mpsMethods.jl")
import .MPSMethods as mt
import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles
using Manopt, Manifolds, Random, ManifoldDiff
using ManifoldDiff: grad_distance, prox_distance


mutable struct Lightcone
    circuit::Vector{Vector{it.ITensor}}     # circuit[i] is the i-th layer, circuit[odd][1] acts on sites (1, 2), circuit[even][1] acts on sites (2, 3)
    d::Int64                                # dimension of the physical space
    size::Int64                             # number of qubits the whole circuit acts on
    depth::Int64                            # circuit depth
    lightbounds::Tuple{Bool, Bool}          # (leftslope == 'light', rightslope == 'light')
    sitesAB::Vector{it.Index}               # siteinds of AB region
    coords                                  # coordinates of all unitaries
    id_coords                               # coordinates of all identities
    layers                                  # coordinates of all gates ordered by layer
    range::Vector{Tuple{Int64, Int64}}      # leftmost and rightmost sites on which each layer acts non-trivially
end

function newLightcone(siteinds::Vector{<:it.Index}, depth; U_array = nothing, lightbounds = (true, true))

    # extract number of sites on which lightcone acts
    sizeAB = length(siteinds)
    isodd(sizeAB) &&
        throw(DomainError(sizeAB, "Choose an even number for sizeAB"))

    # count the minimum sizeAB has to be to construct a circuit of depth 'depth'
    n_light = count(lightbounds)    # count how many boundaries have 'light' type structure
    min_size = n_light*(depth+1) - 2*div(n_light,2)
    sizeAB < min_size &&
        throw(DomainError(sizeAB, "Cannot construct a depth $depth lightcone of with lightbounds $lightbounds with only $sizeAB sites"))
    
    # if U_array is not given, construct an array of random d-by-d unitaries
    d = siteinds[1].space
    if isnothing(U_array)
        n_unitaries = (sizeAB-1)*div(depth,2) + div(sizeAB,2)
        if depth > 2
            dep_m2 = depth-2
            n_unitaries -= n_light*(div(dep_m2+1,2)^2 + mod(dep_m2+1,2)*div(dep_m2+1,2))
        end
        U_array = [mt.random_unitary(d^2) for _ in 1:n_unitaries]
    end

    # finally convert the U_array 
    circuit::Vector{Vector{it.ITensor}} = []
    coords = []
    id_coords = []
    layers_coords = []
    range = []

    k = 1
    for i in 1:depth
        layer_i::Vector{it.ITensor} = []
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1,2)
        lslope, rslope = llim, rlim
        lrange, rrange = 1+mod(i+1,2), sizeAB-mod(i+1,2)
        if lightbounds[1]
            lslope += div(i-1,2)
            lrange += 2*div(i-1,2)
        end
        if lightbounds[2]
            rslope -= div(i-1,2)
            rrange -= 2*div(i-1,2)
        end
        push!(range, (lrange, rrange))

        layer_i_coords = []
        for j in llim:rlim
            inds = it.prime((siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1]), depth-i)
            fullinds = (inds[1], inds[2], inds[1]', inds[2]')
            if j < lslope || j > rslope
                brick = it.delta(inds[1], inds[1]') * it.delta(inds[2], inds[2]')
                push!(id_coords, ((i,j), fullinds))
            else
                brick = it.ITensor(U_array[k], fullinds)
                push!(coords, ((i,j), fullinds))
                k += 1
            end
            push!(layer_i, brick)
            push!(layer_i_coords, ((i,j), fullinds))
        end
        push!(circuit, layer_i)
        push!(layers_coords, layer_i_coords)
    end

    return Lightcone(circuit, d, sizeAB, depth, lightbounds, siteinds, coords, id_coords, layers_coords, range)

end


function lightcone_to_line(lightcone::Vector{Vector{it.ITensor}}, siteinds::Vector{<:it.Index}; lshape = "light", rshape = "light")
    
    # extract depth and number of sites on which lightcone acts
    depth = size(lightcone)[1]
    sizeAB = 2*size(lightcone[1])[1]
    d = it.inds(lightcone[1][1])[1].space

    U_array = []
    for i in 1:depth
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1,2)
        lslope, rslope = llim, rlim
        if lshape == "light"
            lslope += div(i-1,2)
        end
        if rshape == "light"
            rslope -= div(i-1,2)
        end

        for j in lslope:rslope
            inds_ij = [siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]']
            inds_ij = it.prime(inds_ij, depth-i)
            push!(U_array, reshape(Array(lightcone[i][j], inds_ij), (d^2,d^2)))
        end
    end
    return U_array
end


function line_to_lightcone(siteinds::Vector{<:it.Index}, depth; U_array = nothing, lshape = "light", rshape = "light")

    # extract number of sites on which lightcone acts
    sizeAB = length(siteinds)
    isodd(sizeAB) &&
        throw(DomainError(sizeAB, "Choose an even number for sizeAB"))

    # count the minimum sizeAB has to be to construct a circuit of depth 'depth'
    n_light = count((lshape, rshape) .== ("light", "light"))    # count how many boundaries have 'light' type structure
    min_size = n_light*(depth+1) - 2*div(n_light,2)
    sizeAB < min_size &&
        throw(DomainError(sizeAB, "Cannot construct a depth $depth lightcone of type ($lshape, $rshape) with only $sizeAB sites"))

    
    # if U_array is not given, construct an array of random d-by-d unitaries
    if isnothing(U_array)
        d = siteinds[1].space
        n_unitaries = (sizeAB-1)*div(depth,2) + div(sizeAB,2)
        if depth > 2
            dep_m2 = depth-2
            n_unitaries -= n_light*(div(dep_m2+1,2)^2 + mod(dep_m2+1,2)*div(dep_m2+1,2))
        end
        U_array = [mt.random_unitary(d^2) for _ in 1:n_unitaries]
    end

    # finally convert the U_array 
    circuit::Vector{Vector{it.ITensor}} = []
    ij_dict = Dict()
    k = 1
    for i in 1:depth
        layer_i::Vector{it.ITensor} = []
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1,2)
        lslope, rslope = llim, rlim
        if lshape == "light"
            lslope += div(i-1,2)
        end
        if rshape == "light"
            rslope -= div(i-1,2)
        end

        for j in llim:rlim
            if j < lslope || j > rslope
                brick = it.delta(siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)]') * it.delta(siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)+1]')
            else
                brick = it.ITensor(U_array[k], siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]')
                ij_dict[(i,j)] = k
                k += 1
            end
            it.prime!(brick, depth-i)
            push!(layer_i, brick)
        end
        push!(circuit, layer_i)
    end

    # reverse dictionary before returning it
    ij_dict = Dict(value => key for (key, value) in ij_dict)
    return circuit, ij_dict

end


function MPO(lightcone::Lightcone)
    # convert lightcone to mpo by using the ij_dict and the itmps.MPO(it.ITensor) function, keeping track of the deltas for even depths

    circ = lightcone.circuit
    depth = lightcone.depth
    siteinds = lightcone.siteinds

    mpo_list = []
    for layer in lightcone.layers
        qr_list::Vector{it.ITensor} = []
        i = layer[1][1][1]
        if iseven(i)    # add a delta at the edges for even layers
            push!(qr_list, it.prime(it.delta(siteinds[1], siteinds[1]'), depth-i))
        end
        for (pos, inds) in layer
            Q, R = it.qr(circ[pos[1]][pos[2]], inds[1], inds[3])
            push!(qr_list, Q)
            push!(qr_list, R)
        end
        if iseven(i)    # add a delta at the edges for even layers
            push!(qr_list, it.prime(it.delta(siteinds[end], siteinds[end]'), depth-i))
        end
        layer_mpo = itmps.MPO(qr_list)
        push!(mpo_list, layer_mpo)
    end

    return mpo_list
end


"Compute cost and gradient"
function fgLiu(lightcone, reduced_mps::Vector{it.ITensor})

    mps = [it.prime(tensor, lightcone.depth) for tensor in reduced_mps]
    conj_mps = deepcopy(mps)

    zero_vec = [1; [0 for _ in 1:lightcone.d-1]]
    zero_mat = kron(zero_vec, zero_vec')

    # apply each unitary to mps
    mt.contract!(conj_mps, lightcone)
    conj_mps = conj(conj_mps)
    for l in lightcone.range[end][1]:lightcone.range[end][2]
        ind = lightcone.sitesAB[l]
        zero_proj = it.ITensor(zero_mat, ind, ind')
        conj_mps[l] = it.noprime(conj_mps[l]*zero_proj)
    end

    len = length(lightcone.coords)
    grad = []
    for k in 1:len
        mps_low = deepcopy(mps)
        mps_up = deepcopy(conj_mps)
        for l in 1:k-1
            mt.contract!(mps_low, lightcone, l)
        end
        for l in len:-1:k+1
            mt.contract!(mps_up, lightcone, l)
        end

        push!(grad, mt.contract(mps_low, mps_up))
    end

    mt.contract!(mps, lightcone)
    cost = real(Array(mt.contract(mps, conj_mps)))

    return cost, grad
end


function invertMPSLiu(mps::itmps.MPS, sizeAB, tau; d = 2)

    isodd(sizeAB) &&
        throw(DomainError(sizeAB, "Choose an even number for sizeAB"))

    siteinds = it.siteinds(mps)
    k_sites = siteinds[1:sizeAB]
    # select lightcone shape 
    # left shape can be 'light' (/) or 'straight' (|)
    # right shape can be 'light' (\) or 'straight' (|)
    lshape = "light"
    rshape = "light"

    # iterate over regions
    for region in mps

        # for each region, we construct the cost function and its gradient
        # items needed for the cost: whole mps to access inds (actually only region needed for computations)
        # depth, array of unitary matrices which is a point on U(4)^n



    end


end

# Prepare initial state

N = 6
tau = 2
inverter_tau = 4 # 8 IS MAX TO CONTRACT ENVIRONMENT

"Select case:
1: FDQC of depth tau
2: Random MPS of bond dim tau
3: GHZ"
case = 2
proj = []

# unitary = mt.CX * kron(mt.H, mt.Id)
# sites = it.siteinds("Qubit", 4)
# mpo3 = mt.unitary_to_mpo(unitary; siteinds = sites, skip_qudits = 2)''
# mpo2 = mt.unitary_to_mpo(unitary; siteinds = sites, skip_qudits = 1)'
# mpo1 = mt.unitary_to_mpo(unitary; siteinds = sites)
# 
# mpo = mpo3 * mpo2 * mpo1
# results = mt.invertBW(mpo)

# GHZ

# Random bond D mps
# testMPS = rand_MPS(it.siteinds("Qubit", N), linkdims = 4)

let mps
    if case == 1
        # Random FDQC of depth tau
        mps = mt.initialize_fdqc(N, tau)

        if length(proj) > 0
            mps = mt.project_tozero(mps, proj)
        end

    elseif case == 2
        mps = mt.randMPS(it.siteinds("Qubit", N), tau)
    
    elseif case == 3
        mps = mt.initialize_ghz(N)
    end

    # NOTE: if malz is used, inverter_tau sets the value of blocking size q
    global W_list, U_list, fid, sweep, W_tau_list, U_tau_list = mt.invertMPSMalz(mps; eps_malz = 0.1, eps_bell = 0.1, eps_V = 0.3)
    println("Algorithm stopped after $sweep sweeps \nFidelity = $fid")

    #U = U_list[1][3]
    #inputind = it.inds(U_list[1][3])[end]
    #results = mt.invertBW(U, inputind, n_sweeps = 100)

    #W = reshape(Array(W_list[1], it.inds(W_list[1])), tau^2)
    #sites = it.siteinds(2, 4)
    #W = itmps.MPS(W, sites)

end





#unitaries = [reshape(Array(circ[i,j], it.inds(circ[i,j])), (4, 4)) for i in 1:tau, j in 1:div(N,2)]

## function execute(N::Int64; state_depth = 2, n_sweeps = 100, n_runs = 1000)
## 
##     fid1_depth_run::Vector{Vector{Float64}} = []
##     fid2_depth_run::Vector{Vector{Float64}} = []
##     avg_fid1::Vector{Float64} = []
##     avg_fid2::Vector{Float64} = []
## 
##     for depth in 1:5
##         println("Running depth $depth")
##         fid1_run::Vector{Float64} = []
##         fid2_run::Vector{Float64} = []
## 
##         for run in 1:n_runs
##             if mod(run, 100) == 0
##                 println("Run $run of n_runs")
##             end
##             # create random FDQC of depth 2
##             testMPS = initialize_vac(N)
##             siteinds = it.siteinds(testMPS)
##             testMPS = brickwork(testMPS, state_depth)
## 
##             # try to invert it
##             fid1, sweep1 = invertMPS(testMPS, depth, n_sweeps)
##             err1 = 1-sqrt(fid1)
##             push!(fid1_run, err1)
## 
##             # measure middle qubit
##             pos = div(N,2)
##             zero_projs::Vector{it.ITensor} = []
##             for ind in siteinds
##                 vec = [1; [0 for _ in 1:ind.space-1]]
##                 push!(zero_projs, it.ITensor(kron(vec, vec'), ind, ind'))
##             end
##             proj = zero_projs[pos]
##             testMPS[pos] = it.noprime(testMPS[pos]*proj)
##             it.normalize!(testMPS)
## 
##             # try to invert measured one
##             fid2, sweep2 = invertMPS(testMPS, depth, 100)
##             err2 = 1-sqrt(fid2)
##             push!(fid2_run, err2)
##         end
##         push!(fid1_depth_run, fid1_run)
##         push!(fid2_depth_run, fid2_run)
##         push!(avg_fid1, mean(fid1_run))
##         push!(avg_fid2, mean(fid2_run))
##     end
## 
##     return fid1_depth_run, fid2_depth_run, avg_fid1, avg_fid2
## 
## end
## 
## nruns = 10
## err1_all, err2_all, err1, err2 = execute(21, state_depth = 3, n_runs = nruns)
## 
## swapped_results1 = [getindex.(err1_all,i) for i=1:length(err1_all[1])]
## swapped_results2 = [getindex.(err2_all,i) for i=1:length(err2_all[1])]
## 
## Plots.plot(1:5, swapped_results1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err1, lc=:green, primary=false)
## Plots.plot!(1:5, err1, seriestype=:scatter, mc=:green, legend=true, label="Depth-3")
## Plots.plot!(yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"\epsilon = 1-|\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## 
## Plots.plot!(1:5, swapped_results2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_results2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## Plots.plot!(1:5, err2, lc=:red, primary=false)
## Plots.plot!(1:5, err2, seriestype=:scatter, mc=:red, label="Depth-3 + measure", legend=:bottomright)
## #Plots.plot!(title = L"N="*string(N), ylabel = L"\epsilon / M", xlabel = L"\tau")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\err_depth3.pdf");
## 
## 
## fid1, fid2 = 1 .- err1, 1 .- err2
## 
## swapped_fid1 = [1 .- swapped_results1[i, 1] for i in 1:nruns]
## swapped_fid2 = [1 .- swapped_results2[i, 1] for i in 1:nruns]
## 
## Plots.plot(1:5, swapped_fid1, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid1, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, swapped_fid2, lc=:gray90, primary=false)
## Plots.plot!(1:5, swapped_fid2, seriestype=:scatter, mc=:gray90, markersize=:3, primary=false)
## 
## Plots.plot!(1:5, fid1, lc=:green, primary=false)
## Plots.plot!(1:5, fid1, seriestype=:scatter, mc=:green, label="Depth-3", legend=:bottomright)
## Plots.plot!(ylims = (1E-1, 1.2), yscale=:log)
## Plots.plot!(title = L"N=21", ylabel = L"F = |\langle \psi_2|\psi_\tau\rangle|", xlabel = L"\tau")
## Plots.plot!(1:5, fid2, lc=:red, primary=false)
## Plots.plot!(1:5, fid2, seriestype=:scatter, mc=:red, label="Depth-3 + measure")
## #Plots.savefig("D:\\Julia\\MyProject\\Plots\\inverter\\fid_depth3.pdf");