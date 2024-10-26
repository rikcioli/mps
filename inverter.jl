import ITensorMPS as itmps
import ITensors as it
import Plots
using LaTeXStrings, LinearAlgebra, Statistics, Random
using DelimitedFiles


H = [1 1
    1 -1]/sqrt(2)
Id = [1 0
      0 1]
T = [1 0
    0 exp(1im*pi/4)]
X = [0 1
    1 0]
CX = [1 0 0 0
      0 1 0 0
      0 0 0 1
      0 0 1 0]

"Returns random N x N unitary matrix sampled with Haar measure"
function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end     

"Apply 2-qubit matrix U to sites b and b+1 of MPS psi"
function apply(U::Matrix, psi::itmps.MPS, b::Integer; cutoff = 1E-14)
    psi = it.orthogonalize(psi, b)
    s = it.siteinds(psi)
    op = it.op(U, s[b+1], s[b])

    wf = it.noprime((psi[b]*psi[b+1])*op)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(wf, indsb, cutoff = cutoff)
    psi[b] = U
    psi[b+1] = S*V
    return psi
end


function initialize_vac(N::Int)
    nsites = N
    sites = it.siteinds("Qubit", nsites)
    states = ["0" for _ in 1:N]
    psiMPS = itmps.MPS(sites, states)
    return psiMPS
end


function entropy(psi::itmps.MPS, b::Integer)  
    it.orthogonalize!(psi, b)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(psi[b], indsb, cutoff = 1E-14)
    SvN = 0.0
    for n in 1:it.dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    return SvN
  end


function brickwork(psi::itmps.MPS, brick_odd::Matrix, brick_even::Matrix, t::Integer; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    layerOdd = [it.op(brick_odd, sites[b+1], sites[b]) for b in 1:2:(N-1)]
    layerEven = [it.op(brick_even, sites[b+1], sites[b]) for b in 2:2:(N-1)]
    for i in 1:t
        println("Evolving step $i")
        layer = isodd(i) ? layerOdd : layerEven
        psi = it.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end


"Apply depth-t brickwork of 2-local random unitaries"
function brickwork(psi::itmps.MPS, t::Integer; cutoff = 1E-14)
    N = length(psi)
    sites = it.siteinds(psi)
    d = sites[1].space
    for i in 1:t
        layer = [it.op(random_unitary(d^2), sites[b+1], sites[b]) for b in (isodd(i) ? 1 : 2):2:(N-1)]
        psi = it.apply(layer, psi, cutoff = cutoff)
    end
    return psi
end


function polar(block::it.ITensor, inds::Vector{it.Index{Int64}})
    spaceU = reduce(*, [ind.space for ind in inds])
    spaceV = reduce(*, [ind.space for ind in it.uniqueinds(block, inds)])
    #if spaceU > spaceV
    #    throw("Polar decomposition of block failed: make sure link space is bigger than physical space")
    #end
    U, S, V = it.svd(block, inds)
    u, v = it.inds(S)
    
    Vdag = it.replaceind(conj(V)', v', u)
    P = Vdag * S * V

    V = it.replaceind(V', v', u)
    W = U * V

    return W, P
end

"Computes polar decomposition but only returning P (efficient)"
function polar_P(block::Vector{it.ITensor}, inds::Vector{it.Index{Int64}})
    q = length(block)
    block_conj = [conj(tens)' for tens in block]
    left_tensor = block[1] * block_conj[1] * it.delta(inds[1], inds[1]')
    for j in 2:q
        left_tensor *= block[j] * block_conj[j] * it.delta(inds[j], inds[j]')
    end

    low_inds = reduce(it.noncommoninds, [block; inds])
    dim = reduce(*, [ind.space for ind in low_inds])
    mat = reshape(Array(left_tensor, (low_inds', low_inds)), (dim, dim))
    P = sqrt(mat)
    P = it.ITensor(P, (low_inds', low_inds))
    return P
end

"Extends m x n isometry to M x M unitary, where M = max(m, n)"
function iso_to_unitary(V::Union{Matrix, Vector{Float64}})
    nrows, ncols = size(V, 1), size(V, 2)
    dagger = false
    if ncols > nrows
        V = V'
        nrows, ncols = ncols, nrows
        dagger = true
    end
    V = V[:, vec(sum(abs.(V), dims=1) .> 1E-10)]
    nrows, ncols = size(V, 1), size(V, 2)

    bitlenght = length(digits(nrows-1, base=2))     # represent nrows in base 2
    D = 2^bitlenght

    U = zeros(ComplexF64, (D, D))
    U[1:nrows, 1:ncols] = V
    kerU = nullspace(U')
    U[:, ncols+1:D] = kerU

    if dagger
        U = U'
    end

    return U
end

"Performs blocking on an MPS, q sites at a time. Returns the blocked MPS as an array of it, together with the siteinds."
function blocking(mps::Union{Vector{it.ITensor},itmps.MPS}, q::Int)
    N = length(mps)
    newN = div(N, q)
    r = mod(N, q)

    block_mps = [mps[q*(i-1)+1 : q*i] for i in 1:newN] 
    if r != 0
        push!(block_mps, r > 1 ? mps[q * newN : end] : [mps[end]])
    end

    siteinds = reduce(it.noncommoninds, mps[1:end])
    sitegroups = [siteinds[q*(i-1)+1 : q*i] for i in 1:newN]
    if r != 0
        push!(sitegroups, r > 1 ? siteinds[q * newN : end] : [siteinds[end]])
        newN += 1
    end

    return block_mps, sitegroups
end

"Returns mps of Haar random isometries with bond dimension linkdims"
function rand_MPS(sites::Vector{<:it.Index}, linkdims)
    d = sites[1].space
    D = linkdims

    mps = itmps.MPS(sites, linkdims = D)
    links = it.linkinds(mps)

    U0 = it.ITensor(random_unitary(d*D), (sites[1], links[1]', links[1], sites[1]'))
    U_list = [it.ITensor(random_unitary(d*D), (sites[i], links[i-1], links[i], sites[i]')) for i in 2:N-1]
    UN = it.ITensor(random_unitary(d*D), (sites[N], links[N-1], links[N-1]', sites[N]'))

    zero_projs::Vector{it.ITensor} = [it.ITensor([1; [0 for _ in 2:d]], site') for site in sites]
    zero_L = it.ITensor([1; [0 for _ in 2:D]], links[1]')
    zero_R = it.ITensor([1; [0 for _ in 2:D]], links[N-1]')

    U0 *= zero_L
    UN *= zero_R
    U_list = [U0; U_list; UN]

    tensors = [zero_projs[i]*U_list[i] for i in 1:N]

    for i in 1:N
        mps[i] = tensors[i]
    end
    it.orthogonalize!(mps, N)
    it.normalize!(mps)

    return mps
end

"Converts order-4 ITensor to N x N matrix following the order of indices in 'order'"
function tensor_to_matrix(T::it.ITensor, order = [1,2,3,4])
    inds = it.inds(T)
    ord_inds = [inds[i] for i in order]
    M = reshape(Array(T, ord_inds), (ord_inds[1].space^2, ord_inds[1].space^2))
    return M
end


function invertMPS(mps::itmps.MPS, tau::Int64, n_sweeps::Int64)
    mps = conj(mps)
    N = length(mps)
    it.orthogonalize!(mps, N)
    siteinds = it.siteinds(mps)

    L_blocks::Vector{it.ITensor} = []
    R_blocks::Vector{it.ITensor} = []

    # create circuit of identities
    # circuit[i, j] = timestep i unitary acting on qubits (2j-1, 2j) if i odd or (2j, 2j+1) if i even
    #circuit = [it.prime(it.delta(siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)]') * it.delta(siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)+1]'), tau-i) for i in 1:tau, j in 1:div(N,2)]
    # create random circuit instead
    circuit = [it.prime(it.ITensor(random_unitary(4), siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1], siteinds[2*j-mod(i,2)]', siteinds[2*j-mod(i,2)+1]'), tau-i) for i in 1:tau, j in 1:div(N,2)]

    zero_projs::Vector{it.ITensor} = []
    for ind in siteinds
        vec = [1; [0 for _ in 1:ind.space-1]]
        push!(zero_projs, it.ITensor(vec, it.prime(ind, tau)))
    end

    # prepare gates on the edges, which are just identities
    left_deltas = [it.prime(it.delta(siteinds[1], siteinds[1]'), tau-i) for i in 2:2:tau]
    right_deltas = [it.prime(it.delta(siteinds[N], siteinds[N]'), tau-i) for i in 1:2:tau]

    # construct L_1
    leftmost_block = mps[1] * zero_projs[1] 
    if tau > 1
        leftmost_block *= reduce(*, left_deltas)
    end

    push!(L_blocks, leftmost_block)

    # construct R_N
    rightmost_block = mps[N] * zero_projs[N] * reduce(*, right_deltas)
    push!(R_blocks, rightmost_block) 

    # contract everything on the right and save rightmost_block at each intermediate step
    # must be done only the first time, when j=2
    for k in (N-1):-1:3
        # extract right gates associated with site k
        right_gates_k = [circuit[i, div(k,2)+mod(k,2)] for i in (2-mod(k,2)):2:tau]
        if length(right_gates_k) > 0
            rightmost_block *= reduce(*, right_gates_k)
        end
        rightmost_block *= mps[k]*zero_projs[k]
        push!(R_blocks, rightmost_block)
    end
    reverse!(R_blocks)

    # start the loop
    reverse = false
    first_sweep = true
    fid, newfid = 0, 0
    j = 2  
    sweep = 0

    while sweep < n_sweeps

        # extract all gates touching site j
        gates_j = [circuit[i, div(j,2) + mod(i,2)*mod(j,2)] for i in 1:tau]
        L_jm1 = leftmost_block
        R_jp1 = rightmost_block

        # optimize each gate
        for i in 1:tau
            gate_ji = gates_j[i]
            other_gates = [gates_j[1:i-1]; gates_j[i+1:end]]

            env = L_jm1 * mps[j] * zero_projs[j] * R_jp1
            if length(other_gates) > 0
                env *= reduce(*, other_gates)
            end
            env = conj(env)

            inds = it.commoninds(it.prime(siteinds, tau-i), gate_ji)
            U, S, Vdag = it.svd(env, inds, cutoff = 1E-14)
            u, v = it.commonind(U, S), it.commonind(Vdag, S)

            # evaluate fidelity and stop if converged
            newfid = real(tr(Array(S, (u, v))))^2
            #println("Step $j: ", newfid)

            #replace gate_ji with optimized one, both in gates_j (used in this loop) and in circuit
            gate_ji_opt = U * it.replaceind(Vdag, v, u)
            gates_j[i] = gate_ji_opt    
            circuit[i, div(j,2) + mod(i,2)*mod(j,2)] = gate_ji_opt
        end

        ## while loop end conditions

        ## if isapprox(newfid, fid) && (first_sweep == false)
        ##     break
        ## end
        if abs(1 - newfid) < 1E-10
            break
        end
        fid = newfid

        if j == N-1
            first_sweep = false
            reverse = true
            sweep += 1
        end

        if j == 2 && first_sweep == false
            reverse = false
            sweep += 1
        end

        # update L_blocks or R_blocks depending on sweep direction
        if reverse == false
            gates_to_append = gates_j[(1+mod(j,2)):2:end]     # follow fig. 6
            if length(gates_to_append) > 0
                leftmost_block *= reduce(*, gates_to_append)
            end
            leftmost_block *= mps[j]*zero_projs[j]
            if first_sweep == true
                push!(L_blocks, leftmost_block)
            else
                L_blocks[j] = leftmost_block
            end
            rightmost_block = R_blocks[j]       #it should be j+2 but R_blocks starts from site 3
        else
            gates_to_append = gates_j[(2-mod(j,2)):2:end]
            if length(gates_to_append) > 0
                rightmost_block *= reduce(*, gates_to_append)
            end
            rightmost_block *= mps[j]*zero_projs[j]
            R_blocks[j-2] = rightmost_block         #same argument
            leftmost_block = L_blocks[j-2]
        end

        j = reverse ? j-1 : j+1

    end

    return newfid, sweep
end


# Prepare initial state

N = 21
tau = 2
testMPS = initialize_vac(N)
siteinds = it.siteinds(testMPS)

#testMPS = brickwork(testMPS, kron(H, Id), kron(H, Id), 1)
# brick = CX * kron(H, Id)
# testMPS = apply(brick, testMPS, 1)
# for i in 2:N-1
#     global testMPS = apply(CX, testMPS, i)
# end
# testMPS = rand_MPS(it.siteinds("Qubit", N), linkdims = 4)


# create random FDQC of depth 2
testMPS = brickwork(testMPS, 3)

# measure at half chain
zero_projs::Vector{it.ITensor} = []
for ind in siteinds
    vec = [1; [0 for _ in 1:ind.space-1]]
    push!(zero_projs, it.ITensor(kron(vec, vec'), ind, ind'))
end

pos = div(N,2)
proj = zero_projs[pos]

testMPS[pos] = it.noprime(testMPS[pos]*proj)
it.normalize!(testMPS)


fid, sweep = invertMPS(testMPS, 3, 100)
println("Algorithm stopped after $sweep sweeps \nFidelity = $fid")

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