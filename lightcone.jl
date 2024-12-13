
### LIGHTCONE METHODS ###

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
    isodd(sizeAB) && lightbounds[2] == true &&
        throw(DomainError(sizeAB, "Right lightbound can't be true for sizeAB odd"))

    # count the minimum sizeAB has to be to construct a circuit of depth 'depth'
    n_light = count(lightbounds)    # count how many boundaries have 'light' type structure
    min_size = n_light*(depth+1) - 2*div(n_light,2)
    if sizeAB < min_size
        @warn "Cannot construct a depth $depth lightcone with lightbounds $lightbounds with only $sizeAB sites. Attempting to change lightbounds..."
        lightbounds = (true, false)
        n_light = count(lightbounds)    
        min_size = n_light*(depth+1) - 2*div(n_light,2)
        if sizeAB >= min_size
            @warn "Lightbounds changed to $lightbounds"
        else
            throw(DomainError(sizeAB, "Cannot construct a depth $depth lightcone with lightbounds $lightbounds with only $sizeAB sites."))
        end
    end
    
    # if U_array is not given, construct an array of random d-by-d unitaries
    d = siteinds[1].space
    if isnothing(U_array)
        n_unitaries = (sizeAB-1)*div(depth,2) + div(sizeAB,2)
        if depth > 2
            dep_m2 = depth-2
            n_unitaries -= n_light*(div(dep_m2+1,2)^2 + mod(dep_m2+1,2)*div(dep_m2+1,2))
        end
        U_array = [random_unitary(d^2) for _ in 1:n_unitaries]
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
        # limits for circuit indices, left limit is always 1
        # right limit is sizeAB/2 for odd layers, sizeAB/2 or sizeAB/2-1 for even layers depending on the parity of sizeAB
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1,2)
        lslope, rslope = llim, rlim
        # left range is 1 for odd layers, 2 for even layers
        # right range depends on sizeAB: if sizeAB even it's just sizeAB and sizeAB-1; if it's odd it's sizeAB-1 and sizeAB
        lrange, rrange = 1+mod(i+1,2), (iseven(sizeAB) ? sizeAB-mod(i+1,2) : sizeAB-mod(i,2))
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
        # sweep over unitaries, from left to right if depth odd, otherwise from right to left
        # this will be helpful with the og center
        for j in (isodd(i) ? (llim:rlim) : (rlim:-1:llim))
            # prepare inds for current unitary
            inds = it.prime((siteinds[2*j-mod(i,2)], siteinds[2*j-mod(i,2)+1]), depth-i)
            fullinds = [inds[1], inds[2], inds[1]', inds[2]']

            # add a clause for boundary terms: if sizeAB is even, for even layers, a delta should be placed at the extremities
            # we simulate this by decreasing the prime level of the upper index of the first and last gate of the odd layers
            # so that they are connected directly to the odd layer above
            if i < depth
                if j == llim && isodd(i)
                    fullinds[1] = it.prime(inds[1],-1)
                elseif j == rlim
                    # if size even increase the right index of the odd layers, if size odd increase right index of even layers
                    if (iseven(sizeAB) && isodd(i)) || (isodd(sizeAB) && iseven(i))
                        fullinds[2] = it.prime(inds[2],-1)
                    end
                end
            end
            # special case for i=2 since we have to connect the rightmost unitary to the max prime level
            if i == 2 && isodd(sizeAB) && j == rlim
                fullinds[4] = inds[2]''
            end

            fullinds = (fullinds[1], fullinds[2], fullinds[3], fullinds[4])

            # insert unitaries according to lightbounds, inserting identities if we are outside the lightcone
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
        if iseven(i)
            reverse!(layer_i)
        end
        push!(circuit, layer_i)
        push!(layers_coords, layer_i_coords)
    end

    return Lightcone(circuit, d, sizeAB, depth, lightbounds, siteinds, coords, id_coords, layers_coords, range)

end

"Update Lightcone in-place"
function updateLightcone!(lightcone::Lightcone, U_array::Vector{<:Matrix})
    n_unitaries = length(U_array)
    if length(lightcone.coords) != n_unitaries
        throw(BoundsError(n_unitaries, "Number of updated unitaries does not match lightcone structure"))
    end

    for k in 1:n_unitaries
        (i,j), inds = lightcone.coords[k]
        U = U_array[k]
        U_tensor = it.ITensor(U, inds)
        lightcone.circuit[i][j] = U_tensor
    end
end

"Flatten Lightcone to 1D array, from bottom to top"
function Base.Array(lightcone::Lightcone)
    arr::Vector{Matrix} = []
    d = lightcone.d
    for ((i,j), inds) in lightcone.coords
        push!(arr, reshape(Array(lightcone.circuit[i][j], inds), (d^2,d^2)))
    end
    return arr
end

"Apply k-th unitary of lightcone to MPS/MPO psi. Lightcone.size must be equal to length(psi)"
function contract!(psi::Union{itmps.MPS, Vector{it.ITensor}}, lightcone::Lightcone, k::Int64; dagger = false, cutoff = 1E-15)
    l1, l2 = length(psi), lightcone.size
    if l1 != l2
        raise(DomainError(l1, "Cannot apply lightcone of size $l2 to mps of length $l1: the two lengths must be equal"))
    end
    (i,j), inds = lightcone.coords[k]
    U = lightcone.circuit[i][j]
    if dagger
        U = conj(U)
    end

    b = 2*j-mod(i,2)    # where to apply the unitary

    W = psi[b]*psi[b+1]
    suminds = it.commoninds(inds, W)
    length(suminds) < 2 &&
        throw(DomainError(length(suminds), "Number of legs to contract < 2, make sure that indices match"))
    outinds = it.uniqueinds(U, suminds)
    iszero(length(outinds)) &&
        throw(DomainError(length(suminds), "Resulting tensor has fewer output legs, make sure that indices match"))
    W *= U
    
    it.replaceinds!(W, outinds, suminds)
    indsb = it.uniqueinds(psi[b], psi[b+1])
    U, S, V = it.svd(W, indsb, cutoff = cutoff)
    psi[b] = it.replaceinds(U, suminds, outinds)
    psi[b+1] = it.replaceinds(S*V, suminds, outinds) 
end

"Apply lightcone to mps psi, from base to top. Lightcone.size must be equal to length(psi)"
function contract!(psi::Union{itmps.MPS, Vector{it.ITensor}}, lightcone::Lightcone; cutoff = 1E-15)
    for k in 1:length(lightcone.coords)
        contract!(psi, lightcone, k)
    end
end

"Apply vector of Lightcones to MPS, from base to top. The lightcones positions are specified by the ints contained in initial_pos. Prime level of mps siteinds must be the same as lightcones'."
function contract!(mps::itmps.MPS, lightcones::Vector{Lightcone}, initial_pos::Vector{<:Int}; cutoff = 1E-15)
    
    n_lightcones = length(initial_pos)
    
    for l in 1:n_lightcones
        lc = lightcones[l]
        initpos = initial_pos[l]

        for k in 1:length(lc.coords)
            (i,j), inds = lc.coords[k]
            U = lc.circuit[i][j]
            Umat = reshape(Array(U, inds), (inds[1].space^2, inds[1].space^2))
            non_unitarity = norm(Umat'Umat - I)
            if non_unitarity > 1E-14
                coords = (i,j)
                throw(DomainError(non_unitarity, "Non-unitary matrix found in lightcone number $l at coords $coords"))
            end

            b = initpos-1 + 2*j-mod(i,2)
            it.orthogonalize!(mps, b+1)
            norm0 = norm(mps)

            W = mps[b]*mps[b+1]
            suminds = it.commoninds(inds, W)
            outinds = it.uniqueinds(U, suminds)
            W *= U
            
            it.replaceinds!(W, outinds, suminds)
            indsb = it.uniqueinds(mps[b], mps[b+1])
            U, S, V = it.svd(W, indsb, cutoff = cutoff)
            mps[b] = it.replaceinds(U, suminds, outinds)
            mps[b+1] = it.replaceinds(S*V, suminds, outinds) 

            norm1 = norm(mps)
            if abs(norm0-norm1)>1E-14
                throw(DomainError(norm1, "MPS norm has changed during the application of lightcone of initpos $initpos"))
            end
        end
    end

end

"Convert lightcone to MPO"
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