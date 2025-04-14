
### LIGHTCONE METHODS ###

struct Lightcone
    circuit::Vector{ITensor}             # array containing the ITensor unitaries
    inds_arr::Vector{NTuple{4, Index{Int64}}}    # contains the indices of all the tensors
    d::Int64                                # dimension of the physical space
    size::Int64                             # number of qubits the whole circuit acts on
    depth::Int64                            # circuit depth
    site1_empty::Bool                       # if True the first unitary acts on sites 2-3 instead of 1-2
    n_unitaries::Int64                      # number of unitaries
    lightbounds::Tuple{Bool, Bool}          # (leftslope == 'light', rightslope == 'light')
    siteinds::Vector{Index}              # siteinds of AB region
    range::Vector{Tuple{Int64, Int64}}      # leftmost and rightmost sites on which each layer acts non-trivially
    gates_by_site::Vector{Vector{Dict{String,Any}}}    # positions, inds and orientations of all the gates touching each qubit
    gates_by_layer::Vector{Vector{Int64}}     # positions of all gates sorted by depth
    sites_by_gate::Vector{Tuple{Int64, Int64}}  # vector contaning all the sites each gate is acting on
end


function newLightcone(sites::Vector{<:Index}, depth; U_array::Union{Nothing, Vector{<:Matrix}} = nothing, lightbounds = (true, true), site1_empty = false)

    # extract number of sites on which lightcone acts
    sizeAB = length(sites)

    # mode where the first unitary acts on sites 2-3 instead of 1-2
    shift = 0   
    if site1_empty
        depth == 1 &&
            throw(DomainError(depth, "Depth 1 lightcone for site1_empty mode leaves the first site with no unitaries at all and causes problems"))
        if lightbounds[1] || lightbounds[2]
            @warn("Lightbounds can't be true when site1_empty mode, setting them to false")
        end
        lightbounds = (false, false)
        shift = 1 # in many cases leaving the first site empty is just shifting 
        # the current depth by 1, so this can be inserted in a lot of places where
        # there's a mod(i,2) kind of thing, with i in 1:depth
    else
        if isodd(sizeAB) 
            lightbounds[2] &&
                throw(DomainError(sizeAB, "Right lightbound can't be true for sizeAB odd"))
            depth == 1 &&
                throw(DomainError(depth, "Depth 1 lightcone for an odd number of sites leaves the last site empty and causes problems"))
        end
    end

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
    d = sites[1].space
    n_unitaries = div(sizeAB-1,2)*depth + mod(sizeAB-1,2)*div(depth+1-shift,2)
    if depth > 2
        dep_m2 = depth-2
        n_unitaries -= n_light*(div(dep_m2+1,2)^2 + mod(dep_m2+1,2)*div(dep_m2+1,2))
    end
    if isnothing(U_array)
        U_array = [random_unitary(d^2) for _ in 1:n_unitaries]
    else
        n_insert = length(U_array)
        n_insert > n_unitaries &&
            throw(DomainError(n_insert, "Number of inserted unitaries is too large for this lightcone 
            structure\nInserted unitaries: $n_insert\nLightcone unitaries: $n_unitaries\n"))
        if n_insert < n_unitaries
            @warn "Number of inserted unitaries ($n_insert) is lower than maximum capacity ($n_unitaries), filling the missing spaces with identities"
            Ids = [1.0*Id2 for _ in 1:n_unitaries-n_insert]
            U_array = [U_array; Ids]
        end
    end

    # finally convert the U_array 
    circuit::Vector{ITensor} = []
    inds_arr::Vector{NTuple{4, Index{Int64}}} = []

    range::Vector{Tuple{Int64, Int64}} = []
    gates_by_site::Vector{Vector{Dict{String,Any}}} = [[] for _ in 1:sizeAB]
    gates_by_layer::Vector{Vector{Int64}} = []
    sites_by_gate::Vector{Tuple{Int64, Int64}} = []
    k = 1
    for i in 1:depth
        # limits for circuit indices, left limit is always 1
        # right limit is sizeAB/2 for odd layers, sizeAB/2 or sizeAB/2-1 for even layers depending on the parity of sizeAB
        llim, rlim = 1, div(sizeAB,2)-mod(sizeAB+1,2)*mod(i+1-shift,2)
        lslope, rslope = llim, rlim
        # left range is 1 for odd layers, 2 for even layers
        # right range depends on sizeAB: if sizeAB even it's just sizeAB and sizeAB-1; if it's odd it's sizeAB-1 and sizeAB
        # ADDED: shift value which is 0 in normal mode, 1 when first site is empty, in which case the range is swapped
        lrange, rrange = 1+mod(i+1-shift,2), (iseven(sizeAB) ? sizeAB-mod(i+1-shift,2) : sizeAB-mod(i+shift,2))
        if lightbounds[1]
            lslope += div(i-1,2)
            lrange += 2*div(i-1,2)
        end
        if lightbounds[2]
            rslope -= div(i-1,2)
            rrange -= 2*div(i-1,2)
        end
        push!(range, (lrange, rrange))

        layer_i_pos::Vector{Int64} = []
        # sweep over unitaries, from left to right if depth odd, otherwise from right to left
        # this will be helpful with the og center
        for j in (isodd(i) ? (lslope:rslope) : (rslope:-1:lslope))
            # prepare inds for current unitary
            sites_involved = (2*j-mod(i+shift,2), 2*j-mod(i+shift,2)+1)
            inds = prime((sites[sites_involved[1]], sites[sites_involved[2]]), depth-i)
            fullinds = [inds[1], inds[2], inds[1]', inds[2]']

            if !site1_empty
                # add a clause for boundary terms: if sizeAB is even, for even layers, a delta should be placed at the extremities
                # we simulate this by decreasing the prime level of the upper index of the first and last gate of the odd layers
                # so that they are connected directly to the odd layer above
                if i < depth
                    if j == lslope 
                        if lightbounds[1]
                            fullinds[1] = noprime(inds[1])
                        elseif isodd(i)
                            fullinds[1] = prime(inds[1],-1)
                        end
                    end
                    if j == rslope
                        if lightbounds[2]
                            fullinds[2] = noprime(inds[2])
                        # else: if size even decrease the right index of the odd layers, if size odd decrease right index of even layers
                        elseif (iseven(sizeAB) && isodd(i)) || (isodd(sizeAB) && iseven(i))
                            fullinds[2] = prime(inds[2],-1)
                        end
                    end
                end
                # special case for i=2 and odd sizeAB since we have to connect the rightmost unitary to the max prime level
                if i == 2 && isodd(sizeAB) && j == rlim
                    fullinds[4] = inds[2]''
                end

            else
                # add variation for site1_empty mode, which anyway will ALWAYS be with lightbound false
                if i < depth
                    if j == lslope 
                        # decrease left index of even layers
                        if iseven(i)
                            fullinds[1] = prime(inds[1],-1)
                        end
                    end
                    if j == rslope
                        if (isodd(sizeAB) && isodd(i)) || (iseven(sizeAB) && iseven(i))
                            fullinds[2] = prime(inds[2],-1)
                        end
                    end
                end
                # special case for i=2 since we have to connect the leftmost unitary to the max prime level
                # if sizeAB even we also have to connect the rightmost unitary
                if i == 2 
                    if j == llim
                        fullinds[3] = inds[1]''
                    end
                    if j == rlim && iseven(sizeAB)
                        fullinds[4] = inds[2]''
                    end
                end
            end


            fullinds = Tuple(ind for ind in fullinds)

            # insert unitary 
            brick = ITensor(U_array[k], fullinds)
            push!(circuit, brick)
            push!(inds_arr, fullinds)
            
            push!(gates_by_site[sites_involved[1]], Dict([("inds", fullinds), ("pos", k), ("depth", i), ("orientation", "R")]))
            push!(gates_by_site[sites_involved[2]], Dict([("inds", fullinds), ("pos", k), ("depth", i), ("orientation", "L")]))
            push!(sites_by_gate, sites_involved)
            push!(layer_i_pos, k)

            k += 1

        end
        if iseven(i)
            reverse!(layer_i_pos)
        end
        push!(gates_by_layer, layer_i_pos)
    end


    return Lightcone(circuit, inds_arr, d, sizeAB, depth, site1_empty, n_unitaries, lightbounds, sites, range, gates_by_site, gates_by_layer, sites_by_gate)
end

"Update k-th unitary of lightcone in-place"
function updateLightcone!(lightcone::Lightcone, U::AbstractMatrix, k::Integer)
    inds = lightcone.inds_arr[k]
    U_tensor = ITensor(U, inds)
    lightcone.circuit[k] = U_tensor
end

"Update in-place unitary specified by position (site, depth)"
function updateLightcone!(lightcone::Lightcone, U::AbstractMatrix, coords::Tuple{Integer, Integer})
    (site, depth) = coords
    N, site1_empty = lightcone.size, lightcone.site1_empty
    shift = site1_empty ? 1 : 0
    if site == 1 || (site == N && iseven(N))
        iseven(depth+shift) && throw(DomainError(coords, "No gate at specified coords for this structure"))
        gate = lightcone.gates_by_site[site][div(depth+2-shift,2)]
    elseif site == N && isodd(N)
        isodd(depth+shift) && throw(DomainError(coords, "No gate at specified coords for this structure"))
        gate = lightcone.gates_by_site[site][div(depth+shift,2)]
    else     
        depth > length(lightcone.gates_by_site[site]) &&
            throw(DomainError(coords, "No gate at specified coords for this structure"))
        gate = lightcone.gates_by_site[site][depth]
    end
    updateLightcone!(lightcone, U, gate["pos"])
end

"Update Lightcone in-place"
function updateLightcone!(lightcone::Lightcone, U_array::Vector{<:Matrix})
    n_unitaries = length(U_array)
    if lightcone.n_unitaries != n_unitaries
        throw(DomainError(n_unitaries, "Number of updated unitaries does not match lightcone structure"))
    end

    for k in 1:n_unitaries
        U = U_array[k]
        inds = lightcone.inds_arr[k]
        U_tensor = ITensor(U, inds)
        lightcone.circuit[k] = U_tensor
    end
end

"Extract d by d Matrix corresponding to position k"
function Base.Matrix(lightcone::Lightcone, k::Integer)
    d = lightcone.d
    U = reshape(Array(lightcone.circuit[k], lightcone.inds_arr[k]), (d^2,d^2))
    return U
end

"Flatten Lightcone to 1D array, from bottom to top"
function Base.Array(lightcone::Lightcone)
    d = lightcone.d
    arr = [reshape(Array(lightcone.circuit[k], lightcone.inds_arr[k]), (d^2,d^2)) for k in 1:lightcone.n_unitaries]
    return arr
end

"Apply k-th unitary of lightcone to MPS/MPO psi. Lightcone.size must be equal to length(psi)"
function contract!(psi::Union{MPS, Vector{ITensor}}, lightcone::Lightcone, k::Int64; dagger = false, cutoff = 1E-15)
    l1, l2 = length(psi), lightcone.size
    if l1 != l2
        throw(DomainError(l1, "Cannot apply lightcone of size $l2 to mps of length $l1: the two lengths must be equal"))
    end
    inds = lightcone.inds_arr[k]
    U = lightcone.circuit[k]
    if dagger
        U = conj(U)
    end

    b = lightcone.sites_by_gate[k][1]    # where to apply the unitary

    W = psi[b]*psi[b+1]
    suminds = commoninds(inds, W)
    length(suminds) < 2 &&
        throw(DomainError(length(suminds), "Number of legs to contract < 2, make sure that indices match"))
    outinds = uniqueinds(U, suminds)
    iszero(length(outinds)) &&
        throw(DomainError(length(suminds), "Resulting tensor has fewer output legs, make sure that indices match"))
    W *= U
    
    replaceinds!(W, outinds, suminds)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(W, indsb, cutoff = cutoff)
    psi[b] = replaceinds(U, suminds, outinds)
    psi[b+1] = replaceinds(S*V, suminds, outinds) 
end

"Apply lightcone to mps psi, from base to top. Lightcone.size must be equal to length(psi)"
function contract!(psi::Union{MPS, Vector{ITensor}}, lightcone::Lightcone; cutoff = 1E-15)
    for k in 1:lightcone.n_unitaries
        contract!(psi, lightcone, k; cutoff=cutoff)
    end
end

"Apply lightcone to MPS/MPO, from base to top. Lightcones' siteinds must match mps siteinds only in id, not in primelevel.
If dagger is true, all the unitaries are conjugated and the order of application is inverted
If an empty list is passed as entropy_arr karg it will be filled with half-subsystem entropies at each timestep"
function apply!(mps::Union{MPS, MPO}, lightcones::Vector{Lightcone}; dagger = false, cutoff = 1E-15, entropy_arr = nothing)
    N = length(mps)

    if isa(mps, MPO)
        allinds = reduce(vcat, siteinds(mps))
        # determine primelevel of inputinds, which will be the lowest found in allinds
        first2inds = allinds[1:2]   
        plev_in = 0
        while true
            ind = inds(first2inds, plev = plev_in)
            if length(ind) > 0
                break
            end
            plev_in += 1
        end
        ininds = inds(allinds, plev = plev_in)
        outinds = uniqueinds(allinds, ininds)

        new_ininds = siteinds(first2inds[1].space, N)
        for i in 1:N
            replaceind!(mps[i], ininds[i], new_ininds[i])
        end
        noprime!(mps)
        sites = noprime(outinds)
    else
        noprime!(mps)
        sites = siteinds(mps)
    end

    l = 1
    for lc in lightcones
        firstind = noprime(lc.siteinds[1])
        while l < N
            if firstind == noprime(sites[l])   #then apply lightcone here
                depth = lc.depth
                size = lc.size
                # increase plev of siteinds to match lightcone if dagger is false
                # remember that the plev decreases with the layer, so if we start with the gates on top we are already good
                if !dagger
                    for k in l:l+size-1
                        mps[k] = replaceind(mps[k], sites[k], prime(sites[k], depth))
                    end
                end
                # apply lc
                n_unitaries = lc.n_unitaries
                for k in (dagger ? (n_unitaries:-1:1) : (1:n_unitaries)) #apply in reverse if dagger is required
                    inds = lc.inds_arr[k]
                    U = lc.circuit[k]
                    if dagger
                        U = conj(U)
                    end

                    # Umat = reshape(Array(U, inds), (inds[1].space^2, inds[1].space^2))
                    # non_unitarity = norm(Umat'Umat - I)
                    # if non_unitarity > 1E-14
                    #     coords = (i,j)
                    #     throw(DomainError(non_unitarity, "Non-unitary matrix found in lightcone number $l at coords $coords"))
                    # end
                    b = l-1 + lc.sites_by_gate[k][1]
                    orthogonalize!(mps, b+1)
        
                    W = mps[b]*mps[b+1]
                    suminds = commoninds(inds, W)
                    outinds = uniqueinds(U, suminds)
                    W *= U
                    
                    replaceinds!(W, outinds, suminds)
                    indsb = uniqueinds(mps[b], mps[b+1])
                    U, S, V = svd(W, indsb, cutoff = cutoff)
                    mps[b] = replaceinds(U, suminds, outinds)
                    mps[b+1] = replaceinds(S*V, suminds, outinds) 
        
                    # norm1 = norm(mps)
                    # if abs(norm0-norm1)>1E-14
                    #     throw(DomainError(norm1, "MPS norm has changed during the application of lightcone of initpos $initpos"))
                    # end
                    if !isnothing(entropy_arr) && lc.sites_by_gate[k] == (div(size,2), div(size,2)+1) 
                        SvN = 0.0
                        for n in 1:dim(S, 1)
                          p = S[n,n]^2
                          SvN -= p * log(p)
                        end
                        push!(entropy_arr, SvN)
                    end
                end
                l += size
                break
            else
                l += 1
            end
        end
    end
    if dagger
        noprime!(mps)
    end
    if isa(mps, MPO)
        for i in 1:N
            replaceind!(mps[i], sites[i], ininds'[i])
            replaceind!(mps[i], new_ininds[i], ininds[i])
        end
    end
end


function apply!(mps::Union{MPS, MPO}, lightcone::Lightcone; kargs...)
    apply!(mps, [lightcone]; kargs...)
end


"Convert lightcone to MPO"
# must be updated to account for the removal of identities
function MPO(lightcone::Lightcone)
    # convert lightcone to mpo by using the apply(mps, lightcone) function defined above on an MPO of delta tensors
    sites = lightcone.siteinds

    mpo_list = [delta(ind, ind') for ind in sites]
    mpo = MPO(mpo_list)
    apply!(mpo, lightcone)

    return mpo
end