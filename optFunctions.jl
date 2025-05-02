

#this function fixes the phases; each column of a unitary has an arbitrary phase
# by choosing the phase ϕ_i to be the phase of the largest entry in the i'th column
function fix_phase(U)
    ϕs = []
    for v in eachcol(U)
        i = argmax(abs.(v))
        ϕ = atan(imag(v[i]),real(v[i]))
        push!(ϕs,ϕ)
    end
    return U*diagm(exp.(-1im*ϕs))
end

function dist_un(U1::Matrix{T}, U2::Matrix{T}) where {T}
    return dist_un([U1], [U2])
end

"Geodesic distance between two arrays of unitaries"
function dist_un(arrU1::Vector{<:Matrix}, arrU2::Vector{<:Matrix})
    distances_sq = []
    for k in 1:length(arrU1)
        U1, U2 = arrU1[k], arrU2[k]
        V = eigvals(fix_phase(U1)'fix_phase(U2))
        dist_sq = sum(map(x->atan(imag(x),real(x))^2,V))
        push!(distances_sq, dist_sq)
    end
    return sqrt(sum(distances_sq)) 
end

function skew(X::Matrix{T}) where {T}
    return (X - X')/2
end

function skew(arrX::Vector{<:Matrix})
    return map(skew, arrX)
end

"Project arbitrary array of unitaries arrD onto the tangent space at arrU"
function project(arrU::Vector{<:Matrix}, arrD::Vector{<:Matrix})
    #return (arrD .- (arrU .* [D' for D in arrD] .* arrU))/2
    return arrU .* skew([U' for U in arrU] .* arrD)
end

function polar(M::Matrix{T}) where {T}
    U, S, V = svd(M)
    P = V*diagm(S)*V'
    W = U*V'
    return P, W
end

function extractU(M::Matrix{T}) where {T}
    U, S, V = svd(M)
    W = U*V'
    return W
end

function extractU(arrM::Vector{<:Matrix})
    return map(extractU, arrM)
end


#move  U in the direction of X with step length t, 
#X is the gradient obtained using projection.
#return both the "retracted" unitary as well as the tangent vector at the retracted point
# always stabilize unitarity and skewness, it could leave tangent space due to numerical errors
function retract(arrU::Vector{<:Matrix}, arrX::Vector{<:Matrix}, t::Float64)

    # ensure unitarity of arrU
    arrU_unitary = extractU(arrU)

    #non_unitarity = norm(arrU - arrU_polar)/length(arrU)
    #if non_unitarity > 1e-10
    #    @show non_unitarity
    #end
    arrU = arrU_unitary

    arrUinv = [U' for U in arrU]
    arrX_id = arrUinv .* arrX

    # ensure skewness of arrX_id
    #non_skewness = norm(arrX_id - skew(arrX_id))/length(arrU)
    #if non_skewness > 1E-10
    #    @show non_skewness
    #end
    # if non_skewness > 1E-15 + eps()
    #     throw(DomainError(non_skewness, "arrX is not in the tangent space at arrU"))
    # end
    arrX_id = skew(arrX_id)

    # construct the geodesic at the tangent space at unity
    # then move it to the correct point by multiplying by arrU
    arrU_new = arrU .* map(X -> exp(t*X), arrX_id)

    # move arrX to the new tangent space arrU_new
    arrX_new = arrU_new .* arrX_id #move first to the tangent space at unity, then to the new point
    return arrU_new, arrX_new
end

function inner(arrU::Vector{<:Matrix}, arrX::Vector{<:Matrix}, arrY::Vector{<:Matrix})
    return real(tr(arrX'*arrY))
end
function inner(arrX::Vector{<:Matrix}, arrY::Vector{<:Matrix})
    return real(tr(arrX'*arrY))
end

#parallel transport
"""transport tangent vector ξ along the retraction of x in the direction η (same type as a gradient) 
with step length α, can be in place but the return value is used. 
Transport also receives x′ = retract(x, η, α)[1] as final argument, 
which has been computed before and can contain useful data that does not need to be recomputed"""
function transport!(ξ, arrU::Vector{<:Matrix}, η, α, arrU_new::Vector{<:Matrix})
    arrUinv = [U' for U in arrU]
    ξ = arrU_new .* arrUinv .* ξ
    return ξ
end