function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end 

#this function fixes the phases; each column of a unitary has an arbitrary phase
# by choosing the phase ϕ_i to be the phase of the largest entry in the i'th column
function fix_phase(U::AbstractMatrix)
    ϕs = []
    for v in eachcol(U)
        i = argmax(abs.(v))
        ϕ = atan(imag(v[i]),real(v[i]))
        push!(ϕs,ϕ)
    end
    return U*diagm(exp.(-1im*ϕs))
end

function dist_un(U1::AbstractMatrix, U2::AbstractMatrix)
    return dist_un([U1], [U2])
end

"Geodesic distance between two arrays of unitaries"
function dist_un(arrU1::Vector{<:AbstractMatrix}, arrU2::Vector{<:AbstractMatrix})
    dist_sq = 0.0
    for k in eachindex(arrU1)
        U1, U2 = arrU1[k], arrU2[k]
        V = eigvals(fix_phase(U1)'fix_phase(U2))
        dist_sq += sum(map(x->atan(imag(x),real(x))^2,V))
    end
    return sqrt(dist_sq) 
end

function skew(X::AbstractMatrix)
    return (X - X')/2
end

function skew(arrX::Vector{<:AbstractMatrix})
    return map(skew, arrX)
end

function project(U::AbstractMatrix, D::AbstractMatrix)
    return U * skew(U' * D)
end

"Project arbitrary array of unitaries arrD onto the tangent space at arrU"
function project(arrU::Vector{<:AbstractMatrix}, arrD::Vector{<:AbstractMatrix})
    #return (arrD .- (arrU .* [D' for D in arrD] .* arrU))/2
    return arrU .* skew([U' for U in arrU] .* arrD)
end

function polar(M::AbstractMatrix)
    U, S, V = svd(M)
    P = V*diagm(S)*V'
    W = U*V'
    return P, W
end

function extractU(M::AbstractMatrix)
    U, S, V = svd(M)
    W = U*V'
    return W
end

function extractU(arrM::Vector{<:AbstractMatrix})
    return map(extractU, arrM)
end

function retract(U::AbstractMatrix, X::AbstractMatrix, t::Float64)
    U_unitary = extractU(U)
    U = U_unitary

    Uinv = U'
    X_id = Uinv * X
    X_id = skew(X_id)

    U_new = U * exp(t*X_id)
    X_new = U_new * X_id
    return U_new, X_new
end


#move  U in the direction of X with step length t, 
#X is the gradient obtained using projection.
#return both the "retracted" unitary as well as the tangent vector at the retracted point
# always stabilize unitarity and skewness, it could leave tangent space due to numerical errors
function retract(arrU::Vector{<:AbstractMatrix}, arrX::Vector{<:AbstractMatrix}, t::Float64)

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

function inner(arrU::Vector{<:AbstractMatrix}, arrX::Vector{<:AbstractMatrix}, arrY::Vector{<:AbstractMatrix})
    return real(tr(arrX'*arrY))
end
function inner(arrX::Vector{<:AbstractMatrix}, arrY::Vector{<:AbstractMatrix})
    return real(tr(arrX'*arrY))
end

function inner(U::AbstractMatrix, X::AbstractMatrix, Y::AbstractMatrix)
    return real(tr(X'*Y))
end
function inner(X::AbstractMatrix, Y::AbstractMatrix)
    return real(tr(X'*Y))
end

#parallel transport
"""transport tangent vector ξ along the retraction of x in the direction η (same type as a gradient) 
with step length α, can be in place but the return value is used. 
Transport also receives x′ = retract(x, η, α)[1] as final argument, 
which has been computed before and can contain useful data that does not need to be recomputed"""
function transport!(ξ, arrU::Vector{<:AbstractMatrix}, η, α, arrU_new::Vector{<:AbstractMatrix})
    arrUinv = [U' for U in arrU]
    ξ = arrU_new .* arrUinv .* ξ
    return ξ
end




# =========================================================
# Helper: Hermitian projection (analogous to skew)
# =========================================================

function herm(X::AbstractMatrix)
    return (X + X') / 2
end

function herm(arrX::Vector{<:AbstractMatrix})
    return map(herm, arrX)
end

function extractU(M::AbstractMatrix)
    U, S, V = svd(M)
    W = U*V'
    return W
end

# =========================================================
# Left isometry:  V ∈ ℂ^{m×n}, m ≥ n,  V†V = Iₙ
# =========================================================

"""
Project D onto the tangent space of the left-isometry manifold at V.
Tangent condition: V†Z skew-Hermitian
⟹  Z = (I - VV†)D + V·skew(V†D) = D - V·herm(V†D)
"""
function projectL(V::AbstractMatrix, D::AbstractMatrix)
    return D - V * herm(V' * D)
end

# function projectL(arrV::Vector{<:AbstractMatrix}, arrD::Vector{<:AbstractMatrix})
#     return arrD .- arrV .* herm.([V' for V in arrV] .* arrD)
# end

"""
Polar retraction: move V along tangent vector Z with step t.
Returns the new isometry and the transported tangent vector.
"""
function retractL(V::AbstractMatrix, Z::AbstractMatrix, t::Float64)
    V   = extractU(V)       # extractU works for isometries too
    V_new = extractU(V + t * Z)
    Z_new = projectL(V_new, Z)          # transport by re-projection
    return V_new, Z_new
end

function retractL(arrV::Vector{<:AbstractMatrix}, arrZ::Vector{<:AbstractMatrix}, t::Float64)
    arrV     = extractU.(arrV)
    arrV_new = extractU.(arrV .+ t .* arrZ)
    arrZ_new = projectL.(arrV_new, arrZ)  # transport by re-projection
    return arrV_new, arrZ_new
end

"""
Transport tangent vector ξ to the tangent space at V_new (re-projection transport).
Signature mirrors transport! for unitaries.
"""
function transportL!(ξ, arrV::Vector{<:AbstractMatrix}, η, α, arrV_new::Vector{<:AbstractMatrix})
    return projectL(arrV_new, ξ)
end

# =========================================================
# Right isometry:  W ∈ ℂ^{m×n}, m ≤ n,  WW† = Iₘ
# =========================================================


"""
Project D onto the tangent space of the right-isometry manifold at W.
Tangent condition: ZW† skew-Hermitian
⟹  Z = D(I - W†W) + skew(DW†)W = D - herm(DW†)·W
"""
function projectR(W::AbstractMatrix, D::AbstractMatrix)
    return D - herm(D * W') * W
end

# function projectR(arrW::Vector{<:AbstractMatrix}, arrD::Vector{<:AbstractMatrix})
#     return arrD .- herm.([D * W' for (D, W) in zip(arrD, arrW)]) .* arrW
# end

"""
Polar retraction for right isometry.
"""
function retractR(W::AbstractMatrix, Z::AbstractMatrix, t::Float64)
    W     = extractU(W)
    W_new = extractU(W + t * Z)
    Z_new = projectR(W_new, Z)
    return W_new, Z_new
end

function retractR(arrW::Vector{<:AbstractMatrix}, arrZ::Vector{<:AbstractMatrix}, t::Float64)
    arrW     = extractU.(arrW)
    arrW_new = extractU.(arrW .+ t .* arrZ)
    arrZ_new = projectR.(arrW_new, arrZ)
    return arrW_new, arrZ_new
end

"""
Transport tangent vector ξ to the tangent space at W_new (re-projection transport).
"""
function transportR!(ξ, arrW::Vector{<:AbstractMatrix}, η, α, arrW_new::Vector{<:AbstractMatrix})
    return projectR(arrW_new, ξ)
end

# =========================================================
# Random initializers
# =========================================================

function random_left_isometry(m::Int, n::Int)
    # m ≥ n
    F = qr((randn(m, n) + randn(m, n) * im) / sqrt(2))
    return Matrix(F.Q)[:, 1:n]
end

function random_right_isometry(m::Int, n::Int)
    # m ≤ n: just conjugate-transpose a left isometry
    return Matrix(random_left_isometry(n, m)')
end


function projectMixed(arrA::Vector{<:AbstractArray}, arrD::Vector{<:AbstractArray}, ogc::Int)
    projD = similar(arrD)
    for j in 1:ogc-1
        projD[j] = projectL(arrA[j], arrD[j])
    end
    projD[ogc] = arrD[ogc]
    for j in ogc+1:length(arrD)
        projD[j] = projectR(arrA[j], arrD[j])
    end
    return projD
end

function retractMixed(arrA::Vector{<:AbstractArray}, arrD::Vector{<:AbstractArray}, t::Float64, ogc::Int)
    Anew = similar(arrA)
    Dnew = similar(arrD)
    for j in 1:ogc-1
        Anew[j], Dnew[j] = retractL(arrA[j], arrD[j], t)
    end
    Anew[ogc], Dnew[ogc] = (arrA[ogc] + t*arrD[ogc], arrD)
    for j in ogc+1:length(arrD)
        Anew[j], Dnew[j] = retractR(arrA[j], arrD[j], t)
    end
    return Anew, Dnew
end

function innerMixed(arrX::Vector{<:AbstractArray}, arrY::Vector{<:AbstractArray})
    return real(sum(arrX[j][:]'*arrY[j][:] for j in eachindex(arrX)))
end