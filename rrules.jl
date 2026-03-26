using MKL
include("mpsMethods.jl")
import .MPSMethods as mt
import Plots
using ITensors, ITensorMPS
using LaTeXStrings, LinearAlgebra
using Zygote
using MatrixAlgebraKit, TensorOperations
using ChainRulesCore
#using Enzyme
#using HDF5

function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end 

function genPoint()
    return mt.random_unitary(4)
end

function genTanVec(U)
    V = randn(ComplexF64, 4, 4)
    V = mt.skew(V)
    V = U * V
    V /= sqrt(mt.inner(V, V))
    return V
end

function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
    U0 = genPoint()
    V = genTanVec(U0)
    func, grad = computeCostGrad(U0)
    gradV = inner(grad, V) 
    E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end

 
# function ChainRulesCore.rrule(::Type{Matrix{T}}, x::ITensor, rowinds, colinds) where {T}
#     y = Matrix{T}(x, rowinds, colinds)
#     function Matrix_pullback(ȳ)
#         ȳ = unthunk(ȳ)
#         # Convert gradient back to ITensor with the proper indices
#         x̄ = ITensor(ȳ, rowinds, colinds)
#         return (NoTangent(), x̄, NoTangent(), NoTangent())
#     end
#     return y, Matrix_pullback
# end

# WORKS (CHECKED BELOW)
function ChainRulesCore.rrule(::Type{Matrix}, x::ITensor, rowinds, colinds)
    y = Matrix(x, rowinds, colinds)
    function Matrix_pullback(ȳ)
        ȳ = unthunk(ȳ)
        # Convert gradient back to ITensor with the proper indices
        x̄ = ITensor(ȳ, rowinds, colinds)
        return (NoTangent(), x̄, NoTangent(), NoTangent())
    end
    return y, Matrix_pullback
end

# WORKS (CHECKED BELOW)
function ChainRulesCore.rrule(::Type{Matrix}, x::ITensor, xinds)
    y = Matrix(x, xinds)
    function Matrix_pullback(ȳ)
        # Convert gradient back to ITensor with the proper indices
        x̄ = ITensor(unthunk(ȳ), xinds)
        return (NoTangent(), x̄, NoTangent())
    end
    return y, Matrix_pullback
end

# WORKS (CHECKED BELOW)
function ChainRulesCore.rrule(::Type{Array}, x::ITensor, xinds)
    y = Array(x, xinds)
    function Array_pullback(ȳ)
        # Convert gradient back to ITensor with the proper indices
        x̄ = ITensor(unthunk(ȳ), xinds)
        return (NoTangent(), x̄, NoTangent())
    end
    return y, Array_pullback
end






function test_rrule(point::Function, dir::Function, costfunc::Function)
    X = point()
    V = dir()

    grad_ad = Zygote.gradient(costfunc, X)[1]
    gradV = real(dot(grad_ad, V))

    E = t -> begin
        dispX = X + t*V
        return abs(costfunc(dispX) - costfunc(X) - t*gradV)
    end

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end

# Test rrule for ITensor(Array) conversion
ITensor_point = () -> rand(ComplexF64, 4, 4)
ITensor_dir = () -> rand(ComplexF64, 4, 4)
function ITensor_costfunc(X::Array{<:Number})
    shape = size(X)
    xinds = Index(shape[1], "i"), Index(shape[2], "j")
    Xten = ITensor(X, xinds)
    return real(inner(Xten, Xten))
end
plot = test_rrule(ITensor_point, ITensor_dir, ITensor_costfunc)


# Test rrule for Array(ITensor) conversion
Array_point_dir = () -> begin
    i1, i2 = siteinds("Qubit", 2)
    Array_point = () -> randomITensor(i1, i2)
    Array_dir = () -> randomITensor(i1, i2)
    return Array_point, Array_dir
end
function Array_costfunc(X::ITensor)
    Xmat = Array(X, inds(X))
    return real(dot(Xmat, Xmat))
end
first, second = Array_point_dir()
plot = test_rrule(first, second, Array_costfunc)



# Test rrule for MPS(Vector{Itensor}) when applied on a state
# NOW WORKS EVEN WHEN SLICING AND CONVERTING BACK TO MPS
sites = siteinds("Qubit", 2)
mps = MPS(sites, ["0", "0"])
U = mt.random_unitary(4)
function overlap(U::Matrix{T}, mps::MPS; kargs...) where {T}
    N = length(mps)
    sites = siteinds(mps)
    zeros = MPS(sites, ["0" for _ in 1:N])

    gates = [ITensor(U, sites[1]', sites[2]', sites[1], sites[2])]
    
    ansatz = ITensorMPS.apply(gates, zeros)

    check1 = vec(ansatz)
    final = MPS(check1)
    
    return real(inner(mps, final))
end
f = U -> overlap(U, mps)
g = U -> mt.project(U, gradient(f, U)[1])

testgrad = gradient(f, U)[1]

plot = testGrad(genPoint, genTanVec, U -> (f(U), g(U)), mt.inner, mt.retract)


# FULLY MPS WORKS
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)
function testfu(psi::MPS)
    return real(inner(psi, psi))
end
testgrad = gradient(testfu, psi)[1]
typeof(testgrad)
inner(psi, testgrad)

# VECTOR AS INPUT DOESNT SEEM TO WORK
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)[1:2]
function testfu(psi::Vector{ITensor})
    psi_mps = MPS(psi)
    return real(inner(psi_mps, psi_mps))
end
testgrad = gradient(testfu, psi)[1]
typeof(testgrad)
inner(MPS(psi), MPS(testgrad))

# CHECK
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)
function testfu(psi::MPS)
    psivec = psi[1:2]
    psimps = MPS(psivec)
    return real(inner(psimps, psimps))
end
testgrad = gradient(testfu, psi)[1]
typeof(testgrad)
testgrad = mps_from_tangent(testgrad, psi)
inner(psi, testgrad)

# !!! THIS IS PROBABLY DUE TO GAUGE, THE MPS CONSTRUCTOR HAS A PROCEDURE THAT TAKES CARE OF THIS

# ONLY VECTORS DOESNT WORK, AND IT MAKES SENSE: THERE'S NO WAY THE GRADIENT
# COULD KNOW ABOUT THEIR CONSTRAINTS
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)[1:2]
function testfu(psi::Vector{ITensor})
    c1 = conj(psi[1])*conj(psi[2])
    c2 = psi[1]*psi[2]
    res = real(c1*c2)
    return res[]
end
testgrad = gradient(testfu, psi)[1]
typeof(testgrad)
inner(MPS(psi), MPS(testgrad))
#mtest = [Matrix(testgrad[i], inds(testgrad[i])) for i in 1:2]
#mpsi = [Matrix(psi[i], inds(psi[i])) for i in 1:2]
#norm(mtest[1] - 2*mpsi[1]*mpsi[2]*mpsi[2]')
#norm(mtest[2] - 2*mpsi[1]'*mpsi[1]*mpsi[2])



# MIXED NOW WORKS BECAUSE OF rrule(getindex) for MPS
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)
function testfu(psi::MPS)
    c1 = conj(psi[1])*conj(psi[2])
    c2 = psi[1]*psi[2]
    res = real(c1*c2)
    return res[]
end
testgrad = gradient(testfu, psi)[1]
typeof(testgrad)
testgrad = mps_from_tangent(testgrad, psi)
inner(psi, testgrad)
#mtest = [Matrix(testgrad[i], inds(testgrad[i])) for i in 1:2]
#mpsi = [Matrix(psi[i], inds(psi[i])) for i in 1:2]
#norm(mtest[1] - 2*mpsi[1]*mpsi[2]*mpsi[2]')
#norm(mtest[2] - 2*mpsi[1]'*mpsi[1]*mpsi[2])



function truncSimple(ψ::MPS; trunc=NamedTuple())
    sites = siteinds(ψ)
    Aten = ψ[1]*ψ[2]

    A = Matrix(Aten, sites[1], sites[2])
    U, S, Vdg, ϵ = svd_trunc(A, trunc=trunc)

    uind = Index(size(U)[2], "Link, u")
    Uten = ITensor(U, sites[1], uind)
    SVten = ITensor(S*Vdg, uind, sites[2])

    vec = [Uten, SVten]

    ψt = MPS(vec)
    return ψt
end


function ChainRulesCore.rrule(::typeof(truncSimple), ψ::MPS; trunc=NamedTuple())
    sites = siteinds(ψ)
    links = linkinds(ψ)
    Aten = ψ[1]*ψ[2]

    A = Matrix(Aten, sites[1], sites[2])

    ψ1, ψ2 = qr_compact(A)  # needed in the pullback when we need to recreate the input MPS in a gauge inv way
    ψ1ten = ITensor(ψ1, sites[1], links[1])     # can ignore this in the forward mode
    ψ2ten = ITensor(ψ2, links[1], sites[2])

    U, S, Vdg, ϵ = svd_trunc(A; trunc=trunc) # actual compression

    uind = Index(size(U)[2], "Link, u")
    Uten = ITensor(U, sites[1], uind)
    SVten = ITensor(S*Vdg, uind, sites[2])

    vec = [Uten, SVten]

    ψt = MPS(vec)

    function truncSimple_pullback(Δψt)

        Δψt_mat = Δψt[1]*Δψt[2]
        ΔU = Array{ComplexF64}(Δψt_mat*conj(SVten), sites[1], uind)
        ΔSVdg = Array{ComplexF64}(Δψt_mat*conj(Uten), uind, sites[2])
        ΔS = ΔSVdg*(Vdg')
        ΔVdg = S'*ΔSVdg

        ΔA = zero(A)
        MatrixAlgebraKit.svd_trunc_pullback!(ΔA, A, (U, S, Vdg), (ΔU, ΔS, ΔVdg))
        ΔAten = ITensor(ΔA, sites[1], sites[2])

        # now we need to reconstruct the qr we used for gauge fixing, which is STRICTLY NECESSARY
        Δψ1ten = ΔAten*conj(ψ2ten)
        Δψ2ten = ΔAten*conj(ψ1ten)
        Δψ1 = Array{ComplexF64}(Δψ1ten, sites[1], links[1])
        Δψ2 = Array{ComplexF64}(Δψ2ten, links[1], sites[2])
        Δmps = zero(A)
        MatrixAlgebraKit.qr_pullback!(Δmps, A, (ψ1, ψ2), (Δψ1, Δψ2))
        a, b = qr_compact(Δmps)
        Δψ = MPS([ITensor(a, sites[1], links[1]), ITensor(b, links[1], sites[2])])

        Δψ.llim = 1
        Δψ.rlim = 3
        return (NoTangent(), Δψ)
    end
    return ψt, truncSimple_pullback
end

function OGtest(ψ::MPS)
    N = length(ψ)
    return orthogonalize(ψ, N)
end

# This actually works, since we do not propagate the gradient in the gauge dependent directions
function ChainRulesCore.rrule(::typeof(OGtest), ψ::MPS)
    N = length(ψ)
    function OGtest_pullback(Δψ)
        return (NoTangent(), Δψ)
    end
    return orthogonalize(ψ, N), OGtest_pullback
end

# function to orthogonalize an mps to the right that is compatible with AD
function OGR(ψ::MPS)
    N = length(ψ)
    sites = siteinds(ψ)
    links = linkinds(ψ)

    WLRten_current = ψ[1:2]
    WLRlist::Vector{Vector{Matrix{ComplexF64}}} = []
    Wlist = Matrix{ComplexF64}[]
    QRlist = []
    QRmidinds = Index[]
    combs::Vector{Vector{ITensor}} = []
    combinds = []
    vec_final = ITensor[]

    for j in 1:N-1
        WLten, WRten = WLRten_current
        combs_j = ITensor[]
        if j > 1
            combL = combiner(sites[j], QRmidinds[j-1])
            push!(combs_j, combL)
            cLind = combinedind(combL)
            WLten = combL*WLten
        else
            cLind = sites[1]
        end
        if j < N-1
            combR = combiner(sites[j+1], links[j+1])
            push!(combs_j, combR)
            cRind = combinedind(combR)
            WRten = WRten*combR
        else
            cRind = sites[N]
        end
        push!(combs, combs_j)   # store combiners for pullback
        push!(combinds, (cLind, cRind))     # store combined inds
        # convert WLten and WRten into matrices and store them for pullback
        push!(WLRlist, [Array{ComplexF64}(WLten, cLind, links[j]), Array{ComplexF64}(WRten, links[j], cRind)])

        W = Matrix(WLten*WRten, cLind, cRind)
        push!(Wlist, W)
        U, R = qr_compact(W)
        push!(QRlist, (U, R))

        middle_ind = Index(size(U)[2], "Link, u")
        push!(QRmidinds, middle_ind)
        Uten = ITensor(U, cLind, middle_ind)
        Rten = ITensor(R, middle_ind, cRind)
        
        Uten = j==1 ? Uten : Uten*dag(combL)
        Rten = j==N-1 ? Rten : Rten*dag(combR)

        push!(vec_final, Uten)
        if j==N-1
            push!(vec_final, Rten)
        else
            WLRten_current = [Rten, ψ[j+2]]
        end
    end

    ψog = MPS(vec_final)
    ψog.llim = N-1
    ψog.rlim = N+1

    return ψog
end


function ChainRulesCore.rrule(::typeof(OGR), ψ::MPS)
    N = length(ψ)
    sites = siteinds(ψ)
    links = linkinds(ψ)

    WLRten_current = ψ[1:2]
    WLRlist::Vector{Vector{Matrix{ComplexF64}}} = []
    Wlist = Matrix{ComplexF64}[]
    QRlist = []
    QRmidinds = Index[]
    combs::Vector{Vector{ITensor}} = []
    combinds = []
    vec_final = ITensor[]

    for j in 1:N-1
        WLten, WRten = WLRten_current
        combs_j = ITensor[]
        if j > 1
            combL = combiner(sites[j], QRmidinds[j-1])
            push!(combs_j, combL)
            cLind = combinedind(combL)
            WLten = combL*WLten
        else
            cLind = sites[1]
        end
        if j < N-1
            combR = combiner(sites[j+1], links[j+1])
            push!(combs_j, combR)
            cRind = combinedind(combR)
            WRten = WRten*combR
        else
            cRind = sites[N]
        end
        push!(combs, combs_j)   # store combiners for pullback
        push!(combinds, (cLind, cRind))     # store combined inds
        # convert WLten and WRten into matrices and store them for pullback
        push!(WLRlist, [Array{ComplexF64}(WLten, cLind, links[j]), Array{ComplexF64}(WRten, links[j], cRind)])

        W = Matrix(WLten*WRten, cLind, cRind)
        push!(Wlist, W)
        U, R = qr_compact(W)
        push!(QRlist, (U, R))

        middle_ind = Index(size(U)[2], "Link, u")
        push!(QRmidinds, middle_ind)
        Uten = ITensor(U, cLind, middle_ind)
        Rten = ITensor(R, middle_ind, cRind)
        
        Uten = j==1 ? Uten : Uten*dag(combL)
        Rten = j==N-1 ? Rten : Rten*dag(combR)

        push!(vec_final, Uten)
        if j==N-1
            push!(vec_final, Rten)
        else
            WLRten_current = [Rten, ψ[j+2]]
        end
    end

    ψog, MPS_pullback = ChainRulesCore.rrule(MPS, vec_final)
    ψog.llim = N-1
    ψog.rlim = N+1

    function truncOGR_pullback(Δψog)

        _, Δψog_vec = MPS_pullback(Δψog)

        ΔRten = Δψog_vec[N]
        Δψ_vec = ITensor[]
        for j in N-1:-1:1
            @show j
            combs_j = combs[j]
            combinds_j = combinds[j]
            QRmidind_j = QRmidinds[j]
            WL, WR = WLRlist[j]
            W = Wlist[j]
            U, R = QRlist[j]

            # something breaks here, adjoints of qr_pullback are not correct as they are now
            ΔUten = Δψog_vec[j]
            ΔUten = j==1 ? ΔUten : ΔUten*combs_j[1]
            ΔU = Array{ComplexF64}(ΔUten, combinds_j[1], QRmidind_j)
            ΔR = Array{ComplexF64}(ΔRten, QRmidind_j, combinds_j[2])

            ΔW = zero(W)
            MatrixAlgebraKit.qr_pullback!(ΔW, W, (U, R), (ΔU, ΔR))

            ΔWL, ΔWR = ΔW*WR', WL'*ΔW

            ΔWRten = ITensor(ΔWR, links[j], combinds_j[2])
            if j<N-1
                ΔWRten = j==1 ? ΔWRten*dag(combs_j[1]) : ΔWRten*dag(combs_j[2])     # just because if j==1 there's only one combiner and it's the right one
            end
            pushfirst!(Δψ_vec, ΔWRten) # this goes into the final mps
            ΔWLten = ITensor(ΔWL, combinds_j[1], links[j])
            ΔWLten = j==1 ? ΔWLten : ΔWLten*dag(combs_j[1]) # this gets decombined and is prepared for next step

            # update for the next step
            if j>1
                ΔRten = j==2 ? ΔWLten*combs[j-1][1] : ΔWLten*combs[j-1][2]    # just because if j==2 combs[j-1] only has the right combiner (THIS NEEDS TO CHANGE)
            else
                pushfirst!(Δψ_vec, ΔWLten)
            end
        end
        Δψ = MPS(Δψ_vec)
        Δψ.llim = ψ.llim
        Δψ.rlim = ψ.rlim    #if it was normalized, the adjoint should be too in theory

        return (NoTangent(), Δψ)
    end
    return ψog, truncOGR_pullback
end


function truncDMsimple(ψ::MPS; trunc=NamedTuple())

    sites = siteinds(ψ)
    ψdag = dag(ψ)'
    envR = ψ[2]*delta(sites[2], sites[2]')*ψdag[2]
    rho1_ten = ψ[1]*envR*ψdag[1]

    rho1 = Matrix(rho1_ten, sites[1], sites[1]')
    D, U, ϵ = eig_trunc(rho1, trunc=trunc)

    Ulinkind = Index(size(U)[2], "Link, u")
    Uten = ITensor(U, sites[1], Ulinkind)

    ψt2 = dag(Uten)*ψ[1]*ψ[2]
    ψt1 = Uten

    vect = [ψt1, ψt2]
    ψt = MPS(vect)
    return ψt
end

function ChainRulesCore.rrule(::typeof(truncDMsimple), ψ::MPS; trunc=NamedTuple())
    sites = siteinds(ψ)

    ψdag = dag(ψ)'
    envR = ψ[2]*delta(sites[2], sites[2]')*ψdag[2]
    rho1_ten = ψ[1]*envR*ψdag[1]

    rho1 = Matrix(rho1_ten, sites[1], sites[1]')
    D, U, ϵ = eigh_trunc(rho1, trunc=trunc)

    Ulinkind = Index(size(U)[2], "Link, u")
    Uten = ITensor(U, sites[1], Ulinkind)

    ψ12 = ψ[1]*ψ[2]
    ψt2 = dag(Uten)*ψ12
    ψt1 = Uten

    vect = [ψt1, ψt2]
    ψt, MPS_pullback = ChainRulesCore.rrule(MPS, vect)

    function truncDMsimple_pullback(Δψt)
        _, Δψt_vec = MPS_pullback(Δψt)  #vector to MPS
        ΔB1, ΔB2 = Δψt_vec
        cind = commonind(ΔB1, ΔB2)
        ΔB1, ΔB2 = replaceind(ΔB1, cind, Ulinkind), replaceind(ΔB2, cind, Ulinkind)  #extract the two components

        ΔU = Array{ComplexF64}(ΔB1, sites[1], Ulinkind)     # contribution from MPS([U1, B2])
        ΔU += Array{ComplexF64}(ψ12*conj(ΔB2), sites[1], Ulinkind)  # contribution from B2=U1|ψ⟩
        Δrho1 = zero(rho1)
        MatrixAlgebraKit.eigh_trunc_pullback!(Δrho1, rho1, (D, U), (ZeroTangent(), ΔU)) # rho1 to (D, U)
  
        Δψten_rho1 = [noprime(ITensor(Δrho1+Δrho1', sites[1]', sites[1])*ψ[1]), ψ[2]] # contribution from rho1 = Tr2(ψ)
        Δψ_rho1 = MPS(Δψten_rho1)
        Δψten_B2 = [Uten, ΔB2]
        Δψ_B2 = MPS(Δψten_B2) # contribution from B2=U1|ψ⟩
        Δψ = Δψ_rho1 + Δψ_B2 

        return (NoTangent(), Δψ)
    end
    return ψt, truncDM_pullback
end


function truncDM(ψ::MPS; trunc=NamedTuple())
    if length(trunc) == 0
        trunc = (maxrank=maxlinkdim(ψ),)
    end
    N = length(ψ)
    sites = siteinds(ψ)

    envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    for j in N-1:-1:2
        push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    end

    vect = ITensor[]    # store tensors that make the final mps
    errs = Float64[]    # store truncation errors
    ilinks = Index[]    # store intermediate links
    isites = [sites[1]]     # store intermediate site1
    iψ = [ψ[:]]  # store intermediate ψ
    icombiners = ITensor[]      # store intermediate combiners

    for j in 1:N-1
        ψ = iψ[j]
        ψ1 = ψ[1]
        site1 = isites[j]
        rhoj_ten = ψ1*envR[N-j]*dag(ψ1)'

        rhoj = Matrix(rhoj_ten, site1, site1')
        D, U, ϵ = eig_trunc(rhoj, trunc=trunc)
        push!(errs, ϵ)

        Ulinkind = Index(size(U)[2], "Link, l=$(j)")
        push!(ilinks, Ulinkind)
        Uten = ITensor(U, isites[j], Ulinkind)
        if j>1
            push!(vect, dag(icombiners[end])*Uten)
        else
            push!(vect, Uten)
        end    

        if j == N-1
            push!(vect, dag(Uten)*ψ1*ψ[2])
        else
            c12 = combiner(Ulinkind, sites[j+1])
            push!(icombiners, c12)
            ψ1new = c12*(dag(Uten)*ψ1*ψ[2])
            push!(iψ, [ψ1new; ψ[3:end]])
            site1new = combinedind(c12)
            push!(isites, site1new)
        end
    end

    ψt = MPS(vect)

    return ψt
end


function ChainRulesCore.rrule(::typeof(truncDM), ψ::MPS; trunc=NamedTuple())
    maxrank = maxlinkdim(ψ)
    if length(trunc) == 0
        trunc = (maxrank=maxrank,)
    end
    N = length(ψ)
    sites = siteinds(ψ)
    links = linkinds(ψ)

    envR = [ψ[N]*delta(sites[N], sites[N]')*dag(ψ[N])']
    for j in N-1:-1:2
      push!(envR, envR[N-j]*ψ[j]*delta(sites[j], sites[j]')*dag(ψ[j])')
    end

    vect = ITensor[]    # store tensors that make the final mps
    errs = Float64[]    # store truncation errors
    ilinks = Index[]    # store intermediate links
    isites = [sites[1]]     # store intermediate site1
    iψ = [ψ[:]]  # store intermediate ψ
    icombiners = ITensor[]      # store intermediate combiners
    ieigh_trunc = []    # store intermediate rdm and eigh_trunc results
    iψ12::Vector{Vector{ITensor}} = []    # store ψ[1] and ψ[2]

    for j in 1:N-1
        ψ = iψ[j]
        ψ1 = ψ[1]
        site1 = isites[j]
        rhoj_ten = ψ1*envR[N-j]*dag(ψ1)'

        rhoj = Matrix(rhoj_ten, site1, site1')
        D, U, ϵ = eigh_trunc(rhoj, trunc=trunc)
        push!(ieigh_trunc, (rhoj, D, U))
        push!(errs, ϵ)

        Ulinkind = Index(size(U)[2], "Link_trunc, l=$(j)")
        push!(ilinks, Ulinkind)
        Uten = ITensor(U, isites[j], Ulinkind)
        if j>1
            push!(vect, dag(icombiners[end])*Uten)
        else
            push!(vect, Uten)
        end    

        push!(iψ12, ψ[1:2])

        if j == N-1
            push!(vect, dag(Uten)*ψ1*ψ[2])
        else
            c12 = combiner(Ulinkind, sites[j+1])
            push!(icombiners, c12)
            ψ1new = c12*(dag(Uten)*ψ1*ψ[2])
            push!(iψ, [ψ1new; ψ[3:end]])
            site1new = combinedind(c12)
            push!(isites, site1new)
        end
    end

    ψt, MPS_pullback = ChainRulesCore.rrule(MPS, vect)

    function truncDM_pullback(Δψt)
        _, Δψt_vec = MPS_pullback(Δψt)  #vector to MPS
        # we call its elements [ΔB1, ΔB2, ..., ΔBN-1, ΔψN] as the last element is just ΔψN
        # every B is actually just U with split indices, that must be combined accordingly

        ΔψN_pad = Δψt_vec[N] # we already prepare the last element of the returned Δψj_MPS
        # by padding this last ITensor with zeros
        linkfull = links[N-1]
        linktrunc = ilinks[N-1]
        inddiff = linkfull.space - linktrunc.space

        if inddiff > 0     # pad only if needed, otherwise just replace index
            linkcomp = Index(inddiff, "Link_comp, l=$(N-1)")
            tencomp = ITensor(linkcomp, sites[N])
            ΔψN_pad = directsum(linkfull, ΔψN_pad => linktrunc, tencomp => linkcomp)
        else
            ΔψN_pad = replaceind(ΔψN_pad, linktrunc, linkfull)
        end

        # this is the last element of the result, can be computed by unrolling the MPS sum
        Δψj_MPS = [ΔψN_pad + (N-1)*iψ12[N-1][2]]  

        local Δψj_rhoj, Δψj_ψjp1
        local contrj, contrjp1
        for j in N-1:-1:1
            ψj = iψ[j]  
            ψj12 = iψ12[j]
            rhoj, D, U = ieigh_trunc[j]
            site1 = isites[j]
            Ulinkind = ilinks[j]
            ΔBj = Δψt_vec[j]
            if j>1
                combj = icombiners[j-1]
            end

            ΔU = Array{ComplexF64}(j>1 ? combj*ΔBj : ΔBj, site1, Ulinkind)     # contribution from MPS([U1, U2, U3, ψ4])
            # now we need to add the contribution from |ψjp1⟩=Uj|ψj⟩, that is the contribution of Δψjp1 to ΔUj
            # which is obtained by multiplying ψj[1] with contrj ≡ contract(ψj[2:end], Δψjp1).

            if j==N-1
                # for j=N-1 this is simply
                contrj = ψj12[2]*dag(Δψt_vec[N])
            else
                # contrj can be constructed recursively by splitting Δψjp1 into Δψjp1_rhojp1 and Δψjp1_ψjp2
                # the first one is only present when j<N-1, as there's no reduced density matrix computed at the last step
                # it can be computed by reusing the right environment defined in the forward pass
                contr_rhoj = ψj12[2]*dag(prime!(Δψj_rhoj, links[j+1]))*envR[N-j-1]

                # the second one can be defined recursively in terms of contrjp1
                contr_ψjp1 = ψj12[2]*dag(Δψj_ψjp1)*contrjp1

                # sum the two as they have the same inds
                contrj = contr_rhoj + contr_ψjp1
            end
            ΔU += Array{ComplexF64}(ψj12[1]*contrj, site1, Ulinkind)

            Δrhoj = zero(rhoj)
            MatrixAlgebraKit.eigh_trunc_pullback!(Δrhoj, rhoj, (D, U), (ZeroTangent(), ΔU))     # pullback of eigh_trunc
    
            Δψj_rhoj = noprime(ITensor(Δrhoj+Δrhoj', site1', site1)*ψj12[1]) # contribution from rhoj = Tr_(j+1..N)(ψ)
            Δψj_ψjp1 = ITensor(U, site1, Ulinkind)    #contribution from |ψj+1⟩=Uj|ψj⟩

            #Δψj_MPS_rhoj = [Δψj_rhoj; ψj[2:end]]
            #Δψj_MPS_ψjp1 = [Δψj_ψjp1; Δψj_MPS[:]]

            if j>1
                Δψj_rhoj *= dag(combj)
                Δψj_ψjp1 *= dag(combj)
            end

            # now we need to sum them together by padding with zeros the link in Δψj_MPS_ψjp1 if needed
            # at step j, first we pad the truncated linkinds on the right if needed
            # this is only required for Δψj_ψjp1, ad Δψj_rhoj has the full right linkind
            Δψj_ψjp1_pad = Δψj_ψjp1
            linkfullR = links[j]
            linktruncR = Ulinkind
            # NOTE: THIS MAY BE NEGATIVE IF INPUT MPS HAS SOME BOND DIM SMALLER THAN REQUIRED TRUNC -> NEED TO FIX
            inddiffR = linkfullR.space - linktruncR.space   
            if inddiffR > 0
                linkcompR = Index(inddiffR, "Link_comp, l=$(j)")
                tencompR = j>1 ? ITensor(ilinks[j-1], sites[j], linkcompR) : ITensor(sites[j], linkcompR)
                Δψj_ψjp1_pad = directsum(linkfullR, Δψj_ψjp1 => linktruncR, tencompR => linkcompR)
            else
                Δψj_ψjp1_pad = replaceind(Δψj_ψjp1, linktruncR, linkfullR)
            end

            Δψj_rhoj_pad = Δψj_rhoj
            if j>1
                # now we pad the links on the left for both Δψj_ψjp1_pad and Δψj_rhoj_pad
                linkfullL = links[j-1]
                linktruncL = ilinks[j-1]
                inddiffL = linkfullL.space - linktruncL.space
                if inddiffL > 0
                    linkcompL = Index(inddiffL, "Link_comp, l=$(j-1)")
                    tencompL = ITensor(linkcompL, sites[j], linkfullR)  # remember we already padded the right link
                    Δψj_ψjp1_pad = directsum(linkfullL, Δψj_ψjp1_pad => linktruncL, tencompL => linkcompL)
                    Δψj_rhoj_pad = directsum(linkfullL, Δψj_rhoj => linktruncL, tencompL => linkcompL)
                else
                    Δψj_ψjp1_pad = replaceind(Δψj_ψjp1_pad, linktruncL, linkfullL)
                    Δψj_rhoj_pad = replaceind(Δψj_rhoj, linktruncL, linkfullL)
                end
            end
            
            Δψj_MPS_j = Δψj_ψjp1_pad + Δψj_rhoj_pad 
            if j>1
                Δψj_MPS_j += (j-1)*iψ12[j-1][2]
            end
            pushfirst!(Δψj_MPS, Δψj_MPS_j)

            contrjp1 = contrj   # update contrj for next step
        end
        
        return (NoTangent(), MPS(Δψj_MPS))
    end
    return ψt, truncDM_pullback
end


function truncAD(psi::MPS; normalize = false, kargs...)
    N = length(psi)
    sites = siteinds(psi)
    links = linkinds(psi)

    resvec::Vector{ITensor} = [psi[1]]
    reserr::Vector{Float64} = []
    local lind

    for i in 1:N-1
        Aiip1_tmp1 = resvec[i]*psi[i+1]
        local ciind, cip1ind, Aiip1_tmp2, Aiip1_tensor, ci, cip1
        if i > 1
            citmp = combiner(sites[i], lind)
            ciind = Index(size(citmp)[1], "Combiner, c$i")
            ci = replaceind(citmp, combinedind(citmp), ciind)
            Aiip1_tmp2 = ci*Aiip1_tmp1
        else
            ciind = sites[1]
            Aiip1_tmp2 = Aiip1_tmp1
        end
        if i < N-1
            cip1 = combiner(sites[i+1], links[i+1])
            cip1ind = combinedind(cip1)
            Aiip1_tensor = Aiip1_tmp2*cip1
        else
            cip1ind = sites[N]
            Aiip1_tensor = Aiip1_tmp2
        end

        Aiip1 = Matrix(Aiip1_tensor, ciind, cip1ind)
        Ui, Siip1, Vdgiip1, epsi = svd_trunc(Aiip1; kargs...)
        if normalize
            Siip1 /= norm(Siip1)
        end
        lind = Index(size(Ui)[2], "Link, u")
        Ui_tmp = ITensor(Ui, ciind, lind)
        SViip1 = Siip1*Vdgiip1
        SVip1_tmp = ITensor(SViip1, lind, cip1ind)

        Ui_tensor = i==1 ? Ui_tmp : Ui_tmp*dag(ci)
        SVip1_tensor = i==N-1 ? SVip1_tmp : SVip1_tmp*dag(cip1)

        resvec = [resvec[1:(i-1)]; Ui_tensor; SVip1_tensor]
        reserr = [reserr; epsi]
    end

    tpsi = MPS(resvec)

    return tpsi, reserr
end

truncAD_point_dir = () -> begin
    N = 2
    sites = siteinds("Qubit", N)
    truncAD_point = () -> randomMPS(sites, 2)
    truncAD_dir = () -> randomMPS(sites, 2)
    return truncAD_point, truncAD_dir
end
function truncAD_costfunc(psi::MPS)
    tpsi, _ = truncAD(psi; normalize=false)
    return real(inner(psi, tpsi))
end
plot = test_rrule(truncAD_point_dir()..., truncAD_costfunc)


function truncADTO(psi::MPS; normalize = false, kargs...)
    N = length(psi)
    sites = siteinds(psi)
    links = linkinds(psi)

    firstinds = [sites[1], links[1]]
    midinds = [[links[i-1]; sites[i]; links[i]] for i in 2:N-1]
    lastinds = [links[N-1]; sites[N]]
    inds_list = vcat([firstinds], midinds, [lastinds])

    matrices = [Array{ComplexF64}(psi[i], inds_list[i]) for i in 1:N]
    sizes = [size(m) for m in matrices]
    resvec = [matrices[1]]

    reserr::Vector{Float64} = []

    for i in 1:N-1

        shape1 = i>1 ? (sizes[i][1]*sizes[i][2], sizes[i][3]) : sizes[i]
        shape2 = i<N-1 ? (sizes[i+1][1], sizes[i+1][2]*sizes[i+1][3]) : sizes[i+1]
        psi_i = reshape(resvec[i], shape1)
        psi_ip1 = reshape(matrices[i+1], shape2)

        @tensor begin
            Aiip1[a,b] := psi_i[a,d]*psi_ip1[d,b]
        end
        Ui, Siip1, Vdgiip1, epsi = svd_trunc(Aiip1; kargs...)
        if normalize
            Siip1 /= norm(Siip1)
        end

        # Convert Diagonal to dense matrix for TensorOperations compatibility
        Siip1_dense = Matrix(Siip1)
        @tensor begin
            SViip1[a,b] := Siip1_dense[a,c]*Vdgiip1[c,b]
        end

        if i>1
            Ui = reshape(Ui, (div(size(Ui)[1], sizes[i][2]), sizes[i][2], size(Ui)[2]))
        end
        if i<N-1
            SViip1 = reshape(SViip1, (size(SViip1)[1], sizes[i+1][2], div(size(SViip1)[2], sizes[i+1][2])))
        end

        resvec = [resvec[1:(i-1)]; [Ui]; [SViip1]]
        reserr = [reserr; epsi]
    end

    tsizes = size.(resvec)
    new_linkinds = [Index(tsizes[i][1], "Link, u") for i in 2:N]
    firstinds = [sites[1], new_linkinds[1]]
    midinds = [[new_linkinds[i-1]; sites[i]; new_linkinds[i]] for i in 2:N-1]
    lastinds = [new_linkinds[N-1]; sites[N]]
    inds_list = vcat([firstinds], midinds, [lastinds])

    tpsi = MPS([ITensor(resvec[i], inds_list[i]) for i in 1:N])

    return tpsi, reserr
end

# Doesnt really make sense to test this gradient, it's gonna be affected by gauge directions
truncADTO_point_dir = () -> begin
    N = 2
    sites = siteinds("Qubit", N)
    truncADTO_point = () -> randomMPS(sites, 2)
    truncADTO_dir = () -> randomMPS(sites, 2)
    return truncADTO_point, truncADTO_dir
end
function truncADTO_costfunc(psi::MPS)
    tpsi, _ = truncADTO(psi; normalize=false)
    return real(inner(psi, tpsi))
end

test = Zygote.gradient(truncADTO_costfunc, randomMPS(siteinds("Qubit", 2), 2))[1]
plot = test_rrule(truncADTO_point_dir()..., truncADTO_costfunc)


function overlap(U_array::Vector{Matrix{T}}, mps::MPS; trunc=NamedTuple()) where {T}

    N = length(mps)
    sites = siteinds(mps)
    n_unitaries = length(U_array)

    zeromps = MPS(sites, ["0" for _ in 1:N])

    # we prepare the values of j to use in the next section here
    pattern = vcat([1:2:N-1; N-2:-2:2])
    # repeat pattern as needed to match n_unitaries, then truncate to exact length
    # if n_unitaries < length(pattern), only the first k elements are used
    jvals = repeat(pattern, ceil(Int, n_unitaries/length(pattern)))[1:n_unitaries]

    gates = [ITensor(U_array[unit_no], sites[j]', sites[j+1]', sites[j], sites[j+1]) for (unit_no, j) in enumerate(jvals)]
    
    n_twolayers = div(n_unitaries, N-1)
    if mod(n_unitaries, N-1) > 0
        n_twolayers += 1
    end
    #@show n_twolayers
    for j in 1:n_twolayers
        evolvedmps = ITensorMPS.apply(gates[1+(j-1)*(N-1) : min(j*(N-1), n_unitaries)], zeromps)
        # make sure we don't truncate a product state -> need to check what happens for variable bond dim
        zeromps = truncDM(evolvedmps; trunc)
        #zeromps = OGtest(evolvedmps)   #check if OGR works
    end

    return real(inner(mps, zeromps))
end



function genPoint(n_unitaries)
    # generate random point on M
    U0 = [mt.random_unitary(4) for _ in 1:n_unitaries]
    U0dag = [U' for U in U0]
    return U0, U0dag
end

function genTanVec(U, n_unitaries)
    V = [randn(ComplexF64, 4, 4) for _ in 1:n_unitaries]
    V = mt.skew.(V)
    V = U .* V
    V /= sqrt(mt.inner(V, V))
end

function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
    U0, U0dag = genPoint()
    V = genTanVec(U0)
    func, grad = computeCostGrad(U0)
    gradV = inner(grad, V) 
    E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)

    tvals = exp10.(-8:0.1:0)
    plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
    Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
    Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
    return plot
end


# TEST OVERLAP
psi = randomMPS(ComplexF64, siteinds("Qubit", 4), 2)
U_array = [random_unitary(4) for _ in 1:3];
n = length(U_array)

f = arrU -> overlap(arrU, psi; trunc=(maxrank=2,))
gradient(f, U_array)

g = arrU -> mt.project(arrU, gradient(f, arrU)[1])

plot = testGrad(() -> genPoint(n), U -> genTanVec(U, n), arrU -> (f(arrU), g(arrU)), mt.inner, mt.retract)

psiog = OGR(psi)
f = psi -> real(inner(OGR(psi), psi))
f(psi)
gradient(f, psi)



result_time = begin
    times1 = Float64[]
    times2 = Float64[]
    for N in 2:2:100
        tvals1 = Float64[]
        tvals2 = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 2)
            t1 = @elapsed truncDM(psi, trunc=(maxrank=1,))
            t2 = @elapsed truncate!(psi, maxdim=1)
            push!(tvals1, t1)
            push!(tvals2, t2)
        end
        push!(times1, sum(tvals1)/100)
        push!(times2, sum(tvals2)/100)
    end
    times1, times2
end
plot = Plots.plot(2:2:100, result_time[1], label="truncDM")
Plots.plot!(plot, 2:2:100, result_time[2], label="truncate")



result_trunc = begin
    overlapDM = Float64[]
    overlapIT = Float64[]
    for N in 2:2:50
        dm = Float64[]
        it = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 32)
            psit1 = truncDM(psi, trunc=(maxrank=2,))
            psit2 = truncate(psi, maxdim=2)
            push!(dm, real(inner(psit1, psi)))
            push!(it, real(inner(psit2, psi)))
        end
        push!(overlapDM, sum(dm)/100)
        push!(overlapIT, sum(it)/100)
    end
    overlapDM, overlapIT
end
plot = Plots.plot(2:2:50, result_trunc[1], label="truncDM", xlabel=L"$N$", ylabel=L"$\langle \psi |\psi_t\rangle$")
Plots.plot!(plot, 2:2:50, result_trunc[2], label="truncate", yscale=:log)


result_time_scaling = begin
    f = psi -> real(inner(truncDM(psi, trunc=(maxrank=2,)), psi))
    trunc_time = Float64[]
    trunc_grad_time = Float64[]
    for N in 2:2:50
        @show N
        times = Float64[]
        grad_times = Float64[]
        for _ in 1:100
            psi = randomMPS(ComplexF64, siteinds("Qubit", N), 8)
            t1 = @elapsed f(psi)
            t2 = @elapsed gradient(f, psi)
            push!(times, t1)
            push!(grad_times, t2)
        end
        push!(trunc_time, sum(times)/100)
        push!(trunc_grad_time, sum(grad_times)/100)
    end
    trunc_time, trunc_grad_time
end
plot = Plots.plot(2:2:50, result_time_scaling[1], label="truncDM_time", xlabel=L"$N$", ylabel=L"$\langle \psi |\psi_t\rangle$")
Plots.plot!(plot, 2:2:50, result_time_scaling[2], label="grad_truncDM_time")