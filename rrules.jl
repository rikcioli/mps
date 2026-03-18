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


# function mps_from_tangent(t, ψ)
#     data = [isnothing(t.data[i]) ? 0*ψ[i] : t.data[i] for i in eachindex(ψ)]
#     return MPS(data)
# end
# 
# function ITensorMPS.convert(::Type{MPS},
#     t::Tangent{<:Any,<:NamedTuple{(:data,:llim,:rlim)}})
# 
#     data = t.data
#     ψ = MPS(data)
#     ψ.llim = 0
#     ψ.rlim = length(data)+1
#     return ψ
# end

# function ITensorMPS.convert(::Type{MPS}, M::Tangent{Any, @NamedTuple{data::Vector{ITensor}, llim::ZeroTangent, rlim::ZeroTangent}})
#     return MPS(M[1])
# end
 
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



# Technically not needed since already present in ITensorMPS
# BUT ACTUALLY the one present in ITensorMPS also performs projection onto gauge inv subspace
# function ChainRulesCore.rrule(::Type{MPS}, x::Vector{ITensor})
#     ψ = MPS(x)
#     function pullback(ψbar)
#         ψbar = unthunk(ψbar)
#         x̄ = ψbar.data
#         return (NoTangent(), x̄)
#     end
#     return ψ, pullback
# end

# function ChainRulesCore.rrule(::typeof(getindex), ψ::MPS, i::Int)
#     y = ψ[i]
#     function pullback(ȳ)
#         ȳ = unthunk(ȳ)
#         N = length(ψ)
#         databar = Vector{ITensor}(undef, N)
#         for j in 1:N
#             databar[j] = ITensor(inds(ψ[j]))
#         end
#         databar[i] = ȳ
#         ψbar = MPS(databar)
#         ψbar.llim = ψ.llim
#         ψbar.rlim = ψ.rlim
#         return (NoTangent(), ψbar, NoTangent())
#     end
#     return y, pullback
# end



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
    @show S

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


function truncDM(ψ::MPS; trunc=NamedTuple())
    sites = siteinds(ψ)
    ψdag = replace_siteinds(dag(ψ), sites')
    envR = ψ[2]*ψdag[2]
    rho1_ten = ψ[1]*envR*ψdag[1]

    rho1 = Matrix(rho1_ten, sites[1], sites[1]')
    D, V, ϵ = eig_trunc(rho1, trunc=trunc)

    Vlinkind = Index(size(V)[2], "Link, u")
    Vten = ITensor(V, sites[1], Vlinkind)

    ψt2 = dag(Vten)*ψ[1]*ψ[2]
    ψt1 = Vten

    ψt = MPS([ψt1, ψt2])
    return ψt
end

function ChainRulesCore.rrule(::typeof(truncDM), ψ::MPS; trunc=NamedTuple())
    sites = siteinds(ψ)
    links = linkinds(ψ)
    Aten = ψ[1]*ψ[2]

    A = Matrix(Aten, sites[1], sites[2])

    ψ1, ψ2 = qr_compact(A)  # needed in the pullback when we need to recreate the input MPS in a gauge inv way
    ψ1ten = ITensor(ψ1, sites[1], links[1])     # can ignore this in the forward mode
    ψ2ten = ITensor(ψ2, links[1], sites[2])

    U, S, Vdg, ϵ = svd_trunc(A; trunc=trunc) # actual compression
    @show S

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



# This does NOT break when adding custom rrule for MPS, but breaks when re-adding truncsimple
function overlap(U_array::Vector{Matrix{T}}, mps::MPS; trunc=NamedTuple()) where {T}

    N = length(mps)
    sites = siteinds(mps)
    n_unitaries = length(U_array)

    zeromps = MPS(sites, ["0" for _ in 1:N])

    # we prepare the values of j to use in the next section here
    pattern = vcat([1:2:N-1; N-2:-2:2])
    # repeat pattern as needed to match n_unitaries, then truncate to exact length
    # if n_unitaries < length(pattern), only the first k elements are used
    jvals = repeat(pattern, ceil(Int, n_unitaries / length(pattern)))[1:n_unitaries]

    gates = [ITensor(U_array[unit_no], sites[j]', sites[j+1]', sites[j], sites[j+1]) for (unit_no, j) in enumerate(jvals)]
    
    ansatz = ITensorMPS.apply(gates, zeromps)

    #final = ansatz
    #final, err = truncAD(ansatz; normalize=false, kargs...)
    #IMPORTANT: for truncsimple to work, ansatz has to be orthogonalized to 2
    # but orthogonalize! is not compatible with AD -> need to find a better method
    final = truncSimple(ansatz; trunc)

    return real(inner(mps, final))
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
psi = randomMPS(ComplexF64, siteinds("Qubit", 2), 2)
U_array = [random_unitary(4)]
n = length(U_array)


f = arrU -> overlap(arrU, psi; trunc=(maxrank=1,))
gradient(f, U_array)

g = arrU -> mt.project(arrU, gradient(f, arrU)[1])

g(U_array)

plot = testGrad(() -> genPoint(n), U -> genTanVec(U, n), arrU -> (f(arrU), g(arrU)), mt.inner, mt.retract)

U = random_unitary(4)
svd_trunc(U; trunc=(maxrank=2,))