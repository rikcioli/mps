using ITensors, ITensorMPS
using KrylovKit
using SparseArrays, LinearAlgebra
using Plots
using HDF5
using LaTeXStrings
using Printf
using Statistics
ITensors.set_warn_order(28)

function entropy!(psi::MPS, b::Integer)  
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log2(p)
    end
    return SvN
end

function spectrum!(psi::MPS, b::Integer)
    orthogonalize!(psi, b)
    indsb = uniqueinds(psi[b], psi[b+1])
    U, S, V = svd(psi[b], indsb)

    spec = diag(Matrix{Float64}(S, inds(S)))
    return spec
end

function H_spin(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hy::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jx,"Sx",j,"Sx",j+1
        os += Jy,"Sy",j,"Sy",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hx,"Sx",j
        os += hy,"Sy",j
        os += hz,"Sz",j
    end
    os += hx,"Sx",N
    os += hy,"Sy",N
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_spin(sites, Jxy::Real, Jz::Real, hz::Real)
    os = OpSum()
    N = length(sites)
    for j=1:N-1
        os += Jxy/2,"S+",j,"S-",j+1
        os += Jxy/2,"S-",j,"S+",j+1
        os += Jz,"Sz",j,"Sz",j+1
        os += hz,"Sz",j
    end
    os += hz,"Sz",N

    H = MPO(os, sites)
    return H
end

function H_XY(sites, g::Real, hx::Real)
    return H_spin(sites, -(1+g), -(1-g), 0., hx, 0., 0.) 
end

function H_heisenberg(sites, Jx::Real, Jy::Real, Jz::Real, hx::Real, hz::Real)
    return H_spin(sites, Jx, Jy, Jz, hx, 0., hz)
end

function H_heisenberg(sites, Jxy::Real, Jz::Real, hz::Real)
    return H_spin(sites, Jxy, Jz, hz)
end

function initialize_gs(H::MPO, sites; nsweeps = 5, maxdim = [10,20,100,100,200], cutoff = 1e-15, linkdims=2, kwargs...)
    psi0 = random_mps(ComplexF64, sites; linkdims=linkdims)
    energy, psi = dmrg(H,psi0;nsweeps,maxdim,cutoff,kwargs...)
    return energy, psi
end

function XXZ(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_heisenberg(sites, -1., -1., -0.5, -0.1, -0.1)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function XY(N::Int)
    sites = siteinds("S=1/2", N)
    Hamiltonian = H_XY(sites, 0.0, 0.5)
    energy, psi0 = initialize_gs(Hamiltonian, sites; nsweeps = 10, cutoff = 1e-12, maxdim = [10,50,100,100,100,100,100,100,100,100])
    return energy, psi0
end

function ising(N::Int)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., -1.5/2)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end

function ising(N::Int, g::Real)
    sites = siteinds("S=1/2", N; conserve_szparity=true)
    H = H_spin(sites, -1., 0., 0., 0., 0., g)
    psi0 = MPS(sites,"Up")

    E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
               cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
    return E0, psi
end



"""
    connected_corr(psi, op1, op2; i0=nothing)

Connected correlator C(r) = ⟨op1_i0 op2_(i0+r)⟩ - ⟨op1_i0⟩⟨op2_(i0+r)⟩
for r = 1:N-i0, computed from a single reference site i0 (default: center).
Returns (rs, C).
"""
function connected_corr(psi::MPS, op1::String, op2::String; i0::Union{Nothing,Int}=nothing)
    N = length(psi)
    i0 = isnothing(i0) ? div(N, 2) : i0

    corrmat = correlation_matrix(psi, op1, op2)
    onept1 = expect(psi, op1)
    onept2 = expect(psi, op2)

    rs = 1:(N - i0)
    C = [corrmat[i0, i0 + r] - onept1[i0] * onept2[i0 + r] for r in rs]
    return collect(rs), C
end

"""
    connected_corr_avg(psi, op1, op2; rmax)

Connected correlator averaged over all reference sites i0, for distances
r = 1:rmax (default N/4), to reduce noise from a single reference site.
Returns (rs, Cavg).
"""
function connected_corr_avg(psi::MPS, op1::String, op2::String; rmax::Union{Nothing,Int}=nothing)
    N = length(psi)
    rmax = isnothing(rmax) ? div(N, 4) : rmax

    corrmat = correlation_matrix(psi, op1, op2)
    onept1 = expect(psi, op1)
    onept2 = expect(psi, op2)

    rs = 1:rmax
    Cavg = Float64[]
    for r in rs
        vals = [real(corrmat[i, i + r] - onept1[i] * onept2[i + r]) for i in 1:(N - r)]
        push!(Cavg, mean(vals))
    end
    return collect(rs), Cavg
end


function H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.0, hz=0.0)
    N = length(sites)
    os = OpSum()
    for j in 1:N-1
        os += Jxy/2, "S+", j, "S-", j+1
        os += Jxy/2, "S-", j, "S+", j+1     # same coefficient => Hermitian
        os += Jz,    "Sz", j, "Sz", j+1
    end
    for j in 1:N
        os += -hs*(-1)^j, "Sz", j           # STAGGERED — this is what was missing
        os += hz,         "Sz", j
    end
    return MPO(os, sites)
end

# Modify to remove edge states
function H_xxz_boundary(sites; Jxy=1.0, Jz=2.5, hs=0.0, hz=0.0, shift=0.0)
    N = length(sites)
    os = OpSum()
    for j in 1:N-1
        os += Jxy/2, "S+", j, "S-", j+1
        os += Jxy/2, "S-", j, "S+", j+1     # same coefficient => Hermitian
        os += Jz,    "Sz", j, "Sz", j+1
    end
    for j in 1:N
        os += -hs*(-1)^j, "Sz", j           # STAGGERED — this is what was missing
        os += hz,         "Sz", j
    end
    os += Jz/2, "Sz", 1
    os += Jz/2, "Sz", N
    if shift != 0
        os += shift, "Id", 1
    end
    return MPO(os, sites)
end

function H_xxz_afm_boundary(sites; Jxy=1.0, Jz=2.5, hs=0.0, hz=0.0)
    N = length(sites)
    os = OpSum()
    for j in 1:N-1
        os += Jxy/2, "S+", j, "S-", j+1
        os += Jxy/2, "S-", j, "S+", j+1     # same coefficient => Hermitian
        os += Jz,    "Sz", j, "Sz", j+1
    end
    for j in 1:N
        os += -hs*(-1)^j, "Sz", j           # STAGGERED — this is what was missing
        os += hz,         "Sz", j
    end
    os += -0.5, "Sz", 1
    os += +0.5, "Sz", N     # N even
    return MPO(os, sites)
end

# N = 100
# sites = siteinds("S=1/2", N; conserve_qns=true)
# H    = H_xxz(sites; Jxy=1.0, Jz=2.5, hs=0.05)
# psi0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n in 1:N])
# E0, psi = dmrg(H, psi0; nsweeps=20, maxdim=[20,60,100,200,200],
#                cutoff=1e-12, noise=[1e-6,1e-8,1e-10,0.0])
# 
# rs, corr = connected_corr_avg(psi, "Sz", "Sz")
# plot(rs, abs.(corr), marker=:circle, xlabel="r", ylabel="|C(r)|", yscale=:log10,
#      title="Connected ⟨Sz Sz⟩ correlator, XXZ Jz=2.5", legend=false)
# 
# 
# # FERROMAGNETIC 
# N = 100; Jz = -2.5; hz = 1e-6
# sites = siteinds("S=1/2", N; conserve_qns=true)
# H    = H_xxz_boundary(sites; Jxy=-1.0, Jz=Jz, hz=hz)
# 
# Elist = Float64[]
# psilist = MPS[]
# varlist = Float64[]
# varhist_m::Vector{Vector{Float64}} = []
# chihist_m::Vector{Vector{Int64}} = []
# for m in 0:3
#     nsweeps = 10
#     maxiter = 30
#     #psi0 = MPS(sites, [n <= m ? "Dn" : "Up" for n in 1:N])
#     lo = (N-m)÷2 + 1
#     st = [lo <= n < lo + m ? "Dn" : "Up" for n in 1:N]
#     @assert count(==("Dn"), st) == m
#     psi0 = random_mps(sites, st; linkdims=2)
# 
#     varhist = Float64[]
#     chihist = Float64[]
#     for iter in 1:maxiter
#         @show iter
#         noise = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0]
#         E0, psi = dmrg(H, psi0; nsweeps=nsweeps,
#                 #maxdim=[10,20,50,100,100],
#                 maxdim=10,
#                 cutoff=1e-16,
#                 noise = iter==1 ? noise : [1e-9, 1e-10, 0.0])
#         var0 = dot(H, psi, H, psi) - E0^2
#         @show var0
#         push!(varhist, var0)
#         push!(chihist, maxlinkdim(psi))
#         if var0 < 1e-10 || iter == maxiter
#             push!(Elist, E0)
#             push!(psilist, psi)
#             push!(varlist, var0)
#             break
#         else
#             psi0 = psi
#         end
#     end
#     push!(varhist_m, varhist)
#     push!(chihist_m, chihist)
# end


###ACTUAL FERRO

mutable struct VarObserver <: AbstractObserver
    Hs::MPO
    Eref::Float64
    var_tol::Float64
    zpred::Vector{Float64}      # analytic ⟨Sᶻ⟩ profile; empty to skip
    sweeps::Vector{Int}
    energies::Vector{Float64}
    vars::Vector{Float64}
    chis::Vector{Int}
    asym::Vector{Float64}
    resid::Vector{Float64}
end

VarObserver(Hs, Eref; var_tol=0.0, zpred=Float64[]) =
    VarObserver(Hs, Eref, var_tol, zpred,
                Int[], Float64[], Float64[], Int[], Float64[], Float64[])

function ITensorMPS.checkdone!(o::VarObserver; energy, psi, sweep, outputlevel=0, kwargs...)
    es = real(energy) - o.Eref
    v  = real(ITensorMPS.inner(o.Hs, psi, o.Hs, psi)) - es^2
    z  = expect(psi, "Sz")
    a  = maximum(abs.(z .- reverse(z)))
    r  = isempty(o.zpred) ? NaN : maximum(abs.(z .- o.zpred))

    push!(o.sweeps,   sweep)
    push!(o.energies, real(energy))
    push!(o.vars,     v)
    push!(o.chis,     maxlinkdim(psi))
    push!(o.asym,     a)
    push!(o.resid,    r)

    outputlevel > 0 && @printf("  sweep %3d: E=%.12f  var=%.3e  χ=%d  asym=%.2e  resid=%.2e\n",
                               sweep, real(energy), v, maxlinkdim(psi), a, r)
    return v < o.var_tol
end

function droplet_seed(sites, m; cutoff=1e-14)
    N = length(sites)
    m == 0 && return MPS(sites, fill("Up", N))
    L = N - m + 2
    psi = nothing
    for X in 1:(N-m+1)
        amp = sin(pi*X/L)
        abs(amp) < 1e-13 && continue
        p = amp * MPS(sites, [X <= n < X+m ? "Dn" : "Up" for n in 1:N])
        psi = isnothing(psi) ? p : add(psi, p; cutoff=cutoff, maxdim=4m)
    end
    return normalize!(psi)
end

function rho_string(x, m, N)
    L = N - m + 2
    a, b = max(1, x-m+1), min(x, N-m+1)
    (b-a+1)/L - (sin(pi*(2b+1)/L) - sin(pi*(2a-1)/L))/(2L*sin(pi/L))
end

sz_pred(m, N) = [0.5 - rho_string(x, m, N) for x in 1:N]   # define once, outside

N = 100; Jz = -2.5; hz = 0.0
sites = siteinds("S=1/2", N; conserve_qns=true)
Eref = (N-1)*Jz/4 + Jz/2 + hz*N/2        # = -63.075 for your parameters
Hs = H_xxz_boundary(sites; Jxy=-1.0, Jz=Jz, hz=hz, shift=-Eref)
mrange = 0:4

Elist, psilist, varlist = Float64[], MPS[], Float64[]
varhist_m, chihist_m, Ehist_m = Vector{Float64}[], Vector{Int}[], Vector{Float64}[]

for m in mrange
    lo = (N-m)÷2 + 1
    st = [lo <= n < lo + m ? "Dn" : "Up" for n in 1:N]
    @assert count(==("Dn"), st) == m
    #psi0 = random_mps(sites, st; linkdims=4)

    obs = VarObserver(Hs, 0.0; var_tol=1e-12, zpred=sz_pred(m, N))
    psi0 = droplet_seed(sites, m)
    E0, psi = dmrg(Hs, psi0; nsweeps=150, maxdim=32, cutoff=1e-14,
                #noise=[1e-9,1e-10,1e-11,1e-12],
                eigsolve_krylovdim=10, observer=obs, outputlevel=1)
    #E0, psi = dmrg(Hs, psi0; nsweeps=300, maxdim=10, cutoff=1e-12,
    #               noise=[1e-6,1e-7,1e-8,1e-9,1e-10,0.0],
    #               observer=obs, outputlevel=1)

    push!(Elist, E0); push!(psilist, psi); push!(varlist, obs.vars[end])
    push!(varhist_m, obs.vars); push!(chihist_m, obs.chis); push!(Ehist_m, obs.energies)
end


plt_Sz  = plot(xlabel="site", ylabel=L"\langle S^z\rangle", title="XXZ Ferro Jz=-2.5")
plt_var = plot(xlabel="sweep", ylabel=L"\langle H^2\rangle-\langle H\rangle^2",
            title="XXZ Ferro Jz=-2.5")
plt_chi = plot(xlabel="sweep", ylabel=L"\chi", title="XXZ Ferro Jz=-2.5")

for m in mrange
    c = m+1
    plot!(plt_Sz, expect(psilist[c], "Sz"), marker=:circle, ms=2, color=c, label="m=$m")
    plot!(plt_Sz, sz_pred(m, N), lw=2, ls=:dash, color=c, label="m=$m analytic")
    plot!(plt_chi, chihist_m[c], marker=:circle, ms=3, color=c, label="m=$m")
    if m>0
        plot!(plt_var, abs.(varhist_m[c]), marker=:circle, ms=3, color=c, label="m=$m")
    end
end
hline!(plt_var, [N*1e-14]; ls=:dot, color=:black, label="N*cutoff", yscale=:log10)
plt_Sz
plt_chi
plt_var


gaps = diff(Elist)
varlist
    

rs, corr = connected_corr(psi, "Sz", "Sz")
plot(rs, abs.(corr), marker=:circle, xlabel="r", ylabel="|C(r)|", yscale=:log10,
     title="Connected ⟨Sz Sz⟩ correlator, XXZ Jz=2.5", legend=false)




### ANTIFERRO

mutable struct CustomObserver <: AbstractObserver
    Hs::MPO
    Eref::Float64
    var_tol::Float64
    sweeps::Vector{Int}
    energies::Vector{Float64}
    vars::Vector{Float64}
    chis::Vector{Int}
end

CustomObserver(Hs, Eref; var_tol=0.0) =
    CustomObserver(Hs, Eref, var_tol,
                   Int[], Float64[], Float64[], Int[])

function ITensorMPS.checkdone!(o::CustomObserver; energy, psi, sweep, outputlevel=0, kwargs...)
    es = real(energy) - o.Eref
    v  = real(ITensorMPS.inner(o.Hs, psi, o.Hs, psi)) - es^2

    push!(o.sweeps,   sweep)
    push!(o.energies, real(energy))
    push!(o.vars,     v)
    push!(o.chis,     maxlinkdim(psi))

    outputlevel > 0 && @printf("  sweep %3d: E=%.12f  var=%.3e  χ=%d  \n",
                               sweep, real(energy), v, maxlinkdim(psi))
    return v < o.var_tol
end

function kink_density(psi)
    N = length(psi)
    orthogonalize!(psi, 1)
    nk = zeros(N-1)
    for j in 1:N-1
        orthogonalize!(psi, j)
        s1, s2 = siteind(psi, j), siteind(psi, j+1)
        op = ITensors.op("Sz", s1) * ITensors.op("Sz", s2)
        φ  = psi[j] * psi[j+1]
        nk[j] = 0.5 + 2*real(scalar(dag(prime(φ, "Site")) * op * φ)) / norm(φ)^2
    end
    return nk
end

N = 100; Jz = 2.5; hs = 0.0
sites = siteinds("S=1/2", N; conserve_qns=true)
Hs = H_xxz(sites; Jxy=1.0, Jz=Jz, hs=hs)
mrange = 0:0

Elist, psilist, varlist = Float64[], MPS[], Float64[]
varhist_m, chihist_m, Ehist_m = Vector{Float64}[], Vector{Int}[], Vector{Float64}[]

for m in mrange
    lo = (N-m)÷2 + 1
    afm_st = [isodd(n) ? "Up" : "Dn" for n in 1:N]

    @assert count(==("Dn"), afm_st) - count(==("Up"), afm_st) == m

    obs = CustomObserver(Hs, 0.0; var_tol=1e-12)
    #psi0 = MPS(sites, anti_st)
    psi0 = random_mps(sites, afm_st; linkdims=1)
    E0, psi = dmrg(Hs, psi0; nsweeps=50, maxdim=[10, 20, 50, 100], cutoff=1e-15,
                noise=[1e-6,1e-8,1e-10,0.0],
                #eigsolve_krylovdim=10, 
                observer=obs, 
                outputlevel=1)
    push!(Elist, E0); push!(psilist, psi); push!(varlist, obs.vars[end])
    push!(varhist_m, obs.vars); push!(chihist_m, obs.chis); push!(Ehist_m, obs.energies)
end

gaps = diff(Elist)


plt_Sz  = plot(xlabel="site", ylabel=L"\langle S^z\rangle", title="XXZ Antiferro Jz=2.5")
plt_kink = plot(xlabel="site", ylabel=L"\Delta n", title="XXZ Antiferro Jz=2.5")
plt_var = plot(xlabel="sweep", ylabel=L"\langle H^2\rangle-\langle H\rangle^2",
            title="XXZ Antiferro Jz=2.5")
plt_chi = plot(xlabel="sweep", ylabel=L"\chi", title="XXZ Antiferro Jz=2.5")

for m in mrange
    c = m+1
    plot!(plt_Sz, expect(psilist[c], "Sz"), marker=:circle, ms=2, color=c, label="m=$m")

    Δn = kink_density(psilist[c]) .- kink_density(psilist[1])
    plot!(plt_kink, Δn, marker=:circle, ms=2, color=c, label="m=$m")
    plot!(plt_chi, chihist_m[c], marker=:circle, ms=3, color=c, label="m=$m")
    
    plot!(plt_var, abs.(varhist_m[c]), marker=:circle, ms=3, color=c, label="m=$m")
    
end
hline!(plt_var, [N*1e-14]; ls=:dot, color=:black, label="N*cutoff", yscale=:log10)
plt_Sz
plt_kink
plt_chi
plt_var