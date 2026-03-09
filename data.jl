
import Plots
using LinearAlgebra
using CSV
using DataFrames
using JLD2, HDF5
using Glob
using Plots, LaTeXStrings

pathname = "D:\\Julia\\MyProject\\Data\\xy\\invertTrunc\\g0h0.5\\"
df = DataFrame(N=Int[], eps=Float64[], depth=Int[])

Nrange = [20, 40, 60, 80, 100]
eps_array = [0.1, 0.02, 0.004, 0.0008]

for N in Nrange
    for eps in eps_array
        # For each file, push a new row
        depth = load_object(pathname*"trunc_$(N)_$(eps).jld2")
        push!(df, (N=N, eps=eps, depth=depth))
    end
end
CSV.write(pathname*"result.csv", df)




df_list = []
pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\disent\\"

for i in [1,2,3,4,5]
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], nmps=Int[])
    folder = pathname*"mps$i\\"
    Nrange = [20, 40, 60, 80, 100]
    eps_array = [0.1, 0.02, 0.004]
    
    for N in Nrange
        for eps in eps_array
            # For each file, push a new row
            depth = load_object(folder*"trunc_$(N)_$(eps).jld2")
            push!(df, (N=N, eps=eps, depth=depth, nmps=i))
        end
    end
    CSV.write(folder*"result.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(pathname*"result.csv", df_final)


# disent ising, xy

pathname = "D:\\Julia\\MyProject\\Data\\xy\\disent\\g0h0.5\\"
for N in Nrange
    filenames = glob("disent_$(N)_*.jld2", pathname)
    for filename in filenames
        eps = parse(Float64, filename[50+length(string(N)):end-5])
        depth = load_object(filename)
        push!(df, (N=N, eps=eps, depth=depth))
    end
end
CSV.write(pathname*"result.csv", df)



# disent randMPS
Nrange = [20, 40, 60, 80, 100]
df_list = []
folder = "D:\\Julia\\MyProject\\Data\\randMPS\\disent\\"

for i in 1:3
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], nmps=Int[])
    subfolder = folder*"mps$(i)\\"
    Nrange = [20, 40, 60, 80, 100]
    
    for N in Nrange
        filenames = glob("disent_$(N)_*.jld2", subfolder)
        for filename in filenames
            eps = parse(Float64, filename[53+length(string(N)):end-5])
            depth = load_object(filename)
            push!(df, (N=N, eps=eps, depth=depth, nmps=i))
        end
    end
    CSV.write(subfolder*"result.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(folder*"result_old.csv", df_final)


# TIME INVERT XY
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\g0h0.5LC\\"
df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[])

Nrange = [20, 40, 60, 80, 100]
eps_array = [0.1, 0.02, 0.004, 0.0008]

for N in Nrange
    for eps in eps_array
        # For each file, push a new row
        depth = load_object(folder*"invert_$(N)_$(eps).jld2")
        time1 = load_object(folder*"time_invert1_$(N)_0.5.jld2")
        time2 = load_object(folder*"time_invert2_$(N)_$(eps).jld2")
        push!(df, (N=N, eps=eps, depth=depth, time=time1+time2))
    end
end
CSV.write(folder*"result_invert.csv", df)

# TIME TRUNC XY
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\g0h0.5LC\\"
df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[])

Nrange = [20, 40, 60, 80, 100]
eps_array = [0.1, 0.02, 0.004, 0.0008]

for N in Nrange
    for eps in eps_array
        # For each file, push a new row
        depth = load_object(folder*"trunc_$(N)_$(eps).jld2")
        time = load_object(folder*"time_trunc_$(N)_$(eps).jld2")
        push!(df, (N=N, eps=eps, depth=depth, time=time))
    end
end
CSV.write(folder*"result_trunc.csv", df)


# TIME DISENT XY
folder = "D:\\Julia\\MyProject\\Data\\xy\\disent\\g0h0.5new\\"
df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[])

Nrange = [20, 40, 60, 80, 100]
eps_array = [0.1, 0.02, 0.004, 0.0008]

for N in Nrange
    filenames = glob("disent_$(N)_*.jld2", folder)
    for filename in filenames
        eps = parse(Float64, filename[53+length(string(N)):end-5])
        depth = load_object(filename)
        time = load_object(folder*"time_disent_$(N)_$(depth).jld2")
        push!(df, (N=N, eps=eps, depth=depth, time=time))
    end
end
CSV.write(folder*"result.csv", df)


# TIME INVERTFINAL RANDMPS
df_list = []
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\"

for i in 1:5
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[], nmps=Int[])
    subfolder = folder*"mps$(i)\\"
    Nrange = [20, 40, 60, 80, 100]
    eps_array = [0.1, 0.02, 0.004, 0.0008]
    
    for N in Nrange
        for eps in eps_array
            # For each file, push a new row
            depth = load_object(subfolder*"invert_$(N)_$(eps).jld2")
            time1 = load_object(subfolder*"time_invert1_$(N)_0.5.jld2")
            time2 = load_object(subfolder*"time_invert2_$(N)_$(eps).jld2")
            push!(df, (N=N, eps=eps, depth=depth, time=time1+time2, nmps=i))
        end
    end
    CSV.write(subfolder*"result_invert.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(folder*"result_invert.csv", df_final)



# TIME INVERT1 RANDMPS
df_list = []
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\"

for i in 1:5
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[], nmps=Int[])
    subfolder = folder*"mps$(i)\\"
    Nrange = [20, 40, 60, 80, 100, 300, 1000]
    eps_array = [0.5]
    
    for N in Nrange
        for eps in eps_array
            # For each file, push a new row
            params = load_object(subfolder*"$(N)_params.jld2")
            depth = params["start_tau"]
            time1 = load_object(subfolder*"time_invert1_$(N)_0.5.jld2")
            push!(df, (N=N, eps=eps, depth=depth, time=time1, nmps=i))
        end
    end
    CSV.write(subfolder*"result_invert1.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(folder*"result_invert1.csv", df_final)



# TIME TRUNC RANDMPS
df_list = []
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\"

for i in 1:5
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[], nmps=Int[])
    subfolder = folder*"mps$(i)\\"
    Nrange = [20, 40, 60, 80, 100]
    eps_array = [0.1, 0.02, 0.004, 0.0008]
    
    for N in Nrange
        for eps in eps_array
            # For each file, push a new row
            depth = load_object(subfolder*"trunc_$(N)_$(eps).jld2")
            time = load_object(subfolder*"time_trunc_$(N)_$(eps).jld2")
            push!(df, (N=N, eps=eps, depth=depth, time=time, nmps=i))
        end
    end
    CSV.write(subfolder*"result_trunc.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(folder*"result_trunc.csv", df_final)



# DISENT RANDMPS NEW + TIME
Nrange = [20, 40, 60, 80, 100]
df_list = []
folder = "D:\\Julia\\MyProject\\Data\\cluster_copy\\"

for i in 1:5
    # Create an empty DataFrame
    df = DataFrame(N=Int[], eps=Float64[], depth=Int[], time=Float64[], nmps=Int[])
    subfolder = folder*"mps$(i)\\"
    Nrange = [20, 40, 60, 80, 100]
    
    for N in Nrange
        filenames = glob("disent_$(N)_*.jld2", subfolder)
        for filename in filenames
            eps = parse(Float64, filename[51+length(string(N)):end-5])
            depth = load_object(filename)
            time = load_object(subfolder*"time_disent_$(N)_$(depth).jld2")
            push!(df, (N=N, eps=eps, depth=depth, time=time, nmps=i))
        end
    end
    CSV.write(subfolder*"result.csv", df)
    push!(df_list, df)
end

df_final = reduce(vcat, df_list)
CSV.write(folder*"result_disent.csv", df_final)


# RESULTS SCALING XXZ
folder = "D:\\Julia\\MyProject\\Data\\xxz\\"
df = DataFrame(N=Int[], fid=Float64[], depth=Int[], time=Float64[])
filenames = glob("df_*.jld2", folder)
for name in filenames
    df = vcat(df, load_object(name))
end
CSV.write(folder*"df_all.csv", df)


# SCALING RANDOM MPS
path = "D:\\Julia\\MyProject\\Data\\cluster_copy\\"

for i in 1:5
    df = DataFrame(N=Int[], fid=Float64[], depth=Int[], time=Float64[])
    folder = path*"mps$i\\"
    filenames = glob("df_*.jld2", folder)
    for name in filenames
        df = vcat(df, load_object(name))
    end
    CSV.write(folder*"df$i.csv", df)
end



# TIME ERR COMPARISON LIU + SWEEP
pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\sweep\\"

times_liu = []
errs = []
times = []
xs = []


df = load_object(pathname*"df_200_2.jld2") 

for i in [0,2,3,4,5,6]
    df = load_object(pathname*"df_200_$(i).jld2")
    push!(times_liu, df.time)
    params = load_object(pathname*"200_$(i)_params.jld2")
    start_tau = params["start_tau"]
    files_i = glob("invertMPS2_$(i)_invertSweepLC_200_0.01_$(start_tau)_*", pathname)
    pairs_i = [load_object(file) for file in files_i]
    errs_i = [pair["err"] for pair in pairs_i]
    push!(errs, errs_i)
    times_i = [pair["time"] for pair in pairs_i]
    push!(times, times_i)
    x_i = [j for j in start_tau:(length(errs_i)+start_tau-1)]
    push!(xs, x_i)
end

plot = Plots.plot(xs[1][2:end], 1 .- errs[1][2:end], marker = (:circle,5), yscale=:log10, label=L"$\tau = 0$", legend=:bottomright, xlabel = L"\tau", ylabel = L"\epsilon")
Plots.plot!(plot, xs[2], 1 .- errs[2], marker = (:circle,5), label=L"$\tau = 2$")
Plots.plot!(plot, xs[3], 1 .- errs[3], marker = (:circle,5), label=L"$\tau = 3$")
Plots.plot!(plot, xs[4], 1 .- errs[4], marker = (:circle,5), label=L"$\tau = 4$")
Plots.plot!(plot, xs[5], 1 .- errs[5], marker = (:circle,5), label=L"$\tau = 5$")
Plots.plot!(plot, xs[6], 1 .- errs[6], marker = (:circle,5), label=L"$\tau = 6$")

plot = Plots.plot(xs[1][2:end], times[1][2:end], marker = (:circle,5), yscale=:log10, label=L"$\tau = 0$", xlabel = L"\tau", ylabel = L"t")
Plots.plot!(plot, xs[2], times[2], marker = (:circle,5), label=L"$\tau = 2$")
Plots.plot!(plot, xs[3], times[3], marker = (:circle,5), label=L"$\tau = 3$")
Plots.plot!(plot, xs[4], times[4], marker = (:circle,5), label=L"$\tau = 4$")
Plots.plot!(plot, xs[5], times[5], marker = (:circle,5), label=L"$\tau = 5$")
Plots.plot!(plot, xs[6], times[6], marker = (:circle,5), label=L"$\tau = 6$")

plot = Plots.plot(xs[1][2:end], times_liu[1] .+ cumsum(times[1][2:end]), marker = (:circle,5), yscale=:log10, label=L"$\tau = 0$", xlabel = L"\tau", ylabel = L"t_{tot}")
Plots.plot!(plot, xs[2], times_liu[2] .+ cumsum(times[2]), marker = (:circle,5), label=L"$\tau = 2$")
Plots.plot!(plot, xs[3], times_liu[3] .+ cumsum(times[3]), marker = (:circle,5), label=L"$\tau = 3$")
Plots.plot!(plot, xs[4], times_liu[4] .+ cumsum(times[4]), marker = (:circle,5), label=L"$\tau = 4$")
Plots.plot!(plot, xs[5], times_liu[5] .+ cumsum(times[5]), marker = (:circle,5), label=L"$\tau = 5$")
Plots.plot!(plot, xs[6], times_liu[6] .+ cumsum(times[6]), marker = (:circle,5), label=L"$\tau = 6$")



# CHECK WHETHER SWEEP IS NUMERICALLY STABLE: IF IT IS, THEN IT'S OVER
pathname = "D:\\Julia\\MyProject\\Data\\randMPS\\sweep\\sweepcheck\\"

times_liu = []
errs = []
times = []
xs = []

for i in [0]
    df = load_object(pathname*"df_200_$(i).jld2")
    push!(times_liu, df.time)
    params = load_object(pathname*"200_$(i)_params.jld2")
    start_tau = params["start_tau"]
    files_i = glob("invertMPS2_$(i)_invertSweepLC_200_0.01_$(start_tau)_*", pathname)
    pairs_i = [load_object(file) for file in files_i]
    errs_i = [pair["err"] for pair in pairs_i]
    push!(errs, errs_i)
    times_i = [pair["time"] for pair in pairs_i]
    push!(times, times_i)
    x_i = [j for j in start_tau:(length(errs_i)+start_tau-1)]
    push!(xs, x_i)
end

plot = Plots.plot(xs[1][2:end], 1 .- errs[1][2:end], marker = (:circle,5), yscale=:log10, label=L"$\tau = 0$", legend=:bottomright, xlabel = L"\tau", ylabel = L"\epsilon")