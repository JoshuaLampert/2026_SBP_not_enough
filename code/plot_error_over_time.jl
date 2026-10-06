using DelimitedFiles
using DataStructures
using Plots
using LaTeXStrings

# Data structure to hold analysis results
struct AnalysisData
    timesteps::Vector{Int}
    times::Vector{Float64}
    dts::Vector{Float64}
    l2_errors::Matrix{Float64}      # rows: timesteps, columns: variables
    linf_errors::Matrix{Float64}    # rows: timesteps, columns: variables
    cons::Matrix{Float64}           # rows: timesteps, columns: variables
    dsdu_ut::Vector{Float64}
end

"""
    read_analysis_file(filename)

Read analysis data from a .dat file and return an AnalysisData struct.
Automatically detects the number of variables from the column headers.
"""
function read_analysis_file(filename)
    # Read the header to determine the number of variables
    header_line = ""
    open(filename, "r") do f
        header_line = readline(f)
    end

    # Parse header to count variables
    headers = split(header_line)
    l2_cols = findall(col -> startswith(col, "l2_"), headers)
    linf_cols = findall(col -> startswith(col, "linf_"), headers)
    cons_cols = findall(col -> startswith(col, "cons_"), headers)
    dsdu_col = findfirst(col -> col == "dsdu_ut", headers)

    # Read the data, skipping the header line
    data = readdlm(filename, Float64, comments=true, comment_char='#')

    # Extract error data as matrices
    l2_errors = data[:, l2_cols]
    linf_errors = data[:, linf_cols]
    cons = data[:, cons_cols]
    dsdu_ut = data[:, dsdu_col]

    return AnalysisData(
        Int.(data[:, 1]),      # timesteps
        data[:, 2],            # times
        data[:, 3],            # dts
        l2_errors,             # l2_errors matrix
        linf_errors,           # linf_errors matrix
        cons,                  # cons matrix
        dsdu_ut                # dsdu_ut matrix
    )
end

"""
    plot_indices(data; max_points = 250)

Indices of the samples in `data` that are plotted. If the simulation crashed, the adaptive time step collapses and the error
blows up within a vanishing time interval. We omit these samples, so that the curve ends where the simulation breaks down.
A single small time step at the end only stems from hitting the final time exactly and is kept. To keep the plots readable,
at most about `max_points` of the remaining samples are plotted, skipping the initial one.
"""
function plot_indices(data; max_points = 250)
    # the initial sample stores a placeholder instead of a time step
    collapsed = data.dts .< 1e-2 * maximum(data.dts[2:end])
    collapsed[1] = false
    last_regular = findlast(!, collapsed)
    n_samples = length(data.dts) - last_regular <= 1 ? length(data.dts) : last_regular
    stride = max(1, cld(n_samples - 1, max_points))
    indices = collect(2:stride:n_samples)
    last(indices) == n_samples || push!(indices, n_samples)
    return indices
end

function plot_error_over_time(filenames, d, out_filename)
    # Read all data files
    # Use OrderedDict instead of Dict to preserve insertion order
    data_dict = OrderedDict{String,AnalysisData}()

    for filename in filenames
        if isfile(filename)
            label = replace(basename(filename), "_$(d)d_analysis.dat" => "")
            label = replace(label, "_" => " ")
            label = replace(label, "p2" => L"\mathcal{P}_2")
            label = replace(label, "p3" => L"\mathcal{P}_3")
            label = replace(label, "sin cos" => L"\mathcal{T}")
            data_dict[label] = read_analysis_file(filename)
            println("Loaded: $filename ($(length(data_dict[label].times)) time steps)")
        else
            println("Warning: File not found - $filename")
        end
    end

    # Create plots
    if !isempty(data_dict)
        p1 = plot(xlabel="t", ylabel=L"$L^{2}$ Error", yscale=:log10,
            legend_columns=2, legend=(0.84, -0.2), bottom_margin=13 * Plots.mm)

        linestyles = [:solid, :dash, :dot, :dashdot]
        for (i, (label, data)) in enumerate(data_dict)
            # Sum errors across all variables
            l2_sum = sum(data.l2_errors, dims=2)[:]
            indices = plot_indices(data)
            plot!(p1, data.times[indices], l2_sum[indices], label=label, linewidth=2, linestyle=linestyles[mod1(i, length(linestyles))])
        end

        p2 = plot(xlabel="t", ylabel=L"$L^{\infty}$ Error", yscale=:log10, legend=nothing)

        for (i, (label, data)) in enumerate(data_dict)
            # Sum errors across all variables
            linf_sum = sum(data.linf_errors, dims=2)[:]
            indices = plot_indices(data)
            plot!(p2, data.times[indices], linf_sum[indices], label=label, linewidth=2, linestyle=linestyles[mod1(i, length(linestyles))])
        end

        p_errors = plot(p1, p2, layout=(1, 2))
        savefig(p_errors, out_filename)
        println("Error comparison saved to: $out_filename")
    else
        println("No data files found to plot!")
    end
end
