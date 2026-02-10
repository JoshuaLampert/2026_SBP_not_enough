using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
import Optim, ForwardDiff
using ADTypes: ADTypes
using Mooncake: Mooncake
using SummationByPartsOperatorsExtra: function_space_operator, GlaubitzNordströmÖffner2023, legendre_derivative_operator, grid
using Trixi
using OrdinaryDiffEqSSPRK
using Plots: Plots, plot, plot!, savefig
using PrettyTables
using LaTeXStrings

OUT = joinpath(@__DIR__, "figures")
isdir(OUT) || mkdir(OUT)
const EXAMPLE = joinpath(@__DIR__, "examples", "linear_advection_1d.jl")

function solve_equation(D, equations, initial_condition, tspan)
    nodes = grid(D)
    coordinates_min = minimum(nodes)
    coordinates_max = maximum(nodes)
    N = length(nodes)

    CFL = 0.5
    dx = minimum(diff(nodes))
    dt = CFL * dx / abs(only(equations.advection_velocity))
    redirect_stdout(devnull) do
        trixi_include(EXAMPLE, coordinates_min=coordinates_min, coordinates_max=coordinates_max, N=N,
            equations=equations, D=D, tspan=tspan, initial_condition=initial_condition, initial_refinement_level=0,
            abstol = 1e-6, reltol = 1e-6,
            # dt=dt, adaptive=false, alg=SSPRK53()
            )
    end
    l2_error, linf_error = @invokelatest analysis_callback(@invokelatest Main.sol)
    return (@invokelatest Main.semi), (@invokelatest Main.sol), l2_error, linf_error
end

advection_velocity = 2.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_sinpi(x, t, equations)
    x_t = x[1] - equations.advection_velocity[1] * t
    u = sinpi(x_t)
    return SVector(u)
end
function initial_condition_sinpi_higher_frequency(x, t, equations)
    x_t = x[1] - equations.advection_velocity[1] * t
    u = sinpi(2 * x_t)
    return SVector(u)
end
initial_condition1 = initial_condition_sinpi
initial_condition2 = initial_condition_sinpi_higher_frequency

tspan = (0.0, 1.75)
coordinates_min = -1.0
coordinates_max = 1.0
N = 50
nodes = collect(LinRange(coordinates_min, coordinates_max, N))
p_solutions = plot(layout=(1, 2))
linewidth = 2
linestyles = [:solid, :solid, :dash, :dashdot, :dashdotdot, :dot, :solid, :solid, :solid, :solid, :solid, :solid]
global linestyle_counter = 1

step = 100
d = 2

basis = [x -> x^i for i in 0:d]
l2_errors_final_FSBP1_poly = Float64[]
linf_errors_final_FSBP1_poly = Float64[]
l2_errors_final_FSBP2_poly = Float64[]
linf_errors_final_FSBP2_poly = Float64[]
bandwidths_poly = (3, 4, 5, 6, N - 1)
for bandwidth in bandwidths_poly
    println("FSBP b = $bandwidth")
    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        bandwidth=bandwidth, verbose=false,
        options=Optim.Options(g_tol=1e-16, iterations=50000, show_trace=false), opt_alg=Optim.BFGS(),
        autodiff=ADTypes.AutoMooncake(; config=nothing),
    )
    println(rank(Matrix(D)))
    semi1, sol1, l2_error1, linf_error1 = solve_equation(D, equations, initial_condition1, tspan)
    semi2, sol2, l2_error2, linf_error2 = solve_equation(D, equations, initial_condition2, tspan)
    # first (and only) variable at final time step
    push!(l2_errors_final_FSBP1_poly, l2_error1[1, end])
    push!(linf_errors_final_FSBP1_poly, linf_error1[1, end])
    push!(l2_errors_final_FSBP2_poly, l2_error2[1, end])
    push!(linf_errors_final_FSBP2_poly, linf_error2[1, end])

    pd1 = PlotData1D(sol1.u[step], semi1)
    pd2 = PlotData1D(sol2.u[step], semi2)
    plot!(p_solutions, pd1["scalar"], label="b = $bandwidth", title="", xlims=:auto,
        linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=1)
    plot!(p_solutions, pd2["scalar"], label="b = $bandwidth", title="", xlims=:auto,
        linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=2)
    global linestyle_counter += 1
end

basis = [one, identity, sinpi, cospi]
l2_errors_final_FSBP1_sin_cos = Float64[]
linf_errors_final_FSBP1_sin_cos = Float64[]
l2_errors_final_FSBP2_sin_cos = Float64[]
linf_errors_final_FSBP2_sin_cos = Float64[]
bandwidths_sin_cos = (3, 4, 5, 6, N - 1)
for bandwidth in bandwidths_sin_cos
    println("FSBP b = $bandwidth")
    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        bandwidth=bandwidth, verbose=false,
        options=Optim.Options(g_tol=1e-17, iterations=10000, show_trace=false), opt_alg=Optim.BFGS(),
        autodiff=ADTypes.AutoMooncake(; config=nothing),
    )
    println(rank(Matrix(D)))
    semi1, sol1, l2_error1, linf_error1 = solve_equation(D, equations, initial_condition1, tspan)
    semi2, sol2, l2_error2, linf_error2 = solve_equation(D, equations, initial_condition2, tspan)
    # first (and only) variable at final time step
    push!(l2_errors_final_FSBP1_sin_cos, l2_error1[1, end])
    push!(linf_errors_final_FSBP1_sin_cos, linf_error1[1, end])
    push!(l2_errors_final_FSBP2_sin_cos, l2_error2[1, end])
    push!(linf_errors_final_FSBP2_sin_cos, linf_error2[1, end])

    pd1 = PlotData1D(sol1.u[step], semi1)
    pd2 = PlotData1D(sol2.u[step], semi2)
    plot!(p_solutions, pd1["scalar"], label="b = $bandwidth", title="", xlims=:auto,
        linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=1)
    plot!(p_solutions, pd2["scalar"], label="b = $bandwidth", title="", xlims=:auto,
        linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=2)
    global linestyle_counter += 1
end

l2_errors_final_FD1 = Float64[]
linf_errors_final_FD1 = Float64[]
l2_errors_final_FD2 = Float64[]
linf_errors_final_FD2 = Float64[]
FD_orders = (2, 4, 6) # order 8 is unstable
for (order, linestyle) in zip(FD_orders, linestyles)
    println("FD order = $order")
    D_FD = derivative_operator(MattssonNordström2004(), 1, order, coordinates_min, coordinates_max, N)
    println(rank(Matrix(D_FD)))
    semi_FD1, sol_FD1, l2_error_FD1, linf_error_FD1 = solve_equation(D_FD, equations, initial_condition1, tspan)
    semi_FD2, sol_FD2, l2_error_FD2, linf_error_FD2 = solve_equation(D_FD, equations, initial_condition2, tspan)
    # first (and only) variable at final time step
    push!(l2_errors_final_FD1, l2_error_FD1[1, end])
    push!(linf_errors_final_FD1, linf_error_FD1[1, end])
    push!(l2_errors_final_FD2, l2_error_FD2[1, end])
    push!(linf_errors_final_FD2, linf_error_FD2[1, end])

    pd1 = PlotData1D(sol_FD1.u[step], semi_FD1)
    pd2 = PlotData1D(sol_FD2.u[step], semi_FD2)
    if order in (4,)
        plot!(p_solutions, pd1["scalar"], label="FD order $order", title="", xlims=:auto,
            linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=1)
        plot!(p_solutions, pd2["scalar"], label="FD order $order", title="", xlims=:auto,
            linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=2)
        global linestyle_counter += 1
    end
end

D_GLL = legendre_derivative_operator(coordinates_min, coordinates_max, N)
semi_GLL1, sol_GLL1, l2_error_GLL1, linf_error_GLL1 = solve_equation(D_GLL, equations, initial_condition1, tspan)
semi_GLL2, sol_GLL2, l2_error_GLL2, linf_error_GLL2 = solve_equation(D_GLL, equations, initial_condition2, tspan)
# plot!(p_solutions, semi_GLL => sol_GLL, label = "Legendre", plot_title = "", linestyle = :solid)

# t = last(tspan)
t = sol_GLL1.t[step]
pd1 = PlotData1D((x, equation) -> initial_condition1(x, t, equation), semi_GLL1)
plot!(p_solutions, pd1["scalar"], label="analytical", plot_initial=true, title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=1)
t = sol_GLL2.t[step]
pd2 = PlotData1D((x, equation) -> initial_condition2(x, t, equation), semi_GLL2)
plot!(p_solutions, pd2["scalar"], label="analytical", plot_initial=true, title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=2)

# have one legend for all subplots
plot!(subplot=1, legend_column=2, bottom_margin=18 * Plots.mm,
    legend=(0.8, -0.2), legendfontsize=10)

savefig(p_solutions, joinpath(OUT, "advection_solutions_sparse_subplots.pdf"))

# Table
stubhead_label = "Error"
column_labels = [["" for _ in 1:(length(bandwidths_poly)+length(bandwidths_sin_cos)+length(FD_orders))], [[L"b = %$b" for b in bandwidths_poly]..., [L"b = %$b" for b in bandwidths_sin_cos]..., [L"p = %$p" for p in FD_orders]...]]
column_labels[2][length(bandwidths_poly)] = "dense"
column_labels[2][length(bandwidths_poly)+length(bandwidths_sin_cos)] = "dense"
row_labels = [L"$L^2$", L"$L^\infty$"]
data = Matrix{Any}(undef, 2, length(bandwidths_poly) + length(bandwidths_sin_cos) + length(FD_orders))
for (i, err) in enumerate(l2_errors_final_FSBP1_poly)
    data[1, i] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP1_poly)
    data[2, i] = err
end
for (i, err) in enumerate(l2_errors_final_FSBP1_sin_cos)
    data[1, i+length(bandwidths_poly)] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP1_sin_cos)
    data[2, i+length(bandwidths_poly)] = err
end
for (i, err) in enumerate(l2_errors_final_FD1)
    data[1, i+length(bandwidths_poly)+length(bandwidths_sin_cos)] = err
end
for (i, err) in enumerate(linf_errors_final_FD1)
    data[2, i+length(bandwidths_poly)+length(bandwidths_sin_cos)] = err
end
# data[1, end] = l2_error_GLL[1, end]
# data[2, end] = linf_error_GLL[1, end]
pretty_table(data; row_labels, column_labels, stubhead_label,
    alignment=:c, formatters=[fmt__printf("%.1e")],
    merge_column_label_cells=[MergeCells(1, 1, length(bandwidths_poly), L"FSBP with $\mathcal{F} = \mathcal{P}_2$", :c),
        MergeCells(1, length(bandwidths_poly) + 1, length(bandwidths_sin_cos), L"FSBP with $\mathcal{F} = \mathcal{T}$", :c),
        MergeCells(1, length(bandwidths_poly) + length(bandwidths_sin_cos) + 1, length(FD_orders), "Classical FD", :c)],
    backend=:latex, table_format=latex_table_format__booktabs, style=LatexTableStyle(first_line_column_label=String[]),
)

data = Matrix{Any}(undef, 2, length(bandwidths_poly) + length(bandwidths_sin_cos) + length(FD_orders))
for (i, err) in enumerate(l2_errors_final_FSBP2_poly)
    data[1, i] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP2_poly)
    data[2, i] = err
end
for (i, err) in enumerate(l2_errors_final_FSBP2_sin_cos)
    data[1, i+length(bandwidths_poly)] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP2_sin_cos)
    data[2, i+length(bandwidths_poly)] = err
end
for (i, err) in enumerate(l2_errors_final_FD2)
    data[1, i+length(bandwidths_poly)+length(bandwidths_sin_cos)] = err
end
for (i, err) in enumerate(linf_errors_final_FD2)
    data[2, i+length(bandwidths_poly)+length(bandwidths_sin_cos)] = err
end
# data[1, end] = l2_error_GLL[1, end]
# data[2, end] = linf_error_GLL[1, end]
pretty_table(data; row_labels, column_labels, stubhead_label,
    alignment=:c, formatters=[fmt__printf("%.1e")],
    merge_column_label_cells=[MergeCells(1, 1, length(bandwidths_poly), L"FSBP with $\mathcal{F} = \mathcal{P}_2$", :c),
        MergeCells(1, length(bandwidths_poly) + 1, length(bandwidths_sin_cos), L"FSBP with $\mathcal{F} = \mathcal{T}$", :c),
        MergeCells(1, length(bandwidths_poly) + length(bandwidths_sin_cos) + 1, length(FD_orders), "Classical FD", :c)],
    backend=:latex, table_format=latex_table_format__booktabs, style=LatexTableStyle(first_line_column_label=String[]),
)
