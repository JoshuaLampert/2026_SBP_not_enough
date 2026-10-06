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
using Printf: @sprintf
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
            abstol = 1e-10, reltol = 1e-10,
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
    # `g_tol = 0.0`: with a positive gradient tolerance, BFGS stops for b = 4, 5 before the residual
    # falls below the threshold of Remark 2
    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        bandwidth=bandwidth, verbose=false,
        options=Optim.Options(g_tol=0.0, iterations=100_000, show_trace=false), opt_alg=Optim.BFGS(),
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
plot!(p_solutions, pd1["scalar"], label="analytical", title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=1)
t = sol_GLL2.t[step]
pd2 = PlotData1D((x, equation) -> initial_condition2(x, t, equation), semi_GLL2)
plot!(p_solutions, pd2["scalar"], label="analytical", title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=2)

# have one legend for all subplots
plot!(subplot=1, legend_column=2, bottom_margin=18 * Plots.mm,
    legend=(0.8, -0.2), legendfontsize=10)

savefig(p_solutions, joinpath(OUT, "advection_solutions_sparse_subplots.pdf"))

# Tables in the layout of the paper: bandwidths as rows (the last one, N - 1, corresponds to the dense
# operators) and the L2 and Linf errors for F = P_2 and F = T as columns. In each norm, the entries that
# are smallest at the printed precision are set in bold.
format_error(err) = @sprintf("%.1e", err)
# booktabs format of PrettyTables.jl, but with trimmed rules below the merged column labels so that the
# two groups of columns remain visibly separated
const latex_table_format = LatexTableFormat(;
    borders=LatexTableBorders(; top_line="\\toprule", header_line="\\midrule",
                              merged_header_cell_line="\\cmidrule(lr)",
                              middle_line="\\midrule", bottom_line="\\bottomrule"),
    @latex__all_horizontal_lines, @latex__no_vertical_lines,
    horizontal_lines_at_data_rows=:none)
function print_error_table(bandwidths, l2_poly, linf_poly, l2_sin_cos, linf_sin_cos)
    data = hcat(l2_poly, linf_poly, l2_sin_cos, linf_sin_cos)
    row_labels = [b == N - 1 ? "dense" : L"b = %$b" for b in bandwidths]
    column_labels = [["", "", "", ""], [L"$L^2$", L"$L^\infty$", L"$L^2$", L"$L^\infty$"]]
    # columns 1 and 3 contain L2 errors, columns 2 and 4 Linf errors
    is_best(data, i, j) = format_error(data[i, j]) ==
                          format_error(minimum(data[:, isodd(j) ? [1, 3] : [2, 4]]))
    pretty_table(data; row_labels, column_labels,
        alignment=:c, formatters=[fmt__printf("%.1e")],
        highlighters=[LatexHighlighter(is_best, ["textbf"])],
        merge_column_label_cells=[MergeCells(1, 1, 2, L"$\mathcal{F} = \mathcal{P}_2$", :c),
            MergeCells(1, 3, 2, L"$\mathcal{F} = \mathcal{T}$", :c)],
        backend=:latex, table_format=latex_table_format,
        style=LatexTableStyle(first_line_column_label=String[], row_label=String[]),
    )
end

@assert bandwidths_poly == bandwidths_sin_cos

println("Initial condition sin(pi (x - a t)), k = 1 (not shown as a table in the paper):")
print_error_table(bandwidths_poly, l2_errors_final_FSBP1_poly, linf_errors_final_FSBP1_poly,
                  l2_errors_final_FSBP1_sin_cos, linf_errors_final_FSBP1_sin_cos)
println("Classical FD-SBP operators of orders $(FD_orders): L2 errors ",
        join(format_error.(l2_errors_final_FD1), ", "), "; Linf errors ",
        join(format_error.(linf_errors_final_FD1), ", "))
println()
println("Initial condition sin(2 pi (x - a t)), k = 2 (Table 3 of the paper):")
print_error_table(bandwidths_poly, l2_errors_final_FSBP2_poly, linf_errors_final_FSBP2_poly,
                  l2_errors_final_FSBP2_sin_cos, linf_errors_final_FSBP2_sin_cos)
println("Classical FD-SBP operators of orders $(FD_orders): L2 errors ",
        join(format_error.(l2_errors_final_FD2), ", "), "; Linf errors ",
        join(format_error.(linf_errors_final_FD2), ", "))
