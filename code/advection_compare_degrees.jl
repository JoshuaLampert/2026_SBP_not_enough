using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
import Optim, ForwardDiff
using ADTypes: ADTypes
using Mooncake: Mooncake
using SummationByPartsOperatorsExtra: function_space_operator, legendre_derivative_operator, GlaubitzNordströmÖffner2023, mass_matrix, grid
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
            dt=dt, adaptive=false, alg=SSPRK53())
    end
    l2_error, linf_error = @invokelatest analysis_callback(@invokelatest Main.sol)
    return (@invokelatest Main.semi), (@invokelatest Main.sol), l2_error, linf_error
end

function eigenvalue_property_test(D)
    p = diag(mass_matrix(D))
    nu = 1.0
    D_tilde = Matrix(D)
    D_tilde[1, 1] += nu / p[1]
    return eigvals(D_tilde)
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
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
global linestyle_counter = 1

step = 100
l2_errors_final_FSBP1 = Float64[]
linf_errors_final_FSBP1 = Float64[]
l2_errors_final_FSBP2 = Float64[]
linf_errors_final_FSBP2 = Float64[]
ds = (1, 3, 5, 7, 9, 11)
for d in ds
    println("FSBP d = $d")
    basis = [x -> x^i for i = 0:d]
    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        bandwidth=N - 1, verbose=false,
        options=Optim.Options(g_tol=1e-16, iterations=50000, show_trace=false), opt_alg=Optim.BFGS(),
        autodiff=ADTypes.AutoMooncake(; config=nothing),
    )
    println("rank(D) = ", rank(Matrix(D)))
    lambdas = eigenvalue_property_test(D)
    println("real parts of eigenvalues of D_tilde = ", real.(lambdas))
    println(
        "number of eigenvalues with positive real part: ",
        count(x -> x > 1e-14, real.(lambdas)),
    )
    semi1, sol1, l2_error1, linf_error1 = solve_equation(D, equations, initial_condition1, tspan)
    semi2, sol2, l2_error2, linf_error2 = solve_equation(D, equations, initial_condition2, tspan)
    # first (and only) variable at final time step
    push!(l2_errors_final_FSBP1, l2_error1[1, end])
    push!(linf_errors_final_FSBP1, linf_error1[1, end])
    push!(l2_errors_final_FSBP2, l2_error2[1, end])
    push!(linf_errors_final_FSBP2, linf_error2[1, end])
    if d in (1, 3, 5, 7)
        pd1 = PlotData1D(sol1.u[step], semi1)
        pd2 = PlotData1D(sol2.u[step], semi2)
        plot!(p_solutions, pd1["scalar"], label="d = $d", title="", xlims=:auto,
            linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=1,
        )
        plot!(p_solutions, pd2["scalar"], label="d = $d", title="", xlims=:auto,
            linewidth=linewidth, linestyle=linestyles[linestyle_counter], step=step, subplot=2,
        )
        global linestyle_counter += 1
    end
end

l2_errors_final_FD1 = Float64[]
linf_errors_final_FD1 = Float64[]
l2_errors_final_FD2 = Float64[]
linf_errors_final_FD2 = Float64[]
FD_orders = (2, 4, 6) # order 8 is unstable
for (order, linestyle) in zip(FD_orders, linestyles)
    println("FD order = $order")
    D_FD = derivative_operator(MattssonNordström2004(), 1, order, coordinates_min, coordinates_max, N)
    println("rank(D) = ", rank(Matrix(D_FD)))
    lambdas = eigenvalue_property_test(D_FD)
    println("real parts of eigenvalues of D_tilde = ", real.(lambdas))
    println(
        "number of eigenvalues with positive real part: ",
        count(x -> x > 1e-14, real.(lambdas)),
    )
    semi_FD1, sol_FD1, l2_error_FD1, linf_error_FD1 = solve_equation(D_FD, equations, initial_condition1, tspan)
    semi_FD2, sol_FD2, l2_error_FD2, linf_error_FD2 = solve_equation(D_FD, equations, initial_condition2, tspan)
    # first (and only) variable at final time step
    push!(l2_errors_final_FD1, l2_error_FD1[1, end])
    push!(linf_errors_final_FD1, linf_error_FD1[1, end])
    push!(l2_errors_final_FD2, l2_error_FD2[1, end])
    push!(linf_errors_final_FD2, linf_error_FD2[1, end])
end

D_GLL = legendre_derivative_operator(coordinates_min, coordinates_max, N)
semi_GLL1, sol_GLL1, l2_error_GLL1, linf_error_GLL1 = solve_equation(D_GLL, equations, initial_condition1, tspan)
semi_GLL2, sol_GLL2, l2_error_GLL2, linf_error_GLL2 = solve_equation(D_GLL, equations, initial_condition2, tspan)

# t = last(tspan)
t = sol_GLL1.t[step]
pd1 = PlotData1D((x, equation) -> initial_condition1(x, t, equation), semi_GLL1)
plot!(p_solutions, pd1["scalar"], label="analytical", plot_initial=true, title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=1,
)
t = sol_GLL2.t[step]
pd2 = PlotData1D((x, equation) -> initial_condition2(x, t, equation), semi_GLL2)
plot!(p_solutions, pd2["scalar"], label="analytical", plot_initial=true, title="", xlims=:auto,
    xlabel="x", ylabel="u", linewidth=linewidth, linestyle=linestyles[linestyle_counter],
    yrange=(-1.2, 1.2), legend=nothing, subplot=2,
)

# have one legend for all subplots
plot!(subplot=1, legend_column=2, bottom_margin=18 * Plots.mm,
    legend=(0.8, -0.2), legendfontsize=10)

savefig(p_solutions, joinpath(OUT, "advection_solutions_subplots.pdf"))

# Table
stubhead_label = "Error"
column_labels = [
    ["" for _ = 1:(length(ds)+length(FD_orders))],
    [[L"d = %$d" for d in ds]..., [L"p = %$p" for p in FD_orders]...],
]
row_labels = [L"$L^2$", L"$L^\infty$"]
data = Matrix{Any}(undef, 2, length(ds) + length(FD_orders))
for (i, err) in enumerate(l2_errors_final_FSBP1)
    data[1, i] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP1)
    data[2, i] = err
end
for (i, err) in enumerate(l2_errors_final_FD1)
    data[1, i+length(ds)] = err
end
for (i, err) in enumerate(linf_errors_final_FD1)
    data[2, i+length(ds)] = err
end
# data[1, end] = l2_error_GLL[1, end]
# data[2, end] = linf_error_GLL[1, end]
pretty_table(data; row_labels, column_labels, stubhead_label,
    alignment=:c, formatters=[fmt__printf("%.1e")],
    merge_column_label_cells=[
        MergeCells(1, 1, length(ds), "FSBP", :c),
        MergeCells(1, length(ds) + 1, length(FD_orders), "Classical FD", :c)],
    backend=:latex, table_format=latex_table_format__booktabs, style=LatexTableStyle(first_line_column_label=String[]),
)

data = Matrix{Any}(undef, 2, length(ds) + length(FD_orders))
for (i, err) in enumerate(l2_errors_final_FSBP2)
    data[1, i] = err
end
for (i, err) in enumerate(linf_errors_final_FSBP2)
    data[2, i] = err
end
for (i, err) in enumerate(l2_errors_final_FD2)
    data[1, i+length(ds)] = err
end
for (i, err) in enumerate(linf_errors_final_FD2)
    data[2, i+length(ds)] = err
end
# data[1, end] = l2_error_GLL[1, end]
# data[2, end] = linf_error_GLL[1, end]
pretty_table(data; row_labels, column_labels, stubhead_label,
    alignment=:c, formatters=[fmt__printf("%.1e")],
    merge_column_label_cells=[
        MergeCells(1, 1, length(ds), "FSBP", :c),
        MergeCells(1, length(ds) + 1, length(FD_orders), "Classical FD", :c)],
    backend=:latex, table_format=latex_table_format__booktabs, style=LatexTableStyle(first_line_column_label=String[]),
)
