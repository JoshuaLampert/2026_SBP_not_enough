using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using SummationByPartsOperatorsExtra: GlaubitzNordströmÖffner2023, GlaubitzIskeLampertÖffner2026Regularized, function_space_operator, get_optimization_entries, grid
import Optim, ForwardDiff
import Manifolds
using Manopt
using Trixi
using OrdinaryDiffEqSSPRK
using Plots

OUT = joinpath(@__DIR__, "figures")
isdir(OUT) || mkdir(OUT)
const EXAMPLE = joinpath(@__DIR__, "examples", "linear_advection_1d.jl")

function solve_equation(D, equations, initial_condition, tspan, analysis_filename)
    coordinates_min = -1.0
    coordinates_max = 1.0

    CFL = 0.5
    dx = minimum(diff(nodes))
    dt = CFL * dx / abs(only(equations.advection_velocity))
    redirect_stdout(devnull) do
        trixi_include(EXAMPLE, coordinates_min=coordinates_min, coordinates_max=coordinates_max,
            equations=equations, D=D, tspan=tspan, initial_condition=initial_condition, initial_refinement_level=3,
            # dt=dt, adaptive=false, alg=SSPRK53(),
            save_analysis=true, analysis_filename=analysis_filename, analysis_interval=10)
    end
    l2_error, linf_error = @invokelatest analysis_callback(@invokelatest Main.sol)
    return (@invokelatest Main.semi), (@invokelatest Main.sol), l2_error, linf_error
end

advection_velocity = 2.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_sinpi_higher_frequency(x, t, equations)
    x_t = x[1] - equations.advection_velocity[1] * t
    u = sinpi(2 * x_t)
    return SVector(u)
end
function initial_condition_gaussian(x, t, equations)
    xmin = -1.0
    xmax = 1.0
    x_t = mod(x[1] - equations.advection_velocity[1] * t - xmin, xmax - xmin) + xmin
    u = exp(-x_t^2 / 0.1)
    return SVector(u)
end
initial_condition = initial_condition_gaussian

tspan = (0.0, 50.0)
p = 14
ref_xmin = -1.0
ref_xmax = 1.0
nodes = collect(LinRange(ref_xmin, ref_xmax, p + 1))

basis = [one, identity, x -> x^2, x -> x^3]
D_FSBP_non_reg = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
semi_non_reg, sol_non_reg, l2_non_reg, linf_non_reg = solve_equation(D_FSBP_non_reg, equations, initial_condition, tspan, "non_regularized_1d_analysis.dat")

x0 = get_optimization_entries(D_FSBP_non_reg, bandwidth=p)
debug = [:Iteration, :Time, " | ", (:Cost, "f(x): %.6e"), " | ",
    # (:GradientNorm, "||∇f(x)||: %.6e"), " | ",
    DebugFeasibility(["feasible: ", :Feasible, ", total violation: ", :TotalEq]),
    "\n", :Stop]
stopping_criterion = StopAfterIteration(102) |
                     #  StopWhenGradientNormLess(1e-16) |
                     StopWhenCostLess(1e-28)
regularization_functions = [sinpi, cospi]
D_FSBP_reg = function_space_operator(basis, nodes,
    GlaubitzIskeLampertÖffner2026Regularized(); verbose=true, x0=x0,
    regularization_functions=regularization_functions,
    options=(;
        debug=debug,
        stopping_criterion=stopping_criterion,
    ))

semi_reg, sol_reg, l2_reg, linf_reg = solve_equation(D_FSBP_reg, equations, initial_condition, tspan, "regularized_1d_analysis.dat")
println("Non reg P_2: L2 error = $l2_non_reg, L∞ error = $linf_non_reg")
println("Reg P_2: L2 error = $l2_reg, L∞ error = $linf_reg")
println("rank(D_FSBP_non_reg) = $(rank(Matrix(D_FSBP_non_reg))), rank(D_FSBP_reg) = $(rank(Matrix(D_FSBP_reg)))")

# plot(sol_reg, label="regularized", legend=:topright)
# plot(sol_non_reg, label="non-regularized")
# plot!((x, equations) -> initial_condition(x, last(tspan), equations), semi_non_reg,
#     label="analytical", legend=:topright)

include("plot_error_over_time.jl")
OUT_datafiles = joinpath(@__DIR__, "out")
d = 1
filenames = [
    joinpath(OUT_datafiles, "non_regularized_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "regularized_$(d)d_analysis.dat"),
]
out_filename = joinpath(OUT, "error_comparison_$(d)d_regularization.pdf")
plot_error_over_time(filenames, d, out_filename)
