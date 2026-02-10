using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using SummationByPartsOperatorsExtra: GlaubitzNordströmÖffner2023, function_space_operator, legendre_derivative_operator, grid
import Optim, ForwardDiff
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
# D_leg = legendre_derivative_operator(ref_xmin, ref_xmax, p + 1)
# nodes = grid(D_leg)
nodes = collect(LinRange(ref_xmin, ref_xmax, p + 1))

basis = [one, identity, x -> x^2, x -> x^3]
D_FSBP_dense_p3 = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
semi_dense, sol_dense, l2_dense_p3, linf_dense_p3 = solve_equation(D_FSBP_dense_p3, equations, initial_condition, tspan, "dense_p3_1d_analysis.dat")

D_FSBP_sparse_p3 = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
semi_sparse, sol_sparse, l2_sparse_p3, linf_sparse_p3 = solve_equation(D_FSBP_sparse_p3, equations, initial_condition, tspan, "sparse_p3_1d_analysis.dat")

println("Dense P_3: L2 error = $l2_dense_p3, L∞ error = $linf_dense_p3")
println("Sparse P_3: L2 error = $l2_sparse_p3, L∞ error = $linf_sparse_p3")
println("rank(D_FSBP_dense_p3) = $(rank(Matrix(D_FSBP_dense_p3))), rank(D_FSBP_sparse_p3) = $(rank(Matrix(D_FSBP_sparse_p3)))")

basis = [one, identity, sinpi, cospi]
D_FSBP_dense_sin_cos = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
semi_dense_sin_cos, sol_dense_sin_cos, l2_dense_sin_cos, linf_dense_sin_cos = solve_equation(D_FSBP_dense_sin_cos, equations, initial_condition, tspan, "dense_sin_cos_1d_analysis.dat")

D_FSBP_sparse_sin_cos = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
semi_sparse_sin_cos, sol_sparse_sin_cos, l2_sparse_sin_cos, linf_sparse_sin_cos = solve_equation(D_FSBP_sparse_sin_cos, equations, initial_condition, tspan, "sparse_sin_cos_1d_analysis.dat")

println("Dense T: L2 error = $l2_dense_sin_cos, L∞ error = $linf_dense_sin_cos")
println("Sparse T: L2 error = $l2_sparse_sin_cos, L∞ error = $linf_sparse_sin_cos")
println("rank(D_FSBP_dense_sin_cos) = $(rank(Matrix(D_FSBP_dense_sin_cos))), rank(D_FSBP_sparse_sin_cos) = $(rank(Matrix(D_FSBP_sparse_sin_cos)))")

# plot(sol_sparse_sin_cos, label="sin_cos sparse", legend=:topright)
# plot(sol_dense, label="numerical dense")
# plot!((x, equations) -> initial_condition(x, last(tspan), equations), semi_dense,
#     label="analytical", legend=:topright)

include("plot_error_over_time.jl")
OUT_datafiles = joinpath(@__DIR__, "out")
d = 1
filenames = [
    joinpath(OUT_datafiles, "dense_p3_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "sparse_p3_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "dense_sin_cos_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "sparse_sin_cos_$(d)d_analysis.dat"),
]
out_filename = joinpath(OUT, "error_comparison_$(d)d.pdf")
plot_error_over_time(filenames, d, out_filename)
