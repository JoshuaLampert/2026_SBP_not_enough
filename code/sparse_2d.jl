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
const EXAMPLE = joinpath(@__DIR__, "examples", "compressible_euler_2d.jl")

function solve_equation(D, equations, tspan, analysis_filename)
    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    initial_condition = initial_condition_convergence_test
    source_terms = source_terms_convergence_test

    redirect_stdout(stdout) do
        trixi_include(EXAMPLE, coordinates_min=coordinates_min, coordinates_max=coordinates_max,
            equations=equations, D=D, tspan=tspan, initial_condition=initial_condition, source_terms=source_terms, initial_refinement_level=0,
            # abstol=1e-14, reltol=1e-14,
            save_analysis=true, analysis_filename=analysis_filename, analysis_interval=10, alive_callback=TrivialCallback())
    end
    l2_error, linf_error = @invokelatest analysis_callback(@invokelatest Main.sol)
    return (@invokelatest Main.semi), (@invokelatest Main.sol), l2_error, linf_error
end

equations = CompressibleEulerEquations2D(1.4)
tspan = (0.0, 10.0)

p = 49
ref_xmin = -1.0
ref_xmax = 1.0
# D_leg = legendre_derivative_operator(ref_xmin, ref_xmax, p + 1)
# nodes = grid(D_leg)
nodes = LinRange(ref_xmin, ref_xmax, p + 1)

# D_FD = derivative_operator(MattssonNordström2004(), 1, 4, ref_xmin, ref_xmax, p + 1)
# semi_FD, sol_FD, l2_FD, linf_FD = solve_equation(D_FD, equations, tspan, "sparse_FD_2d_analysis.dat")
# println("FD: L2 error = $l2_FD, L∞ error = $linf_FD")
# println("rank(D_FD) = $(rank(Matrix(D_FD)))")

basis = [one, identity, x -> x^2, x -> x^3]
D_FSBP_dense_p3 = function_space_operator(basis, collect(nodes), GlaubitzNordströmÖffner2023();
    verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=true))
semi_dense, sol_dense, l2_dense_p3, linf_dense_p3 = solve_equation(D_FSBP_dense_p3, equations, tspan, "dense_p3_2d_analysis.dat")
D_FSBP_sparse_p3 = function_space_operator(basis, collect(nodes), GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=true))
semi_sparse, sol_sparse, l2_sparse_p3, linf_sparse_p3 = solve_equation(D_FSBP_sparse_p3, equations, tspan, "sparse_p3_2d_analysis.dat")

basis = [one, identity, sinpi, cospi]
D_FSBP_dense_sin_cos = function_space_operator(basis, collect(nodes), GlaubitzNordströmÖffner2023();
    verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=true))
semi_dense, sol_dense, l2_dense_sin_cos, linf_dense_sin_cos = solve_equation(D_FSBP_dense_sin_cos, equations, tspan, "dense_sin_cos_2d_analysis.dat")
D_FSBP_sparse_sin_cos = function_space_operator(basis, collect(nodes), GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=true,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=true))
semi_sparse_sin_cos, sol_sparse_sin_cos, l2_sparse_sin_cos, linf_sparse_sin_cos = solve_equation(D_FSBP_sparse_sin_cos, equations, tspan, "sparse_sin_cos_2d_analysis.dat")
println("Dense P_3: L2 error = $l2_dense_p3, L∞ error = $linf_dense_p3")
println("Sparse P_3: L2 error = $l2_sparse_p3, L∞ error = $linf_sparse_p3")
println("rank(D_FSBP_dense_p3) = $(rank(Matrix(D_FSBP_dense_p3))), rank(D_FSBP_sparse_p3) = $(rank(Matrix(D_FSBP_sparse_p3)))")
println("Dense T: L2 error = $l2_dense_sin_cos, L∞ error = $linf_dense_sin_cos")
println("Sparse T: L2 error = $l2_sparse_sin_cos, L∞ error = $linf_sparse_sin_cos")
println("rank(D_FSBP_dense_sin_cos) = $(rank(Matrix(D_FSBP_dense_sin_cos))), rank(D_FSBP_sparse_sin_cos) = $(rank(Matrix(D_FSBP_sparse_sin_cos)))")

# plot(sol_sparse, label="numerical sparse p2")
# plot(sol_FD, label="numerical FD")
# plot!((x, equations) -> initial_condition_convergence_test_higher_frequency(x, last(tspan), equations), semi_dense,
#       label = "analytical", legend = :topright)
# plot!(sol_sparse_sin_cos, label="numerical sparse sin_cos", legend=:topright)

include("plot_error_over_time.jl")
OUT_datafiles = joinpath(@__DIR__, "out")
d = 2
filenames = [
    joinpath(OUT_datafiles, "dense_p3_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "sparse_p3_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "dense_sin_cos_$(d)d_analysis.dat"),
    joinpath(OUT_datafiles, "sparse_sin_cos_$(d)d_analysis.dat"),
]
out_filename = joinpath(OUT, "error_comparison_$(d)d.pdf")
plot_error_over_time(filenames, d, out_filename)
