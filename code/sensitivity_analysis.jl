using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Sensitivity of the FSBP operators used in `multiblock_sparse.jl` (Figure 2) with respect to the
# initialization of the optimization: the operators are constructed from the default initial guess and
# from random initial guesses, and each operator is characterized by its rank, the resolution of the
# physical mode (see `dispersion_analysis.jl`), and the L2 error of the multiblock simulation of Figure 2.
using LinearAlgebra
using Random
using Printf: Printf, @printf, @sprintf
using Statistics: median
import Optim, ForwardDiff
using Trixi
using OrdinaryDiffEqLowStorageRK
using SummationByPartsOperatorsExtra: function_space_operator, mass_matrix, grid,
                                      GlaubitzNordströmÖffner2023, get_nsigma
using LaTeXStrings
using PrettyTables

const EXAMPLE = joinpath(@__DIR__, "examples", "linear_advection_1d.jl")

# Setup of `multiblock_sparse.jl`
advection_velocity = 2.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
function initial_condition_gaussian(x, t, equations)
    xmin = -1.0
    xmax = 1.0
    x_t = mod(x[1] - equations.advection_velocity[1] * t - xmin, xmax - xmin) + xmin
    u = exp(-x_t^2 / 0.1)
    return SVector(u)
end
tspan = (0.0, 50.0)
p = 14
N = p + 1
nodes = collect(LinRange(-1.0, 1.0, N))
opt_kwargs = (; verbose = false, opt_alg = Optim.BFGS(),
              options = Optim.Options(g_tol = 1e-16, iterations = 5000, show_trace = false))
bases = [(L"\mathcal{P}_3", [one, identity, x -> x^2, x -> x^3]),
         (L"\mathcal{T}", [one, identity, sinpi, cospi])]
ansatzes = [("sparse", 3), ("dense", p)]

function l2_error(D)
    redirect_stdout(devnull) do
        trixi_include(EXAMPLE, coordinates_min = -1.0, coordinates_max = 1.0,
                      equations = equations, D = D, tspan = tspan,
                      initial_condition = initial_condition_gaussian, initial_refinement_level = 3)
    end
    l2, _ = @invokelatest Main.analysis_callback(@invokelatest Main.sol)
    return only(l2)
end

# Bloch matrix and physical mode as in `dispersion_analysis.jl`
function bloch_matrix(D, θ)
    L = Matrix{ComplexF64}(Matrix(D))
    w = diag(Matrix(mass_matrix(D)))
    L[1, 1] += 1 / w[1]
    L[1, end] -= exp(-im * θ) / w[1]
    return L
end

function physical_mode(D, θ)
    x = grid(D)
    W = mass_matrix(D)
    F = eigen(bloch_matrix(D, θ))
    e = exp.(im * θ / 2 .* x)
    overlap = [abs(v' * W * e) / sqrt(real(v' * W * v) * real(e' * W * e))
               for v in eachcol(F.vectors)]
    return -2im * F.values[argmax(overlap)]
end

# points per wavelength needed for an error |Ω - θ| ≤ δ of the physical mode
function points_per_wavelength(D; δ = 1e-2)
    θs = range(1e-3, N * π, length = 4000)
    i = findfirst(θ -> abs(physical_mode(D, θ) - θ) > δ, θs)
    return i === nothing ? NaN : 2π * N / θs[i]
end

# residual ||XW + BV/2|| = ||P (D V - V_x)||, see (2) and Remark 2
function residual(D, basis)
    V = [f(x) for x in nodes, f in basis]
    V_x = [ForwardDiff.derivative(f, x) for x in nodes, f in basis]
    return norm(mass_matrix(D) * (Matrix(D) * V - V_x))
end

# number of eigenvalues of D̃ = D + P^{-1} e_L e_L^T with non-positive real part, see (13)
function n_minus(D)
    Dt = Matrix(D)
    Dt[1, 1] += 1 / mass_matrix(D)[1, 1]
    return count(<=(1e-14), real.(eigvals(Dt)))
end

# Initial guesses: the default of SummationByPartsOperatorsExtra.jl, σ = 0 and s(ρ_i) = 1/N, and random
# perturbations of it with a small and a large standard deviation (10 seeds each)
invsig(p) = log(p / (1 - p))
n_seeds = 10
scales = [(0.1, 0.1), (1.0, 0.5)] # standard deviations of σ and ρ

results = Dict()
for (basis_label, basis) in bases, (ansatz, bandwidth) in ansatzes
    L = get_nsigma(N; bandwidth, size_boundary = 2 * bandwidth, different_values = true,
                   sparsity_pattern = nothing)
    x0s = Union{Nothing, Vector{Float64}}[nothing]
    for (scale_sigma, scale_rho) in scales, seed in 1:n_seeds
        rng = Xoshiro(seed)
        push!(x0s, [scale_sigma .* randn(rng, L); invsig(1 / N) .+ scale_rho .* randn(rng, N)])
    end
    runs = map(x0s) do x0
        kwargs = bandwidth == p ? opt_kwargs : (; opt_kwargs..., bandwidth)
        kwargs = isnothing(x0) ? kwargs : (; kwargs..., x0)
        D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023(); kwargs...)
        run = (; residual = residual(D, basis), rank = rank(Matrix(D)), n_minus = n_minus(D),
               ppw = points_per_wavelength(D), error = l2_error(D))
        @printf("%-6s %-16s %-7s residual = %.1e, rank = %2d, n_- = %d, PPW = %5.1f, L2 error = %.2e\n",
                ansatz, basis_label, isnothing(x0) ? "default" : "random", run.residual, run.rank,
                run.n_minus, run.ppw, run.error)
        return run
    end
    results[(ansatz, basis_label)] = runs
end

# Table: default initial guess and range (minimum -- maximum) over the random initial guesses
format_range(values, fmt) = Printf.format(fmt, minimum(values)) * "--" * Printf.format(fmt, maximum(values))
row_labels = LaTeXString[]
data = Matrix{Any}(undef, length(bases) * length(ansatzes), 6)
for (i, ((basis_label, _), (ansatz, _))) in enumerate((b, a) for b in bases for a in ansatzes)
    default, random = Iterators.peel(results[(ansatz, basis_label)])
    random = collect(random)
    push!(row_labels, latexstring(ansatz, " ", basis_label))
    consistent = count(run -> run.rank == N - 1, random)
    data[i, :] = [default.rank, @sprintf("%.1f", default.ppw), @sprintf("%.1e", default.error),
                  "$consistent/$(length(random))",
                  format_range(getfield.(random, :ppw), Printf.Format("%.1f")),
                  format_range(getfield.(random, :error), Printf.Format("%.1e"))]
    println("$ansatz $basis_label: maximal residual $(maximum(run -> run.residual, results[(ansatz, basis_label)])), " *
            "n_- = 0 for $(count(run -> run.n_minus == 0, random))/$(length(random)) " *
            "random initial guesses, median PPW $(median(getfield.(random, :ppw))), " *
            "median L2 error $(median(getfield.(random, :error)))")
end
column_labels = [["", "", "", "", "", ""],
                 ["rank", "PPW", "error", L"rank $N - 1$", "PPW", "error"]]
# booktabs format of PrettyTables.jl with trimmed rules below the merged column labels, as in
# `advection_compare_sparse.jl`
latex_table_format = LatexTableFormat(;
    borders = LatexTableBorders(; top_line = "\\toprule", header_line = "\\midrule",
                                merged_header_cell_line = "\\cmidrule(lr)",
                                middle_line = "\\midrule", bottom_line = "\\bottomrule"),
    @latex__all_horizontal_lines, @latex__no_vertical_lines,
    horizontal_lines_at_data_rows = :none)
pretty_table(data; row_labels, column_labels, alignment = :c,
             merge_column_label_cells = [MergeCells(1, 1, 3, "default initial guess", :c),
                                         MergeCells(1, 4, 3, "$(length(scales) * n_seeds) random initial guesses", :c)],
             backend = :latex, table_format = latex_table_format,
             style = LatexTableStyle(first_line_column_label = String[], row_label = String[]))
