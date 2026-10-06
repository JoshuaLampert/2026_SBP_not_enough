using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Dispersion and dissipation analysis of the multiblock FSBP-SAT schemes with N = 15 nodes per block
# used in Section 4, following the approach of Gassner and Kopriva (SIAM J. Sci. Comput., 2011).
using LinearAlgebra
using Printf: @sprintf
import Optim, ForwardDiff
import Manifolds
using Manopt
using Trixi
using SummationByPartsOperatorsExtra: function_space_operator, mass_matrix, grid,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzIskeLampertÖffner2026Regularized,
                                      get_optimization_entries, derivative_operator,
                                      MattssonNordström2004, legendre_derivative_operator
using Plots
using LaTeXStrings
using PrettyTables

OUT = joinpath(@__DIR__, "figures")
isdir(OUT) || mkdir(OUT)

# Operators, constructed as in `multiblock_sparse.jl` and `multiblock_regularization.jl`
p = 14
N = p + 1
nodes = collect(LinRange(-1.0, 1.0, N))
opt_kwargs = (; verbose = false, opt_alg = Optim.BFGS(),
              options = Optim.Options(g_tol = 1e-16, iterations = 5000, show_trace = false))
basis_poly = [one, identity, x -> x^2, x -> x^3]
basis_sin_cos = [one, identity, sinpi, cospi]

D_dense_p3 = function_space_operator(basis_poly, nodes, GlaubitzNordströmÖffner2023(); opt_kwargs...)
D_sparse_p3 = function_space_operator(basis_poly, nodes, GlaubitzNordströmÖffner2023();
                                      bandwidth = 3, opt_kwargs...)
D_dense_sin_cos = function_space_operator(basis_sin_cos, nodes, GlaubitzNordströmÖffner2023(); opt_kwargs...)
D_sparse_sin_cos = function_space_operator(basis_sin_cos, nodes, GlaubitzNordströmÖffner2023();
                                           bandwidth = 3, opt_kwargs...)
# passing `options` replaces the package defaults, so Manopt's default stopping criterion is used,
# as in `multiblock_regularization.jl`
x0 = get_optimization_entries(D_dense_p3, bandwidth = p)
D_regularized = function_space_operator(basis_poly, nodes, GlaubitzIskeLampertÖffner2026Regularized();
                                        verbose = false, x0 = x0,
                                        regularization_functions = [sinpi, cospi],
                                        options = (; debug = []))
D_FD = derivative_operator(MattssonNordström2004(), 1, 4, -1.0, 1.0, N)
D_GLL = legendre_derivative_operator(-1.0, 1.0, N)

operators = [(L"sparse $\mathcal{P}_3$", D_sparse_p3),
             (L"dense $\mathcal{P}_3$", D_dense_p3),
             (L"sparse $\mathcal{T}$", D_sparse_sin_cos),
             (L"dense $\mathcal{T}$", D_dense_sin_cos),
             (L"regularized $\mathcal{P}_3$", D_regularized),
             ("FD-SBP, order 4", D_FD),
             ("GLL", D_GLL)]

"""
    bloch_matrix(D, θ)

For the linear advection equation with velocity `a > 0` on a periodic mesh of identical blocks of
length `h`, coupled by the upwind flux, the semidiscretization on block `j` reads

    du_j/dt = -(2a/h) (D u_j + P^{-1} e_L (e_L^T u_j - e_R^T u_{j-1})).

The ansatz `u_j = û exp(i (j θ - ω t))` with `θ = κ h` yields `-iω û = -(2a/h) L(θ) û` with the Bloch
matrix `L(θ) = D + P^{-1} e_L (e_L - exp(-iθ) e_R)^T` returned by this function. The scaled
frequency `Ω = ω h / a` of the modes is `Ω = -2i λ` for the eigenvalues `λ` of `L(θ)`; the exact
relation is `Ω = θ`.
"""
function bloch_matrix(D, θ)
    L = Matrix{ComplexF64}(Matrix(D))
    w = diag(Matrix(mass_matrix(D)))
    L[1, 1] += 1 / w[1]
    L[1, end] -= exp(-im * θ) / w[1]
    return L
end

# Check the semidiscretization underlying the Bloch model against the Jacobian of the Trixi.jl
# semidiscretization used in the numerical experiments (8 periodic blocks). The matrices are compared
# directly, since the eigenvalues of operators with clustered near-zero eigenvalues are too sensitive
# to rounding for a meaningful comparison. The global matrix is block circulant with the blocks
# `-(2a/h) (D + P^{-1} e_L e_L^T)` on the diagonal and `(2a/h) P^{-1} e_L e_R^T` coupling each block
# to its left neighbor, so that its eigenvalues are those of `-(2a/h) L(2πj/nblocks)`. Trixi.jl uses
# the integral of one, i.e., `sum(w)`, as the length of the reference block instead of 2. Both agree
# for every operator that is exact for linear functions; for the regularized operator, which satisfies
# the exactness conditions only up to the residual of the optimization, they differ by about 1e-10,
# which rescales all frequencies uniformly by the same factor.
function check_bloch_model(D; a = 2.0, refinement_level = 3)
    equations = LinearScalarAdvectionEquation1D(a)
    solver = FDSBP(D, surface_integral = SurfaceIntegralStrongForm(flux_lax_friedrichs),
                   volume_integral = VolumeIntegralStrongForm())
    mesh = TreeMesh(-1.0, 1.0, initial_refinement_level = refinement_level, n_cells_max = 10_000,
                    periodicity = true)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver;
                                        boundary_conditions = boundary_condition_periodic)
    J = jacobian_ad_forward(semi)
    nblocks = 2^refinement_level
    h = 2.0 / nblocks
    n = size(D, 1)
    w = diag(Matrix(mass_matrix(D)))
    diagonal_block = Matrix(D)
    diagonal_block[1, 1] += 1 / w[1]
    A = zeros(n * nblocks, n * nblocks)
    for j in 1:nblocks
        rows = ((j - 1) * n + 1):(j * n)
        left = ((mod1(j - 1, nblocks) - 1) * n + 1):(mod1(j - 1, nblocks) * n)
        A[rows, rows] .= diagonal_block
        A[first(rows), last(left)] -= 1 / w[1]
    end
    A .*= -(a * sum(w) / h)
    return norm(J - A) / norm(J)
end

"""
    physical_mode(D, θ)

Return the scaled frequency of the physical mode for the scaled wavenumber `θ`, i.e., of the eigenvector
of the Bloch matrix that is closest to the exact Fourier mode `exp(i θ x / 2)` in the norm of `P`,
together with the corresponding overlap in `[0, 1]`.
"""
function physical_mode(D, θ)
    x = grid(D)
    W = mass_matrix(D)
    F = eigen(bloch_matrix(D, θ))
    e = exp.(im * θ / 2 .* x)
    overlap = [abs(v' * W * e) / sqrt(real(v' * W * v) * real(e' * W * e))
               for v in eachcol(F.vectors)]
    j = argmax(overlap)
    return -2im * F.values[j], overlap[j]
end

# Spurious modes: exactly stationary modes (Ω = 0 for all θ) and the weakest damping of the other
# slow modes (|Ω| < 1) for θ in [π/2, 3π/2], where the physical mode satisfies |Ω| ≥ π/2.
function spurious_modes(D; tol = 1e-8)
    θs = range(π / 2, 3π / 2, length = 201)
    n_stationary = minimum(count(<(tol), abs.(-2im .* eigvals(bloch_matrix(D, θ)))) for θ in θs)
    slow = [Ω for θ in θs for Ω in -2im .* eigvals(bloch_matrix(D, θ)) if tol <= abs(Ω) < 1]
    weakest_damping = isempty(slow) ? NaN : minimum(Ω -> -imag(Ω), slow)
    return n_stationary, weakest_damping
end

for (label, D) in operators
    mismatch = check_bloch_model(D)
    println("Bloch model vs. Trixi.jl Jacobian, $label: relative mismatch $mismatch")
    @assert mismatch < 1e-12
end

# Dispersion (Re Ω) and dissipation (Im Ω) of the physical mode, normalized by the number of nodes.
# The physical mode is plotted only as long as it can be identified, i.e., as long as the overlap of
# the corresponding eigenvector with the exact Fourier mode is at least 0.9. Beyond that, no mode of
# the scheme resembles the Fourier mode.
θs = range(1e-3, N * π, length = 4000)
modes = [[physical_mode(D, θ) for θ in θs] for (_, D) in operators]
min_overlap = 0.9

# errors on a logarithmic scale; values below the lower plot limit are not drawn
floor_value = 1e-12
above_floor(v) = v < floor_value ? NaN : v
p_dispersion = plot(xlabel = L"\theta / N", ylabel = L"|\mathrm{Re}(\Omega) - \theta| / N",
                    yscale = :log10, xlims = (0, 2), ylims = (floor_value, 1),
                    legend_columns = 4, legend = (0.35, -0.2), left_margin = 5 * Plots.mm,
                    bottom_margin = 18 * Plots.mm)
p_dissipation = plot(xlabel = L"\theta / N", ylabel = L"-\mathrm{Im}(\Omega) / N",
                     yscale = :log10, xlims = (0, 2), ylims = (floor_value, 1), legend = nothing)
linestyles = [:solid, :dash, :dot, :dashdot]
for (i, ((label, _), m)) in enumerate(zip(operators, modes))
    identifiable = something(findfirst(((_, overlap),) -> overlap < min_overlap, m), length(m) + 1) - 1
    Ω = first.(m[1:identifiable])
    θ = θs[1:identifiable]
    style = (; label = String(label), linewidth = 2,
             linestyle = linestyles[mod1(i, length(linestyles))])
    plot!(p_dispersion, θ ./ N, above_floor.(abs.(real.(Ω) .- θ) ./ N); style...)
    plot!(p_dissipation, θ ./ N, above_floor.(-imag.(Ω) ./ N); style...)
end
p_relation = plot(p_dispersion, p_dissipation, layout = (1, 2), size = (900, 430))
savefig(p_relation, joinpath(OUT, "dispersion_relation.pdf"))

# Table: points per wavelength needed for an error |Ω - θ| ≤ δ (Gassner and Kopriva, 2011) and
# spurious modes
deltas = (1e-1, 1e-2, 1e-3)
function points_per_wavelength(m, δ)
    i = findfirst(i -> abs(first(m[i]) - θs[i]) > δ, eachindex(θs))
    return i === nothing ? NaN : 2π * N / θs[i]
end
data = Matrix{Any}(undef, length(operators), length(deltas) + 2)
for (i, ((_, D), m)) in enumerate(zip(operators, modes))
    for (j, δ) in enumerate(deltas)
        data[i, j] = @sprintf("%.1f", points_per_wavelength(m, δ))
    end
    n_stationary, weakest_damping = spurious_modes(D)
    data[i, end - 1] = n_stationary
    data[i, end] = isnan(weakest_damping) ? "--" : @sprintf("%.1e", weakest_damping)
end
row_labels = first.(operators)
column_labels = [["", "", "", "", ""],
                 [L"\delta = 10^{-1}", L"\delta = 10^{-2}", L"\delta = 10^{-3}", "stationary", "weakest damping"]]
# booktabs format of PrettyTables.jl with trimmed rules below the merged column labels, as in
# `advection_compare_sparse.jl`
latex_table_format = LatexTableFormat(;
    borders = LatexTableBorders(; top_line = "\\toprule", header_line = "\\midrule",
                                merged_header_cell_line = "\\cmidrule(lr)",
                                middle_line = "\\midrule", bottom_line = "\\bottomrule"),
    @latex__all_horizontal_lines, @latex__no_vertical_lines,
    horizontal_lines_at_data_rows = :none)
pretty_table(data; row_labels, column_labels, alignment = :c,
             merge_column_label_cells = [MergeCells(1, 1, 3, "points per wavelength", :c),
                                         MergeCells(1, 4, 2, "spurious modes", :c)],
             backend = :latex, table_format = latex_table_format,
             style = LatexTableStyle(first_line_column_label = String[], row_label = String[]))
