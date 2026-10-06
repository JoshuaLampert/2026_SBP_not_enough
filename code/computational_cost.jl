# Computational cost of sparse compared to unstructured dense FSBP operators:
# number of nonzero entries, cost of a matrix-vector product, and total runtime
# of the two-dimensional example.
#
# Note that the sparsity of the differentiation matrix only pays off if it is
# also exploited by the implementation. `to_sparse` rebuilds an operator with a
# sparse matrix as storage; since `mul!` for a `MatrixDerivativeOperator` simply
# forwards to the stored matrix, this suffices and no changes to Trixi.jl are
# needed.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using BenchmarkTools
using LinearAlgebra
using SparseArrays
import Optim, ForwardDiff
using Trixi
using SummationByPartsOperatorsExtra: function_space_operator, GlaubitzNordströmÖffner2023,
                                      grid, mass_matrix, MatrixDerivativeOperator,
                                      accuracy_order, source_of_coefficients
using Printf

const EXAMPLE = joinpath(@__DIR__, "examples", "compressible_euler_2d.jl")

# Rebuild `D` with sparse storage. Passing the end points of the grid as `xmin`
# and `xmax` gives a Jacobian of one, i.e., the already scaled nodes, weights,
# and matrix are kept unchanged.
function to_sparse(D)
    x = grid(D)
    return MatrixDerivativeOperator(first(x), last(x), collect(x),
                                    diag(Matrix(mass_matrix(D))), sparse(Matrix(D)),
                                    accuracy_order(D), source_of_coefficients(D))
end

const OPT = (; verbose = false, opt_alg = Optim.BFGS(),
             options = Optim.Options(g_tol = 1e-16, iterations = 5000, show_trace = false))

function time_mul(D, N)
    u = [SVector{4, Float64}(randn(4)) for _ in 1:N]
    du = [zero(SVector{4, Float64}) for _ in 1:N]
    return @belapsed mul!($du, $D, $u, true, true)
end

function run_example(D)
    redirect_stdout(devnull) do
        trixi_include(EXAMPLE; D, initial_refinement_level = 0, tspan = (0.0, 10.0),
                      abstol = 1e-10, reltol = 1e-10)
    end
end

println("Number of nonzero entries and cost of a matrix-vector product")
for (N, d, bandwidth) in ((15, 3, 3), (50, 2, 3), (100, 2, 3))
    nodes = collect(LinRange(-1.0, 1.0, N))
    basis = [x -> x^i for i in 0:d]
    D_dense = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023(); OPT...)
    D_sparse = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                       bandwidth, OPT...)
    nnz_dense = count(!iszero, Matrix(D_dense))
    nnz_sparse = count(!iszero, Matrix(D_sparse))
    t_dense = time_mul(D_dense, N)
    t_sparse_dense_storage = time_mul(D_sparse, N)
    t_sparse_sparse_storage = time_mul(to_sparse(D_sparse), N)
    @printf("N = %3d: nonzeros %5d (dense) vs %5d (sparse)\n", N, nnz_dense, nnz_sparse)
    @printf("         mul!: %6.2f mus (dense), %6.2f mus (sparse, dense storage), %6.2f mus (sparse, sparse storage)\n",
            1e6 * t_dense, 1e6 * t_sparse_dense_storage, 1e6 * t_sparse_sparse_storage)
    @printf("         speedup from sparse storage: %5.1f   (ratio of nonzeros: %5.1f)\n",
            t_dense / t_sparse_sparse_storage, nnz_dense / nnz_sparse)
end

println("\nRuntime of the two-dimensional example with global operators on N = 50 nodes")
nodes = collect(LinRange(-1.0, 1.0, 50))
basis = [one, identity, x -> x^2, x -> x^3]
D_sparse = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                   bandwidth = 3, OPT...)
for (name, D) in (("dense storage ", D_sparse), ("sparse storage", to_sparse(D_sparse)))
    run_example(D)
    l2, _ = (@invokelatest analysis_callback(@invokelatest Main.sol))
    t = @belapsed run_example($D) samples=3 evals=1 seconds=120
    @printf("  %s: %5.2f s, L2 error %.4e\n", name, t, l2[1])
end
