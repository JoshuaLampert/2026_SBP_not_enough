# Dimension of the set of F-exact FSBP operators on a fixed set of nodes.
#
# The exactness condition S*V - P*V_x + B*V/2 = 0 is linear in (S, P), so this
# set is an affine subspace intersected with the cone of positive diagonal P.
# Its dimension is the number of free parameters minus the rank of the linear
# map L(S, P) = S*V - P*V_x, which is computed here for the unstructured dense
# and the sparse ansatz.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
import Optim, ForwardDiff
using SummationByPartsOperatorsExtra: function_space_operator, GlaubitzNordströmÖffner2023,
                                      mass_matrix
using Printf

const OPT = (; verbose = false, opt_alg = Optim.BFGS(),
             options = Optim.Options(g_tol = 1e-16, iterations = 5000, show_trace = false))

"Matrix of `L(S, P) = S * V - P * V_x` for skew-symmetric `S` supported on `pattern`."
function constraint_matrix(pattern, V, V_x)
    N, K = size(V)
    indices = [(i, j) for i in 1:N, j in 1:N if i < j && pattern[i, j]]
    A = zeros(N * K, length(indices) + N)
    for (column, (i, j)) in enumerate(indices)
        E = zeros(N, N)
        E[i, j] = 1
        E[j, i] = -1
        A[:, column] = vec(E * V)
    end
    for k in 1:N
        E = zeros(N, N)
        E[k, k] = -1
        A[:, length(indices) + k] = vec(E * V_x)
    end
    return A
end

"Sparsity pattern of the skew-symmetric part of `D`."
function sparsity_pattern(D)
    P = mass_matrix(D)
    Q = P * Matrix(D)
    return abs.((Q - Q') / 2) .> 1e-14
end

@printf("%-12s %-7s %10s %10s %6s %12s\n",
        "nodes", "ansatz", "parameters", "equations", "rank", "dimension")
for (label, N, basis, basis_derivatives) in
    (("N = 15, P_3", 15, [one, identity, x -> x^2, x -> x^3],
      [x -> zero(x), x -> one(x), x -> 2x, x -> 3x^2]),
     ("N = 50, P_2", 50, [one, identity, x -> x^2],
      [x -> zero(x), x -> one(x), x -> 2x]))
    nodes = collect(LinRange(-1.0, 1.0, N))
    V = hcat((f.(nodes) for f in basis)...)
    V_x = hcat((f.(nodes) for f in basis_derivatives)...)
    D_sparse = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                       bandwidth = 3, OPT...)
    for (name, pattern) in (("dense", trues(N, N)), ("sparse", sparsity_pattern(D_sparse)))
        A = constraint_matrix(pattern, V, V_x)
        r = rank(A)
        @printf("%-12s %-7s %10d %10d %6d %12d\n",
                label, name, size(A, 2), size(A, 1), r, size(A, 2) - r)
    end
end

# Nullity of the resulting operators as a function of N. For the banded ansatz
# the nullspace dimension is bounded independently of N, while it grows linearly
# with N for the unstructured dense ansatz.
println()
@printf("%5s %-7s %8s %12s %12s\n", "N", "ansatz", "nullity", "residual", "||D * 1||")
basis = [one, identity, x -> x^2]
basis_derivatives = [x -> zero(x), x -> one(x), x -> 2x]
for N in (20, 30, 50)
    nodes = collect(LinRange(-1.0, 1.0, N))
    V = hcat((f.(nodes) for f in basis)...)
    V_x = hcat((f.(nodes) for f in basis_derivatives)...)
    B = zeros(N, N)
    B[1, 1] = -1
    B[N, N] = 1
    for (name, kwargs) in (("dense", (;)), ("sparse", (; bandwidth = 3)))
        D_op = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023(); kwargs..., OPT...)
        D = Matrix(D_op)
        P = Matrix(mass_matrix(D_op))
        Q = P * D
        residual = norm((Q - Q') / 2 * V - P * V_x + B * V / 2)
        @printf("%5d %-7s %8d %12.2e %12.2e\n", N, name, N - rank(D), residual, norm(D * ones(N)))
    end
end
