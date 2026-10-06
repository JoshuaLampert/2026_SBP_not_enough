using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Contribution of the spurious stationary modes (Lemma 4) to the error of the linear advection examples, see Section 4.3.
# The semidiscretization du/dt = A u is solved exactly by u(t) = exp(t A) u_0, so that no time integration error is involved.
# Let Pi be the spectral projector onto the spurious stationary subspace, i.e., onto the kernel vectors of A that vanish at
# all block boundary nodes (the elements of U in each block). Since P L + L^T P = (e_L - e_R)(e_L - e_R)^T is positive
# semidefinite for the Bloch matrix L, the eigenvalue zero of A is semisimple and the projector exists. Since Pi exp(t A) = Pi,
# the stationary part of the error is Pi (u_0 - u(t)): it is neither transported nor damped, and it does not grow in time.
# Note that it vanishes whenever t is a multiple of the period, where u(t) = u_0, so we report its maximum over a period
# and compare it with the maximum of the error over the same period.
using LinearAlgebra
using Printf
import Optim, ForwardDiff
import Manifolds
using Manopt
using ADTypes: ADTypes
using Mooncake: Mooncake
using SummationByPartsOperatorsExtra: function_space_operator, mass_matrix, grid,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzIskeLampertÖffner2026Regularized,
                                      get_optimization_entries

"""
    system_matrix(D, nb, a)

System matrix `A` of the semidiscretization of the linear advection equation with velocity `a > 0` on `[-1, 1]` with `nb`
identical periodic blocks coupled by the upwind flux, i.e., the semidiscretization underlying the Bloch matrix in
`dispersion_analysis.jl`. For `nb = 1`, this is a single periodic block as in `advection_compare_degrees.jl`.
"""
function system_matrix(D, nb, a)
    n = size(D, 1)
    w = diag(Matrix(mass_matrix(D)))
    h = 2 / nb
    M = zeros(n * nb, n * nb)
    for j in 1:nb
        r = ((j - 1) * n + 1):(j * n)
        l = ((mod1(j - 1, nb) - 1) * n + 1):(mod1(j - 1, nb) * n)
        M[r, r] .= Matrix(D)
        M[first(r), first(r)] += 1 / w[1]
        M[first(r), last(l)] -= 1 / w[1]
    end
    # the reference block of D has length sum(w)
    return -(a * sum(w) / h) .* M
end

"""
    stationary_error(D, nb, a, ts, u_exact)

Return the maxima over the equidistant times `ts` of the `L^2`-error of the exact solution of the semidiscretization with `nb`
blocks and of the `L^2`-norm of its stationary part, as well as the number of spurious stationary modes.
"""
function stationary_error(D, nb, a, ts, u_exact)
    n = size(D, 1)
    w = diag(Matrix(mass_matrix(D)))
    h = 2 / nb
    xi = grid(D)
    x = vcat([-1 + (j - 1) * h .+ (xi .+ 1) .* h / 2 for j in 1:nb]...)
    W = vcat([w .* h / sum(w) for _ in 1:nb]...)
    A = system_matrix(D, nb, a)
    K_right = nullspace(A)
    K_left = nullspace(Matrix(A'))
    # spurious stationary subspace: kernel vectors of A vanishing at all block boundary nodes
    boundary = sort(vcat([(j - 1) * n + 1 for j in 1:nb], [j * n for j in 1:nb]))
    K_spurious = K_right * nullspace(K_right[boundary, :]; rtol = 1e-8)
    m_spurious = size(K_spurious, 2)
    # projector onto span(K_spurious) along span{1} + range(A), using the basis [1, K_spurious] of ker(A)
    B = hcat(ones(length(x)), K_spurious)
    projector(v) = m_spurious == 0 ? zero(v) : K_spurious * ((K_left' * B) \ (K_left' * v))[2:end]
    l2(v) = sqrt(sum(W .* v .^ 2) / 2) # normalized by the length of the domain, as in Trixi.jl
    u0 = u_exact.(x, 0.0)
    # exact time evolution, advanced from one sample time to the next by the propagator exp(dt A)
    u = exp(first(ts) * A) * u0
    E = length(ts) > 1 ? exp(step(ts) * A) : I
    max_error = max_stationary = 0.0
    for (k, t) in enumerate(ts)
        k > 1 && (u = E * u)
        max_error = max(max_error, l2(u - u_exact.(x, t)))
        max_stationary = max(max_stationary, l2(projector(u0 - u_exact.(x, t))))
    end
    return max_error, max_stationary, m_spurious
end

a = 2.0 # the period of the exact solution on [-1, 1] is 2 / a = 1

# Multiblock operators with N = 15 nodes per block from Figures 2 and 4, constructed as in `multiblock_sparse.jl` and
# `multiblock_regularization.jl`
N = 15
nodes = collect(LinRange(-1.0, 1.0, N))
opt_kwargs = (; verbose = false, opt_alg = Optim.BFGS(),
              options = Optim.Options(g_tol = 1e-16, iterations = 5000, show_trace = false))
basis_poly = [one, identity, x -> x^2, x -> x^3]
basis_sin_cos = [one, identity, sinpi, cospi]
D_dense_p3 = function_space_operator(basis_poly, nodes, GlaubitzNordströmÖffner2023(); opt_kwargs...)
D_sparse_p3 = function_space_operator(basis_poly, nodes, GlaubitzNordströmÖffner2023(); bandwidth = 3, opt_kwargs...)
D_dense_sin_cos = function_space_operator(basis_sin_cos, nodes, GlaubitzNordströmÖffner2023(); opt_kwargs...)
D_sparse_sin_cos = function_space_operator(basis_sin_cos, nodes, GlaubitzNordströmÖffner2023(); bandwidth = 3,
                                           opt_kwargs...)
# passing `options` replaces the package defaults, so Manopt's default stopping criterion is used,
# as in `multiblock_regularization.jl`
D_regularized = function_space_operator(basis_poly, nodes, GlaubitzIskeLampertÖffner2026Regularized();
                                        verbose = false, x0 = get_optimization_entries(D_dense_p3, bandwidth = N - 1),
                                        regularization_functions = [sinpi, cospi],
                                        options = (; debug = []))
operators = (("sparse P_3", D_sparse_p3), ("dense P_3", D_dense_p3), ("sparse T", D_sparse_sin_cos),
             ("dense T", D_dense_sin_cos), ("regularized P_3", D_regularized))
gauss(x, t) = exp(-(mod(x - a * t + 1.0, 2.0) - 1.0)^2 / 0.1)

# Setting of Figures 2 and 4: 8 periodic blocks, last period of the simulation
println("N = 15 nodes per block, 8 periodic blocks, t in [49, 50]")
ts = range(49.0, 50.0; length = 201)
for (name, D) in operators
    e, es, m = stationary_error(D, 8, a, ts, gauss)
    @printf("%-16s stationary modes %3d  max error %.3e  max stationary part %.3e (%5.1f %%)\n",
            name, m, e, es, 100 * es / e)
end

# Refinement in the number of blocks; maxima over the first period t in [0, 1]
println("\nRefinement, maxima over t in [0, 1]")
ts = range(0.0, 1.0; length = 201)
for (name, D) in operators
    println(name)
    previous = nothing
    for nb in (4, 8, 16, 32)
        e, es, _ = stationary_error(D, nb, a, ts, gauss)
        eoc = previous === nothing ? "" :
              @sprintf("EOC %5.2f (error), %5.2f (stationary part)", log2(previous[1] / e), log2(previous[2] / es))
        @printf("  nb = %2d  error %.3e  stationary part %.3e (%5.1f %%)  %s\n", nb, e, es, 100 * es / e, eoc)
        previous = (e, es)
    end
end

# Global operators with N = 50 nodes from Section 3, constructed as in `advection_compare_degrees.jl`;
# one periodic block, u_0(x) = sin(pi x), t = 1.75
println("\nN = 50 nodes, one periodic block, t = 1.75")
N = 50
nodes = collect(LinRange(-1.0, 1.0, N))
for d in (1, 3, 5, 7, 9, 11)
    D = function_space_operator([x -> x^i for i in 0:d], nodes, GlaubitzNordströmÖffner2023();
                                bandwidth = N - 1, verbose = false,
                                options = Optim.Options(g_tol = 1e-16, iterations = 50000, show_trace = false),
                                opt_alg = Optim.BFGS(), autodiff = ADTypes.AutoMooncake(; config = nothing))
    e, es, m = stationary_error(D, 1, a, range(1.75, 1.75; length = 1), (x, t) -> sinpi(x - a * t))
    @printf("dense P_%-2d  rank %2d  stationary modes %2d  error %.3e  stationary part %.3e (%5.1f %%)\n",
            d, rank(Matrix(D)), m, e, es, 100 * es / e)
end
