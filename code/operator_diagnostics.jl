# Diagnostics for the FSBP operators used in the paper: residual of the
# optimization problem, numerical rank (and its sensitivity to the tolerance),
# singular values, and condition numbers of P and of the shifted matrix D_tilde
# used to characterize the eigenvalue property.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
import Optim, ForwardDiff
import Manifolds
using Manopt
using ADTypes: ADTypes
using Mooncake: Mooncake
using SummationByPartsOperatorsExtra: function_space_operator, mass_matrix, grid,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzIskeLampertÖffner2026Regularized,
                                      get_optimization_entries,
                                      derivative_operator, MattssonNordström2004
using Printf

const NU = 1.0

"Numerical rank of `A` for the relative tolerance `rtol` (rank via SVD)."
numrank(sv, rtol) = count(>(rtol * first(sv)), sv)

function diagnose(name, D, basis, basis_derivatives)
    x = grid(D)
    N = length(x)
    M = Matrix(D)
    P = Matrix(mass_matrix(D))
    p = diag(P)
    Q = P * M
    S = (Q - Q') / 2
    B = zeros(N, N); B[1, 1] = -1.0; B[N, N] = 1.0
    V = hcat((f.(x) for f in basis)...)
    V_x = hcat((f.(x) for f in basis_derivatives)...)

    res = norm(S * V - P * V_x + B * V / 2)      # residual ||XW + BV/2||, the square root of the objective
    exa = norm(M * V - V_x)                      # unweighted exactness error
    sv = svdvals(M)
    r_default = rank(M)
    rtol_default = N * eps(Float64)

    D_tilde = copy(M); D_tilde[1, 1] += NU / p[1]
    ev = eigvals(D_tilde)
    n_nonpos = count(<=(1e-14), real.(ev))

    ranks = [numrank(sv, t) for t in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)]

    @printf("%-26s N=%3d  res=%8.2e  exact.err=%8.2e  rank=%3d (N-1=%3d)\n",
            name, N, res, exa, r_default, N - 1)
    @printf("%-26s   sigma_1=%9.3e  sigma_{N-1}=%9.3e  sigma_N=%9.3e  sigma_{N-1}/sigma_1=%8.2e\n",
            "", sv[1], sv[end-1], sv[end], sv[end-1] / sv[1])
    @printf("%-26s   cond(P)=%9.3e  cond(D_tilde)=%9.3e  #Re(lambda)<=0: %3d  min|Re(lambda)|=%8.2e\n",
            "", cond(P), cond(D_tilde), n_nonpos, minimum(abs.(real.(ev))))
    @printf("%-26s   rank at rtol 1e-6,-8,-10,-12,-14,-16: %s   (default rtol=%.2e)\n",
            "", ranks, rtol_default)
    return (; name, N, res, r_default, sv, condP = cond(P), condDt = cond(D_tilde), ranks)
end

polybasis(d) = ([x -> x^i for i in 0:d], [i == 0 ? (x -> zero(x)) : (x -> i * x^(i - 1)) for i in 0:d])
const TRIG = ([one, identity, sinpi, cospi],
              [x -> zero(x), x -> one(x), x -> pi * cospi(x), x -> -pi * sinpi(x)])

# `g_tol = 0.0` on purpose: the norm matrix is parametrized through a logistic function, which damps the
# gradient, so an absolute gradient tolerance can be met before the optimization has converged. With
# `g_tol = 0.0`, BFGS stops once consecutive objective values coincide (or at the iteration limit).
# The remaining scripts mostly use `g_tol = 1e-16`, which yields slightly larger residuals but otherwise the
# same operators for the purposes of the paper; the global operators for T in `advection_compare_sparse.jl`
# use the settings below.
opt = (; verbose = false,
        options = Optim.Options(g_tol = 0.0, iterations = 50000, show_trace = false),
        opt_alg = Optim.BFGS(),
        autodiff = ADTypes.AutoMooncake(; config = nothing))

xmin, xmax = -1.0, 1.0

println("="^110)
println("Section 3: unstructured dense FSBP operators, N = 50 equidistant nodes")
println("="^110)
N = 50
nodes = collect(LinRange(xmin, xmax, N))
for d in (1, 3, 5, 7, 9, 11)
    b, bx = polybasis(d)
    D = function_space_operator(b, nodes, GlaubitzNordströmÖffner2023(); bandwidth = N - 1, opt...)
    diagnose("dense P_$d", D, b, bx)
end

println()
println("="^110)
println("Classical FD-SBP operators for reference, N = 50")
println("="^110)
for order in (2, 4, 6)
    D = derivative_operator(MattssonNordström2004(), 1, order, xmin, xmax, N)
    b, bx = polybasis(order ÷ 2)
    diagnose("FD order $order", D, b, bx)
end

println()
println("="^110)
println("Section 4.1: sparse FSBP operators, N = 50 equidistant nodes")
println("="^110)
for bw in (3, 4, 5, 6)
    b, bx = polybasis(2)
    D = function_space_operator(b, nodes, GlaubitzNordströmÖffner2023(); bandwidth = bw, opt...)
    diagnose("sparse P_2, b=$bw", D, b, bx)
end
# same settings as in `advection_compare_sparse.jl`; more iterations are needed here than for P_2
opt_trig = (; opt..., options = Optim.Options(g_tol = 0.0, iterations = 100_000, show_trace = false))
for bw in (3, 4, 5, 6)
    b, bx = TRIG
    D = function_space_operator(b, nodes, GlaubitzNordströmÖffner2023(); bandwidth = bw, opt_trig...)
    diagnose("sparse T, b=$bw", D, b, bx)
end

println()
println("="^110)
println("Sections 4.1 and 4.2: operators on one block, N = 15 equidistant nodes")
println("="^110)
nodes_block = collect(LinRange(xmin, xmax, 15))
P3 = ([one, identity, x -> x^2, x -> x^3],
      [x -> zero(x), x -> one(x), x -> 2x, x -> 3x^2])
TT = ([one, identity, sinpi, cospi],
      [x -> zero(x), x -> one(x), x -> pi * cospi(x), x -> -pi * sinpi(x)])
opt_block = (; verbose = false, opt_alg = Optim.BFGS(),
             options = Optim.Options(g_tol = 0.0, iterations = 5000, show_trace = false))
D_dense_P3 = function_space_operator(P3[1], nodes_block, GlaubitzNordströmÖffner2023(); opt_block...)
diagnose("dense P_3", D_dense_P3, P3...)
diagnose("dense T", function_space_operator(TT[1], nodes_block, GlaubitzNordströmÖffner2023(); opt_block...), TT...)
diagnose("sparse P_3, b=3", function_space_operator(P3[1], nodes_block, GlaubitzNordströmÖffner2023(); bandwidth = 3, opt_block...), P3...)
diagnose("sparse T, b=3", function_space_operator(TT[1], nodes_block, GlaubitzNordströmÖffner2023(); bandwidth = 3, opt_block...), TT...)

# Regularized operators of Section 4.2: dense, with the augmented basis G = {g_1, ..., g_M}. M = 2 is
# G = {sin(pi x), cos(pi x)} as in Section 4.2; M = 4 and M = 6 add the next trigonometric modes.
# Note that passing `options` replaces the package defaults, so Manopt's default stopping criterion
# for the augmented Lagrangian method is used.
x0 = get_optimization_entries(D_dense_P3, bandwidth = 14)
regularization_pool = [sinpi, cospi, x -> sinpi(2x), x -> cospi(2x), x -> sinpi(3x), x -> cospi(3x)]
for M in (2, 4, 6)
    D_regularized = function_space_operator(P3[1], nodes_block, GlaubitzIskeLampertÖffner2026Regularized();
                                            verbose = false, x0 = x0,
                                            regularization_functions = regularization_pool[1:M],
                                            options = (; debug = []))
    diagnose("regularized P_3, M = $M", D_regularized, P3...)
end
