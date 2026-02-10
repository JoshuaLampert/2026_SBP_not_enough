using SummationByPartsOperatorsExtra
import Optim, ForwardDiff
using KernelInterpolation

function add_noise!(nodes, x_min, x_max, eps = 0.05)
    for i in 2:length(nodes)-1
        nodes[i] += eps * ((x_max - x_min) * rand() + x_min)
    end
end

# Non-equidistant nodes with polynomials
x_min = -1.0
x_max = 1.0
N = 24
nodes = collect(LinRange(x_min, x_max, N))
add_noise!(nodes, x_min, x_max)
basis = [x -> x^i for i in 0:2]

verbose = true
different_values = false # does not converge, but `true` does converge
opt_kwargs = (; options=Optim.Options(g_tol=1e-16, iterations=50000, show_trace=true), opt_alg=Optim.LBFGS())
D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                            different_values, bandwidth = 3, verbose, opt_kwargs...)

# Equidistant nodes with RBFs
nodes = collect(LinRange(x_min, x_max, N))
kernel = ThinPlateSplineKernel{1}()
centers = NodeSet([-0.75, 0.0, 0.75])
basis = LagrangeBasis(centers, kernel)
different_values = false # does not converge, but `true` does converge
D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                            different_values, bandwidth = 3, verbose, opt_kwargs...)
