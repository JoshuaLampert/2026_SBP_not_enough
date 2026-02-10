import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Trixi
using SummationByPartsOperatorsExtra: GlaubitzNordströmÖffner2023, GlaubitzIskeLampertÖffner2026Regularized, function_space_operator, get_optimization_entries
import Optim, ForwardDiff
import Manifolds
using Manopt
using PrettyTables
using Printf
using LaTeXStrings

EXAMPLES_DIR = joinpath(@__DIR__, "examples")

function create_eoc_table(filename, Ds)
    tables_str = []

    initial_refinement_level = 1
    refinements = 4
    for D in Ds
        eocs, errorsmatrix = convergence_test(joinpath(EXAMPLES_DIR, filename), refinements, D=D, abstol=1e-14, reltol=1e-14,
            initial_refinement_level=initial_refinement_level, tspan=(0.0, 1.0))
        table_str = []
        for i in 1:refinements
            K = 2^(i + initial_refinement_level - 1)
            l2 = errorsmatrix[:l2][i, :]
            eoc_first_variable = i == 1 ? "---" : @sprintf("%.2f", eocs[:l2][i-1, 1])
            push!(table_str, [string(K), [@sprintf("%.2e", l2[v]) for v in eachvariable(equations)]..., eoc_first_variable])
        end
        push!(tables_str, table_str)
    end

    if equations isa CompressibleEulerEquations2D
        caption_equation_name = "compressible Euler equations"
        header_variables = [L"$\rho$", L"$\rho v_1$", L"$\rho v_2$", L"$\rho e$"]
        label = "table:eocs_compressible_euler"
    else
        caption_equation_name = "linear advection equation"
        header_variables = ["error"]
        label = "table:eocs_advection"
    end
    column_labels = ["K", header_variables..., "EOC"]
    style = LatexTableStyle(first_line_column_label=String[])
    table_kwargs = (; column_labels, backend=:latex, table_format=latex_table_format__booktabs, style, alignment=:c)

    println("\\begin{table}[htb]")
    println("\\caption{\$L^2\$-errors and EOCs for the $caption_equation_name and function spaces \$\\mathcal{F}=\\mathcal{P}_3\$ and \$\\mathcal{F}=\\mathcal{T}\$} using sparse FSBP operators with bandwidth \$b = 3\$ on \$K\$ blocks with \$N = 15\$ nodes each.")
    println("\\begin{subtable}{.5\\linewidth}")
    println("\\subcaption{\$\\mathcal{F}=\\mathcal{P}_3\$}")
    println("\\centering")
    pretty_table(permutedims(hcat(tables_str[1]...)); table_kwargs...)
    println("\\end{subtable}%")
    println("\\begin{subtable}{.5\\linewidth}")
    println("\\subcaption{\$\\mathcal{F}=\\mathcal{T}\$}")
    println("\\centering")
    pretty_table(permutedims(hcat(tables_str[2]...)); table_kwargs...)
    println("\\end{subtable}%")
    println("\\label{$label}")
    println("\\end{table}")
end

p = 14
ref_xmin = -1.0
ref_xmax = 1.0
nodes = collect(LinRange(ref_xmin, ref_xmax, p + 1))
basis_p3 = [one, identity, x -> x^2, x -> x^3]

# sparsity
D_FSBP_sparse_p3 = function_space_operator(basis_p3, nodes, GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=false,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))

basis_sin_cos = [one, identity, sinpi, cospi]
D_FSBP_sparse_sin_cos = function_space_operator(basis_sin_cos, nodes, GlaubitzNordströmÖffner2023();
    bandwidth=3, verbose=false,
    opt_alg=Optim.BFGS(), options=Optim.Options(g_tol=1e-16, iterations=5000, show_trace=false))
Ds = [D_FSBP_sparse_p3, D_FSBP_sparse_sin_cos]

# create_eoc_table("linear_advection_1d.jl", Ds)
create_eoc_table("compressible_euler_2d.jl", Ds)
