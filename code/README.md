# Numerical Experiments

This directory contains all code required to reproduce the numerical
experiments. First, you need to install Julia, e.g., by downloading
the binaries from the [download page](https://julialang.org/downloads/).
The numerical experiments were performed using Julia v1.13.1.

Some of the scripts benefit considerably from multithreading. We therefore
recommend starting Julia with several threads, e.g.

```bash
julia -t auto --project=.
```

The code builds on the Julia package
[SummationByPartsOperatorsExtra.jl](https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl).

The following list describes which script creates which figure(s) or tables
and the names of the resulting .pdf files:

* Figure 1, Table 1: `advection_compare_degrees.jl` &rarr; `advection_solutions_subplots.pdf`
* Table 2: `solution_set_dimension.jl`
* Table 3: `advection_compare_sparse.jl`
* Figure 2: `multiblock_sparse.jl` &rarr; `error_comparison_1d.pdf`
* Figure 3: `sparse_2d.jl` &rarr; `error_comparison_2d.pdf`
* Table 4: `convergence_test.jl`
* Table 5 and the variation of the number of regularization functions in Section 4.2: `operator_diagnostics.jl`
* Remark 5: `computational_cost.jl`
* Figure 4: `multiblock_regularization.jl` &rarr; `error_comparison_1d_regularization.pdf`
* Figure 5: `multiblock_regularization_2d.jl` &rarr; `error_comparison_2d_regularization.pdf`
* Figure 6, Table 6: `dispersion_analysis.jl` &rarr; `dispersion_relation.pdf`
* Contribution of the stationary modes to the error in Section 4.3: `stationary_modes.jl`
* Table 7: `sensitivity_analysis.jl`

The resulting figures are then saved as .pdf files in a new directory `figures`
inside the folder of this `README.md`. The tables are printed to the screen as $\LaTeX$ code.

Note that some of these scripts are computationally expensive. In particular,
`convergence_test.jl` performs a sequence of two-dimensional simulations on
successively refined meshes and takes on the order of an hour on a current
workstation when Julia is started with a single thread.

In order to execute a script, start Julia in this folder and execute

```julia
julia> include("file_name.jl")
```

in the Julia REPL. To execute the first script from the list above, e.g.,
execute

```julia
julia> include("advection_compare_degrees.jl")
```
