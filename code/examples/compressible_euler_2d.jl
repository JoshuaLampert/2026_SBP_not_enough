using SummationByPartsOperatorsExtra: MattssonNordström2004, derivative_operator
using Trixi
using OrdinaryDiffEqLowStorageRK

N = 15
ref_xmin = -1.0
ref_xmax = 1.0

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

D = derivative_operator(MattssonNordström2004(), 1, 4, ref_xmin, ref_xmax, N)
solver = FDSBP(D,
    surface_integral=SurfaceIntegralStrongForm(flux_hllc),
    volume_integral=VolumeIntegralStrongForm())

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=2, n_cells_max=100_000, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
    source_terms=source_terms, boundary_conditions=boundary_condition_periodic)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
    save_analysis=false, analysis_filename="compressible_euler_2d_analysis.dat", # can be overwritten with trixi_include
    extra_analysis_errors=[:conservation_error])
alive_callback = AliveCallback(analysis_interval=analysis_interval)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

saveat = range(tspan..., length=100)
alg = RDPK3SpFSAL49()
sol = solve(ode, alg; abstol=1.0e-6, reltol=1.0e-6,
    ode_default_options()..., callback=callbacks)
