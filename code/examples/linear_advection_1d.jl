using SummationByPartsOperatorsExtra: MattssonNordström2004, derivative_operator
using Trixi
using OrdinaryDiffEqLowStorageRK

N = 15
ref_xmin = -1.0
ref_xmax = 1.0

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_sinpi(x, t, equations)
    x_t = x[1] - equations.advection_velocity[1] * t
    u = sinpi(x_t)
    return SVector(u)
end
initial_condition = initial_condition_sinpi

D = derivative_operator(MattssonNordström2004(), 1, 4, ref_xmin, ref_xmax, N)
solver = FDSBP(D,
    surface_integral=SurfaceIntegralStrongForm(flux_lax_friedrichs),
    volume_integral=VolumeIntegralStrongForm())

coordinates_min = -2.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=4, n_cells_max=10_000, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver; boundary_conditions=boundary_condition_periodic)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
    save_analysis=false, analysis_filename="linear_advection_1d_analysis.dat", # can be overwritten with trixi_include
    extra_analysis_errors=[:conservation_error])
callbacks = CallbackSet(summary_callback, analysis_callback)

saveat = range(tspan..., length=100)
alg = RDPK3SpFSAL49()
sol = solve(ode, alg; adaptive=true, abstol=1.0e-6, reltol=1.0e-6, dt=1.0, # dt only dummy, which can be overwritten with trixi_include when adaptive=false
    ode_default_options()..., callback=callbacks, saveat=saveat)
