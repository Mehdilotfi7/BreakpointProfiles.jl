using BreakpointProfiles
using DataFrames

optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-5, maxeval=2000)
options = ProfileOptions(samplesize=20, polish=false, smooth_jumps=false)

baseline_objective = p -> sum((p[1] .+ p[2] .* (0:50) .- (5.0 .+ 0.2 .* (0:50))).^2)
baseline = BaselinePLEProblem(
    objective = baseline_objective,
    best_params = [5.0, 0.2],
    best_loss = 0.0,
    lb = [0.0, 0.0],
    ub = [10.0, 1.0],
    param_names = ["u0", "r"]
)
baseline_profiles = profile_all_parameters(baseline; options=options, optimizer=optimizer)
println("Baseline model")
show(ple_summary(baseline_profiles), allcols=true, allrows=true)

function piecewise_linear(params, cps)
    u0, r1, r2 = params
    cp = cps[1]
    values = [t < cp ? u0 + r1 * t : u0 + r1 * cp + r2 * (t - cp) for t in 0:50]
    return reshape(values, 1, :)
end

true_params = [5.0, 0.2, 0.5]
data = piecewise_linear(true_params, [20])
changepoint_objective = (params, cps) -> sum((piecewise_linear(params, cps) .- data).^2)
changepoint_problem = ODEChangepointPLEProblem(
    objective = changepoint_objective,
    data = data,
    loss_fn = (sim, data) -> sum((sim .- data).^2),
    changepoints = [20],
    best_params = true_params,
    best_loss = 0.0,
    lb = [0.0, 0.0, 0.0],
    ub = [10.0, 1.0, 1.0],
    param_names = ["u0", "r1", "r2"],
    n_global = 1,
    n_segment_specific = 1
)

conditional = profile_all_parameters(changepoint_problem; options=options, optimizer=optimizer)
println("\nConditional profiles")
show(ple_summary(conditional), allcols=true, allrows=true)

cp_profile = profile_changepoint(changepoint_problem, 1; window=7, optimizer=optimizer)
println("\nChangepoint profile")
println(cp_profile)

joint = profile_all_parameters_joint(
    changepoint_problem;
    indices=[1],
    window=3,
    n_points=20,
    optimizer=optimizer,
    parallel_over_parameters=false
)
println("\nJoint profile")
show(ple_summary(joint), allcols=true, allrows=true)
