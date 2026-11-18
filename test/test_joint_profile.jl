using BreakpointProfiles
using Test

function joint_test_problem()
    function piecewise_linear(params, cps)
        u0, r1, r2 = params
        cp = cps[1]
        values = [t < cp ? u0 + r1 * t : u0 + r1 * cp + r2 * (t - cp) for t in 0:50]
        return reshape(values, 1, :)
    end
    truth = [5.0, 0.2, 0.5]
    data = piecewise_linear(truth, [20])
    objective = (params, cps) -> sum((piecewise_linear(params, cps) .- data).^2)
    return ODEChangepointPLEProblem(
        objective=objective,
        data=data,
        loss_fn=(sim, data)->sum((sim .- data).^2),
        changepoints=[20],
        best_params=truth,
        best_loss=0.0,
        lb=[0.0, -2.0, -2.0],
        ub=[10.0, 2.0, 2.0],
        param_names=["u0", "r1", "r2"],
        n_global=1,
        n_segment_specific=1,
    )
end

@testset "Joint profile likelihood" begin
    prob = joint_test_problem()
    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-6, maxeval=300)
    reference = joint_reference_fit(prob, 2, optimizer; parallelize_grid=false, n_starts=1)
    @test reference[1] ≈ 0.0 atol=1e-6
    @test reference[3] == [20]

    profile = profile_parameter_joint(prob, 1; n_points=5, window=2,
        optimizer=optimizer, parallelize_grid=false, joint_reference=reference)
    @test profile isa ProfileResult
    @test profile.best_found_loss ≈ 0.0 atol=1e-6
    @test length(profile.values) >= 2
    @test all(isfinite, profile.losses)

    profiles = profile_all_parameters_joint(prob; indices=[1], n_points=5,
        window=2, optimizer=optimizer, parallel_over_parameters=false,
        parallelize_grid=false)
    @test length(profiles) == 1
    @test profiles[1].parameter == "u0"
    @test profiles[1].best_found_loss ≈ 0.0 atol=1e-6
end
