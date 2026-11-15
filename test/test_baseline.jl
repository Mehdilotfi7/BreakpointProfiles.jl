using BreakpointProfiles
using Test

@testset "BaselinePLEProblem" begin
    objective = p -> (p[1] - 3.0)^2 + (p[2] - 5.0)^2
    prob = BaselinePLEProblem(
        objective = objective,
        best_params = [3.0, 5.0],
        best_loss = 0.0,
        lb = [-10.0, -10.0],
        ub = [10.0, 10.0],
        param_names = ["x", "y"]
    )

    @test prob isa ODEChangepointPLEProblem
    @test isempty(prob.changepoints)
    @test prob.n_global == 2
    @test prob.n_segment_specific == 0
    @test prob.param_names == ["x", "y"]
    @test evaluate_loss(prob, [3.0, 5.0]) == 0.0

    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-6, maxeval=1000)
    options = ProfileOptions(samplesize=20, polish=false, smooth_jumps=false)
    prof = profile_parameter(prob, 1; options=options, optimizer=optimizer)

    @test prof isa ProfileResult
    @test prof.parameter == "x"
    @test prof.best_value ≈ 3.0
    @test prof.ci_lower ≈ 3.0 - sqrt(3.8414588206941285) atol=0.15
    @test prof.ci_upper ≈ 3.0 + sqrt(3.8414588206941285) atol=0.15
    @test prof.identifiable
end
