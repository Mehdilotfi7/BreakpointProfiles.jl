using BreakpointProfiles
using Test

# Quadratic baseline objective with known profiles (−2 log L of a Gaussian):
#   a: σ = 0.5   -> 95% CI 5 ± 1.96·0.5 = [4.02, 5.98]
#   b: σ = 0.01  -> CI 0.2 ± 0.0196, well inside its bounds
#   c: does not enter the loss -> flat profile (structurally non-identifiable)
@testset "profile extension" begin
    obj = p -> ((p[1] - 5.0) / 0.5)^2 + ((p[2] - 0.2) / 0.01)^2 + 0.0 * p[3]
    lb, ub = [4.5, 0.0, 0.0], [5.5, 1.0, 1.0]
    prob = BaselinePLEProblem(objective=obj, best_params=[5.0, 0.2, 0.5], best_loss=0.0,
                              lb=lb, ub=ub, param_names=["a", "b", "c"])
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-8, maxeval=2000)
    options = ProfileOptions(samplesize=60, polish=false)
    profiles = profile_all_parameters(prob; options=options, optimizer=opt)
    pa, pb, pc = profiles

    @testset "before extension" begin
        @test !pa.identifiable                       # bounds [4.5, 5.5] are too tight
        @test profile_open_sides(pa) == (true, true)
        @test pb.identifiable
        @test profile_open_sides(pb) == (false, false)
        @test !pc.identifiable
        @test (pa.profile_lb, pa.profile_ub) == (4.5, 5.5)
        @test length(pa.params_trace) == length(pa.values)
    end

    @testset "extended_bounds" begin
        @test all(extended_bounds(4.5, 5.5; left=1.0, right=0.5) .≈ (3.5, 6.0))
        @test all(extended_bounds(2.0, 8.0; left=1.0, right=0.5, mode=:relative) .≈ (1.0, 12.0))
        @test all(extended_bounds(2.0, 8.0; right=0.25) .≈ (2.0, 9.5))
        @test_throws ErrorException extended_bounds(0.0, 1.0; left=0.5, mode=:relative)
        @test_throws ErrorException extended_bounds(1.0, 2.0; left=-0.1)
        @test_throws ErrorException extended_bounds(1.0, 2.0; left=0.1, mode=:log)
    end

    @testset "both sides, existing points kept" begin
        ea = extend_profile(prob, pa; side=:both, left=1.0, right=1.0, options=options, optimizer=opt)
        @test (ea.profile_lb, ea.profile_ub) == (3.5, 6.5)
        @test all(v -> v in ea.values, pa.values)    # nothing recomputed or dropped
        @test length(ea.values) > length(pa.values)
        @test ea.identifiable
        @test isapprox(ea.ci_lower, 5.0 - 1.96 * 0.5; atol=0.02)
        @test isapprox(ea.ci_upper, 5.0 + 1.96 * 0.5; atol=0.02)
        @test ea.best_loss == pa.best_loss && ea.threshold == pa.threshold
        # the other parameters kept their bounds during re-optimisation
        @test all(θ -> lb[2] <= θ[2] <= ub[2] && lb[3] <= θ[3] <= ub[3], ea.params_trace)
    end

    @testset "one side only" begin
        el = extend_profile(prob, pa; side=:left, left=1.0, options=options, optimizer=opt)
        @test (el.profile_lb, el.profile_ub) == (3.5, 5.5)
        @test profile_open_sides(el) == (false, true)
        @test !el.identifiable
        @test isapprox(el.ci_lower, 5.0 - 1.96 * 0.5; atol=0.02)
        er = extend_profile(prob, pa; side=:right, right=0.2, options=options, optimizer=opt)
        @test (er.profile_lb, er.profile_ub) == (4.5, 5.7)
        @test profile_open_sides(er) == (true, true)   # 5.7 < 5.98: still open
    end

    @testset "closed profiles are not touched" begin
        @test extend_profile(prob, pb; left=1.0, right=1.0, options=options, optimizer=opt) === pb
        eb = extend_profile(prob, pb; left=0.1, right=0.1, only_open=false, options=options,
                            optimizer=opt)
        @test eb.identifiable && all(v -> v in eb.values, pb.values)
    end

    @testset "extend_profiles + summary" begin
        ext = extend_profiles(prob, profiles; left=1.0, right=1.0, options=options, optimizer=opt)
        df = extension_summary(profiles, ext)
        @test df.parameter == ["a", "b", "c"]
        @test df.identifiable_before == [false, true, false]
        @test df.identifiable_after == [true, true, false]   # c is flat: stays open
        @test count(df.identifiable_after .& .!df.identifiable_before) == 1
        @test df.open_after == ["none", "none", "both"]
        @test ext[2] === profiles[2]
    end

    @testset "results without stored parameters" begin
        old = ProfileResult(pa.parameter, pa.index, pa.best_value, pa.best_loss, pa.ci_lower,
                            pa.ci_upper, pa.identifiable, pa.threshold, pa.values, pa.losses,
                            0, pa.best_found_loss, pa.best_found_params)
        @test isempty(old.params_trace) && !isfinite(old.profile_lb)
        e = extend_profile(prob, old; left=1.0, right=1.0, options=options, optimizer=opt)
        @test e.identifiable && (e.profile_lb, e.profile_ub) == (3.5, 6.5)
    end
end
