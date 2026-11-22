using BreakpointProfiles
using Test
using Random
using LinearAlgebra
using Statistics

const Δ95 = 3.8414588206941285

# -----------------------------------------------------------------------------
# Helpers: piecewise-linear model, linear in (u0, r1, r2) for a fixed changepoint
# -----------------------------------------------------------------------------
pl_design(cp, ts) = hcat(ones(length(ts)), [min(t, cp) for t in ts], [max(t - cp, 0.0) for t in ts])
pl_sim(θ, cps, ts) = reshape(pl_design(Float64(cps[1]), ts) * θ, 1, :)

function noisy_pl_problem(; seed=11, σ=0.5, cp=20, ts=0.0:50.0)
    rng = MersenneTwister(seed)
    truth = [5.0, 0.2, 0.5]
    data = pl_sim(truth, [cp], ts) .+ σ .* randn(rng, 1, length(ts))
    loss = GaussianNLL(σ)
    obj = (θ, cps) -> loss(pl_sim(θ, cps, ts), data)
    X = pl_design(Float64(cp), ts)
    θ̂ = X \ vec(data)
    prob = ODEChangepointPLEProblem(
        objective=obj, data=data, loss_fn=loss, changepoints=[cp],
        best_params=θ̂, lb=[-10.0, -2.0, -2.0], ub=[20.0, 2.0, 2.0],
        param_names=["u0", "r1", "r2"], n_global=1, n_segment_specific=1)
    return prob, X, θ̂, σ, data, obj
end

@testset "Regression: trial step shrinks on steep profiles" begin
    # loss = 1e4 (θ1 - 3)^2 + (θ2 - 5)^2  →  CI(θ1) = 3 ± sqrt(Δ95 / 1e4)
    prob = BaselinePLEProblem(objective=θ -> 1e4 * (θ[1] - 3)^2 + (θ[2] - 5)^2,
                              best_params=[3.0, 5.0], lb=[-10.0, -10.0], ub=[20.0, 20.0])
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=2000)
    res = profile_parameter(prob, 1; optimizer=opt, options=ProfileOptions(polish=false))
    half = sqrt(Δ95 / 1e4)
    @test res.identifiable
    @test res.ci_lower ≈ 3 - half rtol=1e-4
    @test res.ci_upper ≈ 3 + half rtol=1e-4
    @test length(res.values) >= 10
    # no accepted point beyond the stop margin by more than one step
    @test maximum(res.losses) < 50 * Δ95
end

@testset "Regression: correlated quadratic (nuisance re-optimised)" begin
    # loss = 100 (θ1 - θ2)^2 + (θ1 + θ2 - 2)^2.  Profile of θ1 (θ2 optimised):
    # min_θ2 = 400 (θ1 - 1)^2 / 101 → CI = 1 ± sqrt(Δ95 * 101 / 400)
    prob = BaselinePLEProblem(objective=θ -> 100 * (θ[1] - θ[2])^2 + (θ[1] + θ[2] - 2)^2,
                              best_params=[1.0, 1.0], lb=[-10.0, -10.0], ub=[10.0, 10.0])
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=2000)
    res = profile_parameter(prob, 1; optimizer=opt, options=ProfileOptions(polish=false))
    half = sqrt(Δ95 * 101 / 400)
    @test res.ci_lower ≈ 1 - half rtol=1e-4
    @test res.ci_upper ≈ 1 + half rtol=1e-4
end

@testset "Regression: exact intervals, linear Gaussian model" begin
    prob, X, θ̂, σ, _, _ = noisy_pl_problem()
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=5000)
    profs = profile_all_parameters(prob; optimizer=opt, options=ProfileOptions(polish=false))
    se = sqrt.(diag(inv(X' * X))) .* σ
    for (k, p) in enumerate(profs)
        @test p.identifiable
        @test p.ci_lower ≈ θ̂[k] - sqrt(Δ95) * se[k] rtol=1e-4
        @test p.ci_upper ≈ θ̂[k] + sqrt(Δ95) * se[k] rtol=1e-4
    end
end

@testset "Regression: reference re-fit and better-optimum restart" begin
    # Supply a poor reference; the profile must be anchored at the true optimum.
    prob = BaselinePLEProblem(objective=θ -> (θ[1] - 3)^2 + (θ[2] - 5)^2,
                              best_params=[3.5, 6.0], lb=[-10.0, -10.0], ub=[20.0, 20.0])
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=2000)
    res = profile_parameter(prob, 1; optimizer=opt, options=ProfileOptions(polish=false))
    @test res.best_loss ≈ 0.0 atol=1e-8
    @test res.best_value ≈ 3.0 atol=1e-4
    @test res.ci_lower ≈ 3 - sqrt(Δ95) rtol=1e-4

    # Without the re-fit the restart logic still recovers the optimum.
    res2 = profile_parameter(prob, 1; optimizer=opt,
                             options=ProfileOptions(polish=false, refit_reference=false))
    @test res2.best_loss ≈ 0.0 atol=1e-6
    @test res2.ci_upper ≈ 3 + sqrt(Δ95) rtol=1e-3
end

@testset "Regression: ODE path passes segment parameters" begin
    # du/dt = r_j in segment j  →  piecewise-linear trajectory with a kink at the CP
    rhs!(du, u, p, t) = (du[1] = p[2])    # p = [u0_dummy_global; r_segment]
    ts = 0.0:40.0
    truth = [0.0, 0.2, 1.0]               # global (unused by RHS), r1, r2
    u0 = [1.0]
    data = reshape([1.0 + (t < 15 ? 0.2t : 0.2 * 15 + 1.0 * (t - 15)) for t in ts], 1, :)
    prob = ODEChangepointPLEProblem(
        ode_function=rhs!, u0=u0, tspan=(0.0, 40.0), data=data,
        loss_fn=GaussianNLL(0.1), changepoints=[15], best_params=truth,
        lb=[-1.0, -2.0, -2.0], ub=[1.0, 2.0, 2.0], n_global=1, n_segment_specific=1)

    @test segment_parameters(prob, truth, 1) == [0.0, 0.2]
    @test segment_parameters(prob, truth, 2) == [0.0, 1.0]
    sim = solve_segments(prob, truth)
    @test maximum(abs.(sim .- data)) < 1e-6
    @test evaluate_loss(prob, truth) < 1e-8
    # moving the changepoint must change the dynamics (previously it did not)
    @test evaluate_loss(prob, truth, [10]) > 100
    @test evaluate_loss(prob, truth, [20]) > 100

    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-8, maxeval=2000)
    cpp = profile_changepoint(prob, 1; window=4, min_seg=5, optimizer=opt, polish=false)
    cs = cp_confidence_set(cpp)
    @test cs.best_cp == 15
    @test cs.identifiable
    @test cs.set == [15]
end

@testset "Regression: joint profile never above conditional" begin
    prob, X, θ̂, σ, data, obj = noisy_pl_problem(seed=3, σ=1.0)
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=5000)
    options = ProfileOptions(polish=false)
    ref = joint_reference_fit(prob, 4, opt; parallelize_grid=false, n_starts=1, min_seg=5)
    @test ref[1] <= evaluate_loss(prob, prob.best_params) + 1e-10
    for idx in 1:3
        joint = profile_parameter_joint(prob, idx; window=4, min_seg=5, optimizer=opt,
                                        options=options, parallelize_grid=false,
                                        joint_reference=ref)
        # conditional loss at the same parameter values (closed form: linear LS)
        free = setdiff(1:3, idx)
        cond_loss(v) = begin
            y = vec(data) .- X[:, idx] .* v
            β = X[:, free] \ y
            sum(abs2, y .- X[:, free] * β) / σ^2
        end
        @test all(joint.losses .<= cond_loss.(joint.values) .+ 1e-6)
        @test joint.best_loss <= minimum(cond_loss.(joint.values)) + 1e-6
        # joint profile values are minima over the CP set, so the joint CI contains
        # the conditional CI whenever the joint MLE CP equals the detected CP
        if ref[3] == prob.changepoints
            cond = profile_parameter(prob, idx; optimizer=opt, options=options)
            @test joint.ci_lower <= cond.ci_lower + 1e-6
            @test joint.ci_upper >= cond.ci_upper - 1e-6
        end
    end
end

@testset "Regression: joint grid is fixed and contains detected CPs" begin
    g = BreakpointProfiles._joint_cp_grid([20, 40], 3, 60; min_seg=5)
    @test [20, 40] in g
    @test all(c -> abs(c[1] - 20) <= 3 && abs(c[2] - 40) <= 3, g)
    @test length(g) == 49
    g2 = BreakpointProfiles._joint_cp_grid([20, 26], 3, 60; min_seg=5)
    @test all(c -> c[2] - c[1] >= 5, g2)
end

@testset "Regression: changepoint confidence sets" begin
    flat = CPProfileResult(1, 20, 3, collect(17:23), fill(0.0, 7), 0.0)
    cs = cp_confidence_set(flat)
    @test cs.censored_lower && cs.censored_upper
    @test !cs.identifiable
    lo, hi, id = cp_ci(flat, Δ95)
    @test (lo, hi) == (17, 23)
    @test !id

    # bounded, with a secondary component
    losses = [10.0, 0.5, 0.0, 1.0, 9.0, 2.0, 12.0]
    two = CPProfileResult(1, 20, 3, collect(17:23), losses, 0.0)
    cs2 = cp_confidence_set(two)
    @test cs2.set == [18, 19, 20, 22]
    @test cs2.components == [[18, 19, 20], [22]]
    @test (cs2.lower, cs2.upper) == (18, 20)
    @test cs2.identifiable
    df = cp_summary([flat, two])
    @test df.n_components == [1, 2]
    @test df.identifiable == [false, true]
end

@testset "Regression: write_report" begin
    prob = BaselinePLEProblem(objective=θ -> (θ[1] - 3)^2 + (θ[2] - 5)^2,
                              best_params=[3.0, 5.0], lb=[-10.0, -10.0], ub=[20.0, 20.0])
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-8, maxeval=1000)
    profs = profile_all_parameters(prob; optimizer=opt, options=ProfileOptions(polish=false))
    cpp = [CPProfileResult(1, 20, 2, [18, 19, 20, 21, 22], [9.0, 1.0, 0.0, 2.0, 8.0], 0.0)]
    mktempdir() do dir
        path = write_report(dir, profs; cp_profiles=cpp)
        @test isfile(path)
        txt = read(path, String)
        @test occursin("| par_1 |", txt) || occursin("| x |", txt) || occursin("par", txt)
        @test occursin("Changepoint confidence sets", txt)
        @test isfile(joinpath(dir, "cp_profile_ci.csv"))
    end
end

@testset "Regression: seeded optimisers are reproducible" begin
    f = x -> sum(abs2, x .- [1.0, 2.0]) + 0.1 * sum(sin, 10 .* x)
    opt = MultiStartBOBYQAConfig(n_starts=4, maxeval=300, perturbation=0.5, seed=7)
    a = optimize_ple(f, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], opt)
    b = optimize_ple(f, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], opt)
    @test a[1] == b[1] && a[2] == b[2]
end

@testset "Regression: likelihoods" begin
    sim = [1.0 2.0 3.0]
    dat = [1.5 2.0 2.0]
    @test GaussianNLL(0.5)(sim, dat) ≈ (0.25 + 0 + 1) / 0.25
    @test ProfiledGaussianNLL()(sim, dat) ≈ 3 * log(1.25 / 3)
end

@testset "Regression: local CP search matches exhaustive grid (2 changepoints)" begin
    rng = MersenneTwister(5)
    ts = 0.0:80.0
    design(c) = hcat(ones(length(ts)), [min(t, c[1]) for t in ts],
                     [clamp(t - c[1], 0, c[2] - c[1]) for t in ts], [max(t - c[2], 0.0) for t in ts])
    truth = [2.0, 0.5, -0.3, 0.4]
    y = design([25, 55]) * truth .+ 0.5 .* randn(rng, length(ts))
    obj = (θ, c) -> sum(abs2, design(c) * θ .- y) / 0.25
    θ̂ = design([25, 55]) \ y
    prob = ODEChangepointPLEProblem(objective=obj, data=reshape(y, 1, :), loss_fn=nothing,
                                    changepoints=[25, 55], best_params=θ̂,
                                    lb=fill(-5.0, 4), ub=fill(5.0, 4), n_global=1,
                                    n_segment_specific=1, n_obs=length(ts))
    opt = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-10, maxeval=5000)
    S = CPSearchSpace([25, 55], 3, length(ts); min_seg=5)
    @test [25, 55] in S && !([25, 59] in S) && !([25, 26] in S)
    @test length(enumerate_space(S)) == 49
    g = joint_grid_fit(prob, 3, opt; min_seg=5, parallelize_grid=false, n_starts=1)
    l = joint_grid_fit(prob, 3, opt; min_seg=5, parallelize_grid=false, n_starts=1, cp_search=:local)
    @test l.best_loss ≈ g.best_loss atol=1e-6
    @test length(l.grid) < length(g.grid)
    o = ProfileOptions(polish=false)
    pg = profile_parameter_joint(prob, 2; window=3, min_seg=5, optimizer=opt, options=o,
                                 parallelize_grid=false, grid_fit=g)
    pl = profile_parameter_joint(prob, 2; window=3, min_seg=5, optimizer=opt, options=o,
                                 parallelize_grid=false, grid_fit=l, cp_search=:local)
    @test pl.ci_lower ≈ pg.ci_lower rtol=1e-3
    @test pl.ci_upper ≈ pg.ci_upper rtol=1e-3
end
