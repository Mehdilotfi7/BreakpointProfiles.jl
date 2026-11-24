# =============================================================================
# Profile extension beyond the parameter bounds
# =============================================================================
#
# A profile that reaches a parameter bound before crossing the threshold is
# reported as non-identifiable. That can mean that the likelihood is flat, or
# only that the bound is too tight. `extend_profile` continues such a profile
# beyond the bound of the profiled parameter, keeping every existing point, so
# that the two cases can be told apart. The bounds of all other parameters are
# not changed.

"""
    profile_open_sides(p::ProfileResult) -> (left::Bool, right::Bool)

`true` for a side on which the profile never rises above the threshold, i.e.
the confidence interval is open on that side within the profiled range."""
function profile_open_sides(p::ProfileResult)
    crossed(sel) = any(i -> sel(p.values[i]) && isfinite(p.losses[i]) &&
                            p.losses[i] > p.threshold, eachindex(p.values))
    return (!crossed(v -> v < p.best_value), !crossed(v -> v > p.best_value))
end

"""Original profiling range of `p` (falls back to the problem bounds)."""
function _profile_range(prob, p::ProfileResult)
    lo = isfinite(p.profile_lb) ? p.profile_lb : prob.lb[p.index]
    hi = isfinite(p.profile_ub) ? p.profile_ub : prob.ub[p.index]
    return lo, hi
end

"""
    extended_bounds(lo, hi; left=0.0, right=0.0, mode=:range) -> (new_lo, new_hi)

New profiling range for a parameter currently profiled over `[lo, hi]`.
`left` and `right` are relative amounts (`0.5` = 50 %), never raw values, so
the same setting works for parameters on different scales:

- `mode = :range`    (default) – extend by a fraction of the current range:
  `new_lo = lo - left * (hi - lo)`, `new_hi = hi + right * (hi - lo)`.
- `mode = :relative` – extend by a fraction of the bound itself,
  multiplicatively: `new_lo = lo / (1 + left)`, `new_hi = hi * (1 + right)`.
  Requires `lo > 0` (for `left > 0`) and `hi > 0` (for `right > 0`); the lower
  bound stays positive, so this suits positive rate constants and scales.
"""
function extended_bounds(lo::Real, hi::Real; left::Real=0.0, right::Real=0.0,
                         mode::Symbol=:range)
    left >= 0 && right >= 0 || error("left and right must be non-negative (got $left, $right)")
    lo, hi = Float64(lo), Float64(hi)
    if mode === :range
        w = hi - lo
        w > 0 || error("cannot extend a zero-width range [$lo, $hi] with mode=:range")
        return lo - left * w, hi + right * w
    elseif mode === :relative
        left > 0 && !(lo > 0) && error("mode=:relative needs a positive lower bound (got $lo)")
        right > 0 && !(hi > 0) && error("mode=:relative needs a positive upper bound (got $hi)")
        return (left > 0 ? lo / (1 + left) : lo), (right > 0 ? hi * (1 + right) : hi)
    else
        error("mode must be :range or :relative (got $mode)")
    end
end

"""Copy of `prob` in which only parameter `idx` has new bounds."""
function _with_param_bounds(prob::ODEChangepointPLEProblem, idx::Int, lo::Float64, hi::Float64)
    lb, ub = copy(prob.lb), copy(prob.ub)
    lb[idx], ub[idx] = lo, hi
    return ODEChangepointPLEProblem(
        prob.objective, prob.simulator, prob.ode_function, prob.u0, prob.tspan,
        prob.data, prob.loss_fn, collect(Int, prob.changepoints),
        collect(Float64, prob.best_params), prob.best_loss, lb, ub,
        prob.param_names, prob.n_global, prob.n_segment_specific, prob.n_obs)
end

"""Trace holding the points of an existing profile."""
function _trace_from_profile(p::ProfileResult, fallback_params)
    tr = _ProfileTrace()
    has_params = length(p.params_trace) == length(p.values)
    has_cps = length(p.cps_trace) == length(p.values)
    for i in eachindex(p.values)
        θ = has_params ? p.params_trace[i] : _with_value(fallback_params, p.index, p.values[i])
        cps = has_cps ? p.cps_trace[i] : Int[]
        _trace_push!(tr, p.values[i], p.losses[i], θ, cps)
    end
    _trace_sort!(tr)
    return tr, has_params
end

"""
    extend_profile(prob, profile; side=:both, left=0.5, right=0.5, mode=:range,
                   only_open=true, options=ProfileOptions(),
                   optimizer=default_multistart_bobyqa_config(),
                   samplesize=options.samplesize) -> ProfileResult

Continue an existing conditional (or baseline) profile beyond the bounds of the
profiled parameter, to check whether an interval that is open at a bound closes
further out.

- **Existing points are kept.** The walk starts from the outermost computed
  point on each extended side, warm-started from its optimised parameters, and
  only the new region is computed. The new points are merged with the old ones;
  jump smoothing and the bisection of threshold crossings (if enabled in
  `options`) then run on the merged profile, and the CI and `identifiable` are
  recomputed for the extended range.
- **Relative amounts.** `left`/`right` are fractions (`0.5` = 50 %), interpreted
  by `mode` (see [`extended_bounds`](@ref)): `:range` extends by a fraction of
  the current range, `:relative` by a fraction of the bound value.
- **Sides.** `side = :left`, `:right` or `:both`; with `:both` the two sides may
  use different amounts. With `only_open = true` (default) a side is extended
  only if the profile is open there (see [`profile_open_sides`](@ref)).
- Only the profiled parameter may leave its bounds; all other parameters keep
  the bounds of `prob` during re-optimisation.
- The reference (`best_loss`, threshold) is not changed. If the extension finds
  a loss below the reference, a warning is issued: the optimum lies outside the
  original bounds and the whole analysis should be redone with wider bounds.

`prob` must be the problem the profile was computed for (for profiles from
`profile_all_parameters` with `refit_reference=true`, the reference may have
been re-fitted; the extension uses the profile's own `best_loss` and points).
Joint profiles (`profile_parameter_joint`) are not supported.

Returns a new `ProfileResult` whose `profile_lb`/`profile_ub` give the range
actually profiled."""
function extend_profile(prob::ODEChangepointPLEProblem, p::ProfileResult;
                        side::Symbol=:both, left::Real=0.5, right::Real=0.5,
                        mode::Symbol=:range, only_open::Bool=true,
                        options::ProfileOptions=ProfileOptions(),
                        optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
                        samplesize::Int=options.samplesize)
    side in (:left, :right, :both) || error("side must be :left, :right or :both (got $side)")
    idx = p.index
    1 <= idx <= length(prob.best_params) || error("profile index $idx does not match the problem")
    if length(p.cps_trace) == length(p.values) && !isempty(p.cps_trace) &&
       any(c -> c != p.cps_trace[1], p.cps_trace)
        error("extend_profile supports conditional and baseline profiles only; " *
              "this profile has varying changepoints (joint profile)")
    end
    isempty(p.values) && error("profile has no points")

    lo0, hi0 = _profile_range(prob, p)
    open_left, open_right = profile_open_sides(p)
    do_left = side in (:left, :both) && left > 0 && (open_left || !only_open)
    do_right = side in (:right, :both) && right > 0 && (open_right || !only_open)
    lo_new, hi_new = extended_bounds(lo0, hi0; left=do_left ? left : 0.0,
                                     right=do_right ? right : 0.0, mode=mode)

    fallback = isempty(p.best_found_params) ? prob.best_params : p.best_found_params
    (do_left || do_right) || return p
    tr, has_params = _trace_from_profile(p, fallback)

    margin = p.threshold - p.best_loss
    margin > 0 || error("profile threshold must lie above its reference loss")
    ext = _with_param_bounds(prob, idx, lo_new, hi_new)
    opt = _maybe_polish(optimizer, options)
    inner = _conditional_inner(ext, idx, opt)
    cps = length(p.cps_trace) == length(p.values) ? p.cps_trace[1] : collect(Int, prob.changepoints)

    range_width = hi_new - lo_new
    min_step = max(range_width * options.min_step_factor, 1e-12)
    max_step = max(range_width * options.max_step_factor, min_step)
    n_failed = p.n_failed
    for (dir, go) in ((-1.0, do_left), (1.0, do_right))
        go || continue
        k = dir < 0 ? 1 : length(tr.values)                     # outermost point (trace is sorted)
        start_params = _with_value(tr.params[k], idx, tr.values[k])
        start_loss = tr.losses[k]
        if !has_params || !isfinite(start_loss)
            # no stored optimum at this point: recover it with one inner fit
            start_loss, start_params, _ = inner(tr.values[k], start_params, cps)
            isfinite(start_loss) || (@warn "Cannot restart the profile of $(p.parameter) at $(tr.values[k])"; continue)
        end
        step0 = _initial_step(tr.values[k], range_width, min_step, max_step)
        n_failed += _engine_direction!(tr, ext, idx, start_params, cps, p.best_loss, margin,
                                       inner, options, dir, step0, min_step, max_step,
                                       samplesize; start_loss=start_loss)
    end
    _trace_sort!(tr)

    if options.smooth_jumps && length(tr.values) >= 3
        _engine_smooth!(tr, idx, inner, options.jump_tol * margin)
    end
    if options.refine_ci
        _engine_refine_crossings!(tr, idx, inner, p.best_value, p.threshold, margin,
                                  options.ci_refine_iters, range_width * 1e-7)
    end

    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        tr.values, tr.losses, p.best_value, p.best_loss, lo_new, hi_new, p.threshold)
    best_found, imin = findmin(tr.losses)
    if best_found < p.best_loss - options.better_optimum_tol
        @warn "Extended profile of $(p.parameter) found loss $best_found below the reference " *
              "$(p.best_loss) at $(tr.values[imin]); the optimum lies outside the original " *
              "bounds, so the reference and all intervals should be recomputed with wider bounds."
    end
    return ProfileResult(p.parameter, idx, p.best_value, p.best_loss, ci_lower, ci_upper,
                         identifiable, p.threshold, copy(tr.values), copy(tr.losses), n_failed,
                         best_found, copy(tr.params[imin]), copy(tr.cps),
                         [copy(θ) for θ in tr.params], lo_new, hi_new)
end

"""
    extend_profiles(prob, profiles; kwargs...) -> Vector{ProfileResult}

[`extend_profile`](@ref) for every profile (same keyword arguments; with the
default `only_open = true` closed profiles are returned unchanged). Profiles are
extended in parallel with `Threads.@threads` unless the optimiser is itself
threaded. Compare before and after with [`extension_summary`](@ref)."""
function extend_profiles(prob::ODEChangepointPLEProblem, profiles::Vector{ProfileResult};
                         optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
                         kwargs...)
    out = Vector{ProfileResult}(undef, length(profiles))
    if uses_internal_threads(optimizer)
        for k in eachindex(profiles)
            out[k] = extend_profile(prob, profiles[k]; optimizer=optimizer, kwargs...)
        end
    else
        Base.Threads.@threads for k in eachindex(profiles)
            out[k] = extend_profile(prob, profiles[k]; optimizer=optimizer, kwargs...)
        end
    end
    return out
end

"""
    extension_summary(before, after) -> DataFrame

One row per parameter: profiling range, interval, identifiability and open
sides before and after [`extend_profiles`](@ref). The number of parameters that
became identifiable is `count(df.identifiable_after .& .!df.identifiable_before)`."""
function extension_summary(before::Vector{ProfileResult}, after::Vector{ProfileResult})
    length(before) == length(after) || error("before and after must have the same length")
    df = DataFrame(parameter=String[], range_before=Tuple{Float64,Float64}[],
                   range_after=Tuple{Float64,Float64}[],
                   ci_before=Tuple{Float64,Float64}[], ci_after=Tuple{Float64,Float64}[],
                   identifiable_before=Bool[], identifiable_after=Bool[],
                   open_before=String[], open_after=String[])
    sides(p) = (o = profile_open_sides(p);
                o == (true, true) ? "both" : o[1] ? "left" : o[2] ? "right" : "none")
    for (b, a) in zip(before, after)
        b.index == a.index || error("profiles are not in the same order")
        push!(df, (b.parameter, (b.profile_lb, b.profile_ub), (a.profile_lb, a.profile_ub),
                   (b.ci_lower, b.ci_upper), (a.ci_lower, a.ci_upper),
                   b.identifiable, a.identifiable, sides(b), sides(a)))
    end
    return df
end
