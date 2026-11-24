const CHISQ_95 = 3.8414588206941285

# =============================================================================
# Generic profiling engine
# =============================================================================
#
# The conditional profile (changepoints fixed) and the joint profile
# (changepoints re-optimised on a fixed discrete grid) are computed by the same
# engine. They differ only in the *inner solver*
#
#     inner(value, warm_params, warm_cps) -> (loss, params, cps)
#
# which minimises the loss over all nuisance quantities with parameter `idx`
# fixed to `value`, starting from a warm start.

struct _ProfileTrace
    values::Vector{Float64}
    losses::Vector{Float64}
    params::Vector{Vector{Float64}}
    cps::Vector{Vector{Int}}
end

_ProfileTrace() = _ProfileTrace(Float64[], Float64[], Vector{Float64}[], Vector{Int}[])

function _trace_push!(tr::_ProfileTrace, v, l, θ, cps)
    push!(tr.values, v)
    push!(tr.losses, l)
    push!(tr.params, copy(θ))
    push!(tr.cps, collect(Int, cps))
    return tr
end

function _trace_sort!(tr::_ProfileTrace)
    perm = sortperm(tr.values)
    permute!(tr.values, perm)
    permute!(tr.losses, perm)
    permute!(tr.params, perm)
    permute!(tr.cps, perm)
    return tr
end

_with_value(θ, idx, v) = (x = collect(Float64, θ); x[idx] = v; x)

"""Copy of `opts` with the given fields replaced."""
function _with_options(opts::ProfileOptions; kwargs...)
    vals = Any[get(kwargs, f, getfield(opts, f)) for f in fieldnames(ProfileOptions)]
    return ProfileOptions(vals...)
end

"""Copy of `prob` with a new reference fit (and optionally new changepoints)."""
function _with_reference(prob::ODEChangepointPLEProblem, params, loss;
                         changepoints=prob.changepoints)
    return ODEChangepointPLEProblem(
        prob.objective, prob.simulator, prob.ode_function, prob.u0, prob.tspan,
        prob.data, prob.loss_fn, collect(Int, changepoints),
        collect(Float64, params), Float64(loss), prob.lb, prob.ub,
        prob.param_names, prob.n_global, prob.n_segment_specific, prob.n_obs)
end

function _assemble(n, idx, fixed_val, free, x_free)
    θ = Vector{Float64}(undef, n)
    θ[idx] = fixed_val
    θ[free] .= x_free
    return θ
end

"""
    _build_loss_free(prob, idx, fixed_val, free[, cps])

Loss over the free parameters with parameter `idx` fixed to `fixed_val`."""
function _build_loss_free(prob, idx, fixed_val, free, cps=prob.changepoints)
    n = length(prob.best_params)
    c = collect(Int, cps)
    return x_free -> evaluate_loss(prob, _assemble(n, idx, fixed_val, free, x_free), c)
end

"""
    _optimize_fixed(prob, idx, value, cps, warm, optimizer)

Minimise over all parameters except `idx` (fixed to `value`) at the changepoint
vector `cps`, starting from `warm`. The returned loss is never worse than the
loss at the warm start, which guards against stochastic optimisers returning a
worse point than their initial guess. Returns `(loss, params_full)`."""
function _optimize_fixed(prob, idx, value, cps, warm, optimizer)
    n = length(prob.best_params)
    free = setdiff(1:n, idx)
    lb = prob.lb[free]
    ub = prob.ub[free]
    f = _build_loss_free(prob, idx, value, free, cps)
    x0 = clamp.(collect(Float64, warm[free]), lb, ub)
    l0 = f(x0)
    loss, x = try
        optimize_ple(f, x0, lb, ub, optimizer)
    catch e
        @debug "Inner optimisation failed for parameter $idx at $value" exception = e
        (Inf, x0)
    end
    loss = (loss isa Real && isfinite(loss)) ? Float64(loss) : Inf
    if !(loss <= l0)
        loss, x = l0, x0
    end
    return loss, _assemble(n, idx, value, free, collect(Float64, x))
end

"""Inner solver for the conditional profile: changepoints fixed."""
function _conditional_inner(prob, idx, optimizer)
    cps = collect(Int, prob.changepoints)
    return function (value, warm, warm_cps)
        local loss, θ = _optimize_fixed(prob, idx, value, cps, warm, optimizer)
        return loss, θ, cps
    end
end

function _initial_step(best_val, range_width, min_step, max_step)
    step0 = best_val != 0 ? min(range_width / 50.0, abs(best_val) * 0.05) : range_width / 50.0
    return clamp(step0, min_step, max_step)
end

"""
    _profile_engine(prob, idx, ref_params, ref_cps, ref_loss, margin, inner, options;
                    samplesize=options.samplesize)

D2D-style adaptive profile (Raue et al. 2009; Data2Dynamics `ple.m`).

For each direction:
1. Propose `value ± step`.
2. Trial step: evaluate the loss at the proposal with all nuisance quantities
   held at the *current* profile point (no re-optimisation) and shrink the step
   by `step_factor` until the increase relative to the current point is at
   most `rel_step_increase * margin`.
3. Re-optimise the nuisance quantities with `inner` and record the point.
4. Enlarge the step by `step_factor` and continue until the profile exceeds
   `stop_margin_factor * margin` above the reference or a bound is hit.

`reach = (lo, hi)` forces the lower (upper) direction to continue at least to
`lo` (`hi`) even after the threshold has been exceeded; it is used by the joint
profile to reach admissible secondary modes so that disconnected confidence
regions are not missed.

Afterwards local jumps are smoothed (D2D `pleSmooth`), each threshold crossing
is refined by bisection, and the CI is read off the refined profile.
Returns `(ProfileResult, trace)`."""
function _profile_engine(prob, idx, ref_params, ref_cps, ref_loss, margin, inner,
                         options::ProfileOptions; samplesize::Int=options.samplesize,
                         reach::Tuple{Float64,Float64}=(Inf, -Inf))
    margin > 0 || error("profile threshold margin must be positive (got $margin)")
    best_val = ref_params[idx]
    lower_bound = prob.lb[idx]
    upper_bound = prob.ub[idx]
    range_width = upper_bound - lower_bound
    threshold = ref_loss + margin

    min_step = max(range_width * options.min_step_factor, 1e-12)
    max_step = max(range_width * options.max_step_factor, min_step)
    step0 = _initial_step(best_val, range_width, min_step, max_step)

    tr = _ProfileTrace()
    _trace_push!(tr, best_val, ref_loss, ref_params, ref_cps)
    n_failed = 0

    for direction in (-1.0, 1.0)
        n_failed += _engine_direction!(tr, prob, idx, ref_params, ref_cps, ref_loss,
                                       margin, inner, options, direction, step0,
                                       min_step, max_step, samplesize,
                                       direction < 0 ? reach[1] : reach[2])
    end
    _trace_sort!(tr)

    if options.smooth_jumps && length(tr.values) >= 3
        _engine_smooth!(tr, idx, inner, options.jump_tol * margin)
    end

    if options.refine_ci
        _engine_refine_crossings!(tr, idx, inner, best_val, threshold, margin,
                                  options.ci_refine_iters, range_width * 1e-7)
    end

    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        tr.values, tr.losses, best_val, ref_loss, lower_bound, upper_bound, threshold)

    best_found, imin = findmin(tr.losses)
    result = ProfileResult(
        prob.param_names[idx], idx, best_val, ref_loss, ci_lower, ci_upper,
        identifiable, threshold, copy(tr.values), copy(tr.losses), n_failed,
        best_found, copy(tr.params[imin]), copy(tr.cps),
        [copy(θ) for θ in tr.params], lower_bound, upper_bound
    )
    return result, tr
end

function _engine_direction!(tr, prob, idx, ref_params, ref_cps, ref_loss, margin, inner,
                            options, direction, step0, min_step, max_step, samplesize,
                            reach_val::Float64=(direction < 0 ? Inf : -Inf);
                            start_loss::Float64=ref_loss)
    # `ref_params`/`start_loss` give the point the walk starts from (the
    # reference optimum, or the outermost point of an existing profile when it
    # is extended); `ref_loss` is the profile reference used for stopping.
    lower_bound = prob.lb[idx]
    upper_bound = prob.ub[idx]
    target = options.rel_step_increase * margin

    cur_val = ref_params[idx]
    cur_params = collect(Float64, ref_params)
    cur_cps = collect(Int, ref_cps)
    cur_loss = start_loss
    prev_val = NaN
    prev_params = cur_params
    step = step0
    n_failed = 0
    n_points = 0
    n_attempts = 0

    # Predicted point at `cand`: move along the profile path by linear
    # extrapolation from the last two profile points (D2D `pleExtrapolate`);
    # on the first step only the profiled coordinate moves.
    predict(cand) = begin
        local θp = copy(cur_params)
        if isfinite(prev_val) && cur_val != prev_val
            θp .+= (cand - cur_val) / (cur_val - prev_val) .* (cur_params .- prev_params)
            θp .= clamp.(θp, prob.lb, prob.ub)
        end
        θp[idx] = cand
        θp
    end

    while n_points < samplesize && n_attempts < 2 * samplesize
        n_attempts += 1
        # --- trial step: nuisance at the predicted point, no re-optimisation --
        step = clamp(step, min_step, max_step)
        cand = cur_val
        θ_pred = cur_params
        for redo in 0:options.max_trial_redos
            cand = clamp(cur_val + direction * step, lower_bound, upper_bound)
            cand == cur_val && break
            θ_pred = predict(cand)
            trial_loss = evaluate_loss(prob, θ_pred, cur_cps)
            if !(isfinite(trial_loss) && trial_loss - cur_loss <= target)
                # fall back to the un-extrapolated point if that is better
                θ_flat = _with_value(cur_params, idx, cand)
                flat_loss = evaluate_loss(prob, θ_flat, cur_cps)
                if isfinite(flat_loss) && !(flat_loss >= trial_loss)
                    θ_pred, trial_loss = θ_flat, flat_loss
                end
            end
            isfinite(trial_loss) && trial_loss - cur_loss <= target && break
            # Accept the smallest admissible step (as D2D does) rather than
            # silently truncating the profile.
            (step <= min_step || redo == options.max_trial_redos) && break
            step = max(step / options.step_factor, min_step)
        end
        cand == cur_val && break

        # --- re-optimise the nuisance quantities ----------------------------
        loss, θ, cps = inner(cand, θ_pred, cur_cps)
        if !isfinite(loss)
            n_failed += 1
            step <= min_step && break
            step = max(step / 2, min_step)
            continue
        end
        _trace_push!(tr, cand, loss, θ, cps)
        n_points += 1

        prev_val, prev_params = cur_val, cur_params
        cur_val, cur_params, cur_cps, cur_loss = cand, θ, cps, loss
        # Stop above the threshold, unless a known admissible mode (e.g. another
        # changepoint configuration of the joint problem) lies further out.
        beyond = direction < 0 ? cand <= reach_val : cand >= reach_val
        loss - ref_loss > options.stop_margin_factor * margin && beyond && break
        (cand <= lower_bound || cand >= upper_bound) && break
        step *= options.step_factor
    end
    return n_failed
end

"""
    _engine_smooth!(tr, idx, inner, tol)

D2D `pleSmooth` equivalent. Walking outward from the profile minimum the
profile should not decrease; if a point is higher than its *outer* neighbour by
more than `tol`, it is re-optimised warm-started from the outer neighbour's
optimum (the inner point was trapped in a worse local optimum). Improvements
propagate inward. Repeated until no point improves (at most 5 sweeps)."""
function _engine_smooth!(tr, idx, inner, tol)
    _trace_sort!(tr)
    n = length(tr.values)
    for _ in 1:5
        improved = false
        _, imin = findmin(tr.losses)
        # left side: outer neighbour of k is k-1
        for k in 2:(imin - 1)
            improved |= _smooth_point!(tr, idx, inner, k, k - 1, tol)
        end
        # right side: outer neighbour of k is k+1
        for k in (n - 1):-1:(imin + 1)
            improved |= _smooth_point!(tr, idx, inner, k, k + 1, tol)
        end
        improved || break
    end
    return tr
end

function _smooth_point!(tr, idx, inner, k, from, tol)
    isfinite(tr.losses[from]) || return false
    tr.losses[k] > tr.losses[from] + tol || return false
    loss, θ, cps = inner(tr.values[k], _with_value(tr.params[from], idx, tr.values[k]), tr.cps[from])
    if loss < tr.losses[k] - 1e-10
        tr.losses[k] = loss
        tr.params[k] = θ
        tr.cps[k] = collect(Int, cps)
        return true
    end
    return false
end

"""
    _engine_refine_crossings!(tr, idx, inner, best_val, threshold, margin, iters, vtol)

Refine every threshold crossing of the profile by bisection.
Each midpoint is re-optimised warm-started from the inside bracket point, so
the final CI endpoint is accurate to the bracket width rather than to the
spacing of the adaptive profile grid."""
function _engine_refine_crossings!(tr, idx, inner, best_val, threshold, margin, iters, vtol)
    _trace_sort!(tr)
    n = length(tr.values)
    n < 2 && return tr
    # every adjacent pair straddling the threshold: (inside, outside)
    brackets = Tuple{Int,Int}[]
    for i in 1:(n - 1)
        a, b = tr.losses[i], tr.losses[i + 1]
        (isfinite(a) && isfinite(b)) || continue
        if a <= threshold < b
            push!(brackets, (i, i + 1))
        elseif b <= threshold < a
            push!(brackets, (i + 1, i))
        end
    end
    new_points = Tuple{Float64,Float64,Vector{Float64},Vector{Int}}[]
    for (iin, iout) in brackets
        vin, vout = tr.values[iin], tr.values[iout]
        pin, cin = tr.params[iin], tr.cps[iin]
        for _ in 1:iters
            abs(vout - vin) <= vtol && break
            mid = 0.5 * (vin + vout)
            loss, θ, cps = inner(mid, _with_value(pin, idx, mid), cin)
            isfinite(loss) || break
            push!(new_points, (mid, loss, θ, collect(Int, cps)))
            if loss <= threshold
                vin, pin, cin = mid, θ, cps
            else
                vout = mid
            end
            abs(loss - threshold) <= 1e-3 * margin && break
        end
    end
    for (v, l, θ, c) in new_points
        _trace_push!(tr, v, l, θ, c)
    end
    _trace_sort!(tr)
    return tr
end

# =============================================================================
# Reference fit
# =============================================================================

function _maybe_polish(optimizer::AbstractPLEOptimizer, options::ProfileOptions)
    return options.polish && !(optimizer isa HybridOptimizerConfig) ?
           HybridOptimizerConfig(global_optimizer=optimizer) : optimizer
end

"""
    refit_reference(prob; optimizer=default_multistart_bobyqa_config(), changepoints=prob.changepoints)

Re-optimise all parameters at fixed changepoints starting from
`prob.best_params` and return a copy of `prob` whose `best_params` and
`best_loss` are the improved optimum (never worse than the supplied fit).
Profiles must be anchored at an optimum; a sub-optimal reference shifts the
threshold and invalidates every interval."""
function refit_reference(prob::ODEChangepointPLEProblem;
                         optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
                         changepoints=prob.changepoints)
    cps = collect(Int, changepoints)
    f = θ -> evaluate_loss(prob, θ, cps)
    x0 = clamp.(prob.best_params, prob.lb, prob.ub)
    l0 = f(x0)
    loss, x = try
        optimize_ple(f, x0, prob.lb, prob.ub, optimizer)
    catch e
        @warn "Reference re-fit failed: $e"
        (Inf, x0)
    end
    if !(isfinite(loss) && loss < l0)
        loss, x = l0, x0
    end
    return _with_reference(prob, x, loss; changepoints=cps)
end

"""Verified reference loss, warning if it disagrees with `prob.best_loss`."""
function _verified_reference_loss(prob)
    verified = evaluate_loss(prob, prob.best_params)
    if isfinite(prob.best_loss) &&
       abs(verified - prob.best_loss) > 1e-6 * max(1.0, abs(prob.best_loss))
        @warn "Reference loss mismatch: prob.best_loss=$(prob.best_loss), evaluated=$(verified). Using the evaluated value."
    end
    isfinite(verified) || error("The loss at prob.best_params is not finite")
    return verified
end

"""Threshold margin from an absolute `threshold`, a `delta`, or the χ²₁ default."""
function _margin(threshold, delta, ref_loss)
    delta !== nothing && return Float64(delta)
    threshold !== nothing && return Float64(threshold) - ref_loss
    return CHISQ_95
end

# =============================================================================
# Conditional profile
# =============================================================================

"""
    profile_parameter(prob::ODEChangepointPLEProblem, idx::Int;
                      options=ProfileOptions(),
                      optimizer=default_multistart_bobyqa_config(),
                      threshold=nothing, delta=nothing,
                      method=nothing, n_points=nothing)

Conditional profile likelihood of parameter `idx`: the changepoints are held
fixed at `prob.changepoints` and all other parameters are re-optimised at each
profile point.

The confidence threshold is `reference_loss + delta`. `delta` defaults to the
χ²₁ quantile 3.8415 (valid when the loss is `-2 log L` of a correctly specified
likelihood). Alternatively an absolute `threshold` can be supplied; it is
converted to a margin relative to the loss at `prob.best_params`.

With `options.refit_reference` the reference is first re-optimised; with
`options.allow_better_optimum` the profile is restarted from any better optimum
it discovers (up to `options.max_reference_updates` times).

Keyword arguments:
- `options`: stepping, smoothing and CI settings, see [`ProfileOptions`](@ref).
- `optimizer`: inner optimiser used at every profile point (the main driver of
  runtime and of profile quality).
- `delta` / `threshold`: confidence level, as described above.
- `method`: `:adaptive` (default, D2D-style adaptive steps) or `:fixed` (a
  fixed grid of `n_points` values around the estimate, clipped to the bounds:
  log-spaced over `[best/5, 5·best]` when the parameter is positive and its
  bounds span more than a factor 10, otherwise linear over `best ± half the
  bound range`).
- `n_points`: for `:adaptive`, overrides `options.samplesize` (maximum steps
  per direction); for `:fixed`, the number of grid points (default 20).

A profile that is open at a parameter bound can be continued beyond it with
[`extend_profile`](@ref)."""
function profile_parameter(
    prob::ODEChangepointPLEProblem, idx::Int;
    options::ProfileOptions=ProfileOptions(),
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    threshold::Union{Real,Nothing}=nothing,
    delta::Union{Real,Nothing}=nothing,
    method::Union{Symbol,Nothing}=nothing,
    n_points::Union{Int,Nothing}=nothing
)
    isempty(prob.best_params) && error("ODEChangepointPLEProblem.best_params must be set")
    1 <= idx <= length(prob.best_params) || error("parameter index $idx out of range")
    opt = _maybe_polish(optimizer, options)

    verified = _verified_reference_loss(prob)
    margin = _margin(threshold, delta, verified)
    cur = _with_reference(prob, prob.best_params, verified)
    if options.refit_reference
        cur = refit_reference(cur; optimizer=opt)
    end

    if method !== nothing && method !== :adaptive && method !== :fixed
        @warn "Unknown profile method '$method'; using adaptive D2D stepping."
    end

    result = nothing
    for update in 0:options.max_reference_updates
        result = if method === :fixed
            _profile_fixed(cur, idx, margin, n_points === nothing ? 20 : n_points, opt)
        else
            first(_profile_engine(cur, idx, cur.best_params, cur.changepoints,
                                  cur.best_loss, margin,
                                  _conditional_inner(cur, idx, opt), options;
                                  samplesize=n_points === nothing ? options.samplesize : n_points))
        end
        if options.allow_better_optimum && update < options.max_reference_updates &&
           result.best_found_loss < cur.best_loss - options.better_optimum_tol
            @warn "Parameter $(prob.param_names[idx]): profile found loss $(result.best_found_loss) < reference $(cur.best_loss); restarting from the better optimum."
            cur = refit_reference(_with_reference(cur, result.best_found_params,
                                                  result.best_found_loss); optimizer=opt)
            continue
        end
        break
    end
    if result.best_found_loss < result.best_loss - options.better_optimum_tol
        @warn "Parameter $(prob.param_names[idx]): profile still finds loss $(result.best_found_loss) below the reference $(result.best_loss); the interval is not reliable."
    end
    return result
end

"""
    profile_all_parameters(prob; options=ProfileOptions(),
                           optimizer=default_multistart_bobyqa_config(),
                           indices=nothing, threshold=nothing, delta=nothing,
                           method=nothing, n_points=nothing)

Conditional profiles of every parameter (or of `indices`). All profiles share
one reference fit: the reference is re-fitted once (`options.refit_reference`)
and, if any profile finds a better optimum, the reference is replaced and *all*
profiles are recomputed (`options.allow_better_optimum`). Parameters are
profiled in parallel with `Threads.@threads` unless the optimiser is itself
threaded. Returns a `Vector{ProfileResult}` ordered by parameter index.

Keyword arguments are those of [`profile_parameter`](@ref), plus `indices`
(the parameters to profile, default all). `threshold` and `delta` may also be
vectors with one entry per profiled parameter."""
function profile_all_parameters(
    prob::ODEChangepointPLEProblem;
    options::ProfileOptions=ProfileOptions(),
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    indices::Union{Nothing,Vector{Int}}=nothing,
    threshold::Union{Real,Vector{Float64},Nothing}=nothing,
    delta::Union{Real,Vector{Float64},Nothing}=nothing,
    method::Union{Symbol,Nothing}=nothing,
    n_points::Union{Int,Nothing}=nothing
)
    idxs = indices === nothing ? collect(1:length(prob.best_params)) : indices
    opt = _maybe_polish(optimizer, options)
    verified = _verified_reference_loss(prob)
    margins = [_margin(threshold isa Vector ? threshold[k] : threshold,
                       delta isa Vector ? delta[k] : delta, verified)
               for k in eachindex(idxs)]

    cur = _with_reference(prob, prob.best_params, verified)
    options.refit_reference && (cur = refit_reference(cur; optimizer=opt))
    inner_options = _with_options(options; refit_reference=false,
                                  allow_better_optimum=false, polish=false)

    profiles = Vector{ProfileResult}(undef, length(idxs))
    for update in 0:options.max_reference_updates
        local run_one = k -> profile_parameter(cur, idxs[k]; options=inner_options,
                                         optimizer=opt, delta=margins[k],
                                         method=method, n_points=n_points)
        if uses_internal_threads(opt)
            for k in eachindex(idxs)
                profiles[k] = run_one(k)
            end
        else
            Base.Threads.@threads for k in eachindex(idxs)
                profiles[k] = run_one(k)
            end
        end
        best_found, kbest = findmin([p.best_found_loss for p in profiles])
        if options.allow_better_optimum && update < options.max_reference_updates &&
           best_found < cur.best_loss - options.better_optimum_tol
            @warn "Profile of $(profiles[kbest].parameter) found loss $best_found < reference $(cur.best_loss); re-profiling all parameters from the better optimum."
            cur = refit_reference(_with_reference(cur, profiles[kbest].best_found_params,
                                                  best_found); optimizer=opt)
            continue
        end
        break
    end
    sort!(profiles, by=p -> p.index)
    return profiles
end

"""
    _profile_fixed(prob, idx, margin, n_points, optimizer)

Fixed-grid profiling (fallback used when `method=:fixed`)."""
function _profile_fixed(prob, idx, margin, n_points, optimizer)
    best_val = prob.best_params[idx]
    ref_loss = prob.best_loss
    threshold = ref_loss + margin
    grid = _fixed_grid(prob.lb[idx], prob.ub[idx], best_val, n_points)
    inner = _conditional_inner(prob, idx, optimizer)

    tr = _ProfileTrace()
    n_failed = 0
    # sweep outward from the reference in both directions for good warm starts
    below = reverse(filter(<(best_val), grid))
    above = filter(>(best_val), grid)
    _trace_push!(tr, best_val, ref_loss, prob.best_params, prob.changepoints)
    for side in (below, above)
        warm = copy(prob.best_params)
        for v in side
            loss, θ, cps = inner(v, _with_value(warm, idx, v), prob.changepoints)
            if isfinite(loss)
                warm = θ
            else
                n_failed += 1
            end
            _trace_push!(tr, v, loss, θ, cps)
        end
    end
    _trace_sort!(tr)
    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        tr.values, tr.losses, best_val, ref_loss, prob.lb[idx], prob.ub[idx], threshold)
    best_found, imin = findmin(tr.losses)
    return ProfileResult(
        prob.param_names[idx], idx, best_val, ref_loss, ci_lower, ci_upper,
        identifiable, threshold, tr.values, tr.losses, n_failed, best_found,
        tr.params[imin], tr.cps, [copy(θ) for θ in tr.params],
        prob.lb[idx], prob.ub[idx])
end

function _fixed_grid(lower, upper, best_val, n_points)
    if best_val > 0 && upper > lower && (upper / max(lower, 1e-12) > 10.0)
        gmin = max(lower, best_val / 5.0)
        gmax = min(upper, best_val * 5.0)
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        grid = 10.0 .^ range(log10(gmin), log10(gmax), length=n_points)
    else
        half = 0.5 * (upper - lower)
        gmin = max(lower, best_val - half)
        gmax = min(upper, best_val + half)
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        grid = collect(range(gmin, gmax, length=n_points))
    end
    push!(grid, best_val)
    sort!(grid)
    unique!(grid)
    return grid
end

"""
    ple_ci_interpolated(values, losses, best_val, best_loss, lower_bound, upper_bound, threshold)

Return `(ci_lower, ci_upper, identifiable)` using linear interpolation to find where
the profile crosses `threshold`.  Identifiability is true only if the profile crosses
the threshold on both sides before hitting a parameter bound.
"""
function ple_ci_interpolated(
    values::Vector{Float64},
    losses::Vector{Float64},
    best_val::Float64,
    best_loss::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
    threshold::Float64
)
    perm = sortperm(values)
    svals = values[perm]
    sloss = losses[perm]
    n = length(svals)

    if n < 2
        return (best_val, best_val, false)
    end

    best_pos = argmin(abs.(svals .- best_val))

    # Lower crossing
    ci_lower = lower_bound
    crossed_lower = false
    for i in best_pos:-1:2
        if !isfinite(sloss[i - 1]) || !isfinite(sloss[i])
            continue
        end
        if sloss[i] <= threshold && sloss[i - 1] > threshold
            frac = (threshold - sloss[i]) / (sloss[i - 1] - sloss[i])
            ci_lower = svals[i] - frac * (svals[i] - svals[i - 1])
            crossed_lower = true
            break
        end
    end
    if !crossed_lower
        for i in best_pos:-1:1
            if isfinite(sloss[i]) && sloss[i] <= threshold
                ci_lower = svals[i]
            else
                break
            end
        end
    end

    # Upper crossing
    ci_upper = upper_bound
    crossed_upper = false
    for i in best_pos:(n - 1)
        if !isfinite(sloss[i]) || !isfinite(sloss[i + 1])
            continue
        end
        if sloss[i] <= threshold && sloss[i + 1] > threshold
            frac = (threshold - sloss[i]) / (sloss[i + 1] - sloss[i])
            ci_upper = svals[i] + frac * (svals[i + 1] - svals[i])
            crossed_upper = true
            break
        end
    end
    if !crossed_upper
        for i in best_pos:n
            if isfinite(sloss[i]) && sloss[i] <= threshold
                ci_upper = svals[i]
            else
                break
            end
        end
    end

    identifiable = crossed_lower && crossed_upper &&
                   ci_lower > lower_bound && ci_upper < upper_bound

    return (ci_lower, ci_upper, identifiable)
end

"""
    profile_confidence_intervals(p::ProfileResult)

All maximal intervals of the profile curve that lie below `p.threshold`
(endpoints by linear interpolation; an interval touching the first/last
profile point is open-ended at that point). The profile CI reported in
`p.ci_lower`/`p.ci_upper` is the component containing the estimate; additional
components indicate a disconnected (multimodal) confidence region."""
function profile_confidence_intervals(p::ProfileResult)
    perm = sortperm(p.values)
    v = p.values[perm]
    l = p.losses[perm]
    thr = p.threshold
    out = Tuple{Float64,Float64}[]
    n = length(v)
    inside = isfinite.(l) .& (l .<= thr)
    i = 1
    while i <= n
        if inside[i]
            j = i
            while j < n && inside[j + 1]
                j += 1
            end
            lo = i > 1 && isfinite(l[i - 1]) ?
                 v[i] - (thr - l[i]) / (l[i - 1] - l[i]) * (v[i] - v[i - 1]) : v[i]
            hi = j < n && isfinite(l[j + 1]) ?
                 v[j] + (thr - l[j]) / (l[j + 1] - l[j]) * (v[j + 1] - v[j]) : v[j]
            push!(out, (lo, hi))
            i = j + 1
        else
            i += 1
        end
    end
    return out
end

# Old discrete CI kept for backward compatibility.
"""
    ple_ci(values, losses, best_val, best_loss, lower_bound, upper_bound, threshold)

Return `(ci_lower, ci_upper, identifiable)` using a discrete contiguous-interval
approach.  Prefer `ple_ci_interpolated` for new code.
"""
function ple_ci(
    values::Vector{Float64},
    losses::Vector{Float64},
    best_val::Float64,
    best_loss::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
    threshold::Float64
)
    perm = sortperm(values)
    svals = values[perm]
    sloss = losses[perm]

    valid = isfinite.(sloss) .& (sloss .<= threshold)
    n = length(svals)

    best_pos = findfirst(==(best_val), svals)
    if best_pos === nothing
        best_pos = clamp(searchsortedfirst(svals, best_val), 1, n)
    end

    left = best_pos
    while left > 1 && valid[left - 1]
        left -= 1
    end

    right = best_pos
    while right < n && valid[right + 1]
        right += 1
    end

    ci_lower = svals[left]
    ci_upper = svals[right]
    identifiable = (ci_lower > lower_bound) && (ci_upper < upper_bound)
    return (ci_lower, ci_upper, identifiable)
end

"""
    write_profiles(path, profiles)

Write a summary of the profile results to a CSV file with columns:
`parameter, index, best_value, best_loss, ci_lower, ci_upper, identifiable, threshold, n_failed, best_found_loss`.

Also writes a companion file `<path>_curves.csv` containing the full profile curves with
columns `parameter, index, value, loss, delta_loss`.
"""
function write_profiles(path, profiles::Vector{ProfileResult})
    df = DataFrame(
        parameter = String[],
        index = Int[],
        best_value = Float64[],
        best_loss = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[],
        identifiable = Bool[],
        threshold = Float64[],
        n_failed = Int[],
        best_found_loss = Float64[]
    )
    for p in profiles
        push!(df, (
            p.parameter, p.index, p.best_value, p.best_loss,
            p.ci_lower, p.ci_upper, p.identifiable, p.threshold,
            p.n_failed, p.best_found_loss
        ))
    end
    CSV.write(path, df)

    curves = to_profile_dataframe(profiles)
    curves_path = _curves_path(path)
    CSV.write(curves_path, curves)

    return df
end

function _curves_path(path)
    dir = dirname(path)
    base, ext = splitext(basename(path))
    return joinpath(dir, base * "_curves" * ext)
end


# ============================================================================

