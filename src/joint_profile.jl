"""    _build_loss_free_cp(prob, idx, fixed_val, cps, free)

Loss over the free parameters with parameter `idx` fixed to `fixed_val` at the
changepoint vector `cps`."""
_build_loss_free_cp(prob, idx, fixed_val, cps, free) =
    _build_loss_free(prob, idx, fixed_val, free, cps)

"""    CPSearchSpace(center, window, n_obs; step=1, min_seg=10)

Discrete changepoint search set used by the joint profile:

    T = { τ : τ_j - center_j ∈ {-window, …, window} (multiples of `step`),
          min_seg ≤ τ_j ≤ n_obs - min_seg,  τ_{j+1} - τ_j ≥ min_seg }.

The set is centred on the *detected* changepoints and is the same for the joint
reference fit and for every profile point, so the joint profile is a proper
profile over a fixed nuisance space (never above the conditional profile,
because the detected changepoints belong to `T`). Membership is tested with
`in`; `enumerate_space` lists all elements (for exhaustive search)."""
struct CPSearchSpace
    center::Vector{Int}
    window::Int
    step::Int
    min_seg::Int
    n_obs::Int
end
CPSearchSpace(center, window, n_obs; step::Int=1, min_seg::Int=10) =
    CPSearchSpace(collect(Int, center), window, max(1, step), min_seg, n_obs)

function Base.in(c::AbstractVector{<:Integer}, S::CPSearchSpace)
    length(c) == length(S.center) || return false
    lo, hi = max(S.min_seg, 1), S.n_obs - S.min_seg
    for j in eachindex(c)
        d = c[j] - S.center[j]
        (abs(d) <= S.window && d % S.step == 0 && lo <= c[j] <= hi) || return false
        j > 1 && c[j] - c[j - 1] < S.min_seg && return false
    end
    return true
end

"""Number of changepoint vectors in the box (before separation constraints)."""
function _space_box_size(S::CPSearchSpace)
    isempty(S.center) && return 1
    lo, hi = max(S.min_seg, 1), S.n_obs - S.min_seg
    return prod(count(d -> lo <= c + d <= hi, (-S.window):S.step:S.window) for c in S.center)
end

function enumerate_space(S::CPSearchSpace; max_combinations::Int=5000)
    isempty(S.center) && return [Int[]]
    n_comb = _space_box_size(S)
    n_comb > max_combinations &&
        error("CP grid too large ($n_comb > $max_combinations); reduce window, increase step, or use cp_search=:local")
    lo, hi = max(S.min_seg, 1), S.n_obs - S.min_seg
    ranges = [[c + d for d in (-S.window):S.step:S.window if lo <= c + d <= hi] for c in S.center]
    grid = Vector{Int}[]
    for c in Iterators.product(ranges...)
        v = collect(Int, c)
        v in S && push!(grid, v)
    end
    sort!(grid)
    return grid
end

"""    _joint_cp_grid(cps0, window, n_obs; step=1, min_seg=10, max_combinations=5000)

All changepoint vectors of `CPSearchSpace(cps0, window, n_obs; step, min_seg)`."""
_joint_cp_grid(cps0::Vector{Int}, window::Int, n_obs::Int; step::Int=1, min_seg::Int=10,
               max_combinations::Int=5000) =
    enumerate_space(CPSearchSpace(cps0, window, n_obs; step=step, min_seg=min_seg);
                    max_combinations=max_combinations)

"""Coordinate search over `space` from `start`: move one changepoint by ±step
while `evalc(c, warm_params)` improves. Returns `(loss, params, cps, visited)`."""
function _coordinate_cp_search(evalc, space::CPSearchSpace, start, warm)
    best_c = collect(Int, start)
    best_l, best_θ = evalc(best_c, warm)
    while true
        moved = false
        for j in eachindex(best_c), s in (-space.step, space.step)
            c = copy(best_c)
            c[j] += s
            c in space || continue
            l, θ = evalc(c, best_θ)
            if l < best_l - 1e-10
                best_l, best_θ, best_c = l, θ, c
                moved = true
            end
        end
        moved || break
    end
    return best_l, best_θ, best_c
end

"""
    _joint_inner(prob, idx, space, optimizer; parallel=false, cp_search=:grid, mode_starts)

Inner solver of the joint profile: minimise over all free parameters *and* over
changepoint vectors in the fixed search set `space`.

* `cp_search = :grid` – exhaustive search: every CP vector is optimised from the
  warm start and the best is returned.
* `cp_search = :local` – coordinate search starting from the warm CP vector:
  move single changepoints by one step while the loss improves. Much cheaper
  (needed for many changepoints) but can miss a distant better configuration."""
function _joint_inner(prob, idx, space::CPSearchSpace, optimizer;
                      parallel::Bool=false, cp_search::Symbol=:grid,
                      mode_starts::Dict{Vector{Int},Vector{Float64}}=Dict{Vector{Int},Vector{Float64}}())
    # optimise at one CP vector from the warm start and, for admissible modes,
    # additionally from the mode's own optimum
    fit_cps(value, cps, warm) = begin
        local l1, θ1 = _optimize_fixed(prob, idx, value, cps, warm, optimizer)
        if haskey(mode_starts, cps)
            local l2, θ2 = _optimize_fixed(prob, idx, value, cps,
                                           _with_value(mode_starts[cps], idx, value), optimizer)
            l2 < l1 && return (l2, θ2)
        end
        (l1, θ1)
    end
    if cp_search === :grid
        grid = enumerate_space(space)
        return function (value, warm, warm_cps)
            local ng = length(grid)
            local losses, params, l, ibest
            losses = fill(Inf, ng)
            params = Vector{Vector{Float64}}(undef, ng)
            if parallel && ng > 1
                Base.Threads.@threads for i in 1:ng
                    losses[i], params[i] = fit_cps(value, grid[i], warm)
                end
            else
                for i in 1:ng
                    losses[i], params[i] = fit_cps(value, grid[i], warm)
                end
            end
            l, ibest = findmin(losses)
            return l, params[ibest], copy(grid[ibest])
        end
    elseif cp_search === :local
        return function (value, warm, warm_cps)
            local start, cache
            start = warm_cps in space ? collect(Int, warm_cps) : collect(Int, prob.changepoints)
            cache = Dict{Vector{Int},Tuple{Float64,Vector{Float64}}}()
            evalc = (c, w) -> get!(() -> fit_cps(value, c, w), cache, c)
            l, θ, c = _coordinate_cp_search(evalc, space, start, warm)
            # also start the search from the detected changepoints and from
            # every admissible mode, so that the result does not depend on the
            # path of the profile alone
            for s0 in Iterators.flatten(([collect(Int, prob.changepoints)], keys(mode_starts)))
                s0 == start && continue
                l2, θ2, c2 = _coordinate_cp_search(evalc, space, s0,
                                                   haskey(mode_starts, s0) ?
                                                   _with_value(mode_starts[s0], idx, value) : warm)
                if l2 < l
                    l, θ, c = l2, θ2, c2
                end
            end
            return l, θ, c
        end
    else
        error("Unknown cp_search=$cp_search; use :grid or :local")
    end
end

"""    joint_reference_fit(prob, window, optimizer; step=1, min_seg=10,
                        parallelize_grid=true, n_starts=3, perturbation=0.3,
                        extra_starts=Vector{Float64}[], seed=nothing)

Jointly re-fit all parameters and the changepoint locations over the fixed CP
set `T` (see `_joint_cp_grid`) to obtain the joint maximum-likelihood estimate.
Each CP vector is optimised from `prob.best_params`, from `n_starts - 1` random
perturbations of it (±`perturbation` of the bound range), and from every
vector in `extra_starts`. The result is never worse than the loss at
`(prob.best_params, prob.changepoints)`.

Returns `(best_loss, best_params, best_cps)`."""
function joint_reference_fit(prob::ODEChangepointPLEProblem, window::Int,
                             optimizer::AbstractPLEOptimizer; kwargs...)
    g = joint_grid_fit(prob, window, optimizer; kwargs...)
    return g.best_loss, g.best_params, g.best_cps
end

"""    joint_grid_fit(prob, window, optimizer; kwargs...)

As `joint_reference_fit`, but returns a named tuple with the joint optimum
(`best_loss`, `best_params`, `best_cps`) **and** the optimum at every changepoint
vector of the search set (`grid`, `losses`, `params`). The per-grid optima are
used by the joint profile to locate secondary modes of the confidence region."""
function joint_grid_fit(prob::ODEChangepointPLEProblem, window::Int,
                        optimizer::AbstractPLEOptimizer;
                        step::Int=1, min_seg::Int=10,
                        parallelize_grid::Bool=true, n_starts::Int=3,
                        perturbation::Float64=0.3,
                        extra_starts::Vector{Vector{Float64}}=Vector{Float64}[],
                        seed::Union{Int,Nothing}=nothing,
                        cp_search::Symbol=:grid)
    space = CPSearchSpace(prob.changepoints, window, prob.n_obs; step=step, min_seg=min_seg)
    x0 = clamp.(prob.best_params, prob.lb, prob.ub)
    lb, ub = prob.lb, prob.ub
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    starts = Vector{Vector{Float64}}([x0])
    for _ in 2:n_starts
        push!(starts, clamp.(x0 .+ perturbation .* (ub .- lb) .* (2.0 .* rand(rng, length(x0)) .- 1.0), lb, ub))
    end
    append!(starts, extra_starts)

    # NOTE: all variables assigned inside this closure are declared `local`;
    # otherwise Julia shares them with same-named variables of the enclosing
    # function, which is a data race when grid points run on several threads.
    function fit_at(cps)
        local f = x -> evaluate_loss(prob, x, cps)
        local best_l, best_x = Inf, copy(x0)
        local l0, l, x
        for xs in starts
            l0 = f(xs)
            if l0 < best_l
                best_l, best_x = l0, copy(xs)
            end
            l, x = try
                optimize_ple(f, xs, lb, ub, optimizer)
            catch e
                @debug "Joint reference fit failed for cps=$cps" exception = e
                (Inf, xs)
            end
            if isfinite(l) && l < best_l
                best_l, best_x = l, collect(Float64, x)
            end
        end
        return best_l, best_x
    end

    if cp_search === :local
        # coordinate search from the detected changepoints; every visited CP
        # vector is optimised from all starts (and from the current optimum)
        visited = Dict{Vector{Int},Tuple{Float64,Vector{Float64}}}()
        evalc = (c, w) -> get!(visited, c) do
            local l, x, f, lw, l2, x2
            l, x = fit_at(c)
            f = θ -> evaluate_loss(prob, θ, c)
            lw = f(w)
            lw < l && ((l, x) = (lw, copy(w)))
            l2, x2 = try
                optimize_ple(f, w, lb, ub, optimizer)
            catch
                (Inf, w)
            end
            (isfinite(l2) && l2 < l) ? (Float64(l2), collect(Float64, x2)) : (l, x)
        end
        _coordinate_cp_search(evalc, space, collect(Int, prob.changepoints), x0)
        grid = sort(collect(keys(visited)))
        losses = [visited[c][1] for c in grid]
        params = [visited[c][2] for c in grid]
    else
        grid = enumerate_space(space)
        losses = fill(Inf, length(grid))
        params = Vector{Vector{Float64}}(undef, length(grid))
    end
    ng = length(grid)
    if cp_search === :local
        # already evaluated
    elseif parallelize_grid && !uses_internal_threads(optimizer) && ng > 1
        Base.Threads.@threads for i in 1:ng
            losses[i], params[i] = fit_at(grid[i])
        end
    else
        for i in 1:ng
            losses[i], params[i] = fit_at(grid[i])
        end
    end
    best_loss, ibest = findmin(losses)
    best_params, best_cps = params[ibest], copy(grid[ibest])

    # Polish the winner from its own optimum.
    f_best = x -> evaluate_loss(prob, x, best_cps)
    l2, x2 = try
        optimize_ple(f_best, best_params, lb, ub, optimizer)
    catch
        (Inf, best_params)
    end
    if isfinite(l2) && l2 < best_loss
        best_loss, best_params = l2, collect(Float64, x2)
    end

    # Never worse than the conditional fit, which is a member of the grid.
    l_cond = evaluate_loss(prob, prob.best_params, prob.changepoints)
    if l_cond < best_loss
        @warn "Joint reference fit ($best_loss) is worse than the conditional fit ($l_cond); using the conditional fit. Increase n_starts or use a stronger optimizer."
        best_loss, best_params, best_cps = l_cond, copy(prob.best_params), copy(prob.changepoints)
    end
    # keep the per-grid optima consistent with the (possibly polished) winner
    iw = findfirst(==(best_cps), grid)
    if iw !== nothing && best_loss < losses[iw]
        losses[iw], params[iw] = best_loss, copy(best_params)
    end
    return (best_loss=best_loss, best_params=best_params, best_cps=best_cps,
            grid=grid, losses=losses, params=params)
end

"""    profile_parameter_joint(prob::ODEChangepointPLEProblem, idx;
                            options=ProfileOptions(), window=7, step=1, min_seg=10,
                            optimizer=default_multistart_bobyqa_config(),
                            threshold=nothing, delta=nothing, joint_reference=nothing,
                            cp_search=:grid, parallelize_grid=true, n_points=nothing)

Joint profile likelihood of parameter `idx`: at every profile point the loss is
minimised over all other parameters **and** over the changepoint locations in
the fixed set `T` (all CP vectors within `±window` of the detected
changepoints, spacing `step`, minimum segment length `min_seg`).

The profile is anchored at the joint MLE over `T` (`joint_reference`, computed
with `joint_reference_fit` if not supplied) and the threshold is
`joint_reference_loss + delta` (default `delta = 3.8415`). Every joint profile
value is a minimum over a superset of the conditional problem, so the joint
profile never lies above the conditional profile; because the joint reference
can also be lower than the conditional one, joint intervals are usually, but
not necessarily, wider than conditional intervals.

Uses the same adaptive stepping, smoothing, and bisection refinement as
`profile_parameter`. Returns a `ProfileResult` whose `cps_trace` records the
optimal changepoints at each profile point."""
function profile_parameter_joint(
    prob::ODEChangepointPLEProblem, idx::Int;
    options::ProfileOptions=ProfileOptions(),
    n_points::Union{Int,Nothing}=nothing,
    window::Int=7,
    step::Int=1,
    min_seg::Int=10,
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    threshold::Union{Real,Nothing}=nothing,
    delta::Union{Real,Nothing}=nothing,
    parallelize_grid::Bool=true,
    cp_search::Symbol=:grid,
    joint_reference::Union{Nothing,Tuple{Float64,Vector{Float64},Vector{Int}}}=nothing,
    grid_fit=nothing
)
    isempty(prob.best_params) && error("ODEChangepointPLEProblem.best_params must be set")
    opt = _maybe_polish(optimizer, options)
    verified = _verified_reference_loss(prob)
    margin = _margin(threshold, delta, verified)

    space = CPSearchSpace(prob.changepoints, window, prob.n_obs; step=step, min_seg=min_seg)
    if grid_fit === nothing && joint_reference === nothing
        grid_fit = joint_grid_fit(prob, window, opt; step=step, min_seg=min_seg,
                                  parallelize_grid=parallelize_grid, cp_search=cp_search)
    end
    ref = joint_reference !== nothing ? joint_reference :
          (grid_fit.best_loss, grid_fit.best_params, grid_fit.best_cps)
    ref_loss, ref_params, ref_cps = ref
    if verified < ref_loss
        ref_loss, ref_params, ref_cps = verified, copy(prob.best_params), copy(prob.changepoints)
    end
    ref_cps in space || error("joint reference changepoints $ref_cps are not in the search set")

    # Secondary modes: CP vectors whose own optimum is (nearly) admissible. The
    # profile must reach their parameter values, and their optima are used as
    # additional warm starts, so disconnected confidence regions are found.
    mode_starts = Dict{Vector{Int},Vector{Float64}}()
    reach = (Inf, -Inf)
    if grid_fit !== nothing
        for (c, l, θ) in zip(grid_fit.grid, grid_fit.losses, grid_fit.params)
            isfinite(l) || continue
            if l <= ref_loss + options.stop_margin_factor * margin
                mode_starts[c] = θ
            end
            if l <= ref_loss + margin
                reach = (min(reach[1], θ[idx]), max(reach[2], θ[idx]))
            end
        end
    end

    par = parallelize_grid && !uses_internal_threads(opt)
    inner = _joint_inner(prob, idx, space, opt; parallel=par, cp_search=cp_search,
                         mode_starts=mode_starts)
    samplesize = n_points === nothing ? options.samplesize : n_points

    result = nothing
    for update in 0:options.max_reference_updates
        result, _ = _profile_engine(prob, idx, ref_params, ref_cps, ref_loss, margin,
                                    inner, options; samplesize=samplesize, reach=reach)
        if options.allow_better_optimum && update < options.max_reference_updates &&
           result.best_found_loss < ref_loss - options.better_optimum_tol
            @warn "Joint profile of $(prob.param_names[idx]) found loss $(result.best_found_loss) < joint reference $ref_loss; restarting from the better optimum."
            _, imin = findmin(result.losses)
            r = refit_reference(_with_reference(prob, result.best_found_params,
                                                result.best_found_loss;
                                                changepoints=result.cps_trace[imin]);
                                optimizer=opt)
            ref_loss, ref_params, ref_cps = r.best_loss, r.best_params, r.changepoints
            continue
        end
        break
    end
    return result
end

"""    profile_all_parameters_joint(prob; options=ProfileOptions(), window=7, step=1,
                                 min_seg=10, optimizer=default_multistart_bobyqa_config(),
                                 indices=nothing, threshold=nothing, delta=nothing,
                                 parallel_over_parameters=true, cp_search=:grid,
                                 reference_extra_starts=Vector{Float64}[], n_points=nothing,
                                 joint_reference=nothing, grid_fit=nothing)

Joint profiles of every parameter (or of `indices`), all anchored at one joint
reference fit. If any profile finds a better optimum, the joint reference is
replaced and all profiles are recomputed (`options.allow_better_optimum`).

Keyword arguments (besides those of [`profile_parameter_joint`](@ref)):
- `window`, `step`, `min_seg`: the changepoint search set — every changepoint
  may move within `±window` of its detected position in steps of `step`, and
  no segment may be shorter than `min_seg`. A window that is too narrow cuts
  the profile off; a wide window with several changepoints makes the grid large.
- `cp_search`: `:grid` optimises every changepoint vector of the search set at
  each profile point (exact, cost grows with the grid size); `:local` uses a
  coordinate search from the current changepoints (much cheaper, may miss
  distant modes).
- `parallelize_grid`: fit the changepoint vectors of the grid in parallel
  threads (ignored for optimisers that are threaded themselves).
- `parallel_over_parameters`: profile the parameters in parallel threads.
- `reference_extra_starts`: additional starting vectors for the joint
  reference fit (useful when the default starts miss the global optimum).
- `joint_reference`: a precomputed `(loss, params, cps)` joint optimum; skips
  the reference fit.
- `grid_fit`: a precomputed result of [`joint_grid_fit`](@ref); skips the grid
  fit and supplies the secondary modes used to find disconnected intervals."""
function profile_all_parameters_joint(
    prob::ODEChangepointPLEProblem;
    options::ProfileOptions=ProfileOptions(),
    n_points::Union{Int,Nothing}=nothing,
    window::Int=7,
    step::Int=1,
    min_seg::Int=10,
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    indices::Union{Nothing,Vector{Int}}=nothing,
    threshold::Union{Real,Vector{Float64},Nothing}=nothing,
    delta::Union{Real,Vector{Float64},Nothing}=nothing,
    parallel_over_parameters::Bool=true,
    parallelize_grid::Bool=true,
    cp_search::Symbol=:grid,
    reference_extra_starts::Vector{Vector{Float64}}=Vector{Float64}[],
    joint_reference::Union{Nothing,Tuple{Float64,Vector{Float64},Vector{Int}}}=nothing,
    grid_fit=nothing
)
    idxs = indices === nothing ? collect(1:length(prob.best_params)) : indices
    opt = _maybe_polish(optimizer, options)
    verified = _verified_reference_loss(prob)
    margins = [_margin(threshold isa Vector ? threshold[k] : threshold,
                       delta isa Vector ? delta[k] : delta, verified)
               for k in eachindex(idxs)]

    gfit = grid_fit !== nothing ? grid_fit :
           joint_grid_fit(prob, window, opt; step=step, min_seg=min_seg,
                          parallelize_grid=true, extra_starts=reference_extra_starts,
                          cp_search=cp_search)
    ref = joint_reference !== nothing ? joint_reference :
          (gfit.best_loss, gfit.best_params, gfit.best_cps)
    @info "Joint reference: loss=$(ref[1]) at cps=$(ref[3]) (conditional loss $(verified) at cps=$(prob.changepoints))"

    inner_options = _with_options(options; allow_better_optimum=false, polish=false)
    profiles = Vector{ProfileResult}(undef, length(idxs))
    threaded_params = parallel_over_parameters && !uses_internal_threads(opt)
    for update in 0:options.max_reference_updates
        run_one = k -> profile_parameter_joint(prob, idxs[k]; options=inner_options,
                                               n_points=n_points, window=window, step=step,
                                               min_seg=min_seg, optimizer=opt,
                                               delta=margins[k],
                                               parallelize_grid=parallelize_grid && !threaded_params,
                                               cp_search=cp_search, joint_reference=ref,
                                               grid_fit=gfit)
        if threaded_params
            Base.Threads.@threads for k in eachindex(idxs)
                profiles[k] = run_one(k)
            end
        else
            for k in eachindex(idxs)
                profiles[k] = run_one(k)
            end
        end
        best_found, kbest = findmin([p.best_found_loss for p in profiles])
        if options.allow_better_optimum && update < options.max_reference_updates &&
           best_found < ref[1] - options.better_optimum_tol
            p = profiles[kbest]
            _, imin = findmin(p.losses)
            @warn "Joint profile of $(p.parameter) found loss $best_found < joint reference $(ref[1]); re-profiling all parameters."
            r = refit_reference(_with_reference(prob, p.best_found_params, best_found;
                                                changepoints=p.cps_trace[imin]);
                                optimizer=opt)
            ref = (r.best_loss, r.best_params, r.changepoints)
            continue
        end
        break
    end
    sort!(profiles, by=p -> p.index)
    return profiles
end

"""    JointPLEConfig

Configuration bundle for joint adaptive PLE."""
struct JointPLEConfig
    n_points::Int
    window::Int
    step::Int
    optimizer::AbstractPLEOptimizer
    parallel_over_parameters::Bool
    parallelize_grid::Bool
end

function JointPLEConfig(;
    n_points::Int=100,
    window::Int=7,
    step::Int=1,
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    parallel_over_parameters::Bool=true,
    parallelize_grid::Bool=true
)
    return JointPLEConfig(n_points, window, step, optimizer, parallel_over_parameters, parallelize_grid)
end
