"""
    profile_changepoint(prob::ODEChangepointPLEProblem, cp_idx::Int;
                        window=7, min_seg=10,
                        optimizer=default_multistart_bobyqa_config(), polish=true)

Profile the location of the `cp_idx`-th changepoint: move it over
`original_cp ± window` (the other changepoints stay fixed), re-optimise all
parameters at every candidate location, and record the loss.

Candidates closer than `min_seg` to the data boundaries or to a neighbouring
changepoint are skipped. Candidates are visited outward from the detected
location; each is optimised both from `prob.best_params` and from the optimum
of its inner neighbour, and the better result is kept.

The profile reference `best_loss` is the minimum over the loss at
`prob.best_params` and all profile values, so `losses .- best_loss ≥ 0`.
Returns a `CPProfileResult`."""
function profile_changepoint(
    prob::ODEChangepointPLEProblem, cp_idx::Int;
    window::Int=7,
    min_seg::Int=10,
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    polish::Bool=true
)
    cps = collect(Int, prob.changepoints)
    1 <= cp_idx <= length(cps) || error("cp_idx $cp_idx out of range")
    original_cp = cps[cp_idx]
    opt = polish && !(optimizer isa HybridOptimizerConfig) ?
          HybridOptimizerConfig(global_optimizer=optimizer) : optimizer

    lo_t = max(min_seg, 1)
    hi_t = prob.n_obs - min_seg
    cp_idx > 1 && (lo_t = max(lo_t, cps[cp_idx - 1] + min_seg))
    cp_idx < length(cps) && (hi_t = min(hi_t, cps[cp_idx + 1] - min_seg))

    l_ref = evaluate_loss(prob, prob.best_params, cps)
    results = Dict{Int,Tuple{Float64,Vector{Float64}}}()

    fit_at(cand, warm) = begin
        local new_cps, f, best_l, best_x, l0, l, x
        new_cps = copy(cps)
        new_cps[cp_idx] = cand
        f = θ -> evaluate_loss(prob, θ, new_cps)
        best_l, best_x = Inf, copy(prob.best_params)
        for x0 in (prob.best_params, warm)
            x0 = clamp.(x0, prob.lb, prob.ub)
            l0 = f(x0)
            l, x = try
                optimize_ple(f, x0, prob.lb, prob.ub, opt)
            catch e
                @debug "CP profile optimisation failed at $cand" exception = e
                (Inf, x0)
            end
            if !(isfinite(l) && l <= l0)
                l, x = l0, x0
            end
            if l < best_l
                best_l, best_x = l, collect(Float64, x)
            end
            warm === prob.best_params && break
        end
        best_l, best_x
    end

    if lo_t <= original_cp <= hi_t
        results[original_cp] = fit_at(original_cp, prob.best_params)
    end
    for direction in (-1, 1)
        warm = haskey(results, original_cp) ? results[original_cp][2] : copy(prob.best_params)
        for d in 1:window
            cand = original_cp + direction * d
            (lo_t <= cand <= hi_t) || continue
            results[cand] = fit_at(cand, warm)
            isfinite(results[cand][1]) && (warm = results[cand][2])
        end
    end

    candidate_cps = sort(collect(keys(results)))
    losses = [results[c][1] for c in candidate_cps]
    finite_losses = filter(isfinite, losses)
    best_loss = isempty(finite_losses) ? l_ref : min(l_ref, minimum(finite_losses))
    return CPProfileResult(cp_idx, original_cp, window, candidate_cps, losses, best_loss)
end

"""
    profile_all_changepoints(prob::ODEChangepointPLEProblem; window=7, min_seg=10,
                             optimizer=default_multistart_bobyqa_config(), polish=true)

Profile every changepoint in `prob.changepoints` (in parallel over
changepoints unless the optimiser is threaded). Returns a
`Vector{CPProfileResult}`."""
function profile_all_changepoints(
    prob::ODEChangepointPLEProblem;
    window::Int=7,
    min_seg::Int=10,
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    polish::Bool=true
)
    K = length(prob.changepoints)
    out = Vector{CPProfileResult}(undef, K)
    if uses_internal_threads(optimizer)
        for j in 1:K
            out[j] = profile_changepoint(prob, j; window=window, min_seg=min_seg,
                                         optimizer=optimizer, polish=polish)
        end
    else
        Base.Threads.@threads for j in 1:K
            out[j] = profile_changepoint(prob, j; window=window, min_seg=min_seg,
                                         optimizer=optimizer, polish=polish)
        end
    end
    return out
end

"""
    cp_confidence_set(cp_prof::CPProfileResult[, threshold]; delta=3.8415)

Confidence set for a changepoint location: all candidate locations whose
profile loss is at most `threshold` (default `cp_prof.best_loss + delta`).

Returns a named tuple with fields
* `set`              – all candidate locations in the confidence set,
* `lower`, `upper`   – the contiguous component containing the profile minimum,
* `components`       – all contiguous components (vectors of locations),
* `censored_lower`, `censored_upper` – `true` if the set reaches the first/last
  candidate examined, i.e. the profile was not observed to rise above the
  threshold on that side (the set may extend beyond the window),
* `identifiable`     – `true` if the set is bounded on both sides inside the
  window (`!censored_lower && !censored_upper`),
* `best_cp`          – candidate with the minimum profile loss.

Note: the χ²₁ calibration is heuristic for a discrete changepoint location
(non-regular parameter); a parametric bootstrap gives calibrated sets."""
function cp_confidence_set(cp_prof::CPProfileResult, threshold::Union{Real,Nothing}=nothing;
                           delta::Real=CHISQ_95)
    thr = threshold === nothing ? cp_prof.best_loss + delta : Float64(threshold)
    cands = cp_prof.candidate_cps
    losses = cp_prof.losses
    n = length(cands)
    n == 0 && return (set=Int[], lower=missing, upper=missing, components=Vector{Int}[],
                      censored_lower=true, censored_upper=true, identifiable=false,
                      best_cp=missing)
    perm = sortperm(cands)
    cands = cands[perm]
    losses = losses[perm]
    valid = isfinite.(losses) .& (losses .<= thr)
    _, imin = findmin(map(l -> isfinite(l) ? l : Inf, losses))

    components = Vector{Int}[]
    cur = Int[]
    for i in 1:n
        # contiguous in index *and* in location (no skipped candidates)
        if valid[i] && (isempty(cur) || cands[i] == cands[i - 1] + 1)
            push!(cur, cands[i])
        elseif valid[i]
            push!(components, cur)
            cur = [cands[i]]
        elseif !isempty(cur)
            push!(components, cur)
            cur = Int[]
        end
    end
    isempty(cur) || push!(components, cur)

    set = cands[valid]
    main = something(findfirst(c -> cands[imin] in c, components), 0)
    lower = main == 0 ? missing : first(components[main])
    upper = main == 0 ? missing : last(components[main])
    censored_lower = valid[1]
    censored_upper = valid[n]
    return (set=set, lower=lower, upper=upper, components=components,
            censored_lower=censored_lower, censored_upper=censored_upper,
            identifiable=!isempty(set) && !censored_lower && !censored_upper,
            best_cp=cands[imin])
end

"""
    cp_ci(cp_prof::CPProfileResult, threshold)

Return `(ci_lower, ci_upper, identifiable)` for a changepoint profile, where
`[ci_lower, ci_upper]` is the contiguous component of the confidence set that
contains the profile minimum and `identifiable` is `true` only if the whole
confidence set is bounded inside the profiling window (not censored at either
window edge). See `cp_confidence_set` for secondary components."""
function cp_ci(cp_prof::CPProfileResult, threshold::Real)
    cs = cp_confidence_set(cp_prof, threshold)
    return (cs.lower, cs.upper, cs.identifiable)
end

"""
    write_cp_profiles(path, cp_profiles)

Write a long-format CSV of changepoint profile results with columns:
`cp_index, original_cp, candidate_cp, loss, delta_loss`.
"""
function write_cp_profiles(path, cp_profiles::Vector{CPProfileResult})
    df = DataFrame(
        cp_index = Int[],
        original_cp = Int[],
        candidate_cp = Int[],
        loss = Float64[],
        delta_loss = Float64[]
    )
    for cp in cp_profiles
        for (cand, loss) in zip(cp.candidate_cps, cp.losses)
            push!(df, (cp.cp_index, cp.original_cp, cand, loss, loss - cp.best_loss))
        end
    end
    CSV.write(path, df)
    return df
end
