# Profiling options

This page explains the arguments that control a profile and how each one
changes the result or the runtime. The one-line definitions are also in the
docstrings ([API](api.md)).

## Where the arguments go

| Function | Arguments |
|---|---|
| `ProfileOptions(...)` | 16 settings for stepping, reliability and CI precision (below) |
| `profile_parameter`, `profile_all_parameters` | `options`, `optimizer`, `delta` / `threshold`, `method`, `n_points` (+ `indices`) |
| `profile_changepoint`, `profile_all_changepoints` | `window`, `min_seg`, `optimizer`, `polish` |
| `profile_parameter_joint`, `profile_all_parameters_joint` | the above + `window`, `step`, `min_seg`, `cp_search`, `parallelize_grid`, `parallel_over_parameters`, `reference_extra_starts`, `joint_reference`, `grid_fit` |
| `extend_profile`, `extend_profiles` | `side`, `left`, `right`, `mode`, `only_open`, `options`, `optimizer`, `samplesize` |

## Confidence level and optimiser

- **`delta`** (default `3.8415` = χ²₁ 95 %) or an absolute **`threshold`**: the
  interval is the set of values whose profile loss is at most
  `reference_loss + delta`. This is only a calibrated 95 % interval if the loss
  is `-2 log L` of a correctly specified likelihood.
- **`optimizer`**: the inner optimiser run at every profile point. It is the
  main driver of runtime and of profile quality: a weak optimiser gives jagged
  profiles and spuriously narrow intervals (a point that is not re-optimised
  well lies too high). `MultiStartBOBYQAConfig` is the default;
  `HybridOptimizerConfig` (global + local) suits multimodal losses.

## `ProfileOptions`: stepping (range and resolution)

The profile walks outward from the estimate in both directions with adaptive
steps, as in Data2Dynamics (Raue et al. 2009).

| Option | Default | Effect |
|---|---|---|
| `samplesize` | 100 | Maximum number of steps per direction. If it is too small the walk can stop before the threshold is reached, and the interval is reported open (non-identifiable). |
| `rel_step_increase` | 0.1 | Target loss increase per step, as a fraction of `delta`. Smaller values give more points and smoother curves, at the cost of more fits. |
| `step_factor` | 1.5 | Factor by which the step grows after a successful step and shrinks after a rejected trial step. |
| `min_step_factor` | 1e-6 | Smallest step, as a fraction of the parameter range (`ub - lb`). |
| `max_step_factor` | 0.5 | Largest step, as a fraction of the parameter range. Large values let flat profiles reach the bounds quickly. |
| `stop_margin_factor` | 1.2 | A direction stops once the loss is `1.2 · delta` above the reference (or a bound is reached). |
| `max_trial_redos` | 20 | How often a trial step may be shrunk before the smallest admissible step is accepted. |

Because steps are relative to the parameter range, the same settings work for
parameters on different scales.

## `ProfileOptions`: reliability

| Option | Default | Effect |
|---|---|---|
| `polish` | `true` | Polish every inner fit with a local optimiser. Fewer spurious bumps; slower. |
| `smooth_jumps`, `jump_tol` | `true`, 0.1 | After the walk, points that lie higher than their outer neighbour by more than `jump_tol · delta` are re-optimised from that neighbour (D2D `pleSmooth`). Removes jumps caused by poor local optima. |
| `refit_reference` | `true` | Re-fit the supplied optimum before profiling. A sub-optimal reference shifts the threshold and invalidates every interval. |
| `allow_better_optimum` | `true` | If a profile finds a loss below the reference, the reference is replaced and the profiles are recomputed. Correct, but can multiply the runtime. |
| `max_reference_updates` | 3 | Maximum number of such restarts. |
| `better_optimum_tol` | 1e-3 | How much lower a loss must be to count as a better optimum. |

## `ProfileOptions`: interval precision

| Option | Default | Effect |
|---|---|---|
| `refine_ci` | `true` | Refine each threshold crossing by bisection with re-optimisation, so the interval endpoints do not depend on the spacing of the profile points. |
| `ci_refine_iters` | 12 | Maximum bisection steps per crossing. |

## Changepoint and joint profiles

- **`window`**: a changepoint is profiled (or, in joint profiles, re-optimised)
  within `±window` time indices of its detected position. If the profile is
  still below the threshold at the window edge, the changepoint interval is
  reported as censored (not identifiable) — widen the window.
- **`min_seg`**: candidate positions that would create a segment shorter than
  `min_seg` are skipped.
- **`step`** (joint): spacing of candidate changepoint positions.
- **`cp_search`** (joint): `:grid` fits every changepoint vector of the search
  set at each profile point (exact; the cost grows with the grid, which grows
  quickly with several changepoints); `:local` uses a coordinate search from the
  current changepoints (much cheaper, may miss distant modes).
- **`parallelize_grid`**, **`parallel_over_parameters`**: thread-level
  parallelism; ignored for optimisers that are threaded themselves.
- **`reference_extra_starts`**, **`joint_reference`**, **`grid_fit`**: extra
  starts for, or precomputed results of, the joint reference fit.

## Extending a profile beyond the bounds

A profile that reaches a parameter bound below the threshold is reported as
non-identifiable. That can mean the likelihood is flat in that direction, or
only that the bound is too tight. `extend_profile` continues such a profile
beyond the bound and tells the two cases apart.

```julia
profiles = profile_all_parameters(prob; options=options, optimizer=optimizer)

# extend every open profile by 50 % of its range on both sides
extended = extend_profiles(prob, profiles; left=0.5, right=0.5,
                           options=options, optimizer=optimizer)
df = extension_summary(profiles, extended)
n_new = count(df.identifiable_after .& .!df.identifiable_before)

plot_profiles(extended; reference_bounds=(prob.lb, prob.ub))   # original bounds dotted
```

- **Existing points are kept**: the walk continues from the outermost computed
  point on each side, warm-started from its optimised parameters; only the new
  region is computed. Smoothing and CI refinement then run on the merged
  profile.
- **Relative amounts, not raw values**: `left` and `right` are fractions
  (`0.5` = 50 %).
  - `mode = :range` (default): extend by a fraction of the current range,
    `[lo - left·(hi-lo), hi + right·(hi-lo)]`.
  - `mode = :relative`: extend by a fraction of the bound value,
    `[lo / (1+left), hi · (1+right)]`; the lower bound stays positive, which
    suits positive rate constants (requires positive bounds).
- **Sides**: `side = :left`, `:right` or `:both`; with `:both` the two sides
  may use different amounts (e.g. `left=0.2, right=1.0`). By default
  (`only_open = true`) a side is only extended if the profile is open there
  (see `profile_open_sides`); closed profiles are returned unchanged.
- Only the profiled parameter may leave its bounds; all other parameters keep
  their bounds during re-optimisation.
- The reference loss and threshold are unchanged. If the extension finds a loss
  below the reference, a warning is issued: the optimum lies outside the
  original bounds and the analysis should be repeated with wider bounds.
- Interpretation: an interval that closes only beyond the original bounds means
  the data do determine the parameter, but outside the range that was allowed.
  An interval that stays open after extension indicates a flat likelihood
  (practical or structural non-identifiability).
- Supported for conditional and baseline profiles; not for joint profiles.
