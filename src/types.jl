"""    AbstractPLEProblem

Abstract supertype for all profile-likelihood problems."""
abstract type AbstractPLEProblem end

"""    AbstractPLEOptimizer

Abstract supertype for PLE inner-optimiser backends."""
abstract type AbstractPLEOptimizer end

"""    ODEChangepointPLEProblem{F,M,OF,D,L}

Container for a profile-likelihood problem involving an ODE model with a fixed
set of changepoints.

# Fields
- `objective`: function `(params, changepoints) -> scalar loss`. If provided, it
  is used directly by the profiler. Otherwise the package falls back to `simulator`
  or to the generic ODE segment solver.
- `simulator`: optional function `(params, changepoints) -> simulation matrix`.
- `ode_function`: ODE RHS `(du, u, p, t)` for the generic segment solver. Inside
  segment `j` the solver passes `p = [θ_global; θ_segment_j]`, i.e. the
  `n_global` global parameters followed by the `n_segment_specific` parameters
  of the active segment, so the right-hand side never needs to know the
  changepoint locations.
- `u0`: initial condition for the generic solver.
- `tspan`: time span `(t0, tf)` for the generic solver.
- `data`: observed data, channels × time.
- `loss_fn`: likelihood or loss callable `(sim, data) -> scalar`.
- `changepoints`: sorted vector of detected changepoint indices.
- `best_params`: best-fit parameter vector for the current changepoints.
- `best_loss`: loss evaluated at `best_params`.
- `lb`, `ub`: lower and upper bounds for all parameters (global + segment-specific).
- `param_names`: vector of parameter labels.
- `n_global`: number of global parameters.
- `n_segment_specific`: number of segment-specific parameters per segment.
- `n_obs`: number of time points.
"""
struct ODEChangepointPLEProblem{F,M,OF,D,L}
    objective::F
    simulator::M
    ode_function::OF
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    data::D
    loss_fn::L
    changepoints::Vector{Int}
    best_params::Vector{Float64}
    best_loss::Float64
    lb::Vector{Float64}
    ub::Vector{Float64}
    param_names::Vector{String}
    n_global::Int
    n_segment_specific::Int
    n_obs::Int
end

function ODEChangepointPLEProblem(;
    objective = nothing,
    simulator = nothing,
    ode_function = nothing,
    u0 = Float64[],
    tspan = (0.0, 0.0),
    data,
    loss_fn,
    changepoints,
    best_params,
    best_loss = NaN,
    lb,
    ub,
    param_names = String[],
    n_global,
    n_segment_specific,
    n_obs = size(data, 2)
)
    @assert length(lb) == length(ub) "lower and upper bounds must have the same length"
    @assert length(best_params) == length(lb) "best_params must match bounds length"
    n_expected = n_global + n_segment_specific * (length(changepoints) + 1)
    @assert length(best_params) == n_expected "best_params has length $(length(best_params)) but n_global + n_segment_specific * (n_changepoints + 1) = $n_expected"
    @assert all(lb .<= ub) "lower bounds must not exceed upper bounds"
    if isempty(param_names)
        param_names = parameter_labels(n_global, n_segment_specific, length(changepoints))
    end
    return ODEChangepointPLEProblem(
        objective, simulator, ode_function, u0, tspan,
        data, loss_fn, sort(collect(Int, changepoints)),
        collect(Float64, best_params), Float64(best_loss),
        collect(Float64, lb), collect(Float64, ub), collect(String, param_names),
        n_global, n_segment_specific, n_obs
    )
end

"""    ProfileResult

Result of profiling a single scalar parameter.

# Fields
- `parameter`: label of the profiled parameter.
- `index`: position in the parameter vector.
- `best_value`: MLE value of the profiled parameter.
- `best_loss`: reference loss (MLE).
- `ci_lower`, `ci_upper`: approximate 95% CI bounds.
- `identifiable`: `true` if the profile crosses the threshold on both sides.
- `threshold`: `best_loss + χ²(1, 0.95)`.
- `values`, `losses`: the full profile curve.
- `n_failed`: number of grid points where the inner optimizer returned `Inf`.
- `best_found_loss`: lowest loss found during profiling (may be < `best_loss`
  if the reference optimum was not the true MLE).
- `best_found_params`: full parameter vector at `best_found_loss`.
- `cps_trace`: changepoint vector used at each profile point (constant for
  conditional profiles, re-optimised for joint profiles).
- `params_trace`: full optimised parameter vector at each profile point (empty
  for results built without it). Used by `extend_profile` to continue a profile
  without recomputing it.
- `profile_lb`, `profile_ub`: range over which the parameter was profiled
  (the problem bounds, or the extended range after `extend_profile`);
  `-Inf`/`Inf` when unknown.
"""
struct ProfileResult
    parameter::String
    index::Int
    best_value::Float64
    best_loss::Float64
    ci_lower::Float64
    ci_upper::Float64
    identifiable::Bool
    threshold::Float64
    values::Vector{Float64}
    losses::Vector{Float64}
    n_failed::Int
    best_found_loss::Float64
    best_found_params::Vector{Float64}
    cps_trace::Vector{Vector{Int}}
    params_trace::Vector{Vector{Float64}}
    profile_lb::Float64
    profile_ub::Float64
end

function ProfileResult(parameter, index, best_value, best_loss, ci_lower, ci_upper,
                       identifiable, threshold, values, losses, n_failed,
                       best_found_loss, best_found_params, cps_trace)
    return ProfileResult(parameter, index, best_value, best_loss, ci_lower, ci_upper,
                         identifiable, threshold, values, losses, n_failed,
                         best_found_loss, best_found_params, cps_trace,
                         Vector{Float64}[], -Inf, Inf)
end

function ProfileResult(parameter, index, best_value, best_loss, ci_lower, ci_upper,
                       identifiable, threshold, values, losses, n_failed,
                       best_found_loss, best_found_params)
    return ProfileResult(parameter, index, best_value, best_loss, ci_lower, ci_upper,
                         identifiable, threshold, values, losses, n_failed,
                         best_found_loss, best_found_params,
                         [Int[] for _ in eachindex(values)])
end

"""    CPProfileResult

Result of profiling the location of a single changepoint.

`best_loss` is the reference loss of the profile: the minimum of the loss at the
supplied fit and of all re-optimised losses in the window, so that
`losses .- best_loss >= 0`."""
struct CPProfileResult
    cp_index::Int
    original_cp::Int
    window::Int
    candidate_cps::Vector{Int}
    losses::Vector{Float64}
    best_loss::Float64
end

"""
    parameter_labels(n_global, n_segment_specific, n_cps)

Generate default parameter labels: global parameters first, then segment-specific
parameters for each segment."""
function parameter_labels(n_global::Int, n_segment_specific::Int, n_cps::Int)
    labels = String[]
    for i in 1:n_global
        push!(labels, "par_$(i)")
    end
    for s in 1:(n_cps + 1)
        for i in 1:n_segment_specific
            push!(labels, "par_$(n_global + i)_seg$(s)")
        end
    end
    return labels
end

function parameter_labels(prob::ODEChangepointPLEProblem)
    return prob.param_names
end

Base.length(prob::ODEChangepointPLEProblem) = length(prob.best_params)

"""    ProfileOptions

D2D-style configuration for 1-D profile-likelihood exploration.

# Fields
- `samplesize`: maximum number of outward steps per direction (D2D default: 100).
- `rel_step_increase`: target initial Δloss / threshold_margin for choosing the next
  step size (D2D default: 0.1).
- `step_factor`: factor by which the step is increased/decreased during step-size
  bracketing (D2D default: 1.5).
- `min_step_factor`: minimum step as a fraction of the parameter range.
- `max_step_factor`: maximum step as a fraction of the parameter range.
- `stop_margin_factor`: stop profiling in a direction when Δloss exceeds
  `stop_margin_factor * threshold_margin` (D2D default: 1.2).
- `polish`: if `true`, polish the result of every global/metaheuristic inner
  optimization with a local optimizer (LBFGS/BOBYQA).
- `allow_better_optimum`: if `true` and profiling finds a loss lower than the
  reference `best_loss`, replace the reference with the better optimum and
  restart profiling (see `max_reference_updates`).
- `smooth_jumps`: if `true`, apply D2D `pleSmooth`-style post-processing to remove
  local jumps caused by sub-optimal inner optimizations.
- `jump_tol`: Δloss drop used to flag a local jump during smoothing.
- `max_trial_redos`: maximum number of step-size reductions per outward step.
- `refine_ci`: if `true`, refine each threshold crossing by bisection with
  re-optimisation, so the reported CI endpoint does not depend on the profile
  grid spacing.
- `ci_refine_iters`: maximum number of bisection steps per crossing.
- `refit_reference`: if `true`, re-optimise all free quantities from
  `best_params` before profiling so the profile is anchored at an optimum.
- `max_reference_updates`: when `allow_better_optimum` is `true` and a profile
  finds a loss below the reference by more than `better_optimum_tol`, the
  reference is replaced and profiling restarts (at most this many times).
- `better_optimum_tol`: absolute loss tolerance for declaring a better optimum.
"""
struct ProfileOptions
    samplesize::Int
    rel_step_increase::Float64
    step_factor::Float64
    min_step_factor::Float64
    max_step_factor::Float64
    stop_margin_factor::Float64
    polish::Bool
    allow_better_optimum::Bool
    smooth_jumps::Bool
    jump_tol::Float64
    max_trial_redos::Int
    refine_ci::Bool
    ci_refine_iters::Int
    refit_reference::Bool
    max_reference_updates::Int
    better_optimum_tol::Float64
end

function ProfileOptions(;
    samplesize::Int=100,
    rel_step_increase::Float64=0.1,
    step_factor::Float64=1.5,
    min_step_factor::Float64=1.0e-6,
    max_step_factor::Float64=0.5,
    stop_margin_factor::Float64=1.2,
    polish::Bool=true,
    allow_better_optimum::Bool=true,
    smooth_jumps::Bool=true,
    jump_tol::Float64=0.1,
    max_trial_redos::Int=20,
    refine_ci::Bool=true,
    ci_refine_iters::Int=12,
    refit_reference::Bool=true,
    max_reference_updates::Int=3,
    better_optimum_tol::Float64=1.0e-3
)
    return ProfileOptions(
        samplesize, rel_step_increase, step_factor,
        min_step_factor, max_step_factor, stop_margin_factor,
        polish, allow_better_optimum, smooth_jumps, jump_tol, max_trial_redos,
        refine_ci, ci_refine_iters, refit_reference, max_reference_updates,
        better_optimum_tol
    )
end
