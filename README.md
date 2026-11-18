# BreakpointProfiles.jl

[![CI](https://github.com/Mehdilotfi7/BreakpointProfiles.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Mehdilotfi7/BreakpointProfiles.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/Mehdilotfi7/BreakpointProfiles.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mehdilotfi7/BreakpointProfiles.jl)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mehdilotfi7.github.io/BreakpointProfiles.jl/)

<p align="center">
  <img src="docs/icons/breakpointprofile_icon.png" alt="BreakpointProfiles.jl logo" width="220">
</p>

[Full documentation is available here](https://mehdilotfi7.github.io/BreakpointProfiles.jl/).

Profile-likelihood-based uncertainty quantification for ordinary differential equation (ODE) models with fixed changepoints.

This package is designed to be used **after** a changepoint-detection step (for example with [Mica.jl](https://github.com/Mehdilotfi7/Mica.jl)), once the number and locations of changepoints have been determined. `BreakpointProfiles.jl` then quantifies the uncertainty around the fitted parameters and the detected changepoint locations via profile-likelihood methods.

The adaptive profile-likelihood engine and the confidence-interval extraction in this package were inspired by and validated against the Data2Dynamics (D2D) MATLAB profile-likelihood (Raue et al., *Bioinformatics* 2015; Raue et al., *PLOS ONE* 2013). Users familiar with D2D will recognise the D2D-style adaptive outward stepping, warm-start re-optimisation, and jump-smoothing strategy.

[ProfileLikelihood.jl](https://github.com/DanielVandH/ProfileLikelihood.jl) provides general profile-likelihood analysis for model parameters. BreakpointProfiles.jl complements it with discrete changepoint-location profiles, joint profiling in which changepoints are re-optimized as nuisance parameters, and a common API for baseline, conditional, and changepoint-aware analyses.

## Installation

Install the registered package with:

```julia
using Pkg
Pkg.add("BreakpointProfiles")
```

Or clone the repository and develop it locally:

```julia
using Pkg
Pkg.develop(path="path/to/BreakpointProfiles.jl")
```

## Quick start

```julia
using BreakpointProfiles

# Build a problem from a previously fitted changepoint model
# (see examples/baseline_and_changepoint.jl for a complete workflow)
prob = ODEChangepointPLEProblem(
    objective = (params, cps) -> my_loss(params, cps),
    data = my_data,
    loss_fn = my_loss,
    changepoints = [50, 100],
    best_params = best_fit,
    best_loss = best_loss,
    lb = lb,
    ub = ub,
    n_global = n_global,
    n_segment_specific = n_segment_specific
)

# Conditional profile-likelihood for a parameter (changepoints fixed)
prof = profile_parameter(prob, 1)

# Joint profile-likelihood for a parameter (changepoints re-optimised)
joint_prof = profile_parameter_joint(prob, 1)

# Profile a changepoint location
cp_prof = profile_changepoint(prob, 1; window=7)

# Summarise
summary = ple_summary([prof])
```

## Baseline profile likelihood without changepoints

Use `BaselinePLEProblem` when the model has no changepoints or when you want a
baseline profile before introducing a changepoint model. It uses the same
adaptive D2D-style profiler, optimizers, confidence-interval extraction, and
reporting as the changepoint-aware API.

```julia
objective = p -> (p[1] - 3.0)^2 + (p[2] - 5.0)^2

baseline = BaselinePLEProblem(
    objective = objective,
    best_params = [3.0, 5.0],
    best_loss = 0.0,
    lb = [-10.0, -10.0],
    ub = [10.0, 10.0],
    param_names = ["x", "y"]
)

baseline_profile = profile_parameter(baseline, 1)
baseline_profiles = profile_all_parameters(baseline)
```

## Parallel computation

Start Julia with the desired number of worker threads:

```bash
julia --threads=auto
```

or set `JULIA_NUM_THREADS` before launching Julia. The package parallelizes
independent work at two levels:

- **Across profiles:** `profile_all_parameters` evaluates independent parameter
  profiles on Julia threads when the selected inner optimizer is serial.
  `profile_all_parameters_joint` provides the same option for joint profiles.
  For large models, independent parameters can also be assigned to separate
  cluster jobs, such as SLURM array tasks.
- **Within a profile:** compatible population-based optimizers can evaluate
  their populations on threads. During joint profiling, the discrete
  changepoint grid can also be evaluated in parallel with
  `parallelize_grid=true`.

Avoid nested parallelism that oversubscribes the CPU. When the inner optimizer
uses threads, the batch drivers process profiles serially so that each inner
optimization can use the available workers. For joint profiling,
`parallel_over_parameters=true` disables grid-level threading inside each
profile; setting it to `false` allows the CP grid to use threads instead.

For cluster workflows, write each profile's results to a separate output file
and combine them after all jobs finish. Parallelism changes runtime, not the
profiling method or confidence threshold.

## Features

- **Baseline profile likelihood** for models without changepoints.
- **Conditional profile likelihood** for continuous parameters (changepoints fixed).
- **Joint profile likelihood** for continuous parameters (changepoints treated as discrete nuisance variables and re-optimised).
- **Changepoint-location profiling** with re-optimisation of all other parameters.
- Adaptive profile-likelihood grids with bound handling.
- Multiple optimiser backends: GA (Evolutionary.jl), differential evolution (Metaheuristics.jl), LBFGS (Optim.jl), BOBYQA (NLopt.jl), and multi-start/hybrid pipelines.
- Likelihood options: Gaussian (L2), Laplace (L1), log-scale, and custom losses.
- Identifiability assessment and confidence-interval extraction.
- CSV/DataFrame export, Markdown reports, and plotting utilities.

## L2 vs L1 losses and thresholds

- **Gaussian (L2) losses:** Wilks' theorem applies; use the standard χ² threshold
  (`best_loss + 3.8415` for 1 df, 95%).
- **Laplace (L1) losses:** Wilks' theorem does not apply. Pass a user-derived
  threshold to `profile_parameter(...; threshold=...)`.


## Citation

If you use this package in research, please cite the MICA paper.

## License

MIT
