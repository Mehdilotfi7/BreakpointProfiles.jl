# BreakpointProfiles.jl

Profile-likelihood-based uncertainty quantification for ordinary differential equation (ODE) models with fixed changepoints.

## Installation

The package is not yet registered in the Julia General registry. Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Mehdilotfi7/BreakpointProfiles.jl")
```

Or clone the repository and develop it locally:

```julia
using Pkg
Pkg.develop(path="path/to/BreakpointProfiles.jl")
```

## Quick start

```julia
using BreakpointProfiles

# Define an ODEChangepointPLEProblem (see examples/ for a full COVID-19 example)
prob = ODEChangepointPLEProblem(...)

# Profile a parameter
prof = profile_parameter(prob, 1; n_points=20, method=:adaptive)

# Profile a changepoint
cp_prof = profile_changepoint(prob, 1; window=7)

# Summarise
summary = ple_summary([prof])
```

## Features

- Adaptive profile-likelihood grids with bound handling.
- Multiple optimiser backends: GA (Evolutionary.jl), LBFGS (Optim.jl), direct search (NLopt.jl), and multi-start BOBYQA.
- Likelihood options: Gaussian (L2) and Laplace (L1) negative log-likelihoods, plus custom losses.
- Changepoint location profiling with re-optimisation of all other parameters.
- Identifiability assessment and confidence-interval extraction.
- Bootstrap threshold support for non-Gaussian (e.g. L1) losses.
- CSV/DataFrame export and plotting utilities.

## L1 vs L2 losses and thresholds

- **Gaussian (L2) losses:** Wilks' theorem applies; use the standard χ² threshold
  (`best_loss + 3.8415` for 1 df, 95%).
- **Laplace (L1) losses:** Wilks' theorem does not apply. Pass a user-derived
  threshold to `profile_parameter(...; threshold=...)` or compute one with
  `bootstrap_threshold(data_generator, objective, best_params, idx; ...)`.

The COVID driver (`examples/covid_ple_example.jl` and
`publication/applications/Covid/scripts/covid_ple_bobyqa.jl`) supports both `L1`
and `L2` modes via the `COVID_PLE_LOSS` environment variable.

## Documentation

Build the documentation locally with Documenter.jl:

```bash
julia --project=docs docs/make.jl
```

## Citation

If you use this package in research, please cite the MICA paper.

## License

MIT
