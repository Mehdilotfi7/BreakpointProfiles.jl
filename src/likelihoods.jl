"""    AbstractLikelihood

Abstract supertype for built-in likelihood / loss callables."""
abstract type AbstractLikelihood end

"""
    log_transform(data; threshold=1)

Return `log(max(data, threshold))`.  Useful for epidemiological data with zeros."""
function log_transform(data; threshold=1.0)
    return [v >= threshold ? log(v) : log(threshold) for v in data]
end

"""
    GaussianLogLikelihood

Negative Gaussian log-likelihood on log-transformed observations, scaled to the
`-2 log L` scale used by profile-likelihood methods (e.g. Data2Dynamics).

`sigma` is the standard deviation of the observation noise. Minimising this
objective is equivalent to maximising a Gaussian likelihood, and the standard
χ² threshold of 3.8415 is asymptotically valid for 95% profile-likelihood
confidence intervals (Wilks' theorem).
"""
struct GaussianLogLikelihood <: AbstractLikelihood
    sigma::Float64
end
GaussianLogLikelihood() = GaussianLogLikelihood(1.0)

function (ll::GaussianLogLikelihood)(sim, data)
    n = length(data)
    r = log_transform(sim) .- log_transform(data)
    return sum(x -> x^2 / ll.sigma^2, r) + 2 * n * log(ll.sigma * sqrt(2π))
end

"""
    LaplaceLogLikelihood

Negative Laplace log-likelihood on log-transformed observations.

This corresponds to an L1 loss on log-transformed data, similar to the MICA
default. Because the Laplace likelihood is not Gaussian, Wilks' theorem does
not apply and the standard χ² threshold of 3.84 is **not** rigorously valid
for 95% confidence intervals. Users of L1 losses must supply a
likelihood-appropriate threshold.
"""
struct LaplaceLogLikelihood <: AbstractLikelihood
    sigma::Float64
end
LaplaceLogLikelihood() = LaplaceLogLikelihood(1.0)

function (ll::LaplaceLogLikelihood)(sim, data)
    n = length(data)
    r = log_transform(sim) .- log_transform(data)
    return sum(x -> abs(x) / ll.sigma, r) + n * log(2 * ll.sigma)
end

"""
    CustomLoss{F}

Wrap an arbitrary user-provided loss function `f(sim, data)`."""
struct CustomLoss{F} <: AbstractLikelihood
    f::F
end

function (c::CustomLoss)(sim, data)
    return c.f(sim, data)
end

"""
    GaussianNLL(σ)

Gaussian negative log-likelihood on the original (linear) scale with known
noise standard deviation, on the `-2 log L` scale (constants omitted):

    Σ_k Σ_t ((sim_k(t) - data_k(t)) / σ_k)²

`σ` may be a scalar or one value per data channel. With known σ the χ²₁
threshold 3.8415 gives approximate 95% profile-likelihood intervals."""
struct GaussianNLL <: AbstractLikelihood
    sigma_per_channel::Vector{Float64}
end
GaussianNLL(σ::Real) = GaussianNLL(Float64[σ])

function (ll::GaussianNLL)(sim, data)
    nch = size(data, 1)
    total = 0.0
    for k in 1:nch
        σ = ll.sigma_per_channel[min(k, length(ll.sigma_per_channel))]
        inv_σ2 = 1.0 / (σ * σ)
        for t in 1:size(data, 2)
            d = data[k, t]
            isnan(d) && continue
            r = sim[k, t] - d
            total += r * r * inv_σ2
        end
    end
    return total
end

"""
    ProfiledGaussianNLL()

Gaussian `-2 log L` with a common unknown σ profiled out analytically:

    N · log(RSS / N)

where `RSS` is the residual sum of squares over all non-missing observations
and `N` their number. This is the likelihood implied by the MICA BIC
`N log(RSS/N) + p log N`, so detection and profiling use the same likelihood;
the χ²₁ threshold applies directly on this scale."""
struct ProfiledGaussianNLL <: AbstractLikelihood end

function (::ProfiledGaussianNLL)(sim, data)
    rss = 0.0
    n = 0
    for k in 1:size(data, 1), t in 1:size(data, 2)
        d = data[k, t]
        isnan(d) && continue
        r = sim[k, t] - d
        rss += r * r
        n += 1
    end
    return n * log(max(rss, floatmin(Float64)) / n)
end

"""
    GaussianLogNLL

Gaussian negative log-likelihood on log-transformed multi-channel data, scaled
to the `-2 log L` scale used by profile-likelihood methods (e.g. Data2Dynamics).

Each observed channel `k` is assumed to follow:

    log(obs_k(t)) ~ N(log(sim_k(t)), σ_k²)

so the contribution is:

    Σ_k Σ_t ((log(sim_k(t)) - log(obs_k(t))) / σ_k)²

Constant terms `2n * log(σ_k √(2π))` are omitted because they do not depend on
the model parameters and thus do not affect the profile shape.

This formulation makes the χ² threshold of 3.8415 valid for approximate 95%
profile-likelihood confidence intervals (Wilks' theorem).
"""
struct GaussianLogNLL <: AbstractLikelihood
    sigma_per_channel::Vector{Float64}
end

"""
    GaussianLogNLL(σ::Real)

Convenience: single σ applied to all channels.
"""
GaussianLogNLL(σ::Real) = GaussianLogNLL(Float64[σ])

function (ll::GaussianLogNLL)(sim, data)
    nch = size(data, 1)
    total = 0.0
    for k in 1:nch
        σ = ll.sigma_per_channel[min(k, length(ll.sigma_per_channel))]
        inv_σ2 = 1.0 / (σ * σ)
        for t in 1:size(data, 2)
            s = max(sim[k, t], 1.0)
            d = max(data[k, t], 1.0)
            r = log(s) - log(d)
            total += r * r * inv_σ2
        end
    end
    return total
end

"""
    estimate_channel_sigma(sim, data; data_indices=nothing)

Estimate per-channel σ from the residuals between `sim` and `data` on log scale.

If `data_indices` is provided (e.g. `[5, 6, 7, 9, 11]` for the COVID model),
it maps data rows to simulation rows.  Otherwise a 1:1 mapping is assumed
(data and sim have the same number of rows).

Returns a `Vector{Float64}` of length `size(data, 1)`.
"""
function estimate_channel_sigma(sim, data; data_indices=nothing)
    nch = size(data, 1)
    sigmas = Float64[]
    for k in 1:nch
        r = if data_indices !== nothing
            log_transform(sim[data_indices[k], :]) .- log_transform(data[k, :])
        else
            log_transform(sim[k, :]) .- log_transform(data[k, :])
        end
        push!(sigmas, max(std(r), 1e-6))
    end
    return sigmas
end

"""
    chi2_threshold(df=1, alpha=0.95)

Return the χ² threshold for the given degrees of freedom and confidence level.

For `df=1` and `alpha=0.95` this is the usual 3.8414588 used for approximate
95% profile-likelihood confidence intervals."""
function chi2_threshold(df::Int=1, alpha::Float64=0.95)
    if df == 1 && alpha == 0.95
        return 3.8414588206941285
    end
    error("Non-default chi2_threshold requires Distributions.jl; df=$df, alpha=$alpha")
end

"""
    evaluate_loss(prob::ODEChangepointPLEProblem, params[, cps])

Evaluate the loss for `params` at the changepoint vector `cps` (default:
`prob.changepoints`) using the problem's objective, simulator, or generic ODE
solver, in that order of priority. Non-finite results are mapped to `Inf`."""
evaluate_loss(prob::ODEChangepointPLEProblem, params::AbstractVector) =
    evaluate_loss(prob, params, prob.changepoints)

function evaluate_loss(prob::ODEChangepointPLEProblem, params::AbstractVector, cps::AbstractVector{<:Integer})
    θ = collect(Float64, params)
    c = collect(Int, cps)
    loss = if prob.objective !== nothing
        prob.objective(θ, c)
    elseif prob.simulator !== nothing
        sim = prob.simulator(θ, c)
        prob.loss_fn(sim, prob.data)
    elseif prob.ode_function !== nothing
        sim = solve_segments(prob, θ, c)
        any(isnan, sim) ? Inf : prob.loss_fn(sim, prob.data)
    else
        error("ODEChangepointPLEProblem has no objective, simulator, or ode_function")
    end
    return isfinite(loss) ? Float64(loss) : Inf
end

"""
    segment_parameters(prob, params, j)

Return the parameter vector `[θ_global; θ_segment_j]` passed to the ODE
right-hand side inside segment `j` (1-based)."""
function segment_parameters(prob::ODEChangepointPLEProblem, params::AbstractVector, j::Int)
    ng, ns = prob.n_global, prob.n_segment_specific
    ns == 0 && return collect(Float64, params[1:ng])
    return vcat(params[1:ng], params[(ng + (j - 1) * ns + 1):(ng + j * ns)])
end

"""
    solve_segments(prob, params[, cps])

Integrate `prob.ode_function` piecewise between the changepoints. Segment `j`
(the interval between the `(j-1)`-th and `j`-th changepoint) is integrated with
`segment_parameters(prob, params, j)`, and the state is carried over
continuously across each changepoint. Returns a `states × time` matrix sampled
at integer time steps on `prob.tspan`."""
function solve_segments(prob, params, cps::AbstractVector{<:Integer}=prob.changepoints)
    t0, tf = prob.tspan
    cps_all = sort(Float64.(collect(cps)))
    n_cols = Int(ceil(tf - t0)) + 1
    nan_out = fill(NaN, length(prob.u0), n_cols)

    u0 = copy(prob.u0)
    t_start = t0
    parts = Matrix{Float64}[]

    for cp in cps_all
        t0 < cp < tf || continue
        j = 1 + count(<=(t_start), cps_all)
        sol = _solve_segment(prob, u0, (t_start, cp), segment_parameters(prob, params, j))
        any(isnan, sol) && return nan_out
        if size(sol, 2) > 1
            push!(parts, sol[:, 1:end-1])
        end
        u0 = sol[:, end]
        t_start = cp
    end

    j = 1 + count(<=(t_start), cps_all)
    sol = _solve_segment(prob, u0, (t_start, tf), segment_parameters(prob, params, j))
    any(isnan, sol) && return nan_out
    push!(parts, sol)

    return reduce(hcat, parts)
end

function _solve_segment(prob, u0, tspan, p)
    ode_prob = ODEProblem(prob.ode_function, u0, tspan, p)
    expected = Int(ceil(tspan[2] - tspan[1])) + 1
    sol = try
        solve(
            ode_prob, Tsit5(),
            saveat = 1.0,
            abstol = 1e-8,
            reltol = 1e-6,
            isoutofdomain = (u, p, t) -> any(x -> x < 0, u),
            verbose = false
        )
    catch
        return fill(NaN, length(u0), expected)
    end
    mat = hcat(sol.u...)
    if any(isnan, mat) || size(mat, 2) != expected
        return fill(NaN, length(u0), expected)
    end
    return mat
end
