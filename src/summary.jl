"""
    ple_summary(profiles::Vector{ProfileResult})

Return a `DataFrame` summarising a vector of `ProfileResult` objects."""
function ple_summary(profiles::Vector{ProfileResult})
    df = DataFrame(
        parameter = String[],
        index = Int[],
        best_value = Float64[],
        best_loss = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[],
        identifiable = Bool[],
        threshold = Float64[],
        relative_width = Float64[],
        n_failed = Int[],
        best_found_loss = Float64[],
        n_ci_components = Int[],
        ci_hull_lower = Float64[],
        ci_hull_upper = Float64[]
    )
    for p in profiles
        comps = profile_confidence_intervals(p)
        hull_lo = isempty(comps) ? p.ci_lower : min(p.ci_lower, minimum(first.(comps)))
        hull_hi = isempty(comps) ? p.ci_upper : max(p.ci_upper, maximum(last.(comps)))
        rel_width = if p.best_value != 0
            (p.ci_upper - p.ci_lower) / abs(p.best_value)
        else
            p.ci_upper - p.ci_lower
        end
        push!(df, (
            p.parameter, p.index, p.best_value, p.best_loss,
            p.ci_lower, p.ci_upper, p.identifiable, p.threshold, rel_width,
            p.n_failed, p.best_found_loss, length(comps), hull_lo, hull_hi
        ))
    end
    return df
end

"""
    cp_summary(cp_profiles::Vector{CPProfileResult}[, threshold]; delta=3.8415)

Return a `DataFrame` summarising changepoint confidence sets. If `threshold`
is omitted each profile uses its own reference `best_loss + delta`. Columns:
`cp_index, original_cp, best_cp, ci_lower, ci_upper, identifiable,
censored_lower, censored_upper, n_components, confidence_set`."""
function cp_summary(cp_profiles::Vector{CPProfileResult}, threshold::Union{Real,Nothing}=nothing;
                    delta::Real=CHISQ_95)
    df = DataFrame(
        cp_index = Int[],
        original_cp = Int[],
        best_cp = Int[],
        ci_lower = Int[],
        ci_upper = Int[],
        identifiable = Bool[],
        censored_lower = Bool[],
        censored_upper = Bool[],
        n_components = Int[],
        confidence_set = String[]
    )
    for cp in cp_profiles
        cs = cp_confidence_set(cp, threshold; delta=delta)
        push!(df, (cp.cp_index, cp.original_cp,
                   ismissing(cs.best_cp) ? 0 : cs.best_cp,
                   ismissing(cs.lower) ? 0 : cs.lower,
                   ismissing(cs.upper) ? 0 : cs.upper,
                   cs.identifiable, cs.censored_lower, cs.censored_upper,
                   length(cs.components), join(cs.set, ";")))
    end
    return df
end

"""    is_identifiable(profile::ProfileResult)

Return `true` if the profile crosses the likelihood threshold on both sides
inside the parameter bounds."""
is_identifiable(profile::ProfileResult) = profile.identifiable

"""
    to_profile_dataframe(profiles)

Convert a vector of `ProfileResult` to a long-format `DataFrame` suitable for
custom plotting."""
function to_profile_dataframe(profiles::Vector{ProfileResult})
    df = DataFrame(parameter=String[], index=Int[], value=Float64[], loss=Float64[], delta_loss=Float64[])
    for p in profiles
        for (v, l) in zip(p.values, p.losses)
            push!(df, (p.parameter, p.index, v, l, l - p.best_loss))
        end
    end
    return df
end

"""Plot profile-likelihood curves when Plots.jl is loaded."""
function plot_profiles end

"""Plot changepoint-location profile curves when Plots.jl is loaded."""
function plot_cp_profiles end

function _pchip_eval(x::Vector{Float64}, y::Vector{Float64}, xs::Vector{Float64})
    n = length(x)
    n == 1 && return fill(y[1], length(xs))
    if n == 2
        return [y[1] + (y[2] - y[1]) / (x[2] - x[1]) * (xi - x[1]) for xi in xs]
    end
    h = diff(x)
    d = diff(y) ./ h
    m = zeros(n)
    m[1], m[n] = d[1], d[end]
    for k in 2:(n - 1)
        if d[k - 1] * d[k] <= 0
            m[k] = 0.0
        else
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        end
    end
    m[1] = ((2 * h[1] + h[2]) * d[1] - h[1] * d[2]) / (h[1] + h[2])
    sign(m[1]) != sign(d[1]) && (m[1] = 0.0)
    sign(d[1]) != sign(d[2]) && abs(m[1]) > abs(3 * d[1]) && (m[1] = 3 * d[1])
    m[n] = ((2 * h[end] + h[end - 1]) * d[end] - h[end] * d[end - 1]) / (h[end] + h[end - 1])
    sign(m[n]) != sign(d[end]) && (m[n] = 0.0)
    sign(d[end]) != sign(d[end - 1]) && abs(m[n]) > abs(3 * d[end]) && (m[n] = 3 * d[end])
    out = Vector{Float64}(undef, length(xs))
    for (j, xi) in enumerate(xs)
        k = clamp(searchsortedlast(x, xi), 1, n - 1)
        t = (xi - x[k]) / h[k]
        h00 = 2t^3 - 3t^2 + 1
        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 = t^3 - t^2
        out[j] = h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]
    end
    return out
end

"""Plot parameter and changepoint profiles in a 2 × 4 layout when Plots.jl is loaded."""
function plot_profile_panels end

"""    write_summary(path, summary_df)

Write a summary DataFrame to CSV."""
function write_summary(path, summary_df::DataFrame)
    CSV.write(path, summary_df)
end

"""    print_summary_table(summary_df)

Print a Markdown-formatted summary table to `stdout`."""
function print_summary_table(summary_df::DataFrame)
    println("| parameter | best | CI lower | CI upper | identifiable |")
    println("|---|---|---|---|---|")
    for r in eachrow(summary_df)
        @printf("| %s | %.5g | %.5g | %.5g | %s |\n",
                r.parameter, r.best_value, r.ci_lower, r.ci_upper, r.identifiable)
    end
end

"""
    write_report(out_dir, profiles; cp_profiles=nothing, cp_delta=3.8415)

Write `ple_summary.csv`, `ple_results.csv` (+ curves), optionally the
changepoint profile files, and a Markdown report `ple_report.md`."""
function write_report(out_dir::AbstractString, profiles::Vector{ProfileResult};
                      cp_profiles::Union{Vector{CPProfileResult},Nothing}=nothing,
                      cp_delta::Real=CHISQ_95)
    mkpath(out_dir)
    summary_df = ple_summary(profiles)
    write_summary(joinpath(out_dir, "ple_summary.csv"), summary_df)
    write_profiles(joinpath(out_dir, "ple_results.csv"), profiles)

    lines = String[]
    push!(lines, "# Profile-likelihood analysis")
    push!(lines, "")
    if !isempty(profiles)
        push!(lines, @sprintf("Reference loss: %.6g; threshold: %.6g (Δ = %.4g)",
                              profiles[1].best_loss, profiles[1].threshold,
                              profiles[1].threshold - profiles[1].best_loss))
        push!(lines, "")
    end
    push!(lines, "| parameter | best value | CI lower | CI upper | identifiable |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(summary_df)
        push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s |",
                              r.parameter, r.best_value, r.ci_lower, r.ci_upper, r.identifiable))
    end

    if cp_profiles !== nothing
        cp_df = cp_summary(cp_profiles; delta=cp_delta)
        write_cp_profiles(joinpath(out_dir, "cp_profile_loss.csv"), cp_profiles)
        CSV.write(joinpath(out_dir, "cp_profile_ci.csv"), cp_df)
        push!(lines, "")
        push!(lines, "## Changepoint confidence sets")
        push!(lines, "")
        push!(lines, "| cp | detected | best | CI lower | CI upper | bounded | set |")
        push!(lines, "|---|---|---|---|---|---|---|")
        for r in eachrow(cp_df)
            push!(lines, @sprintf("| %d | %d | %d | %d | %d | %s | %s |",
                                  r.cp_index, r.original_cp, r.best_cp, r.ci_lower,
                                  r.ci_upper, r.identifiable, r.confidence_set))
        end
    end

    open(joinpath(out_dir, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end
    return joinpath(out_dir, "ple_report.md")
end
