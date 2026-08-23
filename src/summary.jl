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
        best_found_loss = Float64[]
    )
    for p in profiles
        rel_width = if p.best_value != 0
            (p.ci_upper - p.ci_lower) / abs(p.best_value)
        else
            p.ci_upper - p.ci_lower
        end
        push!(df, (
            p.parameter, p.index, p.best_value, p.best_loss,
            p.ci_lower, p.ci_upper, p.identifiable, p.threshold, rel_width,
            p.n_failed, p.best_found_loss
        ))
    end
    return df
end

"""
    cp_summary(cp_profiles::Vector{CPProfileResult}, threshold)

Return a `DataFrame` summarising changepoint profile intervals."""
function cp_summary(cp_profiles::Vector{CPProfileResult}, threshold::Float64)
    df = DataFrame(
        cp_index = Int[],
        original_cp = Int[],
        ci_lower = Int[],
        ci_upper = Int[],
        identifiable = Bool[]
    )
    for cp in cp_profiles
        lo, hi, id = cp_ci(cp, threshold)
        if ismissing(lo)
            push!(df, (cp.cp_index, cp.original_cp, 0, 0, false))
        else
            push!(df, (cp.cp_index, cp.original_cp, lo, hi, id))
        end
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

"""
    plot_profiles(profiles::Vector{ProfileResult}; filename=nothing)

Create a multi-panel figure of profile-likelihood curves.

Requires `Plots.jl` to be loaded in the calling environment.  If `Plots` is not
available, the function raises an error."""
function plot_profiles(profiles::Vector{ProfileResult}; filename=nothing)
    if !isdefined(Main, :Plots)
        error("plot_profiles requires Plots.jl to be loaded in the calling environment")
    end
    Plots = Main.Plots
    n = length(profiles)
    ncols = min(4, n)
    nrows = ceil(Int, n / ncols)
    fig = Plots.plot(layout=(nrows, ncols), size=(4 * 300, nrows * 250))
    for (i, p) in enumerate(profiles)
        perm = sortperm(p.values)
        vals = p.values[perm]
        los = p.losses[perm]
        Plots.plot!(fig[i], vals, los, label="", xlabel=p.parameter, ylabel="loss")
        Plots.hline!(fig[i], [p.threshold], label="95% threshold", ls=:dash)
        Plots.vline!(fig[i], [p.best_value], label="best", ls=:dot)
        if p.ci_lower < p.ci_upper
            Plots.vspan!(fig[i], [p.ci_lower, p.ci_upper], label="CI", alpha=0.2)
        end
    end
    if filename !== nothing
        Plots.savefig(fig, filename)
    end
    return fig
end

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
    write_report(out_dir, profiles, cp_profiles=nothing)

Write a Markdown report, summary CSV, and profile curves CSV for a completed PLE
analysis."""
function write_report(out_dir::String, profiles::Vector{ProfileResult};
                    cp_profiles::Union{Vector{CPProfileResult},Nothing}=nothing)
    mkpath(out_dir)
    summary_df = ple_summary(profiles)
    write_summary(joinpath(out_dir, "ple_summary.csv"), summary_df)
    write_profiles(joinpath(out_dir, "ple_results.csv"), profiles)

    lines = String[]
    push!(lines, "# Profile-likelihood analysis")
    push!(lines, "")
    push!(lines, "| parameter | best value | CI lower | CI upper | identifiable |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(summary_df)
        @printf(lines[end], "| %s | %.5g | %.5g | %.5g | %s |",
                r.parameter, r.best_value, r.ci_lower, r.ci_upper, r.identifiable)
    end

    if cp_profiles !== nothing
        threshold = profiles[1].threshold
        cp_df = cp_summary(cp_profiles, threshold)
        write_cp_profiles(joinpath(out_dir, "cp_profile_loss.csv"), cp_profiles)
        CSV.write(joinpath(out_dir, "cp_profile_ci.csv"), cp_df)
        push!(lines, "")
        push!(lines, "## Changepoint profile intervals")
        push!(lines, "")
        push!(lines, "| cp | original | CI lower | CI upper | identifiable |")
        push!(lines, "|---|---|---|---|---|")
        for r in eachrow(cp_df)
            @printf(lines[end], "| %d | %d | %d | %d | %s |",
                    r.cp_index, r.original_cp, r.ci_lower, r.ci_upper, r.identifiable)
        end
    end

    open(joinpath(out_dir, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end
end
