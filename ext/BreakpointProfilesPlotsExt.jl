module BreakpointProfilesPlotsExt

using BreakpointProfiles
using Plots

function BreakpointProfiles.plot_profiles(profiles::Vector{ProfileResult}; filename=nothing,
                                          title::String="Profile-likelihood curves",
                                          reference_bounds=nothing)
    n = length(profiles)
    ncols = min(4, n)
    nrows = ceil(Int, n / ncols)
    fig = plot(layout=(nrows, ncols), size=(300 * ncols, 250 * nrows),
               framestyle=:box, grid=true, gridalpha=0.3)
    n_ident = 0
    for (i, p) in enumerate(profiles)
        p.identifiable && (n_ident += 1)
        perm = sortperm(p.values)
        vals = p.values[perm]
        los = p.losses[perm]
        use_log = all(vals .> 0) && (maximum(vals) / minimum(vals)) > 10.0
        plot!(fig[i], vals, los, seriestype=:path, color=:steelblue, lw=1.0, label="")
        scatter!(fig[i], vals, los, markersize=3, color=:steelblue, label="")
        hline!(fig[i], [p.threshold], color=:crimson, ls=:dash, lw=0.8, label="")
        vline!(fig[i], [p.best_value], color=:darkgreen, ls=:dash, lw=1.0, label="")
        if p.ci_lower < p.ci_upper
            lo, hi = minmax(p.ci_lower, p.ci_upper)
            vline!(fig[i], [lo, hi], color=:gold, alpha=0.25, label="")
        end
        if reference_bounds !== nothing
            rlo, rhi = reference_bounds[1][p.index], reference_bounds[2][p.index]
            vline!(fig[i], [rlo, rhi], color=:gray40, ls=:dot, lw=1.2, label="")
        end
        plot!(fig[i], xlabel="parameter value", ylabel="loss", title=p.parameter,
              titlefontsize=9, xscale=use_log ? :log10 : :identity,
              tickfontsize=6, labelfontsize=7)
    end
    for j in (n + 1):(nrows * ncols)
        plot!(fig[j], axis=([], false), grid=false)
    end
    ref_loss = profiles[1].best_loss
    threshold = profiles[1].threshold
    plot!(fig, plot_title="$title | ref loss=$(round(ref_loss, digits=2)) | " *
                         "threshold=$(round(threshold, digits=2)) | identifiable=$n_ident/$n",
          plot_titlefontsize=11)
    filename !== nothing && savefig(fig, filename)
    return fig
end

function BreakpointProfiles.plot_cp_profiles(cp_profiles::Vector{CPProfileResult}, threshold::Real;
                                             filename=nothing,
                                             title::String="Changepoint profile curves")
    n = length(cp_profiles)
    ncols = min(4, n)
    nrows = ceil(Int, n / ncols)
    fig = plot(layout=(nrows, ncols), size=(400 * ncols, 300 * nrows),
               framestyle=:box, grid=true, gridalpha=0.3)
    n_ident = 0
    for (i, cp) in enumerate(cp_profiles)
        cands = cp.candidate_cps
        los = cp.losses
        lo, hi, id = BreakpointProfiles.cp_ci(cp, Float64(threshold))
        id && (n_ident += 1)
        perm = sortperm(cands)
        cands = cands[perm]
        los = los[perm]
        plot!(fig[i], cands, los, seriestype=:path, color=:steelblue, lw=1.5, label="")
        scatter!(fig[i], cands, los, markersize=3, color=:steelblue, label="")
        hline!(fig[i], [threshold], color=:crimson, ls=:dash, lw=0.8, label="")
        vline!(fig[i], [cp.original_cp], color=:darkgreen, ls=:dash, lw=1.0, label="")
        if !ismissing(lo) && !ismissing(hi)
            vline!(fig[i], [lo], color=:gold, ls=:dash, lw=1.5, label="")
            vline!(fig[i], [hi], color=:gold, ls=:dash, lw=1.5, label="")
        end
        ci_text = ismissing(lo) ? "CI = [-, -]" : "CI = [$lo, $hi]"
        plot!(fig[i], xlabel="candidate changepoint", ylabel="loss",
              title="CP $(cp.cp_index) (original=$(cp.original_cp))\n$ci_text, identifiable=$id",
              titlefontsize=9, tickfontsize=6, labelfontsize=7)
    end
    for j in (n + 1):(nrows * ncols)
        plot!(fig[j], axis=([], false), grid=false)
    end
    plot!(fig, plot_title="$title | threshold=$(round(Float64(threshold), digits=2)) | " *
                         "identifiable=$n_ident/$n", plot_titlefontsize=11)
    filename !== nothing && savefig(fig, filename)
    return fig
end

function BreakpointProfiles.plot_profile_panels(
    profiles::Vector{ProfileResult}, cp_profiles::Vector{CPProfileResult};
    comparison_profiles::Union{Nothing,Vector{ProfileResult}}=nothing,
    filename=nothing, title::String="Profile-likelihood curves")
    length(profiles) <= 6 || error("plot_profile_panels supports at most six parameters")
    length(cp_profiles) <= 2 || error("plot_profile_panels supports at most two changepoints")
    comparison_profiles !== nothing && length(comparison_profiles) != length(profiles) &&
        error("comparison_profiles must contain the same number of profiles")
    fig = plot(layout=(2, 4), size=(1600, 850), framestyle=:box,
               grid=false, bottom_margin=6Plots.mm)
    for (i, profile) in enumerate(profiles)
        values = profile.values
        losses = profile.losses
        perm = sortperm(values)
        values, losses = values[perm], losses[perm]
        minloss = minimum(losses)
        margin = max(0.08 * (profile.threshold - minloss), 1e-6)
        if profile.identifiable && isfinite(profile.ci_lower) && isfinite(profile.ci_upper)
            width = max(profile.ci_upper - profile.ci_lower, 1e-6)
            xlims = (profile.ci_lower - 1.25width, profile.ci_upper + 1.25width)
            ylims = (minloss - 0.05margin, profile.threshold + 0.35margin)
        else
            xlims = (minimum(values), maximum(values))
            ylims = (minloss - 0.05margin, profile.threshold + 0.35margin)
        end
        plot!(fig[i], xlims=xlims, ylims=ylims, xlabel="parameter value",
              ylabel="loss", title=profile.parameter,
              titlefontsize=9, tickfontsize=7, labelfontsize=8, legend=false)
        xs = collect(range(max(xlims[1], minimum(values)),
                           min(xlims[2], maximum(values)), length=400))
        slope_l = length(values) > 1 ? (losses[2] - losses[1]) / (values[2] - values[1]) : 0.0
        slope_r = length(values) > 1 ? (losses[end] - losses[end - 1]) / (values[end] - values[end - 1]) : 0.0
        ys = [x < values[1] ? losses[1] + slope_l * (x - values[1]) :
              x > values[end] ? losses[end] + slope_r * (x - values[end]) :
              BreakpointProfiles._pchip_eval(values, losses, [x])[1] for x in xs]
        plot!(fig[i], xs, ys, color=:steelblue, lw=1.2, label="")
        scatter!(fig[i], values, losses, color=:steelblue, markersize=2.5, label="")
        hline!(fig[i], [profile.threshold], color=:crimson, ls=:dash, lw=1.0, label="")
        vline!(fig[i], [profile.best_value], color=:darkgreen, ls=:dash, lw=1.0, label="")
        if profile.ci_lower < profile.ci_upper
            vspan!(fig[i], [profile.ci_lower, profile.ci_upper], color=:gold, alpha=0.25, label="")
        end
        if comparison_profiles !== nothing
            other = comparison_profiles[i]
            vals, los = other.values, other.losses
            perm = sortperm(vals)
            vals, los = vals[perm], los[perm]
            slope_l = length(vals) > 1 ? (los[2] - los[1]) / (vals[2] - vals[1]) : 0.0
            slope_r = length(vals) > 1 ? (los[end] - los[end - 1]) / (vals[end] - vals[end - 1]) : 0.0
            other_ys = [x < vals[1] ? los[1] + slope_l * (x - vals[1]) :
                        x > vals[end] ? los[end] + slope_r * (x - vals[end]) :
                        BreakpointProfiles._pchip_eval(vals, los, [x])[1] for x in xs]
            plot!(fig[i], xs, other_ys, color=:seagreen, lw=1.2, label="")
            scatter!(fig[i], vals, los, color=:seagreen, markersize=2.5, label="")
            if other.ci_lower < other.ci_upper
                vspan!(fig[i], [other.ci_lower, other.ci_upper], color=:orange, alpha=0.18, label="")
            end
        end
    end
    for (j, cp) in enumerate(cp_profiles)
        i = length(profiles) + j
        lo, hi, _ = BreakpointProfiles.cp_ci(cp, Float64(cp.best_loss + BreakpointProfiles.CHISQ_95))
        candidates, losses = cp.candidate_cps, cp.losses .- minimum(cp.losses)
        threshold = Float64(BreakpointProfiles.CHISQ_95)
        margin = max(0.1 * threshold, 1e-6)
        plot!(fig[i], candidates, losses, color=:steelblue, lw=1.2,
              xlabel="candidate CP", ylabel="loss", title="CP $(cp.cp_index)",
              titlefontsize=9, tickfontsize=7, labelfontsize=8, legend=false,
              ylims=(-0.05margin, threshold + 0.35margin))
        scatter!(fig[i], candidates, losses, color=:steelblue, markersize=2.5, label="")
        hline!(fig[i], [threshold], color=:crimson, ls=:dash, lw=1.0, label="")
        vline!(fig[i], [cp.original_cp], color=:darkgreen, ls=:dash, lw=1.0, label="")
        if !ismissing(lo) && !ismissing(hi)
            vspan!(fig[i], [lo, hi], color=:gold, alpha=0.25, label="")
        end
    end
    filename !== nothing && savefig(fig, filename)
    return fig
end

end
