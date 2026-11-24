using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using BreakpointProfiles

makedocs(
    sitename = "BreakpointProfiles.jl",
    format = Documenter.HTML(
        repolink = "https://github.com/Mehdilotfi7/BreakpointProfiles.jl",
        sidebar_sitename = false
    ),
    modules = [BreakpointProfiles],
    pages = [
        "Home" => "index.md",
        "Profiling options" => "profiling_options.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ],
    remotes = nothing
)

deploydocs(
    repo = "github.com/Mehdilotfi7/BreakpointProfiles.jl.git",
    devbranch = "main",
    push_preview = false
)
