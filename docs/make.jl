using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using BreakpointProfiles

makedocs(
    sitename = "BreakpointProfiles.jl",
    format = Documenter.HTML(repolink = "https://github.com/Mehdilotfi7/BreakpointProfiles.jl"),
    modules = [BreakpointProfiles],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md"
    ],
    remotes = nothing
)

deploydocs(
    repo = "github.com/Mehdilotfi7/BreakpointProfiles.jl.git",
    devbranch = "main",
    push_preview = false
)
