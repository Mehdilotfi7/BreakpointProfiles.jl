using BreakpointProfiles
using Test

@testset "BreakpointProfiles.jl" begin
    include("test_types.jl")
    include("test_baseline.jl")
    include("test_likelihoods.jl")
    include("test_parameter_profile.jl")
    include("test_changepoint_profile.jl")
    include("test_joint_profile.jl")
    include("test_summary.jl")
    include("test_extension.jl")
    include("test_plotting.jl")
    include("test_ple_sanity.jl")
    include("test_regressions.jl")
end
