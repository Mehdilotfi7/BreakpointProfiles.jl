using BreakpointProfiles
using Test
import Plots as P

@testset "Plots extension with renamed import" begin
    profile = synthetic_profile(
        "x", 1, 1.0, 0.0, 0.8, 1.2, true,
        [0.8, 1.0, 1.2], [1.0, 0.0, 1.0]
    )
    cp_profile = CPProfileResult(1, 20, 2, [18, 19, 20, 21, 22],
                                 [2.0, 1.0, 0.0, 1.0, 2.0], 0.0)
    @test plot_profiles([profile]) !== nothing
    @test plot_cp_profiles([cp_profile], 3.8414588206941285) !== nothing
    @test plot_profile_panels([profile], [cp_profile]) !== nothing
end
