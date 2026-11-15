"""
    BaselinePLEProblem(; objective, best_params, best_loss=NaN, lb, ub,
                       param_names=String[])

Construct a profile-likelihood problem without changepoints. `objective` accepts
one parameter vector and returns the scalar loss. The returned problem uses the
same conditional profiling, optimization, reporting, and plotting API as
`ODEChangepointPLEProblem`.
"""
function BaselinePLEProblem(;
    objective,
    best_params,
    best_loss=NaN,
    lb,
    ub,
    param_names=String[]
)
    params = collect(Float64, best_params)
    return ODEChangepointPLEProblem(
        objective = (params, changepoints) -> objective(params),
        data = zeros(0, 0),
        loss_fn = nothing,
        changepoints = Int[],
        best_params = params,
        best_loss = best_loss,
        lb = lb,
        ub = ub,
        param_names = param_names,
        n_global = length(params),
        n_segment_specific = 0,
        n_obs = 0
    )
end
