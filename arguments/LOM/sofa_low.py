OptimizationParams = dict(
    iterations = 10_000,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_max_steps = 10_000,
    densify_until_iter = 5_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    percent_dense = 0.01,
    lambda_dssim = 0.25,
    lambda_dssim_low = 0.2,
    lambda_depth = 0.1,
    lambda_prior = 0.01, 
    tonemapper_lr = 0.0001,
    denoiser_lr = 0.00005,
    depth_threshold = 1.0,
    exposure = 0.49,
    use_denoiser = True
)

ModelParams = dict(
    eval_index = [0, 8, 16, 24]
)

