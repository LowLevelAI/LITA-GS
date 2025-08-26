OptimizationParams = dict(
    iterations = 20_000,
    position_lr_init = 0.00025,
    position_lr_final = 0.0000025,
    position_lr_max_steps = 20_000,
    densify_until_iter = 5_000,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    percent_dense = 0.01,
    lambda_dssim = 0.4,
    lambda_dssim_low = 0.2,
    lambda_depth = 0.1,
    lambda_prior = 0.01, 
    tonemapper_lr = 0.0001,
    denoiser_lr = 0.00005,
    depth_threshold = 1.0,
    densification_interval = 200
)

ModelParams = dict(
    eval_index = [0, 8, 16, 24, 32],
    denoiser_stage = 6,
    denoiser_channel = 64
)
