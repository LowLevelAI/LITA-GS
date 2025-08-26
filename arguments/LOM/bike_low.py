OptimizationParams = dict(
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    percent_dense = 0.01,
    lambda_dssim = 0.0,
    lambda_dssim_low = 0.2,
    lambda_depth = 0.1,
    lambda_prior = 0.01, 
    tonemapper_lr = 0.0001,
    denoiser_lr = 0.00005,
    depth_threshold = 1.0
)

ModelParams = dict(
    eval_index = [0, 8, 16, 32]
)
