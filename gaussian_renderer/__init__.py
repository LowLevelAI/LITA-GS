import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera,  pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, depth_threshold = None, iteration = None):
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold, 
        iteration=iteration, 
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    hdr_rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # [N, 3]
    new_feature = pc.get_new_features # [N, LC]
    
    colors_precomp = torch.cat([hdr_rgb, new_feature], dim=-1)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image_conbine, radii, pixels = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image_conbine[:3, ...] # [3, H, W]
    depth = rendered_image_conbine[3:4, ...]
    prior = rendered_image_conbine[4:5, ...]
    noise = rendered_image_conbine[5:8, ...]
    illu = rendered_image_conbine[8:11, ...]

    denoised_list = pc.denoiser(rendered_image, noise)
    rendered_low = pc.tonemapper(illu * denoised_list[-1])
    
    return {"render": rendered_image,
            "depth": depth,
            "prior": prior,
            "noise": noise,
            "illu": illu,
            "denoised_list": denoised_list,
            "render_low": rendered_low,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}