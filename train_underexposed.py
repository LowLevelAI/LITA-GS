import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, pearson_depth_loss, local_pearson_loss, l2_loss, SmoothLoss, TVLoss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from os import makedirs    
from scene.cameras import Camera
import random
import cv2
import numpy as np
import torchvision
from ciconv2d import CIConv2d

def gray(t): return (
        np.clip((t[0][0] if len(t.shape) == 4 else t[0]).detach().cpu().numpy(), 0, 1) * 255).astype(
    np.uint8)

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict
    
def prior_loss_cal(model, rendered_prior, target_image, a):

    with torch.no_grad():
        target_features = model(target_image.unsqueeze(0))
    
    rendered_W = torch.clamp(rendered_prior, 0, 1)
    target_W = torch.clamp(target_features, 0, 1).squeeze(0)
    WW = l1_loss(target_W, rendered_W)
    loss = WW * a
    return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder 
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training(dataset, save_dir, opt, pipe, testing_iterations, saving_iterations, debug_from):
    #loss_smooth = SmoothLoss().cuda()
    tv_loss = TVLoss()
    
    prior_model = CIConv2d('W', k=3, scale=0.8)
    prior_model = prior_model.cuda()

    set_seed(10) 
    first_iter = 0
    dataset.model_path = save_dir
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(sh_degree=dataset.sh_degree, depth_feat_dim=1, prior_feat_dim=1, noise_feat_dim=3, illu_feat_dim=3, denoiser_stage=dataset.denoiser_stage, denoiser_layers=dataset.denoiser_layers, denoiser_channel=dataset.denoiser_channel, denoiser_mode=dataset.denoiser_mode) 
    scene = Scene(dataset, gaussians, opt.use_depth, opt.use_prior, gap = pipe.interval) 

    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras) 

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_l1_loss_for_log, ema_ssim_loss_for_log =0.0, 0.0

    best_psnr = 0
    best_ssim = 0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        randindex = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam: Camera = viewpoint_stack.pop(randindex)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, depth_threshold = opt.depth_threshold * scene.cameras_extent, iteration = iteration)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"])
        rendered_depth: torch.Tensor = render_pkg["depth"]
        rendered_prior: torch.Tensor = render_pkg["prior"]

        rendered_low: torch.Tensor = render_pkg["render_low"]
        rendered_denoised_list: torch.Tensor = render_pkg["denoised_list"]

        target = opt.exposure
        low_image = viewpoint_cam.low_image
        low_mean = low_image.mean()
        target_image = torch.clamp(low_image * (target/low_mean), 0, 1)
        
        loss = 0 

      
        loss_l1 = l1_loss(rendered_image, target_image)
        loss_ssim = (1.0 - ssim(rendered_image, target_image.unsqueeze(0)))        
        rgb_loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * loss_ssim
        loss += rgb_loss * opt.lambda_rgb
      
        if opt.use_depth:
            gt_depth = viewpoint_cam.gt_depth.unsqueeze(0).to(dataset.data_device) 
            pearson_loss = pearson_depth_loss(rendered_depth[0], gt_depth[0])
            lp_loss = local_pearson_loss(rendered_depth[0], gt_depth[0], 128, 0.5)
            depth_loss = (pearson_loss + lp_loss) * opt.lambda_depth
            
            loss += depth_loss

        if opt.use_prior:
            prior_loss = prior_loss_cal(prior_model, rendered_prior, target_image, opt.lambda_prior)
            loss += prior_loss

        if opt.use_denoiser:
            loss_l1_low = l1_loss(rendered_low, low_image)
            loss_ssim_low = (1.0 - ssim(rendered_low, low_image.unsqueeze(0)))         
            rgb_loss_low = (1.0 - opt.lambda_dssim_low) * loss_l1_low + opt.lambda_dssim_low * loss_ssim_low
            loss += rgb_loss_low

            l2_loss_denoiser = l2_loss(rendered_denoised_list[-1], rendered_image) 
            loss_tv = tv_loss(rendered_denoised_list[-1].unsqueeze(0))
            loss = loss + l2_loss_denoiser + loss_tv
     
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_l1_loss_for_log = 0.4 * loss_l1.item() + 0.6 * ema_l1_loss_for_log
            ema_ssim_loss_for_log = 0.4 * loss_ssim.item() + 0.6 * ema_ssim_loss_for_log


            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "L1": f"{ema_l1_loss_for_log:.{4}f}", "SSIM": f"{ema_ssim_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            if iteration in testing_iterations:
                psnr_test, ssim_test = evaluate(opt.use_denoiser, iteration, testing_iterations, scene, render, (pipe, background))
                if psnr_test >= best_psnr or ssim_test >= best_ssim or iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    if psnr_test >= best_psnr:
                        best_psnr = psnr_test
                    if ssim_test >= best_ssim:
                        best_ssim = ssim_test          
        
                print("\n[Best PNSR: {}   Best SSIM: {}]".format(best_psnr, best_ssim))

            # Densification
            if iteration < opt.densify_until_iter: 
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)  

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity_3dgs()
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    
def evaluate(use_denoiser, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if iteration in testing_iterations:
        with open(scene.model_path + "/eval_result.txt", "a") as f:
            f.write(str(iteration) + '  total_points:  ' + str(scene.gaussians.get_xyz.shape[0]) + '\n')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['name'] == 'train':
                continue
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                render_path = os.path.join(scene.model_path, config['name'], "ours_{}".format(iteration), "renders")

                makedirs(render_path, exist_ok=True)

                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    if use_denoiser:
                        image = torch.clamp(render_result["denoised_list"][-1], 0.0, 1.0)
                    else:
                        image = torch.clamp(render_result["render"][-1], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                with open(scene.model_path + "/eval_result.txt", "a") as f:
                    f.write(str(iteration) + '  PSNR:  ' + str(psnr_test) + '  SSIM:  ' + str(ssim_test) + '\n')

        torch.cuda.empty_cache()
        return psnr_test, ssim_test

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000] + list(range(1500, 7_0001, 100)) + list(range(6000, 30_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000] + list(range(6000, 30_0001, 5000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default =None)
    parser.add_argument("--configs", type=str, default = "")


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Optimizing " + args.model_path)

    #safe_state(args.quiet)
    training(dataset=lp.extract(args),
             save_dir=args.model_path,
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
