import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import torch.nn as nn


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, use_depth=False, use_prior=False, \
                 load_iteration=None, shuffle=False, resolution_scales=[1.0], gap = 1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, use_depth, use_prior, args.eval_index)
        else:
            assert False, "Could not recognize scene type!"
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        # train poses
        num_train_samples = len(scene_info.train_cameras)
        self.name_list_according_to_cameras = []
        for i in range(num_train_samples):
            image_name = scene_info.train_cameras[i].image_name
            self.name_list_according_to_cameras.append(image_name)
        # test poses
        num_test_samples = len(scene_info.test_cameras)
        self.name_list_according_to_cameras_test = []
        for i in range(num_test_samples):
            image_name = scene_info.test_cameras[i].image_name
            self.name_list_according_to_cameras_test.append(image_name)


        self.R_q = nn.Parameter(torch.zeros(size=(num_train_samples, 3), dtype=torch.float32), requires_grad = True)
        self.T_q = nn.Parameter(torch.zeros(size=(num_train_samples, 3), dtype=torch.float32), requires_grad = True)
        parameters = [{'params': [self.R_q], 'lr': 5e-4, "name": "R_q"}, {'params': [self.T_q], 'lr': 5e-4, "name": "T_q"}]
        self.optimizer_pose = torch.optim.Adam(parameters, lr=5e-4)
        self.scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, 
                                                                milestones=list(range(0, 800, 30)),
                                                                gamma=0.9, last_epoch=-1)
        self.R_q_test = nn.Parameter(torch.zeros(size=(num_test_samples, 3), dtype=torch.float32), requires_grad = True)
        self.T_q_test = nn.Parameter(torch.zeros(size=(num_test_samples, 3), dtype=torch.float32), requires_grad = True)
        parameters_test = [{'params': [self.R_q_test], 'lr': 5e-4, "name": "R_q_test"}, {'params': [self.T_q_test], 'lr': 5e-4, "name": "T_q_test"}]
        self.optimizer_test = torch.optim.Adam(parameters_test, lr=5e-3)
        self.scheduler_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_test, 
                                                                milestones=list(range(0 , 200, 20)),
                                                                gamma=0.9, last_epoch=-1)      

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loading Training Cameras: {len(self.train_cameras[resolution_scale])} .")
            
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loading Test Cameras: {len(self.test_cameras[resolution_scale])} .")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
            path_denoiser = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), 'denoiser.pth')
            path_tonemapper = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), 'tonemapper.pth')
    

            state_dict1 = load_state_dict(path_denoiser, location='cpu')
            state_dict2 = load_state_dict(path_tonemapper, location='cpu')
    
            try:
                self.gaussians.denoiser.load_state_dict(state_dict1, strict=True)
                print('Loading denoiser parameters successful')
            except:
                print('Loading denoiser parameters error')

            try:
                self.gaussians.tonemapper.load_state_dict(state_dict2, strict=True)
                print('Loading tonemapper parameters successful')
            except:
                print('Loading tonemapper parameters error')

            gaussians.denoiser.cuda()
            gaussians.tonemapper.cuda()

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras)  

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        tonemapper_path = os.path.join(self.model_path, "point_cloud/iteration_{}/tonemapper.pth".format(iteration))
        denoiser_path = os.path.join(self.model_path, "point_cloud/iteration_{}/denoiser.pth".format(iteration))

        torch.save(self.gaussians.tonemapper.state_dict(), tonemapper_path)
        torch.save(self.gaussians.denoiser.state_dict(), denoiser_path)


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def step(self):
        self.optimizer_pose.step()
        self.optimizer_pose.zero_grad()
        
    def step_test(self):
        self.optimizer_test.step()
        self.optimizer_test.zero_grad()   