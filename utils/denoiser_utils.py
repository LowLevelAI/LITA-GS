import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F



class ExposureControl(nn.Module):
    def __init__(self, level=0.5):
        super(ExposureControl, self).__init__()
        self.level = torch.nn.Parameter(torch.tensor([level]), requires_grad=True)
    def forward(self, input):

        clamped_level = torch.clamp(self.level, 0.45, 0.55)

        low_mean = input.mean()
        target_image = torch.clamp(input * (clamped_level/low_mean), 0, 1)
        
        return target_image



class BaseModel(nn.Module):
    def __init__(self, channels=3, layers=6, mode=1):
        super(BaseModel, self).__init__()

        self.mode = mode

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def apply_gaussian_blur_cuda(self, channel, kernel_size=3):
        # 生成 3x3 高斯核
        sigma = kernel_size / 6.0  # 通常，sigma 和 kernel 大小相关
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=channel.device)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = x / x.sum()
    
        # 生成 2D 高斯核（通过两个1D核的外积）
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    
        # 调整形状为 [1, 1, kernel_size, kernel_size]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
        # 调整输入张量为 [1, 1, H, W] 进行卷积
        channel_4d = channel.unsqueeze(0).unsqueeze(0)
    
        # 使用 2D 卷积进行高斯模糊
        blurred = F.conv2d(channel_4d, kernel_2d, padding=kernel_size // 2)
    
        return blurred.squeeze(0).squeeze(0)

    def decreasing_sigmoid(self, x):
        return 1 / (1 + np.exp(x))
    
    def cal_input(self, reflectance, denoised, noise_gs, iter_num):

        a = denoised.squeeze(0)
        noise_ = denoised - torch.stack([self.apply_gaussian_blur_cuda(a[i], kernel_size=3) for i in range(a.shape[0])]).unsqueeze(0).cuda()

        if iter_num == 0:
            if self.mode == 1:
                scale = 1
            else:
                scale = 0.5
            result = (1 - scale) * noise_ + scale * noise_gs.unsqueeze(0)
        else:
            
            if self.mode == 1:
                result = denoised
            else:
                result = noise_
            #- 0.5 * (reflectance - denoised)
        return result       

    def forward(self, reflectance, denoised, noise_gs, iter_num):

        input = self.cal_input(reflectance, denoised, noise_gs, iter_num)
        
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        output = input + fea  ## output: N_(k+1)   fea: - F_{PDM}(\hat{N}_k)

        denoised = torch.clamp(reflectance - output, 0, 1)
        return denoised

class Denoiser(nn.Module):
    def __init__(self, K=3, hidden_channel=64, layers=6, mode=1):
        super(Denoiser, self).__init__()
        self.K = K
        self.net = nn.ModuleList()
        for i in range(self.K):
            self.net.append(BaseModel(channels=hidden_channel, layers=layers, mode=mode))   

    def forward(self, reflectance, noise):
        #print(reflectance.shape)  #[3, 375, 500]
        #print(noise.shape) [3, 375,500]
        denoised_list = []
        denoised = reflectance.unsqueeze(0)
        for i in range(self.K):
            denoised = self.net[i](reflectance, denoised, noise, i)
            #print(denoised.shape)  #[1, 3, 375, 500]
            denoised_list.append(denoised.squeeze(0))
        return denoised_list