# The model for location-aware single image reflection removal.

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from collections import OrderedDict

class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class ResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf.
    # Which contains SE-layer implements.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SElayer(channel, se_reduction)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x)
        x = x * self.res_scale
        out = x + res
        return out


class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(2, 1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        pool_out = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(pool_out)
        return self.sigmoid(x)


class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x 


class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)
        
        out = x + res
        return out 


class SingleLaplacian(nn.Module):

    def __init__(self, device, dim=3):
        super(SingleLaplacian, self).__init__()

        # 2D laplacian kernel (2D LOG operator.).
        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable kernel.
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        # self.kernel = Variable(torch.FloatTensor(laplacian_kernel).to(device))
    
    def forward(self, x):
        # pyramid module in 4 scales.
        lap = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)

        return lap


class LaplacianPyramid(nn.Module):
    # filter laplacian LOG kernel, kernel size: 3.
    # The laplacian Pyramid is used to generate high frequency images.

    def __init__(self, device, dim=3):
        super(LaplacianPyramid, self).__init__()

        # 2D laplacian kernel (2D LOG operator).
        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)
        # learnable laplacian kernel
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        # self.kernel = Variable(torch.FloatTensor(laplacian_kernel).to(device))
    
    def forward(self, x):
        # pyramid module for 4 scales.
        x0 = F.interpolate(x, scale_factor=0.125, mode='bilinear')
        x1 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bilinear')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bilinear')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bilinear')

        return torch.cat([lap_0, lap_1, lap_2, lap_3], 1)


class LRM(nn.Module):

    def __init__(self, device):
        super(LRM, self).__init__()
        
        # Laplacian blocks 
        self.lap_pyramid = LaplacianPyramid(device, dim=6) # multi-scale laplacian submodules (RDMs)
        # self.lap_single = SingleLaplacian(device, dim=6)

        self.det_conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        
        # SE-resblocks(ReLU)
        self.det_conv1 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv2 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv3 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4_1 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4_2 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1)

        # Convolutional blocks for encoding laplacian features. 
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(24, 32, 3, 1, 1),
            nn.PReLU()
            )
        
        # SE-resblocks(P-ReLU)
        self.det_conv6 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv7 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv8 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv9 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv10 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv11 = ResidualBlock(32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        
        # Activations.
        self.p_relu = nn.PReLU()
        self.relu = nn.ReLU()

        # Convolutional block for RCMap_{i+1}
        self.det_conv_mask0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
            )
        
        # LSTM block.
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 * 4, 32 * 2, 3, 1, 1),
            nn.Sigmoid()
            )
        
        # Convolutional block for R_{i+1}
        self.det_conv_mask1 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.ReLU()
            )

        # Auto-Encoder.
        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.ReLU()
            )
        
        # Convolutional Block Attention Modules.
        self.cbam_block0 = ResidualCbamBlock(64, norm=None, cbam_reduction=2)

        self.cbam_block1 = ResidualCbamBlock(128, norm=None, cbam_reduction=4)
        self.cbam_block2 = ResidualCbamBlock(128, norm=None, cbam_reduction=4)

        self.cbam_block3 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)
        self.cbam_block4 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)
        self.cbam_block5 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)

    
    def forward(self, I, T, h, c):
        # I: original image.
        # T: transmission image.
        # h, c: hidden states for LSTM block in stage_1.

        x = torch.cat([I, T], 1)
        # get laplacian(frequency) information of [I,T].
        lap = self.lap_pyramid(x)

        # ----- Stage1 -----
        # encode [I, T].
        x = self.det_conv0(x)
        # se-resblock layer1 for [I, T] features.
        x = F.relu(self.det_conv1(x))
        x = F.relu(self.det_conv2(x))
        x = F.relu(self.det_conv3(x))
        # se-resblock layer2 for [I, T] features.
        x = F.relu(self.det_conv4(x))
        x = F.relu(self.det_conv4_1(x))
        x = F.relu(self.det_conv4_2(x))
        
        # encode [I_lap, T_lap].
        lap = self.det_conv5(lap)
        # se-resblock layer3 for [I_lap, T_lap] features (p-relu for activation.)
        lap = self.p_relu(self.det_conv6(lap))
        lap = self.p_relu(self.det_conv7(lap))
        lap = self.p_relu(self.det_conv8(lap))
        # predict RCMap from laplacian features.
        c_map = self.det_conv_mask0(lap)
        # se-resblock layer4 for [I_lap, T_lap] features (p-relu for activation.)
        lap = self.p_relu(self.det_conv9(lap))
        lap = self.p_relu(self.det_conv10(lap))
        lap = self.p_relu(self.det_conv11(lap))
        # suppress transmission features.
        lap = (1 - c_map) * lap
        
        # concat image & laplacian feature and recurrent features.
        x = torch.cat([x, lap, h], 1)
        
        # lstm.
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        reflect = self.det_conv_mask1(h)

        # ------ Stage2 ------ 
        # predict T_{i+1} with input: R_{i+1}, T_i, C_{i+1}.
        x = torch.cat([I, T, reflect, c_map], 1)
        x = self.conv1(x)
        x = self.cbam_block0(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        # feature spatial & channel attention.
        x = self.cbam_block1(x)
        x = self.cbam_block2(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # feature spatial & channel attention.
        x = self.cbam_block3(x)
        x = self.cbam_block4(x)
        x = self.cbam_block5(x)

        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)

        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)

        return h, c, c_map, reflect, frame1, frame2, x


class LocationAwareSIRR(nn.Module):

    def __init__(self, opts, device):
        super(LocationAwareSIRR, self).__init__()
        self.device = device
        self.opts  = opts
        self.visual_names = ['fake_Ts', 'fake_Rs', 'rcmaps', 'I']
        
        self.netG_T = LRM(device).to(device)

    def setup(self):
        # setup setting of the LRM.
        self.load_networks()
    
    def load_networks(self):
        # load parameters for the network: netG_T.
        model_name = 'model.pth'
        load_path = os.path.join(self.opts.model_dir, model_name)
        # net = self.netG_T.module
        state_dict = torch.load(load_path, map_location=str(self.device))
        print('Load the model from %s' % load_path)
        self.netG_T.load_state_dict(state_dict)
        return

    def get_current_visuals(self):
        # get the current visuals results.
        visual_result = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_result[name] = getattr(self, name)
        
        return visual_result
    
    def get_image_paths(self):
        # return the current image paths which are used to load current data.
        return self.image_paths
    
    def set_input(self, input):
        # load images dataset from dataloader.
        with torch.no_grad():
            self.I = input['I'].to(self.device)
            self.image_paths = input['I_path']

    def init(self):
        b,c,h,w = self.I.shape
        self.h = Variable(torch.zeros(b, 64, h, w, device=self.device))
        self.c = Variable(torch.zeros(b, 64, h, w, device=self.device))
        
        self.fake_T = self.I.clone().detach()
        self.fake_Ts = [self.fake_T]

        self.fake_R = torch.zeros_like(self.I, device=self.device)
        self.fake_Rs = [self.fake_R]

        self.rcmaps = []
        
    def forward(self):
        self.init()

        for i in range(3):
            self.h, self.c, self.c_map, self.fake_R, self.fake_T4, self.fake_T2, self.fake_T = \
                self.netG_T(self.I, self.fake_Ts[-1], self.h, self.c)

            self.rcmaps.append(self.c_map)
            self.fake_Rs.append(self.fake_R)
            self.fake_Ts.append(self.fake_T)
        
        # adjust the pixel values ranges.
        for i in range(len(self.fake_Ts)):
            self.fake_Ts[i] = torch.clamp(self.fake_Ts[i], min=0, max=1)
        for i in range(len(self.fake_Rs)):
            self.fake_Rs[i] = torch.clamp(self.fake_Rs[i], min=0, max=1)
    
    def eval(self):
        self.netG_T.eval()
    
    def inference(self):
        with torch.no_grad():
            self.forward()
