import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from .resnet_updated import conv3x3, conv1x1
from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet

import numpy as np
import math
import itertools


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        
        if self.upsample is not None:
            x = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    


class Decoder(nn.Module):
    def __init__(self, in_planes=256, wh=14, n_kpts=10):
        super(Decoder, self).__init__()
        
        self.K = n_kpts
        # feat_dim = in_planes + self.K
        
        self.layer1 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample(int(wh*2), mode='bilinear')); in_planes /= 2
        self.layer2 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample(int(wh*4), mode='bilinear')); in_planes /= 2
        self.layer3 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample(int(wh*8), mode='bilinear')); in_planes /= 2
        self.layer4 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample(int(wh*16), mode='bilinear')); in_planes /= 2
        self.layer5 = BasicBlock(int(in_planes)+self.K, max(int(in_planes/2), 32),
                                 upsample=nn.Upsample(int(wh*32), mode='bilinear'))
        # self.layer5 = BasicBlock(int(in_planes)+self.K, max(int(in_planes/2), 32),
        #                          upsample=None)
        in_planes = max(int(in_planes/2), 32)
        self.conv_final = nn.Conv2d(int(in_planes), 3, kernel_size=1, stride=1)
        
    def forward(self, x, heatmap):
        # import pdb; pdb.set_trace()
        
        # feat = x[0] * heatmap[0].sum(1, keepdim=True)
        # x = torch.cat((feat, heatmap[0]), dim=1)
        x = torch.cat((x[0], heatmap[0]), dim=1)
        x = self.layer1(x)
        x = torch.cat((x, heatmap[1]), dim=1)
        x = self.layer2(x)
        x = torch.cat((x, heatmap[2]), dim=1)
        x = self.layer3(x)
        x = torch.cat((x, heatmap[3]), dim=1)
        x = self.layer4(x)
        
        x = torch.cat((x, heatmap[4]), dim=1)
        x = self.layer5(x)
        x = self.conv_final(x)
        
        return x

    
class Model(nn.Module):
    def __init__(self, n_kpts=10, output_dim=200, pretrained=True, cls_kpts=5):
        super(Model, self).__init__()
        self.K = n_kpts
        self.K_cls = cls_kpts
        
        channel_settings = [2048, 1024, 512, 256]
        output_shape = (64, 64) 
        self.output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kpts)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.decoder = Decoder(in_planes=2048, wh=4, n_kpts=self.K) 
        
        feat_channels = [2048, 1024, 512, 256]
        laterals = []
        for i in range(len(feat_channels)):
            laterals.append(self._lateral(feat_channels[i], (64, 64))) 
        self.laterals = nn.ModuleList(laterals)
        # upsampling and 1x1 conv layer
        
        self.classifier = nn.Linear(256*len(feat_channels)*self.K_cls, output_dim)  #  + 2048
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)
        
    
    def _simple_lateral(self, input_size, output_shape):
        out_dim = 256
        
        layers = []
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(input_size, out_dim,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
    
    def _lateral(self, input_size, output_shape):
        out_dim = 256
        
        layers = []
        layers.append(nn.Conv2d(input_size, out_dim,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_dim, out_dim,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
    def run_pose(self, x):
        x_res = self.encoder(x)
        kpt_feat, kpt_out = self.kptNet(x_res)
        
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
        
        confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]
        
        u_x, u_y, covs = self._mapTokpt(heatmap)
        
        return kpt_out, (u_x, u_y), confidence
    
    
    def forward(self, x, tr_x, transform=None, pseudo_run=False):
        
        if pseudo_run:
            output = self.run_pose(x)
            return output
            
        x_res = self.encoder(x)
        cls_feat_banks = self.concat_features(x_res) # feature block for image classification
        
        tr_x_res = self.encoder(tr_x)
        
        tr_kpt_feat, tr_kpt_out = self.kptNet(tr_x_res)  # keypoint for reconstruction
        kpt_feat, kpt_out = self.kptNet(x_res)
        
        # Classification module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs = self._mapTokpt(heatmap)
        
        _cls_feats = self._kptAttend(cls_feat_banks, heatmap[:,:self.K_cls,:,:]) 
        
        ## image feature        
        cls_feats = _cls_feats
        cls_feats = cls_feats.view(-1, cls_feats.size(1)*cls_feats.size(2))  # (N, K, C) -> (N, K*C)
        
        id_feats = self.classifier(cls_feats)
                
        # Reconstruction module
        tr_heatmap = tr_kpt_out[-1].view(-1, self.K, tr_kpt_out[-1].size(2) * tr_kpt_out[-1].size(3))
        tr_heatmap = self.ch_softmax(tr_heatmap)
        tr_heatmap = tr_heatmap.view(-1, self.K, tr_kpt_out[-1].size(2), tr_kpt_out[-1].size(3))
                
        tr_u_x, tr_u_y, tr_covs = self._mapTokpt(tr_heatmap)
        tr_kpt_conds = []
        prev_size = 2
        std_in = [0.1, 0.1, 0.01, 0.01, 0.001]
        
        for i in range(0, len(std_in)):
            prev_size = prev_size * 2
            
            hmaps = self._kptTomap(tr_u_x, tr_u_y, H=prev_size, W=prev_size, inv_std=std_in[i], normalize=False)
            tr_kpt_conds.append(hmaps)
        
        recon = self.decoder(x_res, tr_kpt_conds)
                
        return id_feats, recon, (tr_u_x, tr_u_y), heatmap, kpt_out[-1], _cls_feats, tr_kpt_out, (u_x, u_y), tr_kpt_conds[-2]
        
    def concat_features(self, feats):
        
        output = self.laterals[0](feats[0])
        for i in range(1, len(self.laterals)):
            _output = self.laterals[i](feats[i])
            output = torch.cat((output, _output), dim=1)
        
        return output
        
    def _mapTokpt(self, heatmap):
        # heatmap: (N, K, H, W)    
            
        H = heatmap.size(2)
        W = heatmap.size(3)
        
        s_y = heatmap.sum(3)  # (N, K, H)
        s_x = heatmap.sum(2)  # (N, K, W)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
        u_x = (x * s_x).sum(2) / s_x.sum(2)
        
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
        
        # Covariance
        var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
        var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
        cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
        return u_x, u_y, (var_x, var_y, cov)
    
    
    def _kptTomap(self, u_x, u_y, inv_std=1.0/0.1, H=16, W=16, normalize=False):
        # u_x: (N x K)
        # u_y: (N x K)
        
        mu_x = u_x.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        mu_y = u_y.unsqueeze(2) #.unsqueeze(3)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, H))
        x = torch.reshape(x, (1, 1, W))
        
        g_y = (mu_y - y).pow(2)
        g_x = (mu_x - x).pow(2)
        
        g_y = g_y.unsqueeze(3)
        g_yx = g_y + g_x
        
        g_yx = torch.exp(- g_yx / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
                
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        
        return g_yx
    
   
    def _kptAttend(self, feat, heatmap):
        # feat: (N, C, H, W)
        # heatmap: (N, K, H, W)
        
        aggr = feat * heatmap[:,0,:,:].unsqueeze(1)
        
        aggr = aggr.sum(2).sum(2) # (ORIGINAL ONE)
        aggr = aggr.unsqueeze(1)  # (ORIGINAL ONE)
        for i in range(1, heatmap.size(1)):
            _aggr = feat * heatmap[:,i,:,:].unsqueeze(1)
            
            _aggr = _aggr.sum(2).sum(2)
            _aggr = _aggr.unsqueeze(1)
            aggr = torch.cat((aggr, _aggr), dim=1)
        
        return aggr 
    