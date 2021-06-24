import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from .resnet_updated import conv3x3, conv1x1
from .resnet_updated import resnetbank50all as resnetbank50
from .hourglass import HourglassNet
from .hourglass import Bottleneck as hgBottleneck
from .globalNet import globalNet
from .reconstruction import KptUNet

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
    def __init__(self, n_kpts=10, output_dim=200, pretrained=True, cls_kpts=5, geometry=True, pseudo=True):
        super(Model, self).__init__()
        self.K = n_kpts
        self.K_cls = cls_kpts
        
        channel_settings = [2048, 1024, 512, 256]
        output_shape = (64, 64)  #(56, 56), (28,28) (32, 32)
        self.output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kpts)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.decoder = Decoder(in_planes=2048, wh=4, n_kpts=self.K) #+45)
        
        feat_channels = [2048, 1024, 512, 256]
        laterals = []
        for i in range(len(feat_channels)):
            laterals.append(self._lateral(feat_channels[i], (64, 64)))  # (128, 128)
        self.laterals = nn.ModuleList(laterals)
        # upsampling and 1x1 conv layer
        
        # self.part_classifier = nn.Linear(256*len(feat_channels)*self.K_cls, output_dim)
        self.classifier = nn.Linear(256*len(feat_channels)*self.K_cls, output_dim)  #  + 2048
        # self.fc = nn.Linear(2048, output_dim)
        
        self.geometry = geometry
        if geometry:
            self.gmtr_fc = nn.Linear(n_kpts*2, n_kpts)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)
        self.pseudo = pseudo
        
    
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
    
    
    def forward(self, x, tr_x, transform=None, appr_x=None, pseudo_run=False):
        
        if pseudo_run:
            output = self.run_pose(x)
            return output
            
        x_res = self.encoder(x)
        cls_feat_banks = self.concat_features(x_res) #[x_res[0], x_res[1]])  # feature block for image classification
        
        tr_x_res = self.encoder(tr_x)
        
        tr_kpt_feat, tr_kpt_out = self.kptNet(tr_x_res)  # keypoint for reconstruction
        kpt_feat, kpt_out = self.kptNet(x_res)
        
        # Classification module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
        
        # confidence score as the maximum of normalized heatmap
        # if the prediction is uncertain, heatmap will look like a huge blurred map
        confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]
        
        u_x, u_y, covs = self._mapTokpt(heatmap)
        # kpt_conds = self._kptTomap(u_x, u_y, H=self.output_shape[0], W=self.output_shape[1], 
        #                            inv_std=0.01, normalize=False)  # 0.0005
        
        _cls_feats = self._kptAttend(cls_feat_banks, heatmap[:,:self.K_cls,:,:]) 
        #kpt_conds) #kpt_out[-1][:,:self.K_cls,:,:] -> error, kpt_conds[:,:self.K_cls,:,:]->not working
        
        ## image feature        
        cls_feats = _cls_feats
        cls_feats = cls_feats.view(-1, cls_feats.size(1)*cls_feats.size(2))  # (N, K, C) -> (N, K*C)
        
        id_feats = self.classifier(cls_feats)
        
        # img_feat = self.avgpool(x_res[0])
        # img_feat = img_feat.view(-1, img_feat.size(1))
        # img_part_feats = torch.cat((img_feat, cls_feats), dim=1)
        # img_feat = self.fc(img_feat)
        
        # id_feats = (id_feats + img_feat) / 2.0
        # id_feats = self.classifier(img_part_feats)
        
        # Reconstruction module
        tr_heatmap = tr_kpt_out[-1].view(-1, self.K, tr_kpt_out[-1].size(2) * tr_kpt_out[-1].size(3))
        tr_heatmap = self.ch_softmax(tr_heatmap)
        tr_heatmap = tr_heatmap.view(-1, self.K, tr_kpt_out[-1].size(2), tr_kpt_out[-1].size(3))
        
        # tr_confidence = tr_heatmap.max(dim=-1)[0].max(dim=-1)[0]
        
        tr_u_x, tr_u_y, tr_covs = self._mapTokpt(tr_heatmap)
        tr_kpt_conds = []
        prev_size = 2
        std_in = [0.1, 0.1, 0.01, 0.01, 0.001] #0.01] # 0.001]
        # std_in = [0.1, 0.01, 0.01, 0.001]
        # std_in = [10, 10, 10, 10, 10]
        
        for i in range(0, len(std_in)):
            prev_size = prev_size * 2
            
            hmaps = self._kptTomap(tr_u_x, tr_u_y, H=prev_size, W=prev_size, inv_std=std_in[i], normalize=False)
            tr_kpt_conds.append(hmaps)
        
        recon = self.decoder(x_res, tr_kpt_conds)
        
        # Use transformed image feature for classification?
        
        # if self.geometry:
        if appr_x is not None:            
            appr_res = self.encoder(appr_x)
            appr_kpt_feat, appr_kpt_out = self.kptNet(appr_res)
            
            appr_heatmap = appr_kpt_out[-1].view(-1, self.K, appr_kpt_out[-1].size(2) * appr_kpt_out[-1].size(3))
            appr_heatmap = self.ch_softmax(appr_heatmap)
            appr_heatmap = appr_heatmap.view(-1, self.K, appr_kpt_out[-1].size(2), appr_kpt_out[-1].size(3))
            confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]

            appr_u_x, appr_u_y, appr_covs = self._mapTokpt(appr_heatmap)
                        
            flip_x = x.flip(-1)
            
            flip_x_res = self.encoder(flip_x)
            flip_kpt_feat, flip_kpt_out = self.kptNet(flip_x_res)
            
            flip_heatmap = flip_kpt_out[-1].view(-1, self.K, flip_kpt_out[-1].size(2) * flip_kpt_out[-1].size(3))
            flip_heatmap = self.ch_softmax(flip_heatmap)
            flip_heatmap = flip_heatmap.view(-1, self.K, flip_kpt_out[-1].size(2), flip_kpt_out[-1].size(3))
            flip_confidence = flip_heatmap.max(dim=-1)[0].max(dim=-1)[0]
            
            flip_u_x, flip_u_y, flip_covs = self._mapTokpt(flip_heatmap)
            
            anchor_pt = torch.cat((u_x, u_y), dim=1)
            # pos_pt = torch.cat((tr_u_x, tr_u_y), dim=1)
            pos_pt = torch.cat((appr_u_x, appr_u_y), dim=1)
            neg_pt = torch.cat((flip_u_x, flip_u_y), dim=1)
            
            # anchor_pt = torch.cat((u_x, u_y, confidence), dim=1)
            # pos_pt = torch.cat((tr_u_x, tr_u_y, tr_confidence), dim=1)
            # neg_pt = torch.cat((flip_u_x, flip_u_y, flip_confidence), dim=1)
            
            anchor_pt = self.gmtr_fc(anchor_pt)
            anchor_pt = F.relu(anchor_pt)
            # anchor_pt = F.normalize(anchor_pt, dim=1)
            pos_pt = self.gmtr_fc(pos_pt)
            pos_pt = F.relu(pos_pt)
            # pos_pt = F.normalize(pos_pt, dim=1)
            neg_pt = self.gmtr_fc(neg_pt)
            neg_pt = F.relu(neg_pt)
            # neg_pt = F.normalize(neg_pt, dim=1)
            
            return id_feats, recon, (tr_u_x, tr_u_y), heatmap, tr_kpt_conds[-1], _cls_feats, tr_kpt_out, confidence, (flip_u_x, flip_u_y), (anchor_pt, pos_pt, neg_pt)
        
        return id_feats, recon, (tr_u_x, tr_u_y), heatmap, kpt_out[-1], _cls_feats, tr_kpt_out, confidence, (u_x, u_y), tr_kpt_conds[-2]
        
    def concat_features(self, feats):
        
        output = self.laterals[0](feats[0])
        for i in range(1, len(self.laterals)):
            _output = self.laterals[i](feats[i])
            output = torch.cat((output, _output), dim=1)
        
        """
        output = self.laterals[-1](feats[3])
        for i in range(len(feats)-2):
            _output = self.laterals[i](feats[i])
            output = torch.cat((output, _output), dim=1)
        """ 
        return output
        
    def _mapTokpt(self, heatmap):
        # heatmap: (N, K, H, W)    
            
        H = heatmap.size(2)
        W = heatmap.size(3)
        
        s_y = heatmap.sum(3)  # (N, K, H)
        s_x = heatmap.sum(2)  # (N, K, W)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
        # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
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
        
        # g_y = torch.exp(- ((mu_y - y).pow(2) / (2 * inv_std) ) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        # g_x = torch.exp(- ((mu_x - x).pow(2) / (2 * inv_std) ) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        # g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
        # g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))
        
        # g_y = g_y.unsqueeze(3)
        # g_yx = torch.matmul(g_y, g_x)  # (N, K, H, W)
        # g_yx = g_y * g_x
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        # g_yx = g_yx + 1e-9
        # import pdb; pdb.set_trace()
        
        # am = g_yx.max(dim=2)[0].max(dim=2)[0]
        # g_yx = g_yx / am.view(g_yx.size(0), g_yx.size(1), 1, 1)
        
        return g_yx
    
    def _kptTolimb(self, u_x, u_y, cov_u, H=16, W=16, inv_std=1.0/0.1, concat=None, normalize=False):
        # Known class, known pair of keypoints
        # AnimalPose:
        # kpt_labels = {'L_Eye': 0, 'R_Eye': 1, 'Nose': 2, 'L_EarBase': 3, 'R_EarBase': 4,
        #           'Throat': 5, 'Withers': 6, 'L_F_Elbow': 7, 'R_F_Elbow': 8,
        #           'L_B_Elbow': 9, 'R_B_Elbow': 10, 'L_F_Paw': 11, 'R_F_Paw': 12,
        #           'L_B_Paw': 13, 'R_B_Paw': 14, 'TailBase': 15, 'L_F_Knee': 16, 'R_B_Knee': 17,
        #           'R_F_Knee': 18, 'L_B_Knee': 19}
        """
        # pairs = [[3,4], [4,5], [4,6], [6,7]]
        pairs = [[0,2], [0,1], [1,2], [0,3], [1,3],
                 [3,5], [5,7], [3,4], [4,6], [6,8]] #, [4,9]]
        # pairs = [[0,1], [1,2], [2,3], [3,4], [4,5], 
        #          [5,6], [6,7], [7,8], [8,9]]
        pairs = np.asarray(pairs)
        """
        pairs = list(itertools.combinations(range(self.K), 2))
        pairs = np.asarray(pairs)
        # pairs = list(itertools.product(range(self.K_cls), range(self.K_cls,self.K)))
        # pairs = np.asarray(pairs)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, 1, H, 1, 1))
        x = torch.reshape(x, (1, 1, 1, W, 1))
        
        mu_x = u_x.unsqueeze(2)  # (N, K, 1)
        mu_y = u_y.unsqueeze(2)  # (N, K, 1)
        
        mu_u_x = mu_x[:,pairs[:,0],:]
        mu_u_y = mu_y[:,pairs[:,0],:]
                
        mu_v_x = mu_x[:,pairs[:,1],:]
        mu_v_y = mu_y[:,pairs[:,1],:]
        
        alpha = torch.linspace(0, 1.0, H).cuda()
        eq_x = alpha * mu_u_x + (1 - alpha) * mu_v_x  # Representative points (N, K, #points)
        eq_y = alpha * mu_u_y + (1 - alpha) * mu_v_y
        eq_x = eq_x.unsqueeze(2).unsqueeze(3)
        eq_y = eq_y.unsqueeze(2).unsqueeze(3)
        
        min_dist = ((x - eq_x).pow(2) + (y - eq_y).pow(2)) #.sqrt()  # (N, K, H, W, #pts)
        min_dist, _ = torch.min(min_dist, dim=4)
        
        # limb heatmap (from predefined sets)
        g_yx = torch.exp( - min_dist / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        # g_yx = torch.exp(- ((x - eq_x).pow(2) + (y - eq_y).pow(2)) / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        
        if concat is not None:
            g_yx = torch.cat([concat, g_yx], dim=1)
        
        return g_yx        
    
    
#     def _kptToMultivariate(self, u_x, u_y, covs, H=16, W=16, normalize=False, const=0.1):
#         det = (covs[0] * covs[1] - covs[2].pow(2))
#         # compute covariance for each kpt
        
#         cov_mat1 = 
    
    
    def _kptTodist(self, u_x, u_y, covs, inv_std=1.0/0.1, H=16, W=16, normalize=False, conc_const=0.1):
        # u_x: (N x K)
        # u_y: (N x K)
        # covs: (var_x, var_y, cov) -> (N, K, 2, 2) each variable (N, K)
        # For now, use independent x, y distribution
        
        # conc_const = 0.1  # 1.0, 0.1, 0.01
        # covs[0] = conc_const * covs[0]
        # covs[1] = conc_const * covs[1]
        
        # when using covariance matrix, cov should be psd
        # det = conc_const * conc_const * (covs[0] * covs[1] - covs[2].pow(2))
        # det = (covs[0] * covs[1] - covs[2].pow(2))
        # det = covs[0] * covs[1] - covs[2].pow(2)  # (N, K)
        # covs[0] = covs[0] + 1e-6
        # covs[1] = covs[1] + 1e-6
        # Normalize covariance? 
        # import pdb; pdb.set_trace()
        
        # For keypoints with low uncertainty -> more localized
        # high uncertainty -> high localized
        # import pdb; pdb.set_trace()
        # det thr
        # uncertainty = torch.argsort(det, dim=1)
        # conc_const = torch.ones(det.size(0), self.K).cuda() * 0.1
        # for ii in range(0, 5):
        #     kpt_mask[torch_index, high_idx[:,ii]] = 1
        # conc_const[covs[0] > 0.1] = 1
        # conc_const[covs[1] > 0.1] = 1
        
        # det = conc_const * conc_const * det
        
        cov_mat1 = torch.stack((conc_const * covs[1], conc_const * covs[2]), dim=2)
        cov_mat2 = torch.stack((conc_const * covs[2], conc_const * covs[0]), dim=2)
        # cov_mat1 = torch.stack((covs[1], covs[2]), dim=2)  # (N, K, 2)
        # cov_mat2 = torch.stack((covs[2], covs[0]), dim=2)
        cov_mat = torch.stack((cov_mat1, cov_mat2), dim=3)  # (N, K, 2, 2)
        # cov_mat = cov_mat * conc_const
        
        det = (cov_mat[:,:,0,0] * cov_mat[:,:,1,1] - cov_mat[:,:,0,1] * cov_mat[:,:,1,0]).abs()
        
        inv_covmat = torch.inverse(cov_mat)
        inv_covmat = inv_covmat.unsqueeze(2)
                
        # y = torch.reshape(self.H_tensor, (1, 1, H, 1))
        # x = torch.reshape(self.W_tensor, (1, 1, 1, W))
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
                        
        mu_x = u_x.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        mu_y = u_y.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        
        y_minus_mu = y - mu_y  # (N, K, H, 1)
        y_minus_mu = y_minus_mu.repeat((1,1,1,W))
        x_minus_mu = x - mu_x  # (N, K, 1, W)
        x_minus_mu = x_minus_mu.repeat((1,1,H,1))
        # import pdb; pdb.set_trace()
        xy_minus_mu = torch.stack((y_minus_mu, x_minus_mu), dim=4) # (N, K, H, W, 2)
        xy_minus_mu = xy_minus_mu.reshape((-1, self.K, H*W, 1, 2))
        xy_minus_mu_tr = xy_minus_mu.transpose(3,4)
        
        # den = 1.0 / (2 * math.pi * det * conc_const).sqrt()  # (N, K)
        den = 1.0 / 2 * math.pi * det.sqrt()  # (N, K)
        den = den.reshape((den.size(0), den.size(1), 1))
        
        num = torch.matmul(xy_minus_mu, inv_covmat)
        num = torch.matmul(num, xy_minus_mu_tr)
        num = torch.exp( - num / 2.0 )  # (N, K, H*W, 1)
        num = num.reshape((num.size(0), num.size(1), num.size(2)))
        
        g_yx = (den * num).reshape((num.size(0), num.size(1), H, W))
        # pdb.set_trace()
        """
        num = torch.matmul(xy_minus_mu, inv_covmat)
        num = torch.matmul(num, xy_minus_mu_tr)
        num = num + 1.0
        
        g_yx = (1.0 / num).reshape((num.size(0), num.size(1), H, W))
        """
        
#         var_y = covs[1].unsqueeze(2).unsqueeze(3)  # (N, K, 1)
#         var_x = covs[0].unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        
#         g_y = torch.exp(- ((y - mu_y).pow(2) / (2 * var_y * conc_const) ) ) / (2 * math.pi * var_y * conc_const).sqrt()
#         g_x = torch.exp(- ((x - mu_x).pow(2) / (2 * var_x * conc_const) ) ) / (2 * math.pi * var_x * conc_const).sqrt()
        
#         # g_y = g_y.unsqueeze(3)
#         _g_yx = g_y * g_x  # (N, K, H, W)
                
        # g_yx = torch.matmul(g_y, g_x)  # (N, K, H, W)
        
        # BINARY MASK?????????
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        # g_yx = g_yx + 1e-6
        
        return g_yx
        
    def _kptAttend(self, feat, heatmap):
        # feat: (N, C, H, W)
        # heatmap: (N, K, H, W)
        """
        # import pdb; pdb.set_trace()
        aggr_feat = feat.unsqueeze(1)  # (N, 1, C, H, W)
        aggr_hmap = heatmap.unsqueeze(2)  # (N, K, 1, H, W)
        
        aggr = aggr_feat * aggr_hmap  # (N, K, C, H, W)
        # aggr = aggr.view(-1, aggr.size(2), aggr.size(3), aggr.size(4))
        # aggr = self.maxpool(aggr)
        # aggr = aggr.view(heatmap.size(0), heatmap.size(1), -1)
        
        aggr = aggr.sum(3).sum(3)  # MAXPOOL? AVGPOOL? now: weighted average
        """
        aggr = feat * heatmap[:,0,:,:].unsqueeze(1)
        # aggr = self.avgpool(aggr)
        # aggr = aggr.view(aggr.size(0), 1, -1)
        
        aggr = aggr.sum(2).sum(2) # (ORIGINAL ONE)
        aggr = aggr.unsqueeze(1)  # (ORIGINAL ONE)
        for i in range(1, heatmap.size(1)):
            _aggr = feat * heatmap[:,i,:,:].unsqueeze(1)
            # _aggr = self.avgpool(_aggr)
            # _aggr = _aggr.view(_aggr.size(0), 1, -1)
            
            _aggr = _aggr.sum(2).sum(2)
            _aggr = _aggr.unsqueeze(1)
            aggr = torch.cat((aggr, _aggr), dim=1)
        
        return aggr 
    