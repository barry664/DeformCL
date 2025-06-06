import torch
import torch.nn.functional as F
import torch.nn as nn
from itertools import product

class BasicFeatureSampling(nn.Module):

    def __init__(self):
        super(BasicFeatureSampling, self).__init__()
 

    def forward(self, voxel_features, vertices, pad_img_shape):

        center = vertices[:, :, None, None]  # (Batchsize, N_points, 1, 1, 3)
        features = F.grid_sample(voxel_features, center, mode='bilinear', padding_mode='border', align_corners=True)
        # (Batchsize, feature_dims, N_points, 1, 1)
        features = features[:, :, :, 0, 0].transpose(2, 1)  # (Batchsize, N_points, feature_dims)

        return features

class NeighborhoodFeatureSampling(nn.Module):

    def __init__(self, features_count):
        super(NeighborhoodFeatureSampling, self).__init__()

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda()

        torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)

    def forward(self, voxel_features, vertices, img_shape):

        H, W, D = img_shape
        self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2/(D-1), 2/(W-1), 2/(H-1)]]])[None]

        self.shift = self.shift.to(voxel_features.device)
        neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None]  # (Batchsize, n_points, 27 ,1, 3)
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)

        features = features[:, :, :, :, 0]  # (batchsize, feature_dims, n_points, 27)
        features = self.sum_neighbourhood(features)[:, :, :, 0].transpose(2, 1)  # (batchsize, n_points, feature_dims)

        return features

class LearntNeighbourhoodSampling(nn.Module):

    def __init__(self, features_count):
        super(LearntNeighbourhoodSampling, self).__init__()

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda()

        # torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)
        self.shift_delta = nn.Conv1d(features_count, 27*3, kernel_size=(1), padding=0).cuda()
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff_1 = nn.Linear(features_count + 3, features_count)
        self.feature_diff_2 = nn.Linear(features_count, features_count) 

        self.feature_center_1 = nn.Linear(features_count + 3, features_count)
        self.feature_center_2 = nn.Linear(features_count, features_count)

    def forward(self, voxel_features, vertices):

        B, N, _ = vertices.shape
        # divide by stride
        center = vertices[:, :, None, None]
        features = F.grid_sample(voxel_features, center, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 27, 1, 3)
        shift_delta[:,:,0,:,:] = shift_delta[:,:,0,:,:] * 0 # setting first shift to zero so it samples at the exact point
 
        # neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] + shift_delta
        neighbourhood = vertices[:, :, None, None] + shift_delta
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, :, 0]
        features = torch.cat([features, neighbourhood.permute(0,4,1,2,3)[:,:,:,:,0]], dim=1)

        features_diff_from_center = features - features[:,:,:,0][:,:,:,None] # 0 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        
        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres =  features[:,:,:,0].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center 
        return features
