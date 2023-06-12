import torch
import torch.nn as nn
from torch.nn.functional import pairwise_distance
from pytorch3d.loss import chamfer_distance

# define losses
def voxel_loss(voxel_src, voxel_tgt):
    loss = nn.BCELoss()
    # implement some loss for binary voxel grids
    prob_loss = loss(voxel_src, voxel_tgt.float())
    return prob_loss

def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # loss_chamfer, _ = chamfer_distance(point_cloud_tgt, point_cloud_src) 
    
    # implement chamfer loss from scratch
    dist_matrix = torch.cdist(point_cloud_src, point_cloud_tgt)  # 計算距離矩陣

    chamfer_dist_A = torch.min(dist_matrix ** 2, dim=2)[0]  
    chamfer_dist_B = torch.min(dist_matrix ** 2, dim=1)[0]  

    chamfer_loss = torch.mean(chamfer_dist_A, dim=1) + torch.mean(chamfer_dist_B, dim=1) 

    loss_chamfer = chamfer_loss.mean()

    return loss_chamfer

# def smoothness_loss(mesh_src):
# 	# loss = 
# 	# implement laplacian smoothening loss
# 	return loss_laplacian

class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)**2
        dist2 = torch.sqrt(dist2)**2

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)
        
        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3