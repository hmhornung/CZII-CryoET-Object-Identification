import torch
import torch.nn as nn
from monai.networks.nets import UNet

class MaskToPointUNet(nn.Module):
    def __init__(self, dropout):
        super(MaskToPointUNet, self).__init__()
        self.seg_unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=7,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            dropout=dropout
        )
        
        self.pts_unet = UNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=7,
            channels=(64, 128, 256),
            strides=(2, 2),
            num_res_units=1,
            dropout=dropout
        )
        
        self.full = True
    
    def forward(self, x):
        segmentation = torch.softmax(self.seg_unet(x), dim=1)
        
        if not self.full:
            return segmentation
        
        x = self.norm(torch.cat((segmentation, x), dim=1), dims=(1))
        
        x = torch.softmax(self.pts_unet(x), dim=1)
        
        return x
    
    def norm(self, tensor, dims):
        min_vals = torch.amin(tensor, dim=dims, keepdim=True)
        max_vals = torch.amax(tensor, dim=dims, keepdim=True)
        return (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    
    def full_pass(self, full):
        self.full = full
    
    def load_seg(self, seg):
        weights = torch.load(seg)
        self.seg_unet.load_state_dict(weights)
        
    def load_pts(self, pts):
        weights = torch.load(pts)
        self.pts_unet.load_state_dict(weights)
        
    def train_seg(self):
        self.seg_unet.train()
        self.seg_unet.requires_grad_(True)
        
        self.pts_unet.eval()
        self.pts_unet.requires_grad_(False)
    
    def train_pts(self):
        self.seg_unet.eval()
        self.seg_unet.requires_grad_(False)
        
        self.pts_unet.train()
        self.pts_unet.requires_grad_(True)
    
    def train_full(self):
        self.seg_unet.train()
        self.seg_unet.requires_grad_(True)
        
        self.pts_unet.train()
        self.pts_unet.requires_grad_(True)
    