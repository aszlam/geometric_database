from typing import Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from models.view_encoders.channel_encoders.depth.abstract_depth_encoder import (
    AbstractDepthEncoder,
)


class get_model(nn.Module):
    def __init__(self, num_classes, feature_only=True):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.featurize = nn.Identity() if feature_only else nn.LogSoftmax(dim=-1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.featurize(x)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


class Pointnet2DepthEncoder(AbstractDepthEncoder):
    def __init__(
        self,
        representation_dim: int,
        device: Union[str, torch.device],
        accumulate_operation: Callable = lambda x: torch.mean(x, dim=-2, keepdim=False),
        load_saved_model: bool = False,
        model_path: str = "",
        *args,
        **kwargs
    ):
        super().__init__()
        self.representation_dim = representation_dim
        self.pointnet_model = get_model(13)
        # Now load the model first.
        if load_saved_model:
            model_weights = torch.load(model_path)
            self.pointnet_model.load_state_dict(model_weights["model_state_dict"])
            self.pointnet_model.to(device)
            self.pointnet_model.requires_grad_(False)
        # Now replace the last conv layer.
        self.pointnet_model.conv2 = nn.Conv1d(128, representation_dim, 1)
        self.pointnet_model.conv2.requires_grad_(True)
        self.device = device

        self.accumulate_operation = accumulate_operation

    def to(self, device: Union[str, torch.device]) -> AbstractDepthEncoder:
        model = self.pointnet_model.to(device)
        model.requires_grad_(False)
        model.conv2.requires_grad_(True)
        self.pointnet_model = model
        self.device = device
        return self

    def encode_view(
        self, depth_view: torch.Tensor, rgb_view: torch.Tensor
    ) -> torch.Tensor:
        preprocessed_result = super().preprocess_view(
            depth_view=depth_view, rgb_data=rgb_view
        )
        per_point_rep = self.pointnet_model(preprocessed_result)
        # Now accumulate the per-point representations.
        return self.accumulate_operation(per_point_rep)


if __name__ == "__main__":
    import torch

    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
