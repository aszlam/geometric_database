from typing import Callable, Union
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from utils.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from models.view_encoders.channel_encoders.depth.abstract_depth_encoder import (
    AbstractDepthEncoder,
)


class get_model(nn.Module):
    def __init__(self, num_class, feature_only=True):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(
            global_feat=False, feature_transform=True, channel=9
        )
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.featurize = nn.Identity() if feature_only else nn.LogSoftmax(dim=-1)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = self.featurize(x.view(-1, self.k))
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class PointnetDepthEncoder(AbstractDepthEncoder):
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
        self.pointnet_model.k = representation_dim
        self.pointnet_model.conv4 = nn.Conv1d(128, representation_dim, 1)
        self.pointnet_model.conv4.requires_grad_(True)
        self.device = device

        self.accumulate_operation = accumulate_operation

    def to(self, device: Union[str, torch.device]) -> AbstractDepthEncoder:
        model = self.pointnet_model.to(device)
        model.requires_grad_(False)
        model.conv4.requires_grad_(True)
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
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))
