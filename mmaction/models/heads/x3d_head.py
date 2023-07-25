# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from collections import OrderedDict

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class X3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        fc1_bias (bool): If the first fc layer has bias. Default: False.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 pretrained=None,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 fc1_bias=False):
        super().__init__(num_classes, in_channels, loss_cls)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.pretrained = pretrained
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.in_channels = in_channels
        self.mid_channels = 2048
        self.num_classes = num_classes
        self.fc1_bias = fc1_bias

        self.fc1 = nn.Linear(
            self.in_channels, self.mid_channels, bias=self.fc1_bias)
        self.fc2 = nn.Linear(self.mid_channels, self.num_classes)

        self.relu = nn.ReLU()

        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise NotImplementedError

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if self.pretrained:
            model_CKPT = torch.load(self.pretrained)
            if 'state_dict' in model_CKPT:
                model_CKPT = model_CKPT['state_dict']
            model_CKPT_change = OrderedDict()
            for k, v in model_CKPT.items():
                if k == 'cls_head.fc1.weight':
                    model_CKPT_change['weight'] = v
            self.fc1.load_state_dict(model_CKPT_change)
        else:
            normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, T, H, W]
        assert self.pool is not None
        x = self.pool(x)
        # [N, in_channels, 1, 1, 1]
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        x = self.fc1(x)
        # [N, 2048]
        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc2(x)
        # [N, num_classes]
        return cls_score
