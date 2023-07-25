# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
import torch


class IDM(nn.Module):
    def __init__(self, channel=64):
        super(IDM, self).__init__()
        self.channel = channel
        self.adaptiveFC1 = nn.Linear(2*channel, channel)
        self.adaptiveFC2 = nn.Linear(channel, int(channel/2))
        self.relu = nn.ReLU()
        self.adaptiveFC3 = nn.Linear(int(channel/2), 2)
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.init_std = 0.01
    
    def init_weights(self):
        normal_init(self.adaptiveFC1, std=self.init_std)
        normal_init(self.adaptiveFC2, std=self.init_std)
        normal_init(self.adaptiveFC3, std=self.init_std)


    def forward(self, x):

        if (not self.training):
            return x

        # torch.Size([B*2N, 3, 16, 384, 288])
        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        x_s = split[0].contiguous() # [B*N, C, L, H, W] N 表示 N_clips
        x_t = split[1].contiguous()

        x_embd_s = torch.cat((self.avg_pool(x_s.detach()).squeeze(), self.max_pool(x_s.detach()).squeeze()), 1)     # [B*N, 2*C]
        x_embd_t = torch.cat((self.avg_pool(x_t.detach()).squeeze(), self.max_pool(x_t.detach()).squeeze()), 1)     # [B*N, 2*C]
        # x_embd_s = torch.cat((self.avg_pool(x_s).squeeze(), self.max_pool(x_s).squeeze()), 1)     # [B*N, 2*C]
        # x_embd_t = torch.cat((self.avg_pool(x_t).squeeze(), self.max_pool(x_t).squeeze()), 1)     # [B*N, 2*C]

        x_embd_s, x_embd_t = self.adaptiveFC1(x_embd_s), self.adaptiveFC1(x_embd_t) # [B*N, C]
        x_embd = x_embd_s+x_embd_t
        x_embd = self.relu(self.adaptiveFC2(x_embd))
        lam = self.adaptiveFC3(x_embd)
        lam = self.softmax(lam) # [B*N, 2]
        x_inter = lam[:, 0].reshape(-1,1,1,1,1)*x_s + lam[:, 1].reshape(-1,1,1,1,1)*x_t
        out = torch.cat((x_s, x_t, x_inter), 0) # [B*3N, C, L, H, W] N 表示 N_clips
        return out, lam



@BACKBONES.register_module()
class IDMC3D(nn.Module):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        out_dim (int): The dimension of last layer feature (after flatten).
            Depends on the input shape. Default: 8192.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self,
                 pretrained=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 out_dim=8192,
                 dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.idm = IDM(channel=64)

        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1)#(input_size,hidden_size,num_layers)
        self.h0 = torch.randn(1, 81, 512).cuda() #(num_layers,batch,output_size)
        self.c0 = torch.randn(1, 81, 512).cuda() #(num_layers,batch,output_size)

        self.fc6 = nn.Linear(out_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                normal_init(param)
            else:
                nn.init.zeros_(param)

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            self.idm.init_weights()

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1a(x)
        x = self.pool1(x)
        if self.training:
            x, attention_lam = self.idm(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)       # torch.Size([15, 512, 1, 9, 9])

        # LSTM layer 
        if self.training:
            ns = x.size(0)
            assert (ns%3==0)
            split = torch.split(x, int(ns/3), 0)
            x_s = split[0].contiguous() # torch.Size([5, 512, 1, 9, 9])
            x_s = x_s.view(x_s.shape[0], x_s.shape[2], x_s.shape[3], x_s.shape[4], x_s.shape[1]) # torch.Size([5, 1, 9, 9, 512])
            split_shape = x_s.shape
            x_s = x_s.reshape((x_s.shape[0], -1, x_s.shape[-1]))    # torch.Size([5, 81, 512])

            x_t = split[1].contiguous()
            x_t = x_t.view(x_t.shape[0], x_t.shape[2], x_t.shape[3], x_t.shape[4], x_t.shape[1])
            x_t = x_t.reshape((x_t.shape[0], -1, x_t.shape[-1]))

            x_mixed = split[2].contiguous()
            x_mixed = x_mixed.view(x_mixed.shape[0], x_mixed.shape[2], x_mixed.shape[3], x_mixed.shape[4], x_mixed.shape[1])
            x_mixed = x_mixed.reshape((x_mixed.shape[0], -1, x_mixed.shape[-1]))

            x_s_out, (x_s_hn, x_s_cn) = self.rnn(x_s, (self.h0, self.c0))   # torch.Size([5, 81, 512])
            x_t_out, (x_t_hn, x_t_cn) = self.rnn(x_t, (self.h0, self.c0))   # torch.Size([5, 81, 512])
            x_mixed_out, (x_mixed_hn, x_mixed_cn) = self.rnn(x_mixed, (self.h0, self.c0))   # torch.Size([5, 81, 512])

            x_s_out = x_s_out.reshape(split_shape).view(split_shape[0], split_shape[4], split_shape[1], split_shape[2], split_shape[3]) # torch.Size([5, 512, 1, 9, 9])
            x_t_out = x_t_out.reshape(split_shape).view(split_shape[0], split_shape[4], split_shape[1], split_shape[2], split_shape[3])
            x_mixed_out = x_mixed_out.reshape(split_shape).view(split_shape[0], split_shape[4], split_shape[1], split_shape[2], split_shape[3])

            x_out = torch.cat((x_s_out, x_t_out, x_mixed_out), 0)   # torch.Size([15, 512, 1, 9, 9])
            return x_out, attention_lam
        else:
            x_t = x.contiguous()    # torch.Size([5, 512, 1, 9, 9])
            x_t = x_t.view(x_t.shape[0], x_t.shape[2], x_t.shape[3], x_t.shape[4], x_t.shape[1])    # torch.Size([5, 1, 9, 9, 512])
            split_shape = x_t.shape
            x_t = x_t.reshape((x_t.shape[0], -1, x_t.shape[-1]))    # torch.Size([5, 81, 512])
            x_t_out, (x_t_hn, x_t_cn) = self.rnn(x_t, (self.h0, self.c0))   # torch.Size([5, 81, 512])
            x_t_out = x_t_out.reshape(split_shape).view(split_shape[0], split_shape[4], split_shape[1], split_shape[2], split_shape[3])
            x_out = x_t_out
            return x_out


        # x = x.flatten(start_dim=1)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # if self.training:
        #     return x_out, attention_lam
        # return x_out
