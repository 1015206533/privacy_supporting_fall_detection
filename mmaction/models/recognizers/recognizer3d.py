# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from mmaction.utils import get_root_logger
from ..builder import RECOGNIZERS
from .base import BaseRecognizer, GRL


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        # imgs.shape_v1_1, torch.Size([1, 10, 3, 16, 384, 288])
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # imgs.shape_v1_2, torch.Size([10, 3, 16, 384, 288])
        losses = dict()

        domain_loss_lambda = 1.0
        if 'domain_loss_lambda' in kwargs:
            domain_loss_lambda = kwargs.pop('domain_loss_lambda')

        x = self.extract_feat(imgs) # train_x.shape_v1, torch.Size([10, 432, 16, 12, 9])
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        flip_x = GRL.apply(x, 1.0)
        domain_score = self.domain_head(flip_x)
        domain_gt_labels = kwargs.pop('domain_label').squeeze()
        loss_domain = self.domain_head.loss(domain_score, domain_gt_labels, **kwargs)

        cls_score = self.cls_head(x)    #   cls_score_size, torch.Size([10, 2])
        #logger = get_root_logger()
        #logger.info(f'cls_score_size, {cls_score.shape}, {domain_score.shape}, {imgs.shape}')
        gt_labels = labels.squeeze()    # train_labels.shape_v1, torch.Size([1, 10])    train_gt_labels.shape_v1, torch.Size([10])
        
        loss_cls = self.cls_head.loss(cls_score[domain_gt_labels==1], gt_labels[domain_gt_labels==1], **kwargs)
        if torch.isnan(loss_cls['loss_cls']).any():
            losses['loss_all'] = domain_loss_lambda * loss_domain['loss_cls']
        else:
            losses['loss_all'] = loss_cls['loss_cls'] + domain_loss_lambda * loss_domain['loss_cls']
        
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses['loss_all'] = loss_cls['loss_cls']

        # losses.update(loss_cls)
        #logger = get_root_logger()
        #logger.info(f'domain_loss_lambda, {domain_loss_lambda}')
        #logger.info(f'cls_score, {cls_score}')
        #logger.info(f'gt_labels, {gt_labels}')
        #logger.info(f'domain_score, {domain_score}')
        #logger.info(f'domain_gt_labels, {domain_gt_labels}')
        #logger.info(f'self.cls_head.fc2.weight, {self.cls_head.fc2.weight}')
        #logger.info(f'self.domain_head.fc2.weight, {self.domain_head.fc2.weight}')

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        #logger = get_root_logger()
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        #logger.info(f'imgs.shape_v1, {imgs.shape}, {batches}, {num_segs}')
        # imgs.shape_v1, torch.Size([1, 10, 3, 16, 384, 288]), 1, 10
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # imgs.shape_v2, torch.Size([10, 3, 16, 384, 288]), 1, 10
        #logger.info(f'imgs.shape_v2, {imgs.shape}, {batches}, {num_segs}')

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        # test_feat.shape_v1, torch.Size([10, 432, 16, 12, 9])
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        #logger.info(f'cls_score_v1, {cls_score.shape}, {cls_score}')
        # test_cls_score.shape_v1, torch.Size([10, 2])
        cls_score = self.average_clip(cls_score, num_segs)
        # test_cls_score_v2, torch.Size([1, 2])
        #logger.info(f'cls_score_v2, {cls_score.shape}, {cls_score}')
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
