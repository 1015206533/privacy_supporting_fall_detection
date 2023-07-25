# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

from mmaction.utils import get_root_logger
from ..builder import RECOGNIZERS
from .base import BaseRecognizer, GRL


@RECOGNIZERS.register_module()
class IDMRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        logger = get_root_logger()
        # imgs.shape_v1_1, torch.Size([1, 10, 3, 16, 384, 288])，
        # 10对应的是rgb和depth各5个clip采样序列个数合并在一起的长度，1表示batchsize，16表示的是序列长度
        ns = imgs.size(1)
        assert (ns%2==0)
        split = torch.split(imgs, int(ns/2), 1)
        x_s = split[0].contiguous()
        x_s = x_s.reshape((-1, ) + x_s.shape[2:])    # [B*N, 3, 16, 384, 288]
        x_t = split[1].contiguous()
        x_t = x_t.reshape((-1, ) + x_t.shape[2:])    # [B*N, 3, 16, 384, 288]
        x_all = torch.cat((x_s, x_t), 0)    # [B*2N, 3, 16, 384, 288]
                                            # torch.Size([10, 3, 16, 256, 256])

        # domain label 
        domain_gt_labels_ori = kwargs.pop('domain_label')
        split_domain = torch.split(domain_gt_labels_ori, int(ns/2), 1)
        domain_s = split_domain[0].contiguous().squeeze()
        domain_t = split_domain[1].contiguous().squeeze()
        domain_gt_labels = torch.cat((domain_s, domain_t), 0)

        # cls label 
        split_label = torch.split(labels, int(ns/2), 1)
        label_s = split_label[0].contiguous().squeeze()
        label_t = split_label[1].contiguous().squeeze()
        gt_labels = torch.cat((label_s, label_t), 0)

        # config parameter
        domain_loss_lambda = 1.0
        if 'domain_loss_lambda' in kwargs:
            domain_loss_lambda = kwargs.pop('domain_loss_lambda')
        pseudo_label_target = 0.8
        if 'pseudo_label_target' in kwargs:
            pseudo_label_target = kwargs.pop('pseudo_label_target')
        bridge_label_target = 0.8
        if 'bridge_label_target' in kwargs:
            bridge_label_target = kwargs.pop('bridge_label_target')
        loss_type_list = ['all_label_loss', 'loss_rgb_cls', 'loss_domain', 'loss_pseudo', 'loss_bridge_feat', 'xbm_triplet_loss']
        if 'loss_type_list' in kwargs:
            loss_type_list = kwargs.pop('loss_type_list')
        is_loss_adaptative = True 
        if 'is_loss_adaptative' in kwargs:
            is_loss_adaptative = kwargs.pop('is_loss_adaptative')

        # backbone net and get feature map
        x, attention_lam = self.extract_feat(x_all) # train_x.shape_v1, torch.Size([B*3N, 432, 16, 8, 8])
                                                    # C3D torch.Size([15, 512, 1, 9, 9]), torch.Size([5, 2])
        feats_s, feats_t, feats_mixed = x.split(x.size(0) // 3, dim=0)  # [B*N, 432, 16, 8, 8]
        feat_s_t = torch.cat((feats_s, feats_t), 0) # [B*2N, 432, 16, 8, 8]

        # cls head net
        cls_score = self.cls_head(x)    #   cls_score_size, torch.Size([10, 2])
        cls_score_prob = F.softmax(cls_score, dim=1)
        cls_score_s, cls_score_t, cls_score_mixed = cls_score.split(cls_score.size(0)//3, dim=0)
        cls_score_prob_s, cls_score_prob_t, cls_score_prob_mixed = cls_score_prob.split(cls_score_prob.size(0)//3, dim=0)
        cls_score_s_t = torch.cat((cls_score_s, cls_score_t), 0)
        cls_score_prob_s_t = torch.cat((cls_score_prob_s, cls_score_prob_t), 0)

        # pseudo depth score and label and loss
        # pseudo_label_target = 0.8
        pseudo_pos_score = cls_score_s_t[(domain_gt_labels==0) & (cls_score_prob_s_t[:, 1] > pseudo_label_target)]
        pseudo_pos_label = torch.from_numpy(np.array([1]*pseudo_pos_score.shape[0]))
        pseudo_pos_label = pseudo_pos_label.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        pseudo_pos_label = pseudo_pos_label.to(torch.int64)
        pseudo_neg_score = cls_score_s_t[(domain_gt_labels==0) & (cls_score_prob_s_t[:, 0] > pseudo_label_target)]
        pseudo_neg_label = torch.from_numpy(np.array([0]*pseudo_neg_score.shape[0]))
        pseudo_neg_label = pseudo_neg_label.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        pseudo_neg_label = pseudo_neg_label.to(torch.int64)
        pseudo_score = torch.cat((pseudo_pos_score, pseudo_neg_score), axis=0)
        pseudo_label = torch.cat((pseudo_pos_label, pseudo_neg_label), axis=0)
        pseudo_loss_cls = self.cls_head.loss(pseudo_score, pseudo_label, **kwargs)  # depth数据伪标签交叉熵loss

        # real rgb and depth label cls loss 
        all_label_loss = self.cls_head.loss(cls_score_s_t, gt_labels, **kwargs) # rgb样本和depth样本的真实标签loss，用于计算数据分类上限
        #all_label_loss = self.cls_head.loss(cls_score_t, label_t, **kwargs) # 仅depth样本的真实标签loss，用于计算数据分类上限
        rgb_label_loss = self.cls_head.loss(cls_score_s, label_s, **kwargs) # 仅rgb样本的真实标签loss，用于计算直接迁移的效果

        # domain head net and domain loss 
        flip_feat_s_t = GRL.apply(feat_s_t, 1.0)
        domain_score = self.domain_head(flip_feat_s_t)
        loss_domain = self.domain_head.loss(domain_score, domain_gt_labels, **kwargs)   # depth数据和rgb数据的domain loss

        # bridge feature loss
        dist_mixed2s = ((feats_mixed-feats_s)**2).sum((1,2,3,4), keepdim=True)
        dist_mixed2t = ((feats_mixed-feats_t)**2).sum((1,2,3,4), keepdim=True)
        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()
        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (attention_lam*dist_mixed).sum(1, keepdim=True)
        loss_bridge_feat = lam_dist_mixed.mean()        # IDM 模块 feature 的 bridge loss ，用于更新IDM模块

        # bridge prob loss
        # bridge_label_target = 0.8
        label_t_ = (cls_score_prob_t > bridge_label_target).to(torch.int64)  
        label_t_thres = label_t_[label_t_.sum(axis=1) == 1] 
        if label_t_thres.size(0) > 0:
            label_s_ = torch.zeros((label_s.size(0), 2)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).scatter_(1, label_s.unsqueeze(1).to(torch.int64), 1)
            label_s_thres = label_s_[label_t_.sum(axis=1) == 1] 
            label_mixed = attention_lam[label_t_.sum(axis=1) == 1, 0].view(-1, 1).detach()*label_s_thres + attention_lam[label_t_.sum(axis=1) == 1, 1].view(-1, 1).detach()*label_t_thres
            soft_label_mixed = label_mixed*0.9 + 0.05    # 起到平滑所用
            cls_score_mixed_thres = cls_score_mixed[label_t_.sum(axis=1) == 1]
            loss_bridge_prob = (- soft_label_mixed*F.log_softmax(cls_score_mixed_thres, 1)).mean(0).sum()     # IDM 模块概率 bridge loss， 不更新IDM 模块
        else:
            loss_bridge_prob = torch.zeros(torch.Size([])).cuda()
        
        # xbm triplet loss
        if label_t_thres.size(0) > 0:
            targets = torch.cat((label_s_thres[:, 1], label_t_thres[:, 1]), 0)
            feats_s_ = feats_s[label_t_.sum(axis=1) == 1]
            feats_t_ = feats_t[label_t_.sum(axis=1) == 1]
            feats_s_t_ = torch.cat((feats_s_, feats_t_), 0)
            self.xbm.enqueue_dequeue(feats_s_t_.view(feats_s_t_.size(0), -1).detach(), targets.detach())
            xbm_feats, xbm_targets = self.xbm.get()
            xbm_trilpet_loss = self.criterion_tri_xbm(feats_s_t_.view(feats_s_t_.size(0), -1), targets, xbm_feats, xbm_targets)
        else:
            xbm_trilpet_loss = torch.zeros(torch.Size([])).cuda()

        # xbm diverse loss 
        xbm_attention_lam = self.diverse_xbm.get()
        self.diverse_xbm.enqueue_dequeue(attention_lam.detach())
        if xbm_attention_lam.size(0) > 0:
            attention_lam_ = torch.cat((attention_lam, xbm_attention_lam.detach()), 0)
        else: 
            attention_lam_ = attention_lam
        mu = attention_lam_.mean(0)
        std = ((attention_lam_-mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
        xbm_diverse_loss = -std.sum()

        # loss process
        rgb_label_loss_final = rgb_label_loss['loss_cls']
        domain_loss_final = domain_loss_lambda * loss_domain['loss_cls']
        pseudo_loss_final = None 
        if not torch.isnan(pseudo_loss_cls['loss_cls']).any():
            pseudo_loss_final = 0.1 * pseudo_loss_cls['loss_cls']
        else:
            pseudo_loss_final = torch.zeros(torch.Size([])).cuda()
        idm_bridge_loss_final = 0.0001 * loss_bridge_feat
        xbm_trilpet_loss_final = 0.0001 * xbm_trilpet_loss

        
        ###################################################################################################
        # # solid loss ratio
        losses = dict()
        if not is_loss_adaptative:
            # upper bound
            if "all_label_loss" in loss_type_list:
                losses['all_label_loss'] = all_label_loss['loss_cls']

            # rgb real label loss
            if "loss_rgb_cls" in loss_type_list:
                losses['loss_rgb_cls'] = rgb_label_loss_final

            # add domain loss 
            if "loss_domain" in loss_type_list:
                losses['loss_domain'] = domain_loss_final
            
            # add depth pseudo label cls loss
            if "loss_pseudo" in loss_type_list:
                losses['loss_pseudo'] = pseudo_loss_final

            # add bridge feature loss
            if "loss_bridge_feat" in loss_type_list:
                losses['loss_bridge_feat'] = idm_bridge_loss_final

            # add xbm triplet loss
            if "xbm_triplet_loss" in loss_type_list:
                losses['xbm_triplet_loss'] = xbm_trilpet_loss_final

        ###################################################################################################
        # # network loss adaptation
        adaptation_feature_map_v0 = self.adaptation_pool(x)
        adaptation_feature_map_v1 = adaptation_feature_map_v0.view(adaptation_feature_map_v0.shape[0], -1)
        adaptation_feature_map_v2 = self.adaptation_fc(adaptation_feature_map_v1)
        adaptation_feature_map_v3 = F.relu(adaptation_feature_map_v2)
        adaptation_feature_map_v4 = self.adaptation_fc2(adaptation_feature_map_v3)
        adaptation_feature_map_v5 = F.relu(adaptation_feature_map_v4)
        adaptation_feature_map_v6 = self.adaptation_fc3(adaptation_feature_map_v5)
        adaptation_feature_map_v7 = adaptation_feature_map_v6.mean(axis=0)
        adaptation_feature_map_v8 = F.softmax(adaptation_feature_map_v7, dim=0)
        
        #losses = dict()
        if is_loss_adaptative:
            # # rgb real label loss
            #losses['loss_rgb_cls'] = rgb_label_loss_final
            losses['loss_rgb_cls'] = adaptation_feature_map_v8[0] * rgb_label_loss_final

            # # add domain loss 
            #losses['loss_domain'] = domain_loss_final
            losses['loss_domain'] = adaptation_feature_map_v8[1] * domain_loss_final
            
            # # add depth pseudo label cls loss
            #losses['loss_pseudo'] = pseudo_loss_final
            losses['loss_pseudo'] = adaptation_feature_map_v8[2] * pseudo_loss_final

            # # add bridge feature loss
            #losses['loss_bridge_feat'] = idm_bridge_loss_final
            losses['loss_bridge_feat'] = adaptation_feature_map_v8[3] * idm_bridge_loss_final

            # # add xbm triplet loss
            #losses['xbm_triplet_loss'] = xbm_trilpet_loss_final
            losses['xbm_triplet_loss'] = adaptation_feature_map_v8[4] * xbm_trilpet_loss_final

            # logger.info(f'network loss adaptation, {adaptation_feature_map_v8}')

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # logger = get_root_logger()
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        # logger.info(f'imgs.shape_v1, {imgs.shape}, {batches}, {num_segs}')
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
            feat = self.extract_feat(imgs)  # [10, 432, 16, 8, 8]
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
        # cls_score_prob = F.softmax(cls_score, dim=1)
        # logger.info(f'test_cls_score_prob, {cls_score_prob.shape}, {cls_score_prob}')
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
