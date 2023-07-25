# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import torch

from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS
from mmaction.utils import get_root_logger


@DATASETS.register_module()
class IDMUdaRawframeDataset(BaseDataset):
    """UDA Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1 0
        some/directory-2 122 1 1
        some/directory-3 258 2 0
        some/directory-4 234 2 0
        some/directory-5 295 3 1
        some/directory-6 121 3 1

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:04}.png',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 **kwargs):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)
        self.short_cycle_factors = kwargs.get('short_cycle_factors',
                                              [0.5, 0.7071])
        self.default_s = kwargs.get('default_s', (224, 224))

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []

        rgb_file, depth_file = self.ann_file.split('&&')
        rgb_f = open(rgb_file, 'r')
        rgb_infos = rgb_f.readlines()
        rgb_f.close()
        depth_f = open(depth_file, 'r')
        depth_infos = depth_f.readlines()
        depth_f.close()

        for i in range(len(rgb_infos)):
            if i >= len(depth_infos):
                continue
            rgb_info = rgb_infos[i].strip().split(' ')
            depth_info = depth_infos[i].strip().split(' ')
            video_info = {}
            video_info['rgb_frame_dir'] = osp.join(self.data_prefix, rgb_info[0])
            video_info['depth_frame_dir'] = osp.join(self.data_prefix, depth_info[0])
            video_info['rgb_total_frames'] = int(rgb_info[1])
            video_info['depth_total_frames'] = int(depth_info[1])
            video_info['rgb_label'] = int(rgb_info[2])
            video_info['depth_label'] = int(depth_info[2])
            video_info['rgb_domain'] = int(rgb_info[3])
            video_info['depth_domain'] = int(depth_info[3])
            video_infos.append(video_info)
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""

        def pipeline_for_a_sample(idx):
            results = copy.deepcopy(self.video_infos[idx])
            results['filename_tmpl'] = self.filename_tmpl
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            return self.pipeline(results)

        if isinstance(idx, tuple):
            index, short_cycle_idx = idx
            last_resize = None
            for trans in self.pipeline.transforms:
                if isinstance(trans, Resize):
                    last_resize = trans
            origin_scale = self.default_s
            long_cycle_scale = last_resize.scale

            if short_cycle_idx in [0, 1]:
                # 0 and 1 is hard-coded as PySlowFast
                scale_ratio = self.short_cycle_factors[short_cycle_idx]
                target_scale = tuple(
                    [int(round(scale_ratio * s)) for s in origin_scale])
                last_resize.scale = target_scale
            res = pipeline_for_a_sample(index)
            last_resize.scale = long_cycle_scale
            return res
        else:
            return pipeline_for_a_sample(idx)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)
