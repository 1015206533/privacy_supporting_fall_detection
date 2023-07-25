# common setting
work_dir = './work_dirs/x3d_ur_fall_uda/20230725_V01_C3D' 
total_epochs = 200
checkpoint_config = dict(interval=20)
log_config = dict(interval=40, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])

# dataset settings
train_dataset_type = 'IDMUdaRawframeDataset'
val_dataset_type = 'UdaRawframeDataset'
data_root = ''
data_root_val = ''

##########################################################################
# kinetics 700 数据集
ann_file_train = 'data/kinetics_700/annotation/idm_train_rgb_unsim.txt&&data/kinetics_700/annotation/idm_train_depth_unsim.txt'
ann_file_val = 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt'
ann_file_test = 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt'

# NTU-RGB-D 数据集
#ann_file_train = 'data/ntu_rgb_d/annotation/shuffle_idm_train_rgb.txt&&data/ntu_rgb_d/annotation/shuffle_idm_train_depth.txt'
#ann_file_val = 'data/ntu_rgb_d/annotation/shuffle_idm_test_depth.txt'
#ann_file_test = 'data/ntu_rgb_d/annotation/shuffle_idm_test_depth.txt'

# model setting
model = dict(  # Config of the model
    type='IDMRecognizer3D',  # Type of the recognizer
    backbone=dict(
        type='IDMC3D',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',  # noqa: E501
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=2,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    domain_head=dict(  # Dict for classification head
        type='I3DHead',
        num_classes=2,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=dict(aux_info=['domain_label'])) # Std value for linear layer initiation


img_norm_cfg = dict(  # Config of image normalization used in data pipeline
    mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
    std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
    to_bgr=False)  # Whether to convert channels from RGB to BGR


train_pipeline = [  # List of training pipeline steps
    dict(  # Config of SampleFrames
        type='UdaSampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=5,
        test_mode=False),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='UdaRawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='UdaResize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    dict(  # Config of Normalize
        type='UdaNormalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='UdaFormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCTHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label', 'domain_label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs', 'label', 'domain_label'])  # Keys to be converted from image to tensor
]

val_pipeline = [  # List of validation pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=5,
        test_mode=True),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    dict(  # Config of Normalize
        type='Normalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCTHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label', 'domain_label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs', 'label', 'domain_label'])  # Keys to be converted from image to tensor
]

test_pipeline = [  # List of testing pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=5,
        test_mode=True),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    dict(  # Config of Normalize
        type='Normalize',  # Normalize pipeline
        **img_norm_cfg),  # Config of image normalization
    dict(  # Config of FormatShape
        type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
        input_format='NCTHW'),  # Final image shape format
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label', 'domain_label'],  # Keys of input
        meta_keys=[]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['imgs', 'label', 'domain_label'])  # Keys to be converted from image to tensor
]


data = dict(  # Config of data
    videos_per_gpu=1,  # Batch size of each single GPU
    workers_per_gpu=1,  # Workers to pre-fetch data for each single GPU
    train_dataloader=dict(  # Additional config of train dataloader
        drop_last=True),  # Whether to drop out the last batch of data in training
    val_dataloader=dict(  # Additional config of validation dataloader
        videos_per_gpu=1),  # Batch size of each single GPU during evaluation
    test_dataloader=dict(  # Additional config of test dataloader
        videos_per_gpu=1),  # Batch size of each single GPU during testing
    train=dict(  # Training dataset config
        type=train_dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        #filename_tmpl='{:04}.png',
        pipeline=train_pipeline),
    val=dict(  # Validation dataset config
        type=val_dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        #filename_tmpl='{:04}.png',
        pipeline=val_pipeline),
    test=dict(  # Testing dataset config
        type=val_dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        #filename_tmpl='{:04}.png',
        pipeline=test_pipeline))


# optimizer
optimizer = dict(
    # Config used to build optimizer, support (1). All the optimizers in PyTorch
    # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
    # which are built on `constructor`, referring to "tutorials/5_new_modules.md"
    # for implementation.
    type='SGD',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
    lr=0.001,  # Learning rate, see detail usages of the parameters in the documentation of PyTorch
    momentum=0.9,  # Momentum,
    weight_decay=0.0001)  # Weight decay of SGD L2 regularization param
optimizer_config = dict(  # Config used to build the optimizer hook
    grad_clip=dict(max_norm=40, norm_type=2))  # Use gradient clip


# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    step=[10, 200],
    gamma=0.1)  # Steps to decay the learning rate  default multiply 0.1
evaluation = dict(  # Config of evaluation during training
    interval=1,  # Interval to perform evaluation
    gpu_collect=True,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],  # Metrics to be performed
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3))), # Set top-k accuracy to 1 and 3 during validation
    save_best='top1_acc')  # set `top_k_accuracy` as key indicator to save best checkpoint
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3)))) # Set top-k accuracy to 1 and 3 during testing. You can also use `--eval top_k_accuracy` to assign evaluation metrics


# runtime settings
pseudo_label_target = 0.8
bridge_label_target = 0.8
loss_type_list = ['all_label_loss', 'loss_rgb_cls', 'loss_domain', 'loss_pseudo', 'loss_bridge_feat', 'xbm_triplet_loss']
is_loss_adaptative = True 
domain_loss_lambda = 0.1
#cudnn_benchmark = True
find_unused_parameters = True
test_cfg = dict(average_clips='score')
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
load_from = None  # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once


