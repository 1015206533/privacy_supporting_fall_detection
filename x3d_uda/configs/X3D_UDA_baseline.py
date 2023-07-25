# common setting
work_dir = './work_dirs/x3d_ur_fall_uda/20221017_V03_uda_unsim' 
#work_dir = './work_dirs/x3d_ur_fall_uda/test'
total_epochs = 16
checkpoint_config = dict(interval=2)
log_config = dict(interval=40, hooks=[dict(type='TextLoggerHook'),])

# dataset settings
dataset_type = 'UdaRawframeDataset'
data_root = ''
data_root_val = ''

##########################################################################
# kinetics 700 数据集
# 相似类别切分
#ann_file_train = 'data/kinetics_700/annotation/shuffle_train_sim.txt'
#ann_file_val = 'data/kinetics_700/annotation/shuffle_test_sim_depth.txt'
#ann_file_test = 'data/kinetics_700/annotation/shuffle_test_sim_rgb_depth.txt'

# 不相似类别切分
ann_file_train = 'data/kinetics_700/annotation/shuffle_train_unsim.txt'
ann_file_val = 'data/kinetics_700/annotation/shuffle_test_unsim_depth.txt'
ann_file_test = 'data/kinetics_700/annotation/shuffle_test_unsim_rgb_depth.txt'

##########################################################################
# NTU RGB D 数据集
# ann_file_train = 'data/ntu_rgb_d/annotation/shuffle_subject_train_file.txt'
# ann_file_val = 'data/ntu_rgb_d/annotation/shuffle_subject_test_file_depth.txt'
# ann_file_test = 'data/ntu_rgb_d/annotation/shuffle_subject_test_file_rgb_depth.txt'

#ann_file_train = 'data/ntu_rgb_d/annotation/shuffle_camera_train_file.txt'
#ann_file_val = 'data/ntu_rgb_d/annotation/shuffle_camera_test_file_depth.txt'
#ann_file_test = 'data/ntu_rgb_d/annotation/shuffle_camera_test_file_rgb_depth.txt'

# ann_file_train = 'data/ntu_rgb_d/annotation/shuffle_train_file.txt'
# ann_file_val = 'data/ntu_rgb_d/annotation/shuffle_test_file_depth.txt'
# ann_file_test = 'data/ntu_rgb_d/annotation/shuffle_test_file_rgb_depth.txt'

##########################################################################
# PKU MMD 数据集
# ann_file_train = 'data/pku_mmd/annotation/shuffle_train_file.txt'
# ann_file_val = 'data/pku_mmd/annotation/shuffle_test_file_depth.txt'
# ann_file_test = 'data/pku_mmd/annotation/shuffle_test_file_rgb_depth.txt'

##########################################################################
# UR FALL 数据集
# ann_file_train = 'data/UR_Fall/annotation/shuffle_annotation_train.txt'
# ann_file_val = 'data/UR_Fall/annotation/shuffle_annotation_d_test.txt'
# ann_file_test = 'data/UR_Fall/annotation/shuffle_annotation_test.txt'

# model setting
model = dict(  # Config of the model
    type='Recognizer3D',  # Type of the recognizer
    backbone=dict(  # Dict for backbone
        type='X3D',  # Name of the backbone
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2,
        pretrained='checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',  # The url/site of the pretrained model
        norm_eval=False),  # Whether to set BN layers to eval mode when training
    cls_head=dict(  # Dict for classification head
        type='X3DHead',  # Name of classification head
        num_classes=2,  # Number of classes to be classified.
        pretrained='checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',
        in_channels=432,  # The input channels of classification head.
        dropout_ratio=0.4,  # Probability in dropout layer
        init_std=0.01),
    domain_head=dict(  # Dict for classification head
        type='X3DHead',  # Name of classification head
        num_classes=2,  # Number of classes to be classified.
        pretrained='checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth',
        in_channels=432,  # The input channels of classification head.
        dropout_ratio=0.4,  # Probability in dropout layer
        init_std=0.01),
    train_cfg=dict(aux_info=['domain_label'])) # Std value for linear layer initiation


img_norm_cfg = dict(  # Config of image normalization used in data pipeline
    mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
    std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
    to_bgr=False)  # Whether to convert channels from RGB to BGR


train_pipeline = [  # List of training pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=False),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    # dict(  # Config of MultiScaleCrop
    #     type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
    #     input_size=224,  # Input size of the network
    #     scales=(1, 0.875, 0.75, 0.66),  # Scales of width and height to be selected
    #     random_crop=False,  # Whether to randomly sample cropping bbox
    #     max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
    # dict(  # Config of Resize
    #     type='Resize',  # Resize pipeline
    #     scale=(224, 224),  # The scale to resize images
    #     keep_ratio=False),  # Whether to resize with changing the aspect ratio
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
val_pipeline = [  # List of validation pipeline steps
    dict(  # Config of SampleFrames
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=True),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    # dict(  # Config of MultiScaleCrop
    #     type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
    #     input_size=224,  # Input size of the network
    #     scales=(1, 0.875, 0.75, 0.66),  # Scales of width and height to be selected
    #     random_crop=False,  # Whether to randomly sample cropping bbox
    #     max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
    # dict(  # Config of Resize
    #     type='Resize',  # Resize pipeline
    #     scale=(224, 224),  # The scale to resize images
    #     keep_ratio=False),  # Whether to resize with changing the aspect ratio
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
        num_clips=10,
        test_mode=True),  # Number of clips to be sampled
    dict(  # Config of RawFrameDecode
        type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
    dict(  # Config of Resize
        type='Resize',  # Resize pipeline
        keep_ratio=False,
        scale=(256, 256)),  # The scale to resize images
    # dict(  # Config of MultiScaleCrop
    #     type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
    #     input_size=224,  # Input size of the network
    #     scales=(1, 0.875, 0.75, 0.66),  # Scales of width and height to be selected
    #     random_crop=False,  # Whether to randomly sample cropping bbox
    #     max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
    # dict(  # Config of Resize
    #     type='Resize',  # Resize pipeline
    #     scale=(224, 224),  # The scale to resize images
    #     keep_ratio=False),  # Whether to resize with changing the aspect ratio
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
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        #filename_tmpl='{:04}.png',
        pipeline=train_pipeline),
    val=dict(  # Validation dataset config
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        #filename_tmpl='{:04}.png',
        pipeline=val_pipeline),
    test=dict(  # Testing dataset config
        type=dataset_type,
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
    weight_decay=0.0001)  # Weight decay of SGD
optimizer_config = dict(  # Config used to build the optimizer hook
    grad_clip=dict(max_norm=40, norm_type=2))  # Use gradient clip


# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    step=[40, 80])  # Steps to decay the learning rate
evaluation = dict(  # Config of evaluation during training
    interval=1,  # Interval to perform evaluation
    gpu_collect=True,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],  # Metrics to be performed
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3))), # Set top-k accuracy to 1 and 3 during validation
    save_best='top1_acc')  # set `top_k_accuracy` as key indicator to save best checkpoint
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3)))) # Set top-k accuracy to 1 and 3 during testing. You can also use `--eval top_k_accuracy` to assign evaluation metrics


# runtime settings
domain_loss_lambda = 0.1
#cudnn_benchmark = True
find_unused_parameters = True
test_cfg = dict(average_clips='score')
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
load_from = None  # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once


