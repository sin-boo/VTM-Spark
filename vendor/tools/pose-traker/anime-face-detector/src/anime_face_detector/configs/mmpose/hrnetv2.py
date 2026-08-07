# mmpose 1.x config for HRNetV2 anime face landmark detection

# codec configuration
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
)

# model configuration
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True,  # Output all branches for concat
            ),
        ),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=270,  # 18+36+72+144 = 270 (concat of all HRNet outputs)
        out_channels=28,
        deconv_out_channels=None,
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,  # Disabled - requires proper dataset metainfo
    ),
)

# flip pairs for flip augmentation
flip_indices = [4, 3, 2, 1, 0, 10, 9, 8, 7, 6, 5, 19, 18, 17, 22, 21, 20, 13, 12, 11, 16, 15, 14, 23, 26, 25, 24, 27]

# test pipeline
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

# test dataloader (required for inference_topdown)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        data_mode='topdown',
        ann_file='',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

# dataset meta information
dataset_info = dict(
    dataset_name='anime_face',
    paper_info=dict(),
    keypoint_info={
        0: dict(name='kpt-0', id=0, color=[255, 255, 255], type='', swap='kpt-4'),
        1: dict(name='kpt-1', id=1, color=[255, 255, 255], type='', swap='kpt-3'),
        2: dict(name='kpt-2', id=2, color=[255, 255, 255], type='', swap=''),
        3: dict(name='kpt-3', id=3, color=[255, 255, 255], type='', swap='kpt-1'),
        4: dict(name='kpt-4', id=4, color=[255, 255, 255], type='', swap='kpt-0'),
        5: dict(name='kpt-5', id=5, color=[255, 255, 255], type='', swap='kpt-10'),
        6: dict(name='kpt-6', id=6, color=[255, 255, 255], type='', swap='kpt-9'),
        7: dict(name='kpt-7', id=7, color=[255, 255, 255], type='', swap='kpt-8'),
        8: dict(name='kpt-8', id=8, color=[255, 255, 255], type='', swap='kpt-7'),
        9: dict(name='kpt-9', id=9, color=[255, 255, 255], type='', swap='kpt-6'),
        10: dict(name='kpt-10', id=10, color=[255, 255, 255], type='', swap='kpt-5'),
        11: dict(name='kpt-11', id=11, color=[255, 255, 255], type='', swap='kpt-19'),
        12: dict(name='kpt-12', id=12, color=[255, 255, 255], type='', swap='kpt-18'),
        13: dict(name='kpt-13', id=13, color=[255, 255, 255], type='', swap='kpt-17'),
        14: dict(name='kpt-14', id=14, color=[255, 255, 255], type='', swap='kpt-22'),
        15: dict(name='kpt-15', id=15, color=[255, 255, 255], type='', swap='kpt-21'),
        16: dict(name='kpt-16', id=16, color=[255, 255, 255], type='', swap='kpt-20'),
        17: dict(name='kpt-17', id=17, color=[255, 255, 255], type='', swap='kpt-13'),
        18: dict(name='kpt-18', id=18, color=[255, 255, 255], type='', swap='kpt-12'),
        19: dict(name='kpt-19', id=19, color=[255, 255, 255], type='', swap='kpt-11'),
        20: dict(name='kpt-20', id=20, color=[255, 255, 255], type='', swap='kpt-16'),
        21: dict(name='kpt-21', id=21, color=[255, 255, 255], type='', swap='kpt-15'),
        22: dict(name='kpt-22', id=22, color=[255, 255, 255], type='', swap='kpt-14'),
        23: dict(name='kpt-23', id=23, color=[255, 255, 255], type='', swap=''),
        24: dict(name='kpt-24', id=24, color=[255, 255, 255], type='', swap='kpt-26'),
        25: dict(name='kpt-25', id=25, color=[255, 255, 255], type='', swap=''),
        26: dict(name='kpt-26', id=26, color=[255, 255, 255], type='', swap='kpt-24'),
        27: dict(name='kpt-27', id=27, color=[255, 255, 255], type='', swap=''),
    },
    skeleton_info={},
    joint_weights=[1.0] * 28,
    sigmas=[0.025] * 28,
    flip_indices=flip_indices,
)
