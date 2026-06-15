# mmdetection-3.3.0/projects/vasco/configs/vasco_single_videoswin3d_nofpn_temporal8.py

_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']
custom_imports = dict(
    imports=['projects.vasco', 'projects.vasco.metrics'],
    allow_failed_imports=False)

# ===== 数据设置 =====
dataset_type = 'VascoClipDataset'
data_root = '/home/tili/masterwork/transformer/VASCO/'
train_ann = data_root + 'vasco_train/train.json'
train_img = data_root + 'vasco_train/'
val_ann   = data_root + 'vasco_val/val.json'
val_img   = data_root + 'vasco_val/'

data_preprocessor = dict(
    type='VascoDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(
        type='SeqResizeFlipPad',
        short_edge_choices=(720, 864, 960, 1080),
        max_long_edge=1600,
        crop_size=(896, 1600)),
    dict(type='PackVascoInputs'),
]
val_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(
        type='SeqResizeFlipPad',
        short_edge_choices=(1080,),
        max_long_edge=1600,
        crop_size=(896, 1600)),
    dict(type='PackVascoInputs'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_img),
        temporal_window=8,
        temporal_span=20,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_img),
        temporal_window=8,
        temporal_span=20,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetric', ann_file=val_ann, metric='bbox'),
    dict(
        type='VascoActionDirMetric',
        iou_thr=0.6,
        num_actions=5,
        num_dirs=6,
        prefix='vasco'),
    dict(type='VascoSegMetric', num_classes=7, prefix='seg'),
]
test_evaluator = val_evaluator

# ===== 模型：一路 Video Swin3D（无 FPN），backbone 直接接 4 个头 =====
model = dict(
    type='VascoVideoRCNN',   # 你刚刚新建的 detector 类
    data_preprocessor=dict(type='VascoDataPreprocessor'),

    # 一路 backbone：Video Swin 3D
    backbone=dict(
        _delete_=True,
        type='VideoSwin3DBackbone',
        pretrained='/home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth',
        window_size=(8, 7, 7),
        out_indices=(3,),
        with_cp=True,
        arch=dict(
            embed_dims=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24)),
    ),

    # 无 FPN：检测直接用末帧最后一层特征 (C=768, stride=32)
    neck=None,

    # ----- RPN 头（单层特征） -----
    rpn_head=dict(
        type='RPNHead',
        in_channels=768,      # 对应 Video Swin 
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[32]),    
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0)),

    # ----- RoIHead：检测 + 动作/方向（共用同一份 RoI 特征） -----
    roi_head=dict(
        _delete_=True,
        type='VascoSinglePathRoIHead',   # 你需要从 vasco_roi_head 精简出一个一路版
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2),
            out_channels=768,        # 从 backbone 最后层取特征
            featmap_strides=[32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=768,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=28,          # thing 类别数
            reg_class_agnostic=True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='L1Loss',
                loss_weight=1.0)),
        temporal_aggregator=dict(
            type='TemporalAlignAggregator',
            in_channels=768,
            k=7, bias_last=0.5, max_offset_px=5,
            detach_prev=False, gamma_max=0.5, gamma_scale=1.0,
            consistency_weight=0.2, heads=4, layers=1
        ),
        # 动作 / 方向头：直接用 bbox RoI 特征
        action_dir_head=dict(
            type='ActionDirHead',
            in_channels=768,
            roi_feat_size=7,
            hidden_dim=1024,
            with_mlp=True,
            num_actions=5,
            num_dirs=6,
            loss_action=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0),
            loss_dir=dict(
                type='CrossEntropyLoss',
                loss_weight=1.5)),
        # ----- 语义分割头：用 backbone 第一层末帧特征 (C=96) -----
        semantic_head=dict(
            type='SemanticHead',
            in_channels=768,          # 对应 Video Swin 第一层输出通道
            num_classes=8,           # 和你双路里保持一致（含背景）
            ce_weight=1.0,
            dice_weight=0.5,
            ignore_index=255),
        aux_detach=False),

    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300),
    )


# ===== 训练 / 测试配置（沿用双路） =====
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=48,
    val_interval=2)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        weight_decay=5e-2),
    clip_grad=dict(max_norm=30, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='MultiStepLR',
        milestones=[27, 33],
        gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=True,
        max_keep_ckpts=2,
        save_last=True,
        save_best=[
            'coco/bbox_mAP',
            'vasco/action_acc',
            'vasco/dir_acc',
            'seg/mIoU'],
        rule=[
            'greater',
            'greater',
            'greater',
            'greater']))