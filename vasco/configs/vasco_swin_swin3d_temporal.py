# mmdetection-3.3.0/projects/vasco/configs/vasco_videoswin3d_fpn_temporal5.py
_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']
custom_imports = dict(imports=['projects.vasco', 'projects.vasco.metrics'], allow_failed_imports=False)

# ===== 数据 =====
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
        type='SeqColorJitter',
        brightness=0.3,   # 可先用 0.2, 后续看效果再微调
        contrast=0.2,
        saturation=0.1,
        p=0.8),
    dict(type='SeqResizeFlipPad',
         short_edge_choices=(720, 864, 960, 1080),
         max_long_edge=1600,
         crop_size=(896, 1600)),
    dict(type='PackVascoInputs'),
]
val_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResizeFlipPad', 
         short_edge_choices=(1080,), 
         max_long_edge=1600,
         crop_size=(896, 1600)),
    dict(type='PackVascoInputs'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4 ,num_workers=2, persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file=train_ann, data_prefix=dict(img=train_img),
        temporal_window=8, temporal_span=20,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file=val_ann, data_prefix=dict(img=val_img),
        temporal_window=8, temporal_span=20, test_mode=True,
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetric', ann_file=val_ann, metric='bbox'),
    dict(type='VascoActionDirMetric', iou_thr=0.6, num_actions=2, num_dirs=5, prefix='vasco'),
    dict(type='VascoSegMetric', num_classes=6, prefix='seg'),
    dict(type='VascoEgoStopGoMetric', prefix='ego'),
]
test_evaluator = val_evaluator

# ===== 模型：Swin3D + FPN=5 + all-temporal =====
# 模型
model = dict(
    type='VascoDualPathRCNN',
    data_preprocessor=dict(type='VascoDataPreprocessor'),
    # 2D 主链（COCO 检测预训练）
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',  # 或 ViTDet
        embed_dims=96, depths=[2,2,18,2], num_heads=[3,6,12,24],
        window_size=7, drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint='/home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth')
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96,192,384,768], out_channels=256, num_outs=4
    ),
    # 3D 辅助链（Kinetics 预训练 / 或 2D→3D 膨胀初始化）
    backbone_3d=dict(
        type='VideoSwin3DBackbone',
        pretrain='/home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth',  
        window_size=(8, 7, 7),
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        arch=dict(embed_dims=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24)),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[1.,1.,1.,1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        _delete_=True,
        type='VascoDualPathRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256, featmap_strides=[4,8,16,32]
        ),
        bbox_head=dict(   # 仅用于 get_targets / loss 的参数来源（前向由 RoIHead 自己做）
            type='Shared2FCBBoxHead',
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=28, reg_class_agnostic=True,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0) 
        ),
        temporal_aggregator=dict(
            type='TemporalAlignAggregator',
            k=7, bias_last=0.5, max_offset_px=5,
            detach_prev=False, gamma_max=0.5, gamma_scale=1.0,
            consistency_weight=0.2, heads=4, layers=1
        ),
        action_dir_head=dict(
            type='ActionDirHead',
            in_channels=256, roi_feat_size=7, hidden_dim=1024, with_mlp=True,
            num_actions=2, num_dirs=5,
            loss_action=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_dir=dict(type='CrossEntropyLoss', loss_weight=1.0)
        ),

        semantic_head=dict(
            type='SemanticHead',
            in_channels=(256, 256, 256), num_classes=7, ce_weight=1.0, dice_weight=0.5, ignore_index=255
        ),
        ego_stopgo_head=dict(
            type='EgoStopGoHead',
            # in_channels=256,
            in_channels=1280,  # E3: 1024 (multi-stage global) + 256 (roi token)
            ego_num_choices=5,
            ego_emb_dim=16,
            hidden_dim=128,
            num_classes=2,
            dropout=0.0,
            loss=dict(type='CrossEntropyLoss', loss_weight=0.3),
            class_weight=[12.0, 1.0],  # [stop, go]
            ignore_index=-1,
        ),
        ego_topk=10,
        ego_roi_score_thr=0.2,
        ego_roi_use_rpn_in_train=True,
        aux_detach=False,
    ),
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.3),
                 sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
                 allowed_border=-1, pos_weight=-1, debug=False),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
                  sampler=dict(type='RandomSampler', num=512, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True),
                  debug=False)
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.7), max_per_img=300)
    )
)

# ===== 优化与调度 =====
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=5e-2),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/1000, by_epoch=False, begin=0, end=1500),
    dict(type='MultiStepLR', milestones=[12, 18], gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=10, by_epoch=True, max_keep_ckpts=2, save_last=True,
        save_best=['coco/bbox_mAP','vasco/action_acc','vasco/dir_acc','seg/mIoU','ego/stopgo_mF1','ego/stopgo_acc'],
        rule=['greater','greater','greater','greater','greater','greater'])
)