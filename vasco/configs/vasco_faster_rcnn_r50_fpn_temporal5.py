# mmdetection-3.3.0/projects/vasco/configs/vasco_faster_rcnn_r50_fpn_temporal5.py

_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

custom_imports = dict(imports=['projects.vasco', 'projects.vasco.metrics'], allow_failed_imports=False)

# 数据
dataset_type = 'VascoClipDataset'
data_root = '/home/tili/masterwork/transformer/VASCO/'
train_ann = data_root + 'vasco_train/train.json'
train_img = data_root + 'vasco_train/'
val_ann   = data_root + 'vasco_val/val.json'
val_img   = data_root + 'vasco_val/'

# 使用自定义预处理与三段 pipeline
data_preprocessor = dict(
    type='VascoDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResizeFlipPad',
         short_edge_choices=(720, 864, 960, 1080),
         max_long_edge=1600,
         crop_size=(896, 1600)),
    dict(type='PackVascoInputs'),
]
val_pipeline = val_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResizeFlipPad',
         short_edge_choices=(1080,),   # 固定到一档
         max_long_edge=1600),
    dict(type='PackVascoInputs'),
]
# val_pipeline = train_pipeline
test_pipeline = val_pipeline


train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_img),
        temporal_window=5, temporal_span=20,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_img),
        temporal_window=5, temporal_span=20,
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetric', ann_file=val_ann, metric='bbox'),
    dict(type='ActionDirMetric', iou_thr=0.5, num_actions=5, num_dirs=6, prefix='vasco'),
    dict(type='VascoSegMetric', num_classes=7, prefix='seg'),
]

test_evaluator = val_evaluator


# === 模型 ===
model = dict(
    type='VascoRCNN',
    data_preprocessor=dict(
        type='VascoDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    # 统一 FPN 到 4 层，避免 RoIExtractor 越界
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    roi_head=dict(
        type='VascoRoIHead',
        # RoIExtractor 明确 4 层
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56,
        ),
        bbox_head=dict(  # 28 thing 类
            type='Shared2FCBBoxHead',
            num_classes=28,
            in_channels=256,
            fc_out_channels=1024,
            reg_class_agnostic=True,  # 类无关回归
            # 更抗背景（早期收敛更稳）
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            # 回归损失可偏软一点
            loss_bbox=dict(type='SmoothL1Loss', beta=1/9, loss_weight=0.5),
        ),
        temporal_aggregator=dict(
            type='VascoRoITemporalAgg',
            method='mean_conv1d',
            T=5,
            detach_prev_frames=True,
            temporal_dropout=0.2
        ),
        action_dir_head=dict(
            type='ActionDirHead',
            in_channels=256,
            roi_feat_size=7,
            fc_out_channels=1024,
            num_actions=5,  # 含 none
            num_dirs=6,     # 含 none
            # 先降低权重，避免抢梯度
            loss_action=dict(type='CrossEntropyLoss', loss_weight=0.5),
            loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.5)
        ),
        semantic_head=dict(
            type='SemanticHead',
            in_channels=256,
            num_classes=7,
            ce_weight=0.5,   # 降权
            dice_weight=0.2
        ),
    ),
    # —— detector 内部的 rpn/rcnn 训练 cfg ——（增加正样本，放宽阈值）
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner',
                          pos_iou_thr=0.3, neg_iou_thr=0.3,
                          min_pos_iou=0.3, match_low_quality=True),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False)
        ),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner',
                          pos_iou_thr=0.3,  # 0.4→0.3，给早期更多正样本
                          neg_iou_thr=0.3, min_pos_iou=0.0,
                          match_low_quality=True),
            sampler=dict(type='RandomSampler', num=2048, pos_fraction=0.7,
                         neg_pos_ub=-1, add_gt_as_proposals=False)
        )
    ),
    # —— detector 的测试 cfg（正常验证） ——
    test_cfg=dict(
        rpn=dict(nms_pre=4000, max_per_img=2000,
                 nms=dict(type='nms', iou_threshold=0.9), min_bbox_size=0),
        rcnn=dict(score_thr=0.01,
                  nms=dict(type='soft_nms', iou_threshold=0.6, min_score=0.0),
                  max_per_img=300)
    ),
)

# === 优化与调度 ===
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=2e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=10, norm_type=2),  # 稳定早期
)

# 训练到 24 epoch，正常 warmup 与多段衰减
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/1000, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', milestones=[16, 22], gamma=0.1),
]

# === 评价器（保留你已有） ===
val_evaluator = [
    dict(type='CocoMetric', ann_file=val_ann, metric='bbox'),
    dict(type='ActionDirMetric', iou_thr=0.5, num_actions=5, num_dirs=6, prefix='vasco'),
    dict(type='VascoSegMetric', num_classes=7, prefix='seg'),
]
test_evaluator = val_evaluator

# === Checkpoint（先不以 coco/bbox_mAP 做 best，等指标起来再加） ===
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=True,
        max_keep_ckpts=2,
        save_last=True,
        save_best=['vasco/action_acc', 'vasco/dir_acc', 'seg/mIoU'],
        rule=['greater', 'greater', 'greater'],
    ),
)
