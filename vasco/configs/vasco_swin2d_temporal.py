# # mmdetection-3.3.0/projects/vasco/configs/vasco_videoswin3d_fpn_temporal5.py
# _base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']
# custom_imports = dict(imports=['projects.vasco', 'projects.vasco.metrics', 'projects.vasco.hooks'], allow_failed_imports=False)

# # ===== 数据 =====
# dataset_type = 'VascoClipDataset'
# data_root = '/home/tili/masterwork/transformer/VASCO/'
# train_ann = data_root + 'vasco_train/train.json'
# train_img = data_root + 'vasco_train/'
# val_ann   = data_root + 'vasco_val/val.json'
# val_img   = data_root + 'vasco_val/'

# data_preprocessor = dict(
#     type='VascoDataPreprocessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_size_divisor=32)

# train_pipeline = [
#     dict(type='LoadMultiImagesFromFile'),
#     dict(
#         type='SeqColorJitter',
#         brightness=0.3,   # 可先用 0.2, 后续看效果再微调
#         contrast=0.2,
#         saturation=0.1,
#         p=0.5),
#     dict(type='SeqResizeFlipPad',
#          short_edge_choices=(864, 1080), 
#          max_long_edge=1600,
#          crop_size=(896, 1600)),
#     dict(type='PackVascoInputs'),
# ]
# val_pipeline = [
#     dict(type='LoadMultiImagesFromFile'),
#     dict(type='SeqResizeFlipPad', 
#          short_edge_choices=(1080,), 
#          max_long_edge=1600,
#          crop_size=(896, 1600)),
#     dict(type='PackVascoInputs'),
# ]
# test_pipeline = val_pipeline

# train_dataloader = dict(
#     batch_size=4 ,num_workers=2, persistent_workers=True,
#     collate_fn=dict(type='pseudo_collate'),
#     dataset=dict(
#         type=dataset_type, data_root=data_root,
#         ann_file=train_ann, data_prefix=dict(img=train_img),
#         temporal_window=12, temporal_span=20,
#         filter_cfg=dict(filter_empty_gt=True, min_size=1),
#         pipeline=train_pipeline
#     )
# )
# val_dataloader = dict(
#     batch_size=4, num_workers=2, persistent_workers=True,
#     collate_fn=dict(type='pseudo_collate'),
#     dataset=dict(
#         type=dataset_type, data_root=data_root,
#         ann_file=val_ann, data_prefix=dict(img=val_img),
#         temporal_window=12, temporal_span=20, test_mode=True,
#         pipeline=val_pipeline
#     )
# )
# test_dataloader = val_dataloader

# val_evaluator = [
#     dict(type='CocoMetric', ann_file=val_ann, metric='bbox'),
#     dict(
#         type='VascoActionDirMetric',
#         iou_thr=0.5,
#         num_actions=2,
#         num_dirs=6,
#         prefix='vasco'),
#     dict(type='VascoSegMetric', num_classes=6, prefix='seg'),
#     # dict(type='VascoEgoStopGoMetric', prefix='ego'),
# ]
# test_evaluator = val_evaluator

# custom_hooks = [
#     dict(
#         type='VascoValLossHook',
#         interval=1,
#         max_batches=40,
#         prefix='val'
#     )
# ]

# # 模型
# model = dict(
#     type='VascoSinglePathRCNN',
#     data_preprocessor=dict(type='VascoDataPreprocessor'),
#     # 2D 主链（COCO 检测预训练）
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',  
#         # embed_dims=96, depths=[2,2,18,2], num_heads=[3,6,12,24], # imgsmall, ade20ksmall
#         # embed_dims=128, depths=[2,2,18,2], num_heads=[4,8,16,32], # imgbase, ade20kbase
#         embed_dims=192, depths=[2,2,18,2], num_heads=[6,12,24,48], # ade20klarge
#         window_size=7, drop_path_rate=0.1,
#         frozen_stages=4,
#         init_cfg=dict(
#             type='Pretrained', 
#             checkpoint='/home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_ade20k_20220318_015320-48d180dd.pth',
#         )
#     ),
#     neck=dict(
#         _delete_=True,
#         type='FPN',
#         # in_channels=[96, 192, 384, 768], out_channels=256, num_outs=4 # [embed_dims, 2*embed_dims, 4*embed_dims, 8*embed_dims]
#         # in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=4
#         # in_channels=[128, 256, 512, 1024], out_channels=512, num_outs=4
#         in_channels=[192, 384, 768, 1536], out_channels=256, num_outs=4
#     ),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256, feat_channels=256,
#         # in_channels=512, feat_channels=512,
#         anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32]),
#         bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[1.,1.,1.,1.]),
#         loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)
#     ),
#     roi_head=dict(
#         _delete_=True,
#         type='VascoSinglePathRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
#             out_channels=256, 
#             # out_channels=512,
#             featmap_strides=[4,8,16,32]
#         ),
#         bbox_head=dict(   # 仅用于 get_targets / loss 的参数来源（前向由 RoIHead 自己做）
#             type='Shared2FCBBoxHead',
#             in_channels=256, 
#             # in_channels=512,
#             fc_out_channels=1024, roi_feat_size=7,
#             num_classes=28, reg_class_agnostic=True,
#             bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
#             loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0) 
#         ),
#         temporal_aggregator=dict(
#             type='TemporalMSDeformAttnAggregator',
#             embed_dims=256,
#             # embed_dims=512,
#             num_levels=48,   # t*4
#             num_heads=8,
#             num_points=4,
#             dropout=0.0,
#             # make embeddings long enough
#             time_max_len=12,   # t 
#             level_max_len=48,  # t*4
#         ),
#         # temporal_aggregator=None,
#         action_dir_head=dict(
#             type='ActionDirHead',
#             # in_channels=256, roi_feat_size=7, hidden_dim=1024, with_mlp=True,
#             in_channels=256, roi_feat_size=7, hidden_dim=512, with_mlp=True,
#             action_hidden_dim=256, dir_hidden_dim=512,
#             action_dropout=0.3, dir_dropout=0.2,
#             # in_channels=512, roi_feat_size=7, hidden_dim=512, with_mlp=True,
#             num_actions=2, num_dirs=6, dropout=0.2,
#             loss_action=dict(type='CrossEntropyLoss', loss_weight=0.5),
#             loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.5),
#             action_class_weight=None,
#             dir_class_weight=None,
#             static_dir_loss_weight= 1.0,
#             dynamic_dir_loss_weight= 2
#         ),
#         # action_dir_head=None,

#         semantic_head=dict(
#             type='SemanticHead',
#             in_channels=(256, 256, 256, 256), 
#             # in_channels=(512, 512, 512),
#             num_classes=7, ce_weight=1, dice_weight=1.0, ignore_index=255,
#             dropout_ratio=0.2,label_smoothing=0.1,
#             bg_id=6, bg_weight=0.1
#         ),
#         # semantic_head=None,
#         # ego_stopgo_head=dict(
#         #     type='EgoStopGoHead',
#         #     num_det_classes=29,
#         #     num_action_classes=2,
#         #     num_dir_classes=6,
#         #     ego_num_choices=5,
#         #     class_emb_dim=32,
#         #     action_emb_dim=16,
#         #     dir_emb_dim=16,
#         #     seg_in_dim=32,
#         #     seg_proj_dim=32,
#         #     choice_emb_dim=16,
#         #     hidden_dim=256,
#         #     num_classes=2,
#         #     dropout=0.2,
#         #     class_weight=[12.0, 1.0],
#         #     ignore_index=-1,
#         #     label_smoothing=0.0,
#         #     use_focal_ce=False,
#         # ),
#         ego_stopgo_head=None,
#         # ego_topk=10,
#         # ego_roi_score_thr=0.2,
#         # ego_roi_use_rpn_in_train=True,
#         aux_detach=False,
#         ad_topk_per_gt = 10,
#     ),
#     train_cfg=dict(
#         rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.3),
#                  sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
#                  allowed_border=-1, pos_weight=-1, debug=False),
#         rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
#                   sampler=dict(type='RandomSampler', num=512, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
#                   debug=False)
#     ),
#     test_cfg=dict(
#         rpn=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
#         rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.7), max_per_img=300)
#     )
# )

# # ===== 优化与调度 =====
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=5e-2),
#     # optimizer=dict(_delete_=True, type='AdamW', lr=5e-5, weight_decay=1e-1),
#     clip_grad=dict(max_norm=10, norm_type=2),
#     # paramwise_cfg=dict(custom_keys={
#     #     'roi_head.bbox_head': dict(lr_mult=0.5),
#     #     'rpn_head': dict(lr_mult=1),
#     #     'roi_head.semantic_head': dict(lr_mult=0.5),
#     #     'roi_head.action_dir_head': dict(lr_mult=0.5)})
# )

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
# param_scheduler = [
#     dict(type='LinearLR', start_factor=1.0/1000, by_epoch=False, begin=0, end=1500),
#     dict(type='MultiStepLR', milestones=[12, 18], gamma=0.1),
# ]

# default_hooks = dict(
#     checkpoint=dict(
#         type='CheckpointHook', interval=10, by_epoch=True, max_keep_ckpts=1, save_last=False,
#         # save_best=['coco/bbox_mAP','vasco/action_mF1','vasco/dir_mF1','seg/mIoU'],
#         # rule=['greater','greater','greater','greater']
#         )
# )

#==================================================
#==================================================

# 后置ego
# mmdetection-3.3.0/projects/vasco/configs/vasco_videoswin3d_fpn_temporal5.py
_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']
custom_imports = dict(imports=['projects.vasco', 'projects.vasco.metrics', 'projects.vasco.hooks'], allow_failed_imports=False)

# ===== Ego-only training: load frozen perception checkpoint =====
# Replace this path with the selected Det+Seg+AD checkpoint, e.g. epoch_13.pth or epoch_20.pth.
load_from = '/home/tili/masterwork/transformer/mmdetection-3.3.0/work_dirs/vasco_swin_2d-ade20klarge-proposalfreeze-segv2-allweight1-topk10adclassweightdir-2s-det+adv4+seg-egonone/epoch_20.pth'

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
        p=0.5),
    dict(type='SeqResizeFlipPad',
         short_edge_choices=(864, 1080), 
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
    dict(
        type='VascoActionDirMetric',
        iou_thr=0.6,
        num_actions=2,
        num_dirs=6,
        prefix='vasco'),
    dict(type='VascoSegMetric', num_classes=6, prefix='seg'),
    dict(type='VascoEgoStopGoMetric', prefix='ego'),
]
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='VascoValLossHook',
        interval=1,
        max_batches=None,
        prefix='val'
    )
]

# 模型
model = dict(
    type='VascoSinglePathRCNN',
    data_preprocessor=dict(type='VascoDataPreprocessor'),
    # 2D 主链（COCO 检测预训练）
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',  
        # embed_dims=96, depths=[2,2,18,2], num_heads=[3,6,12,24], # imgsmall, ade20ksmall
        # embed_dims=128, depths=[2,2,18,2], num_heads=[4,8,16,32], # imgbase, ade20kbase
        embed_dims=192, depths=[2,2,18,2], num_heads=[6,12,24,48], # ade20klarge
        window_size=7, drop_path_rate=0.1,
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/home/tili/masterwork/transformer/mmdetection-3.3.0/projects/vasco/configs/upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_ade20k_20220318_015320-48d180dd.pth',
        )
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        # in_channels=[96, 192, 384, 768], out_channels=256, num_outs=4 # [embed_dims, 2*embed_dims, 4*embed_dims, 8*embed_dims]
        # in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=4
        # in_channels=[128, 256, 512, 1024], out_channels=512, num_outs=4
        in_channels=[192, 384, 768, 1536], out_channels=256, num_outs=4
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256, feat_channels=256,
        # in_channels=512, feat_channels=512,
        anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[1.,1.,1.,1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        _delete_=True,
        type='VascoSinglePathRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256, 
            # out_channels=512,
            featmap_strides=[4,8,16,32]
        ),
        bbox_head=dict(   # 仅用于 get_targets / loss 的参数来源（前向由 RoIHead 自己做）
            type='Shared2FCBBoxHead',
            in_channels=256, 
            # in_channels=512,
            fc_out_channels=1024, roi_feat_size=7,
            num_classes=28, reg_class_agnostic=True,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0) 
        ),
        temporal_aggregator=dict(
            type='TemporalMSDeformAttnAggregator',
            embed_dims=256,
            # embed_dims=512,
            num_levels=32, # t*4
            num_heads=8,
            num_points=4,
            dropout=0.0,
            # make embeddings long enough
            time_max_len=8,  #t
            level_max_len=32, #t*4
        ),
        # temporal_aggregator=None,
        action_dir_head=dict(
            type='ActionDirHead',
            # in_channels=256, roi_feat_size=7, hidden_dim=1024, with_mlp=True,
            in_channels=256, roi_feat_size=7, hidden_dim=512, with_mlp=True,
            action_hidden_dim=256, dir_hidden_dim=512,
            action_dropout=0.3, dir_dropout=0.2,
            # in_channels=512, roi_feat_size=7, hidden_dim=512, with_mlp=True,
            num_actions=2, num_dirs=6, dropout=0.2,
            loss_action=dict(type='CrossEntropyLoss', loss_weight=0.5),
            loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.5),
            action_class_weight=None,
            dir_class_weight=None,
            static_dir_loss_weight= 1.0,
            dynamic_dir_loss_weight= 2
        ),
        # action_dir_head=None,

        semantic_head=dict(
            type='SemanticHead',
            in_channels=(256, 256, 256, 256), 
            # in_channels=(512, 512, 512),
            num_classes=7, ce_weight=1, dice_weight=1.0, ignore_index=255,
            dropout_ratio=0.2,label_smoothing=0.1,
            bg_id=6, bg_weight=0.1,
            return_feat=True,
        ),
        # semantic_head=None,
        ego_stopgo_head=dict(
            type='EgoStopGoHead',
            num_det_classes=28,
            num_action_classes=2,
            num_dir_classes=6,

            class_emb_dim=32,
            action_emb_dim=16,
            dir_emb_dim=16,

            roi_feat_dim=256,
            obj_proj_dim=256,
            seg_feat_dim=256,
            seg_feat_proj_dim=256,

            use_choice=False,

            use_ad=True,
            use_light_branch=False,
            use_seg_feat=True,

            ego_num_choices=5,
            choice_emb_dim=16,
            light_class_ids=[6,7,10],
            hidden_dim=256,
            num_classes=2,
            dropout=0.2,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            class_weight=[12.0, 1.0],
            ignore_index=-1,
            label_smoothing=0.0,
            use_focal_ce=True,
            focal_gamma=2.0,
        ),
        # ego_stopgo_head=None,
        ego_topk=10,
        ego_roi_score_thr=0.2,
        ego_roi_use_rpn_in_train=True,
        aux_detach=False,
        ad_topk_per_gt = 10,
    ),
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.3),
                 sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
                 allowed_border=-1, pos_weight=-1, debug=False),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
                  sampler=dict(type='RandomSampler', num=512, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
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
    # Ego-only training: freeze perception modules by zero learning rate.
    # Only roi_head.ego_stopgo_head is optimized.
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.0, decay_mult=0.0),
        'neck': dict(lr_mult=0.0, decay_mult=0.0),
        'rpn_head': dict(lr_mult=0.0, decay_mult=0.0),
        'roi_head.bbox_head': dict(lr_mult=0.0, decay_mult=0.0),
        'roi_head.temporal_aggregator': dict(lr_mult=0.0, decay_mult=0.0),
        'roi_head.action_dir_head': dict(lr_mult=0.0, decay_mult=0.0),
        'roi_head.semantic_head': dict(lr_mult=0.0, decay_mult=0.0),
        'roi_head.ego_stopgo_head': dict(lr_mult=1.0, decay_mult=1.0),
    })
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/1000, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[6, 9], gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=10, by_epoch=True, max_keep_ckpts=1, save_last=False,
        save_best=['ego/stopgo_mF1'],
        rule=['greater']
        )
)
