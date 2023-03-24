_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=20)))

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/voc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
           'potted plant', 'sheep', 'couch', 'train', 'tv')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train.txt',
            img_prefix=data_root + 'VOC2012/',
            pipeline=train_pipeline),
        # dict(type='CocoDataset',
        #      ann_file=[
        #          'data/voc_de/annotations/voc12seg_i2i_ann.json',
        #          # 'data/voc_de/annotations/voc0712_i2i_ann.json',
        #      ],
        #      img_prefix=[
        #          'data/voc_de/images'
        #      ],
        #      pipeline=train_pipeline,
        #      classes=CLASSES,
        # ),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='mAP', save_best='mAP')

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='fixed')

# runtime settings, try to reimplement dall-e's baseline
checkpoint_config = dict(max_keep_ckpts=3)
runner = dict(type='EpochBasedRunner', max_epochs=20)

