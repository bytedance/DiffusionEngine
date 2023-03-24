_base_ = [
    './dino_4scale_r50_8x2_12e_coco.py'
]

data = dict(
    train=dict(
        ann_file=[
            'data/coco/annotations/instances_train2017.json',
            'data/coco_de/annotations/ann.json',
        ],
        img_prefix=[
            'data/coco/train2017/',
            'data/coco_de/images',
        ]),
)
evaluation = dict(save_best='bbox_mAP')
lr_config = dict(step=[11])
runner = dict(max_epochs=12)
checkpoint_config = dict(max_keep_ckpts=3)
