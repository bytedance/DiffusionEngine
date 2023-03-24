_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
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
checkpoint_config = dict(max_keep_ckpts=3)
evaluation = dict(interval=1, classwise=True, metric='bbox', save_best='bbox_mAP')

