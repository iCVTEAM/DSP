_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    roi_head=dict(
        bbox_head=dict(num_classes=10)))

data_root = '/path/to/RUOD/'
metainfo = {
    'classes': ('holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish'),
}

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='RUOD_ANN/instances_train.json',
        data_prefix=dict(img='RUOD_pic/train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='RUOD_ANN/instances_test.json',
        data_prefix=dict(img='RUOD_pic/test/'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'RUOD_ANN/instances_test.json')
test_evaluator = val_evaluator