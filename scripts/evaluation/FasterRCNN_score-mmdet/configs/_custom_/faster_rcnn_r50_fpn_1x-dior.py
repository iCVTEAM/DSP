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
        bbox_head=dict(num_classes=20)))

data_root = '/path/to/DIOR-VOC/'
metainfo = {
    'classes': ('vehicle', 'baseballfield', 'groundtrackfield', 'windmill', 'bridge', 'overpass', 'ship', 'airplane', 'tenniscourt', 'airport',
                'expressway-service-area', 'basketballcourt', 'stadium', 'storagetank', 'chimney', 'dam', 'expressway-toll-station', 'golffield', 'trainstation', 'harbor'),
}

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file='')
test_evaluator = val_evaluator

# visualizer = dict(
#     type='DetLocalVisualizer',
#     vis_backends=[dict(type='LocalVisBackend')],
#     name='visualizer',
#     text_scale=1.5,
#     line_width=3,
#     alpha=0.8
# )