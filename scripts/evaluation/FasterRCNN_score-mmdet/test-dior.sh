#!/bin/bash

EXP_DIR=${1%/}
VIZ_DIR=$2

CMD="python -m tools.test \
configs/_custom_/faster_rcnn_r50_fpn_1x-dior.py $DSP_PROJECT_DIR/pretrained/evaluation/mmdet/faster_rcnn_r50_fpn_1x-dior/epoch_12.pth \
--cfg-options \
test_dataloader.dataset.ann_file=$(realpath gt_jsons/dior/xml_result_novel.json) \
test_dataloader.dataset.data_prefix.img=$EXP_DIR/image \
test_evaluator.ann_file=$(realpath gt_jsons/dior/xml_result_novel.json) \
default_hooks.logger.out_dir=$EXP_DIR/mmdet_logs \
test_evaluator.classwise=True"

if [ -n "$VIZ_DIR" ]; then
    CMD="$CMD --show-dir $VIZ_DIR"
fi

echo "$CMD"
eval "$CMD"