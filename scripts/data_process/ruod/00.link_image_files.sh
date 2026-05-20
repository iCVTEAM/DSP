#!/usr/bin/env bash
mkdir -p $DSP_PROJECT_DIR/data/RUOD
ln -s /path/to/RUOD/RUOD_pic $DSP_PROJECT_DIR/data/RUOD/images # Replace with your actual path
cp ./008431.jpg $DSP_PROJECT_DIR/data/RUOD/images/train  # overwrite original image with corrected (rotated) version