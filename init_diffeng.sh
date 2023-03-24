#!/usr/bin/env bash

pip install -U openmim;
mim install mmcv-full;
pip install -v -e .;
pip install diffdist;
pip install open_clip_torch;
pip install gradio;