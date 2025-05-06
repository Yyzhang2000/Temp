#!/bin/bash

# Set visible GPU (optional if you're not using torchrun/DDP)
export CUDA_VISIBLE_DEVICES=0

# Run FixMatch with adaptive thresholding on CIFAR-10 (4000 labeled)
python train4.py \
  --dataset cifar10 \
  --num-labeled 4000 \
  --arch wideresnet \
  --batch-size 32 \
  --total-steps 204800 \
  --lr 0.03 \
  --mu 7 \
  --lambda-u 1 \
  --use-ema \
  --ema-decay 0.999 \
  --global-thresh-base 0.7 \
  --class-conf-factor 0.1 \
  --confusion-factor 0.05 \
  --initial-class-threshold 0.9 \
  --ema-decay-stats 0.9 \
  --num-workers 4 \
  --gpu-id 0 \
  --out result/cifar10_4000_multiview_adaptive
