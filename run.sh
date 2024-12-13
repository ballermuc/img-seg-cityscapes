#!/bin/bash

# Environment setup
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONHASHSEED=0

# General training parameters
BATCH_SIZE_TRAINING=24
BATCH_SIZE_VALIDATION=12
BATCH_SIZE_TEST=6
LEARNING_RATE=5e-4
EPOCHS=50
OPTIMIZER="Adam"
NUM_WORKERS=16

# Model-specific parameters
MODEL="DeepLabV3P+"
ENCODER="efficientnet-b4"
ENCODER_WEIGHTS="imagenet"
PATCH_SIZE=512

# Loss functions
LOSS="ce"  # Choose from 'ce', 'dice', 'focal'
CLASS_WEIGHTS="none"  # Class weights as "none" or "11:10,12:10"

# DeepLabV3+ specific parameters
ENCODER_OUTPUT_STRIDE=8  # Choose between 8 and 16
DECODER_ATROUS_RATES="6,12,18"
DECODER_CHANNELS=512

# Unet-specific parameters
DECODER_USE_BATCHNORM="True"  # Choose from 'True', 'False', 'inplace'
DECODER_ATTENTION_TYPE="None"  # Choose from 'None', 'scse'

# Ensure CLASS_WEIGHTS is passed correctly
CLASS_WEIGHTS_ARG=""
if [ "$CLASS_WEIGHTS" != "none" ]; then
  CLASS_WEIGHTS_ARG="--class_weights=${CLASS_WEIGHTS}"
fi

# Execute the training script with dynamically passed arguments
python3 ./train.py \
  --batch_size_training=${BATCH_SIZE_TRAINING} \
  --batch_size_validation=${BATCH_SIZE_VALIDATION} \
  --batch_size_test=${BATCH_SIZE_TEST} \
  --learning_rate=${LEARNING_RATE} \
  --epochs=${EPOCHS} \
  --optimizer=${OPTIMIZER} \
  --num_workers=${NUM_WORKERS} \
  --model=${MODEL} \
  --encoder=${ENCODER} \
  --encoder_weights=${ENCODER_WEIGHTS} \
  --patch_size=${PATCH_SIZE} \
  --loss=${LOSS} \
  $CLASS_WEIGHTS_ARG \
  --encoder_output_stride=${ENCODER_OUTPUT_STRIDE} \
  --decoder_atrous_rates=${DECODER_ATROUS_RATES} \
  --decoder_channels=${DECODER_CHANNELS} \
  --decoder_use_batchnorm=${DECODER_USE_BATCHNORM} \
  --decoder_attention_type=${DECODER_ATTENTION_TYPE}
