#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="0"
WEIGHTS='imagenet'
TRAIN_PATH='/media/storage2/etoropov/datasets/telenav_ai_dataset/train_data'
VALIDATION_PATH='/media/storage2/etoropov/datasets/telenav_ai_dataset/train_data'

echo 'Parameters:'
echo 'TRAIN_PATH:' $TRAIN_PATH
echo 'VALIDATION_PATH:' $VALIDATION_PATH
echo 'WEIGHTS:' $WEIGHTS
echo '------------------------------------'
TELENAV_HOME=/home/scotty/etoropov/Telenav.AI
PYTHONPATH=$TELENAV_HOME:$TELENAV_HOME/apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3  # Remove annoying TF debugging info.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -u retinanet/train.py \
    --imagenet-weights \
    --steps 35000 --multi-gpu 1 --batch-size 1 --evaluate_score_threshold 0.5 \
    traffic_signs $TRAIN_PATH $VALIDATION_PATH
