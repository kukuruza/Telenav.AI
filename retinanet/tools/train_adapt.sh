#!/usr/bin/env bash
set -e
WEIGHTS='imagenet'
TRAIN_SRC_PATH='/media/storage2/etoropov/datasets/telenav_ai_dataset/train_data'
TRAIN_DST_PATH='/media/storage2/etoropov/datasets/scotty/4096x2160_5Hz__2018_6_15_10_10_38'

echo 'Parameters:'
echo 'TRAIN_PATH:' $TRAIN_PATH
echo 'VALIDATION_PATH:' $VALIDATION_PATH
echo 'WEIGHTS:' $WEIGHTS
echo '------------------------------------'
TELENAV_HOME=/home/scotty/etoropov/Telenav.AI
PYTHONPATH=$TELENAV_HOME:$TELENAV_HOME/apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3  # Remove annoying TF debugging info.

python3 -u $TELENAV_HOME/retinanet/train_adapt.py \
    --imagenet-weights \
    --steps 35000 --batch-size 1 \
    ${@:1} \
    traffic_signs $TRAIN_SRC_PATH $TRAIN_DST_PATH
