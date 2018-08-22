#!/usr/bin/env bash
set -e
ANNOTATIONS='/home/etoropov/datasets/BDD/annotation_list_train_v2.csv'
CLASSES='/home/etoropov/datasets/BDD/class_list.csv'
TRAIN_DST_PATH='/home/etoropov/datasets/scotty_2018_6_7'

TELENAV_HOME=/home/etoropov/projects/Telenav.AI
PYTHONPATH=$TELENAV_HOME:$TELENAV_HOME/apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3  # Remove annoying TF debugging info.

#    traffic_signs $TRAIN_SRC_PATH 
python3 -u $TELENAV_HOME/retinanet/train_adapt.py \
    ${@:1} \
    csv $ANNOTATIONS $CLASSES $TRAIN_DST_PATH

