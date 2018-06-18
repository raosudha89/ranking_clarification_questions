#!/bin/bash

DATA_DIR=data
#SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
#SITE_NAME=superuser.com
SITE_NAME=askubuntu_unix_superuser

SCRIPTS_DIR=src/evaluation
#MODEL=baseline_pq
#MODEL=baseline_pa
#MODEL=baseline_pqa
MODEL=evpi

python $SCRIPTS_DIR/evaluate_model_with_human_annotations.py \
												--human_annotations_filename $DATA_DIR/$SITE_NAME/human_annotations \
												--model_predictions_filename $DATA_DIR/$SITE_NAME/test_predictions_${MODEL}.out.epoch13 \
