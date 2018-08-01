#!/bin/bash

DATA_DIR=data
EMB_DIR=embeddings
#SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
SITE_NAME=superuser.com

SCRIPTS_DIR=src/models

python $SCRIPTS_DIR/load_data.py	--post_data_tsv $DATA_DIR/$SITE_NAME/post_data.tsv \
									--qa_data_tsv $DATA_DIR/$SITE_NAME/qa_data.tsv \
									--train_ids $DATA_DIR/$SITE_NAME/train_ids \
									--tune_ids $DATA_DIR/$SITE_NAME/tune_ids \
									--test_ids $DATA_DIR/$SITE_NAME/test_ids \
									--vocab $EMB_DIR/vocab.p
