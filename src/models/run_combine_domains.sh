#!/bin/bash

DATA_DIR=data
UBUNTU=askubuntu.com
UNIX=unix.stackexchange.com
SUPERUSER=superuser.com
SCRIPTS_DIR=src/models
SITE_NAME=askubuntu_unix_superuser

mkdir -p $DATA_DIR/$SITE_NAME

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_train.p \
										$DATA_DIR/$UNIX/post_vectors_train.p \
										$DATA_DIR/$SUPERUSER/post_vectors_train.p \
										$DATA_DIR/$SITE_NAME/post_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_train.p \
										$DATA_DIR/$UNIX/ques_list_vectors_train.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_train.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_train.p \
										$DATA_DIR/$UNIX/ans_list_vectors_train.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_train.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_train.p \
										$DATA_DIR/$UNIX/post_ids_train.p \
										$DATA_DIR/$SUPERUSER/post_ids_train.p \
										$DATA_DIR/$SITE_NAME/post_ids_train.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_tune.p \
										$DATA_DIR/$UNIX/post_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/post_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/post_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_tune.p \
										$DATA_DIR/$UNIX/ques_list_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_tune.p \
										$DATA_DIR/$UNIX/ans_list_vectors_tune.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_tune.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_tune.p \
										$DATA_DIR/$UNIX/post_ids_tune.p \
										$DATA_DIR/$SUPERUSER/post_ids_tune.p \
										$DATA_DIR/$SITE_NAME/post_ids_tune.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_vectors_test.p \
										$DATA_DIR/$UNIX/post_vectors_test.p \
										$DATA_DIR/$SUPERUSER/post_vectors_test.p \
										$DATA_DIR/$SITE_NAME/post_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ques_list_vectors_test.p \
										$DATA_DIR/$UNIX/ques_list_vectors_test.p \
										$DATA_DIR/$SUPERUSER/ques_list_vectors_test.p \
										$DATA_DIR/$SITE_NAME/ques_list_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/ans_list_vectors_test.p \
										$DATA_DIR/$UNIX/ans_list_vectors_test.p \
										$DATA_DIR/$SUPERUSER/ans_list_vectors_test.p \
										$DATA_DIR/$SITE_NAME/ans_list_vectors_test.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/post_ids_test.p \
										$DATA_DIR/$UNIX/post_ids_test.p \
										$DATA_DIR/$SUPERUSER/post_ids_test.p \
										$DATA_DIR/$SITE_NAME/post_ids_test.p

cat $DATA_DIR/$UBUNTU/human_annotations $DATA_DIR/$UNIX/human_annotations $DATA_DIR/$SUPERUSER/human_annotations > $DATA_DIR/$SITE_NAME/human_annotations
