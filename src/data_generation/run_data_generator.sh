#!/bin/bash

DATADUMP_DIR=stackexchange #Directory containing xml files
EMB_DIR=embeddings
DATA_DIR=data
SITE_NAME=askubuntu.com
#SITE_NAME=unix.stackexchange.com
#SITE_NAME=superuser.com

SCRIPTS_DIR=ranking_clarification_questions/src/data_generation
LUCENE_DIR=ranking_clarification_questions/lucene

mkdir -p $DATA_DIR/$SITE_NAME

rm -r $DATA_DIR/$SITE_NAME/post_docs
rm -r $DATA_DIR/$SITE_NAME/post_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

rm -r $DATA_DIR/$SITE_NAME/ques_docs
rm -r $DATA_DIR/$SITE_NAME/ques_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/ques_docs

python $SCRIPTS_DIR/data_generator.py   --posts_xml $DATADUMP_DIR/$SITE_NAME/Posts.xml \
                                        --comments_xml $DATADUMP_DIR/$SITE_NAME/Comments.xml \
                                        --posthistory_xml $DATADUMP_DIR/$SITE_NAME/PostHistory.xml \
                                        --word_embeddings $EMB_DIR/word_embeddings.p \
                                        --vocab $EMB_DIR/vocab.p \
										--lucene_dir $LUCENE_DIR \
                                        --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs \
                                        --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts.txt \
                                        --site_name $SITE_NAME \
                                        --post_data_tsv $DATA_DIR/$SITE_NAME/post_data.tsv
                                        --qa_data_tsv $DATA_DIR/$SITE_NAME/qa_data.tsv

