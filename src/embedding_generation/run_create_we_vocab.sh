#!/bin/bash

SCRIPTS_DIR=ranking_clarification_questions/src/embedding_generation
EMB_DIR=ranking_clarification_questions/embeddings/

python $SCRIPTS_DIR/create_we_vocab.py $EMB_DIR/vectors.txt $EMB_DIR/word_embeddings.p $EMB_DIR/vocab.p

