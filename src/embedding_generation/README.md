# Prerequisite

* Download and compile GLoVE code from: https://nlp.stanford.edu/projects/glove/

# NOTE

* Word embeddings pretrained on stackexchange datadump (version year 2017) can be found here: https://go.umd.edu/stackexchange_embeddings 

# Steps to retrain word embeddings

1. Download all domains of stackexchange from: https://archive.org/download/stackexchange
2. Extract text from all Posts.xml, Comments.xml and PostHistory.xml
3. Save the combined data under: stackexchange/stackexchange_datadump.txt
4. cd ranking_clarification_questions; sh src/embedding_generation/run_glove.sh
5. cd ranking_clarification_questions; sh src/embedding_generation/run_create_we_vocab.sh
