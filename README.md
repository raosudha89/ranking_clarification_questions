This repository contains information about the data and the code for the paper below:

<i><a href="https://arxiv.org/abs/1805.04655">
Learning to Ask Good Questions: Ranking Clarification Questions using Neural Expected Value of Perfect Information</a></i><br/>
Sudha Rao (raosudha@cs.umd.edu) and Hal Daumé III (hal@umiacs.umd.edu)<br/>
To appear in the proceedings of The 2018 Association of Computational Lingusitics (ACL 2018)

Download the clarification questions dataset from google drive here: https://go.umd.edu/clarification_questions_dataset <br/>
And copy the directory clarification_questions_dataset/data to ranking_clarification_questions/data

Download word embeddings trained on stackexchange datadump here: https://go.umd.edu/stackexchange_embeddings <br/>
And copy the directory stackexchange_embeddings/embeddings to ranking_clarification_questions/embeddings

The above dataset contains clarification questions for these three sites of stackexchange:
1. askubuntu.com
2. unix.stackexchange.com
3. superuser.com

To run models on a combination of the three sites above, check ranking_clarification_questions/src/models/README

To generate clarification questions for a different site of stackexchange, check ranking_clarification_questions/src/data_generation/README

To retrain word embeddings on a newer version of stackexchange datadump, check ranking_clarification_questions/src/embedding_generation/README

Please contact Sudha Rao (raosudha@cs.umd.edu) if you have any questions or any feedback.
