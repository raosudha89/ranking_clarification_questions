# Prerequisites

* Install lasagne: http://lasagne.readthedocs.io/en/latest/user/installation.html
* Install numpy, scipy
* Version information:
Python 2.7.5
Theano 0.9.0dev5
Lasagne 0.2.dev1
Cuda 8.0.44
Cudnn 5.1

# Loading data 

Load data from askubuntu.com

* Set "SITE_NAME=askubuntu.com" in ranking_clarification_questions/src/models/run_load_data.sh 
* cd ranking_clarification_questions; sh src/models/run_load_data.sh

Load data from unix.stackexchange.com

* Set "SITE_NAME=unix.stackexchange.com" in ranking_clarification_questions/src/models/run_load_data.sh 
* cd ranking_clarification_questions; sh src/models/run_load_data.sh

Load data from superuser.com

* Set "SITE_NAME=superuser.com" in ranking_clarification_questions/src/models/run_load_data.sh 
* cd ranking_clarification_questions; sh src/models/run_load_data.sh

Combine data from three domains

* cd ranking_clarification_questions; sh src/models/run_combine_domains.sh
* cat data/askubuntu.com/human_annotations data/unix.stackexchange.com/human_annotations data/superuser.com/human_annotations > askubuntu_unix_superuser/human_annotations

# Running neural baselines on the combined data

Neural(p,q)

* Set "MODEL=baseline_pq" in ranking_clarification_questions/src/models/run_main.sh
* cd ranking_clarification_questions; sh src/models/run_main.sh

Neural(p,a)

* Set "MODEL=baseline_pa" in ranking_clarification_questions/src/models/run_main.sh
* cd ranking_clarification_questions; sh src/models/run_main.sh

Neural(p,q,a)

* Set "MODEL=baseline_pqa" in ranking_clarification_questions/src/models/run_main.sh
* cd ranking_clarification_questions; sh src/models/run_main.sh

# Runing EVPI model on the combined data

* Set "MODEL=evpi" in ranking_clarification_questions/src/models/run_main.sh
* cd ranking_clarification_questions; sh src/models/run_main.sh

# Runing evaluation

* cd ranking_clarification_questions; sh src/evaluation/run_evaluation.sh
