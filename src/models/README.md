# Prerequisites

* Install lasagne: http://lasagne.readthedocs.io/en/latest/user/installation.html
* Install numpy, scipy

# First load the data: 

* Load data from askubuntu.com

1. Set "SITE_NAME=askubuntu.com" in ranking_clarification_questions/src/models/run_load_data.sh 
2. cd ranking_clarification_questions; sh src/models/run_load_data.sh

* Load data from unix.stackexchange.com

1. Set "SITE_NAME=unix.stackexchange.com" in ranking_clarification_questions/src/models/run_load_data.sh 
2. cd ranking_clarification_questions; sh src/models/run_load_data.sh

* Load data from superuser.com

1. Set "SITE_NAME=superuser.com" in ranking_clarification_questions/src/models/run_load_data.sh 
2. cd ranking_clarification_questions; sh src/models/run_load_data.sh

* Combine data from three domains

1. cd ranking_clarification_questions; sh src/models/run_combine_domains.sh
2. cat data/askubuntu.com/human_annotations data/unix.stackexchange.com/human_annotations data/superuser.com/human_annotations > askubuntu_unix_superuser/human_annotations

# -----------------------------------------------------------------------------------

# Run different neural baselines on the combined data:

* Neural(p,q)

1. Set "MODEL=baseline_pq" in ranking_clarification_questions/src/models/run_main.sh
2. cd ranking_clarification_questions; sh src/models/run_main.sh

* Neural(p,a)

1. Set "MODEL=baseline_pa" in ranking_clarification_questions/src/models/run_main.sh
2. cd ranking_clarification_questions; sh src/models/run_main.sh

* Neural(p,q,a)

1. Set "MODEL=baseline_pqa" in ranking_clarification_questions/src/models/run_main.sh
2. cd ranking_clarification_questions; sh src/models/run_main.sh

# -----------------------------------------------------------------------------------

# Run EVPI model on the combined data:

1. Set "MODEL=evpi" in ranking_clarification_questions/src/models/run_main.sh
2. cd ranking_clarification_questions; sh src/models/run_main.sh

# -----------------------------------------------------------------------------------

# Run evaluation

1. cd ranking_clarification_questions; sh src/evaluation/run_evaluation.sh
