files you can actually run:
==========================
admm_trainer.py - used to train ADMM using distance in wordnet
    - requires access to a dump of wordnet pairwise similarities, one is available at /cl/nldata/books_google_ngrams_eng/wordnet_similarities_max.npy
    - by default will copy this sim file to a shared memory file in TMP in case multiple procs are running

relational_admm_trainer.py - used to train ADMM using wordnet relations

evaluation/ - contains interfaces to semeval, grefenstette, and wordsim eval sets, plus one script evaluation.py that will run all
    usage:
    python evaluation/evaluation.py MODEL_DIRECTORIES --plot_interval PLOT_INTEVAL --limit LIMIT
        - caches stats in .pkl files in each model directory, reads these if necessary
        - flag --no_new will only read existing cached stats

plot_admm_stats.py - plots cached ADMM stats from stats.pkl files in model directories
    usage:
    python plot_admm_stats.py MODEL_DIRECTORIES --limit LIMIT

some dependent files
================

models.py - contains all Theano models

policies.py - gradient descent schedules

corpus_creator.py - build an hd5 file from Berkeley format corpus

dump_wordnet_similarities.py - used to dump the pairwise similarities needed by admm_trainer.py
