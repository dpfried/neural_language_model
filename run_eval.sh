#!/bin/zsh
python evaluation/evaluation.py /cl/work/dfried/models/adagrad/{tensor_fix_no_init_0.05/no_adagrad,no_init_0.05/,relational_no_init_0.05/,only_semantic/no_adagrad,only_syntactic/no_adagrad} --plot_interval 100 --limit 1500 --average_embeddings --no_new
