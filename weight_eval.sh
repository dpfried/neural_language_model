#!/bin/zsh
python evaluation/evaluation.py --average_embeddings /cl/work/dfried/models/adagrad/only_syntactic/no_adagrad /cl/work/dfried/models/socher_dataset_1-10/{ntn,transe,gd}/only_sem /cl/work/dfried/models/socher_dataset_1-12/*/* --plot_interval 100 --limit 2000 --no_new
