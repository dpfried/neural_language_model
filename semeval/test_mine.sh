#!/bin/bash
#
#   examples.sh
#
#   Peter Turney
#   February 10, 2012
#
#
#   This shell script shows how you can use the Perl scripts to evaluate
#   the performance of your algorithm. The role of your algorithm is simulated
#   by the first Perl script, "random_scale.pl", which generates random
#   ratings for the word pairs. You would substitute your own algorithm for
#   "random_scale.pl", using some non-random method to assign ratings to the pairs.
#
#
#   random_scale.pl <input file of word pairs> <output file of rated pairs>
#
#   - assign random scale ratings to a list of word pairs, as a baseline
#

training_or_testing=$1
suffix=$2
echo -----------------------
echo RUNNING random_scale.pl
echo -----------------------
#
python ~/code/nlm/semeval.py $training_or_testing/Phase1Answers/Phase1Answers-$suffix.txt NLM/$training_or_testing/NLMScaled-$suffix.txt
#
#   maxdiff_to_scale.pl <input file of MaxDiff answers> <output file of rated pairs>
#
#   - convert MaxDiff answers to a list of word pairs rated on a scale
#   - see http://en.wikipedia.org/wiki/MaxDiff
#
echo ---------------------------
echo RUNNING maxdiff_to_scale.pl
echo ---------------------------
#
maxdiff_to_scale.pl $training_or_testing/Phase2Answers/Phase2Answers-$suffix.txt NLM/$training_or_testing/TurkerScaled-$suffix.txt
#
#   score_scale.pl <input file of Gold Standard pair ratings> 
#                  <input file of pair ratings to be evaluated> <output file of results>
#
#   - calculate the Spearman correlation between two sets of rated word pairs
#   - see http://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
#
echo ----------------------
echo RUNNING score_scale.pl
echo ----------------------
#
score_scale.pl NLM/$training_or_testing/TurkerScaled-$suffix.txt NLM/$training_or_testing/NLMScaled-$suffix.txt \
               NLM/$training_or_testing/SpearmanNLMScaled-$suffix.txt
#
#   scale_to_maxdiff.pl <input file of MaxDiff questions> <input file of scaled pairs> 
#                       <output file of answers to MaxDiff questions>
#
#   - given a set of MaxDiff questions and a set of word pairs rated on a scale,
#     answer the MaxDiff questions
#
echo ---------------------------
echo RUNNING scale_to_maxdiff.pl
echo ---------------------------
#
scale_to_maxdiff.pl $training_or_testing/Phase2Questions/Phase2Questions-$suffix.txt NLM/$training_or_testing/NLMScaled-$suffix.txt \
                    NLM/$training_or_testing/NLMMaxDiff-$suffix.txt
#
#   score_maxdiff.pl <input file of Mechanical Turk answers to MaxDiff questions> 
#                    <input file of MaxDiff answers to be evaluated> <output file of results>
#
#   - evaluate a set of answers to MaxDiff questions by comparing them with
#     the majority vote of Mechanical Turkers
#
echo ------------------------
echo RUNNING score_maxdiff.pl
echo ------------------------
#
score_maxdiff.pl $training_or_testing/Phase2Answers/Phase2Answers-$suffix.txt \
                 NLM/$training_or_testing/NLMMaxDiff-$suffix.txt NLM/$training_or_testing/MaxDiffNLM-$suffix.txt
#
#
#
