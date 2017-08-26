# MetaphoricChange

Data and code for the experiments in 

* Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of CoNLL 2017. Vancouver, Canada.
           
### Test Set, Annotation Data and Results
 
Find the test set, the annotation data and the results [here](http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/metaphoric_change.en.html).

In `./dataset/` we provide the reduced test set version (as described in the paper) which is also transformed to a suitable input format for the measure scripts.


### Copyright

The measure code is based on the scripts in 

* V. Shwartz, E. Santus, and D. Schlechtweg. 2016. Hypernyms under Siege: Linguistically-motivated Artillery for Hypernymy Detection. CoRR abs - 1612 - 04460. [repository](https://github.com/vered1986/UnsupervisedHypernymy)

Wherever part of the code is copyrighted this is indicated in the respective file.


### Usage Note

The scripts should be run directly from the main directory. If you wish to do otherwise, you may have to change the path you add to the path attribute in `sys.path.append('./modules/')` in the scripts.


### Reproduction of Results

In order to reproduce the results described in the paper proceed in the following way:

1. Download the [DTA corpus](http://www.deutschestextarchiv.de/download) (DTA-Kernkorpus und Ergänzungstexte, TCF-Version vom 11. Mai 2016)
2. Obtain standard cooccurrence matrices for relevant time periods from corpus files with `./dsm/create_diachronic_cooc_files.py`, and an exemplar cooccurrence matrice for the whole corpus period with `./dsm/create_exemplar_cooc.py` and the test set `./dataset/testset_metaphoric_change_reduced_transformed.txt`
3. Transform matrices to pickle format with `./dsm/apply_ppmi_plmi.py`
4. Get word frequency ranks for relevant time periods from corpus files with `./dsm/get_freqs.py`
5. Calculate unscored results for non-normalized measures H and H_2 from standard cooccurrence matrices with `./measures/H.py` and `./measures/H_2.py` (and test set)
6. Normalize word frequency ranks with `./normalization/Freq_n.py`, and make unscored results from normalized frequency ranks with `./measures/Freq.py`
7. Get word entropy ranks for relevant time periods from corpus files with `./measures/H_rank.py`
8. Calculate unscored results for normalized word entropy by OLS from word frequency and entropy ranks with `./normalization/H_OLS.py`
9. Calculate unscored results for normalized word entropy by MON from exemplar cooccurrence matrice with `./normalization/H_MON.py` (right now, the script can only take a global number of contexts for all target words in the test set as input. This makes it tedious to calculate the measure in the case that you want to calculate it with the maximum number of contexts n possible for each target word, bounded by the smaller word frequency of the target word in one of the matrices, as you will have to specify n individually for each target word.)
10. Calculate predicted ranks from unscored results for each measure and relevant pairs of time periods with `./evaluation/score_results.py`
11. Calculate Spearman's rank correlation coefficient of gold rank and predicted ranks with `./evaluation/rank_correlation.py`

Please do not hesitate to write an email in case of any questions.
