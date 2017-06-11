# MetaphoricChange
Data and code for the experiments in: "German in Flux: Detecting Metaphoric Change via Word Entropy". Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. CoNLL 2017.

Find the test set, the annotation data and the results [here](http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/metaphoric_change.en.html).


The code is based on the scripts in 

V. Shwartz, E. Santus, and D. Schlechtweg. 2016. Hypernyms under Siege: Linguistically-motivated Artillery for Hypernymy Detection. CoRR abs - 1612 - 04460. [link](https://github.com/vered1986/UnsupervisedHypernymy)

Wherever part of the code is copyrighted this is indicated in the respective file.

Usage note:

The scripts in source/measures/ should be run directly from their directory. If you wish to do otherwise, you may have to change the path you add to the path attribute in sys.path.append('../') in the respective measure script.