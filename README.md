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