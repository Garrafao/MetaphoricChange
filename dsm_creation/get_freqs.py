import sys
sys.path.append('./modules/')

import os
import codecs
from xml.etree.ElementTree import ElementTree
from docopt import docopt
from collections import Counter
import logging

from dsm_module import *


class BorderReached(Exception): 
    pass

def main():
    """
    Get word frequencies from various slices of DTA as described in:
    
        Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
            2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
            CoNLL 2017. Vancouver, Canada.
    """

    # Get the arguments
    args = docopt("""Get word frequencies from various slices of DTA.

    Usage:
        get_freqs.py <dta-tcf_dir> <outDir> <frequency_file> <time_boundary_1-time_boundary_n>
        
    Arguments:
        <dta-tcf_dir> = the DTA-tcf directory
        <outDir> = the directory of the sparse matrix output files
        <frequency_file> = the file containing frequent lemmas (>= MIN_FREQ)
        <time_boundary_1-time_boundary_n> = the time boundaries to slice up the corpus

    """)
    
    dtatcfdir = args['<dta-tcf_dir>']
    freq_file = args['<frequency_file>']
    outDir = args['<outDir>']      
    time_boundaries_str = args['<time_boundary_1-time_boundary_n>']
    time_boundaries = time_boundaries_str.split("-")
    tagset = ['N', 'V', 'AD']
    join_sign = ':'

    
    # Load the frequent words file
    with codecs.open(freq_file, "r", "utf-8") as f_in:
        freq_words = set([line.strip() for line in f_in])

    
    logging.basicConfig(filename= outDir + 'log_' + "get_freqs_" + time_boundaries_str + '.txt', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info(args)
         
     
    # build every interval seperately   
    for t in range (1,len(time_boundaries)):

        interval_start = int(time_boundaries[t-1])
        interval_end = int(time_boundaries[t])
        
        print "Building corpus from %d to %d..."  % (interval_start, interval_end)
        
        print "Building lemma list..."            
        lemma_list = []
        
        # build a list of lemmas over all relevant files
        for root, dirs, files in os.walk(dtatcfdir):
            for n, filename in enumerate(files):
                if ".tcf" in filename:
                    dta_file = os.path.join(root, filename)
                    print 'Reading dta file %d/%d...' % (n, len(files))
                    tree = ElementTree()
                    tree.parse(dta_file)
                    
                    # just consider files from the time interval
                    file_boundary = get_file_boundary(time_boundaries, tree)
                    if not file_boundary == interval_end or file_boundary == 0:
                        continue

                    lemmas = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}lemmas")
                    POStags = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}POStags")

                    for i, lemma_elem in enumerate(lemmas):
                        try:
                            lemma = lemma_elem.text
                            pos = POStags[i].text
                            target = lemma + join_sign + pos[0] # lemma + first char of POS, e.g. run-v / run-n
                                                        
                            if not target in freq_words or not True in [pos.startswith(x) for x in tagset]:
                                continue
                            
                            lemma_list.append(target)
                        except:
                            pass
                            print "token skipped..."
                            
        corpus_size = len(lemma_list)                      
        print "Interval %d-%d has size %d" % (interval_start, interval_end, corpus_size)
        logging.info("Interval %d-%d has size %d" % (interval_start, interval_end, corpus_size))

        #build frequency file from lemmas
        outputpath = outDir + "get_freqs_%d-%d_freq.txt" % (interval_start, interval_end)
        print 'Building frequency file to ' + outputpath + "..."
        lemma_count = Counter(lemma_list)
        
        # Rank the lemmas
        lemmas_ranked = sorted(lemma_count, key=lambda x: -(lemma_count[x]))

        with open(outputpath, 'w') as f_out:
            for lemma in lemmas_ranked:
                print >> f_out, '\t'.join((lemma.encode('utf-8'), str(float(lemma_count[lemma]))))


if __name__ == '__main__':
    main()
