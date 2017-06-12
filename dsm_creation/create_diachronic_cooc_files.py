import sys
sys.path.append('./modules/')

import os
import codecs
from xml.etree.ElementTree import ElementTree
from docopt import docopt
from collections import defaultdict
import logging

from dsm_module import *


MIN_FREQ = 5

class BorderReached(Exception): 
    pass


def main():
    """
    Create co-occurence matrices from the tcf-version of Deutsches Textarchiv (http://www.deutschestextarchiv.de/download) 
    as described in:
    
        Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
            2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
            CoNLL 2017. Vancouver, Canada, 2017.
    """

    # Get the arguments
    args = docopt("""Create co-occurence matrices in the format 'lemma1:POS lemma2:POS freq' for different time intervals.

    Usage:
        create_diachronic_cooc_files.py [--equal_sized_corpora] [--build_frequency_file] <dta-tcf_dir> <outDir> <frequency_file> <time_boundary_1-time_boundary_n> <window_size>
        
    Arguments:
        <dta-tcf_dir> = the DTA-tcf directory
        <frequency_file> = the file containing frequent lemmas (>= MIN_FREQ)
        <outDir> = the directory of the sparse matrix output files
        <time_boundary_1-time_boundary_n> = the time boundaries to slice up the corpus
        <window_size> = the linear distance of context words to consider
        
    Options:
        --equal_sized_corpora  for comparability
        --build_frequency_file  in case there is no frequency file yet
        
    """)
    
    dtatcfdir = args['<dta-tcf_dir>']
    freq_file = args['<frequency_file>']
    outDir = args['<outDir>']      
    time_boundaries_str = args['<time_boundary_1-time_boundary_n>']
    time_boundaries = time_boundaries_str.split("-")
    window_size = int(args['<window_size>'])
    is_equal_sized = args['--equal_sized_corpora']
    is_build_frequency_file = args['--build_frequency_file']
    tagset = ['N', 'V', 'AD']
    args['tagset'] = tagset   
    join_sign = ':'

    
    logging.basicConfig(filename= outDir + 'log_' + time_boundaries_str + '.txt', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info(args)

    # build frequency file if wished
    if is_build_frequency_file:
        build_frequency_file(dtatcfdir, freq_file, MIN_FREQ, join_sign)     
    
    # Load the frequent words file
    with codecs.open(freq_file, "r", "utf-8") as f_in:
        freq_words = set([line.strip() for line in f_in])

    # set upper bound to infinity as default in case corpus size doesn't matter
    corpus_size_upper_bound = float('inf')
        
    # if corpus size matters, get smallest interval size as upper bound for all
    if is_equal_sized:
        interval_size_dict = get_interval_sizes(dtatcfdir, time_boundaries, freq_words, tagset, join_sign)
        smallest_interval_size = get_smallest_interval_size(interval_size_dict)
        corpus_size_upper_bound = smallest_interval_size
            
    # build every interval seperately
    for t in range (1,len(time_boundaries)):

        interval_start = int(time_boundaries[t-1])
        interval_end = int(time_boundaries[t])
        
        print "Building corpus from %d to %d..."  % (interval_start, interval_end)

        cooc_mat = defaultdict(lambda: defaultdict(lambda: 0))
        
        corpus_size = 0       
        try:
            for root, dirs, files in os.walk(dtatcfdir):
                n = 0
                for filename in files:
                    n += 1
                    if ".tcf" in filename: 
                        dta_file = os.path.join(root, filename)
                        print 'Reading dta file %d/%d...' % (n, len(files))
                        tree = ElementTree()
                        tree.parse(dta_file)
                        
                        # just consider files from the time interval
                        file_boundary = get_file_boundary(time_boundaries, tree)
                        if not file_boundary == interval_end or file_boundary == 0:
                            continue
    
                        for sentence in make_sentences(tree):
                            
                            try:
                                # cuts rare and function (words with non-desired POS-tag) words
                                sentence = clean_sentence(sentence, freq_words, tagset, join_sign)
                            except TypeError:
                                print 'Sentence Cleaning failed.'
                                continue
                                
    
                            token_id = 0
                            for (word, lemma, pos, index) in sentence:
    
                                # check whether upper bound is reached
                                if corpus_size >= corpus_size_upper_bound:
                                    raise BorderReached                
                                # count the context in window size
                                try:
                                    for i in range(1,window_size+1):
                                        counter(cooc_mat, sentence, token_id, lemma, pos, i, join_sign)                    
                                except IndexError: 
                                    pass
                                
                                try:
                                    for i in range(1,window_size+1):
                                        counter(cooc_mat, sentence, token_id, lemma, pos, -i, join_sign)                    
                                except IndexError: 
                                    pass  
                                
                                corpus_size += 1
                                token_id += 1
                                
        except BorderReached:
            pass
                                
        print "Interval %d-%d has size %d" % (interval_start, interval_end, corpus_size)
        logging.info("Interval %d-%d has size %d" % (interval_start, interval_end, corpus_size))
        # Print in sparse matrix format
        build_sm(cooc_mat, outDir, interval_start, interval_end)

    
if __name__ == '__main__':
    main()
