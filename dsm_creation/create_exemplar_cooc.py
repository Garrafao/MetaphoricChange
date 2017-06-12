import sys
sys.path.append('./modules/')

import os
import codecs
from xml.etree.ElementTree import ElementTree
from docopt import docopt
from collections import defaultdict
import logging

from dsm_module import *


def main():
    """
    Create exemplar co-occurence matrix from the tcf-version of Deutsches Textarchiv (http://www.deutschestextarchiv.de/download) 
    as described in:
    
        Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
            2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
            CoNLL 2017. Vancouver, Canada.
    """

    # Get the arguments
    args = docopt("""Create exemplar co-occurence matrix where each row corresponds to one occurrence 
                        of a word. Each line will have the form 'lemma1:POS:year:identifier lemma2:POS freq'.


    Usage:
        create_exemplar_cooc.py <testset_file> <dta-tcf_dir> <outDir> <frequency_file> <window_size>
        
    Arguments:
        <testset_file> = a file containing term-pairs corresponding to developments with their
                       starting points and the type of the development
        <dta-tcf_dir> = the DTA-tcf directory
        <outDir> = the directory of the sparse matrix output files
        <frequency_file> = the file containing frequent lemmas (>= MIN_FREQ)
        <window_size> = the linear distance of context words to consider

    """)
    
    testset_file = args['<testset_file>']
    dtatcfdir = args['<dta-tcf_dir>']
    outDir = args['<outDir>']      
    freq_file = args['<frequency_file>']
    window_size = int(args['<window_size>'])
    tagset = ['N', 'V', 'AD']
    args['tagset'] = tagset
    join_sign = ':'


    # Load the term-pairs
    with codecs.open(testset_file, "r", "utf-8") as f_in:
        targets = set([tuple(line.strip().split('\t'))[0] for line in f_in])
    
    # Load the frequent words file
    with codecs.open(freq_file, "r", "utf-8") as f_in:
        freq_words = set([line.strip() for line in f_in])

    
    logging.basicConfig(filename= outDir + 'log_' + 'create_exemplar_cooc' + '.txt', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info(args)
    
    print outDir + 'log_' + 'create_exemplar_cooc' + '.txt'
    
    print "Building matrice..."
    
    cooc_mat = defaultdict(lambda: defaultdict(lambda: 0))
     
    corpus_size = 0
    for root, dirs, files in os.walk(dtatcfdir):
        n = 0
        for filename in files:
            n += 1
            if ".tcf" in filename: 
                dta_file = os.path.join(root, filename)
                print 'Reading dta file %d/%d...' % (n, len(files))
                tree = ElementTree()
                tree.parse(dta_file)
                
                # get publication date of file
                file_date = get_file_date(tree)

                for sentence in make_sentences(tree):
                    
                    try:
                        # cuts rare and function (words with non-desired POS-tag) words
                        sentence = clean_sentence(sentence, freq_words, tagset, join_sign)
                    except TypeError:
                        print 'Sentence Cleaning failed'
                        continue
                        
                    token_id = 0
                    for (word, lemma, pos, index) in sentence:
                                                                 
                        # check if current word is target
                        target = lemma + join_sign + pos[0] # lemma + first char of POS

                        if not target in targets:
                            corpus_size += 1
                            token_id += 1
                            continue
                        
                        identifier = str(file_date) + join_sign + str(corpus_size)

                        # count the context in window size
                        try:
                            for i in range(1,window_size+1):
                                counter_exemplar(cooc_mat, sentence, token_id, lemma, pos, i, identifier, join_sign)                    
                        except IndexError: 
                            pass
                        
                        try:
                            for i in range(1,window_size+1):
                                counter_exemplar(cooc_mat, sentence, token_id, lemma, pos, -i, identifier, join_sign)                    
                        except IndexError: 
                            pass  
                        
                        corpus_size += 1
                        token_id += 1
               
        print "Corpus has size %d" % corpus_size
        logging.info("Corpus has size %d" % corpus_size)
        # Print in sparse matrix format
        build_sm(cooc_mat, outDir + "exemplar_", 1485, 1926)


if __name__ == '__main__':
    main()
