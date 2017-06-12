"""       
This module is a collection of methods useful for extraction of co-occurrence matrices from 
the tcf-version of Deutsches Textarchiv (http://www.deutschestextarchiv.de/download) as 
described in:
    
    Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
        2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
        CoNLL 2017. Vancouver, Canada.
"""

import sys
sys.path.append('./modules/')

import numpy as np

import os
from xml.etree.ElementTree import ElementTree
from collections import defaultdict, Counter
import logging

from warnings import warn
from scipy.sparse import csr_matrix

from composes import *
from composes.matrix.sparse_matrix import SparseMatrix
                         

def make_sentences(tree):
    """
    Makes sentence stream from dta file xml tree    
    :param tree: the xml tree
    :return: sentence list with each word as (word, lemma, pos, index) (yield)
    """
 
    tokens = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}tokens")
    lemmas = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}lemmas")
    POStags = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}POStags")
    sentences = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}sentences")
    
    end_ids = []
    
    for sentence in sentences:
        token_ids = sentence.attrib["tokenIDs"].split()
        end_word_id = token_ids[len(token_ids)-1]
        end_ids.append(end_word_id)

   
    s = []
    
    for i, token in enumerate(tokens):
        try:
            token_id = token.attrib["ID"]
    
            word = token.text
            lemma = lemmas[i].text
            pos = POStags[i].text
            index = i
            
            s.append((word, lemma, pos, index))
                
            if token_id in end_ids:
                yield s
                s = []
        except:
            pass
            print "token skipped..."

         
            
def clean_sentence(sentence, freq_words, tagset, join_sign):
    """
    Cleans sentence from rare words and words lacking a specific POS  
    :param sentence: list of tuples with form (word, lemma, pos, index)
    :param freq_words: list of words with sufficient frequency
    :param tagset: list of POS-tags to include
    :param join_sign: sign to join lemma + first char of POS
    :return: cleaned sentence
    """
    
    sentence_cleaned = [] 
    
    for (word, lemma, pos, index) in sentence:
        target = lemma + join_sign + pos[0] # lemma + first char of POS
        if not target in freq_words or not True in [pos.startswith(x) for x in tagset]:
            continue
        else:
            sentence_cleaned.append((word, lemma, pos, index))
    
    return sentence_cleaned


def counter(cooc_mat, sentence, token_id, lemma, pos, i, join_sign):
    """
    Increases co-occurrence count for target (lemma + POS)
    :param cooc_mat: co-occurrence matrix
    :param sentence: list of tuples with form (word, lemma, pos, index)
    :param token_id: position of target in sentence
    :param lemma: target lemma
    :param pos: target POS
    :param i: position of context word in sentence
    :param join_sign: sign to join lemma + first char of POS
    """
    
    index = i+token_id    
    
    # avoid negative indexes
    if index < 0:
        raise IndexError    
    
    # count c_lemma+c_pos at position i as context of lemma+pos
    _, c_lemma, c_pos, _ = sentence[index]
    
    target_lemma = lemma + join_sign + pos[0] # lemma + first char of POS

    context = c_lemma + join_sign + c_pos[0] # lemma + first char of POS
    cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1
    
    
def counter_exemplar(cooc_mat, sentence, token_id, lemma, pos, i, identifier, join_sign):
    """
    Increases co-occurrence count for target occurrence (lemma + POS)
    :param cooc_mat: co-occurrence matrix
    :param sentence: list of tuples with form (word, lemma, pos, index)
    :param token_id: position of target in sentence
    :param lemma: target lemma
    :param pos: target POS
    :param i: position of context word in sentence
    :param identifier: unique identifier for each occurrence
    :param join_sign: sign to join lemma + first char of POS
    """

    index = i+token_id
    
    # avoid negative indexes
    if index < 0:
        raise IndexError 
                    
    # count c_lemma+c_pos at position i as context of lemma+pos
    _, c_lemma, c_pos, _ = sentence[index]
        
    target_lemma = lemma + join_sign + pos[0] + join_sign + identifier # lemma + first char of POS

    context = c_lemma + join_sign + c_pos[0] # lemma + first char of POS
    cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1
        

def build_frequency_file(dtatcfdir, freq_file, MIN_FREQ, join_sign):
    """
    Builds file with all lemma + POS pairs above certain frequency threshold. 
    :param dtatcfdir: path to directory with dta tcf files
    :param freq_file: path to frequency file
    :param MIN_FREQ: frequency threshold
    :param join_sign: sign to join lemma + first char of POS
    """
    
    # build frequency file from lemmas
    outputpath = freq_file
    print 'Building frequency file to ' + outputpath + "..."
    lemma_count = Counter(build_lemma_list(dtatcfdir, join_sign))
    frequent_lemmas = filter(lambda x: lemma_count[x] >= MIN_FREQ, lemma_count)
    with open(outputpath, 'w') as f_out:
        for lemma in frequent_lemmas:
            print >> f_out, lemma.encode('utf-8')
            
            
def get_interval_sizes(dtatcfdir, time_boundaries, freq_words, tagset, join_sign):
    """
    Get token frequency in specific time intervals of dta. 
    :param dtatcfdir: path to directory with dta tcf files
    :param time_boundaries: list of time boundaries seperating intervals
    :param freq_words: list of words with sufficient frequency
    :param tagset: list of POS-tags to include
    :param join_sign: sign to join lemma + first char of POS
    :return: dictionary, mapping file boundaries to token frequencies
    """
    
    interval_size_dict = defaultdict(int)  
        
    print "Getting interval sizes..."
        
    # get the corpus size of each time interval speciified in the input
    for root, dirs, files in os.walk(dtatcfdir):
        for filename in files:
            if ".tcf" in filename: 
                dta_file = os.path.join(root, filename)
                file_boundary = get_file_boundary(time_boundaries, tree)
                if file_boundary == 0:
                    continue
            
                tree = ElementTree()
                tree.parse(dta_file)
                
                lemmas = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}lemmas")
                POStags = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}POStags")
                
                for i, lemma_elem in enumerate(lemmas):
                    try:
                        lemma = lemma_elem.text 
                        pos = POStags[i].text
                        target = lemma + join_sign + pos[0] # lemma + first char of POS, e.g. run-v / run-n
                        
                        # only count lemmas which are frequent enough and have the respective POS-tags
                        if not target in freq_words or not (pos.startswith(x) for x in tagset):
                                continue
                        interval_size_dict[file_boundary] += 1
                    except:
                        pass
                        print "token skipped..."

    print interval_size_dict
    logging.info(interval_size_dict)
        
    return interval_size_dict
    

def get_smallest_interval_size(interval_size_dict):
    """
    Get token frequency in specific time intervals of dta. 
    :param interval_size_dict: dictionary, mapping time intervals to token frequencies
    :return: integer, the smallest token frequency
    """
    
    smallest_interval = sorted(interval_size_dict, key=lambda x: interval_size_dict[x])[0]
    smallest_interval_size = interval_size_dict[smallest_interval]
    return smallest_interval_size


def build_lemma_list(dtatcfdir, join_sign):
    """
    Build list of all lemma + POS pairs. 
    :param dtatcfdir: path to directory with dta tcf files
    :param join_sign: sign to join lemma + first char of POS
    :return: list of strings, the lemma + POS pairs
    """
    
    lemma_list = []

    print "Building lemma list..."    
    
    # build a list of lemmas over all files for building the frequency file
    for root, dirs, files in os.walk(dtatcfdir):
        for filename in files:
            if ".tcf" in filename:
                print filename
                dta_file = os.path.join(root, filename)
                
                tree = ElementTree()
                tree.parse(dta_file)
                lemmas = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}lemmas")
                POStags = tree.find("{http://www.dspin.de/data/textcorpus}TextCorpus/{http://www.dspin.de/data/textcorpus}POStags")

                for i, lemma_elem in enumerate(lemmas):
                    try:
                        lemma = lemma_elem.text 
                        pos = POStags[i].text
                        target = lemma + join_sign + pos[0] # lemma + first char of POS
                        lemma_list.append(target)
                    except:
                        print "token skipped..."        
                        pass
    
    return lemma_list
    
    
def get_file_date(tree):
    """
    Get publication date from dta file xml tree. 
    :param tree: the xml tree
    :return: int, the publication date
    """
    
    date = tree.find("{http://www.dspin.de/data/metadata}MetaData/{http://www.dspin.de/data/metadata}source/{http://www.clarin.eu/cmd/}CMD/{http://www.clarin.eu/cmd/}Components/{http://www.clarin.eu/cmd/}teiHeader/{http://www.clarin.eu/cmd/}fileDesc/{http://www.clarin.eu/cmd/}sourceDesc/{http://www.clarin.eu/cmd/}biblFull/{http://www.clarin.eu/cmd/}publicationStmt/{http://www.clarin.eu/cmd/}date").text    
    return date

    
def get_file_boundary(time_boundaries, tree):
    """
    Get nearest later time boundary for dta file. 
    :param time_boundaries: list of time boundaries seperating intervals
    :param tree: the dta file xml tree
    :return: int, the time boundary
    """
    
    # get the time interval a file falls in    
    date = tree.find("{http://www.dspin.de/data/metadata}MetaData/{http://www.dspin.de/data/metadata}source/{http://www.clarin.eu/cmd/}CMD/{http://www.clarin.eu/cmd/}Components/{http://www.clarin.eu/cmd/}teiHeader/{http://www.clarin.eu/cmd/}fileDesc/{http://www.clarin.eu/cmd/}sourceDesc/{http://www.clarin.eu/cmd/}biblFull/{http://www.clarin.eu/cmd/}publicationStmt/{http://www.clarin.eu/cmd/}date").text    
    boundary_number = len(time_boundaries)
    if date < time_boundaries[0] or date >= time_boundaries[boundary_number-1]:
        return 0
    else:
        for i in range(1, boundary_number):
            if date < time_boundaries[i]:
                return int(time_boundaries[i])


def build_sm(cooc_mat, outDir, interval_start, interval_end):
    """
    Export co-occurrence matrix in DISSECT's sparse matrix format.
    :param cooc_mat: co-occurrence matrix
    :param outDir: path to output file
    :param interval_start: start of matrix's time interval
    :param interval_end: end of matrix's time interval
    """

    cols = []    
    
    with open(outDir+"matrix-dta-"+str(interval_start)+"-"+str(interval_end)+".sm", 'w') as f_out:
        
        file = open(outDir+"matrix-dta-"+str(interval_start)+"-"+str(interval_end)+".rows", "w")
        
        for target, contexts in cooc_mat.iteritems():
            
            file.write((target + "\n").encode('utf-8'))
        
            for context, freq in contexts.iteritems():
                print >> f_out, ' '.join((target, context, str(freq))).encode('utf-8')
                cols.append(context)
            
            cols = list(set(cols))        
        
        f_out.close()
        file.close()
            
    file1 = open(outDir+"matrix-dta-"+str(interval_start)+"-"+str(interval_end)+".cols", "w")

    for context in cols:
                
        file1.write((context + "\n").encode('utf-8'))

    file1.close()    


def extract_indexing_structs_mod(input_list, field_list):
    """
    Extract index map for list of inputs.
    :param input_list: list of inputs
    :param field_list: list of field indexes
    :return: tuple, lists of maps from ids to strings and backwards
    """
    str2id = {}
    id2str = []
    no_fields = len(field_list)

    str2id_list = [str2id.copy() for i in xrange(no_fields)]
    id2str_list = [list(id2str) for i in xrange(no_fields)]
    index_list = [0 for i in xrange(no_fields)]
    max_field = max(field_list)

    for line in input_list:
        if line.strip() != "":
            elements = line.strip().split()
            if len(elements) <= max_field:
                warn("Invalid input line:%s. Skipping it" % line.strip())
            else:
                for field_idx, field in enumerate(field_list):
                    current_str = elements[field]
                    if not current_str in str2id_list[field_idx]:
                        str2id_list[field_idx][current_str] = index_list[field_idx]
                        id2str_list[field_idx].append(current_str)
                        index_list[field_idx] += 1

    for id2str in id2str_list:
        if not id2str:
            raise ValueError("Found no valid data in input!")
    return (id2str_list, str2id_list)


def read_sparse_space_data_mod(input_list, row2id, column2id, dtype=np.double):
    """
    Transform matrix in tuple structure to DISSECT's sparse matrix format
    :param input_list: list of inputs
    :param row2id: dictionary, mapping rows to ids
    :param column2id: dictionary, mapping columns to ids
    :param dtype: data type of cell values
    :return: sparse matrix
    """
    
    f = input_list

    no_lines = sum(1 for line in f if line != ())

    row = np.zeros(no_lines, dtype=np.int32)
    col = np.zeros(no_lines, dtype=np.int32)

    data = np.zeros(no_lines, dtype=dtype)

    i = 0
    for line in f:
         if line != ():
            line_elements = line
            if len(line_elements) >= 3:
                [word1, word2, count] = line_elements[0:3]
                if word1 in row2id and word2 in column2id:
                    row[i] = row2id[word1]
                    col[i] = column2id[word2]
                    data[i] = dtype(count)
                    i += 1
                    if i % 1000000 == 0:
                        print "Progress...%d" % i
            else:
                raise ValueError("Invalid row: %s, expected at least %d fields"
                                 % (line.strip(), 3))

    # eliminate the extra zeros created when word1 or word2 is not row2id or col2id!!
    data = data[0:i]
    row = row[0:i]
    col = col[0:i]

    m = SparseMatrix(csr_matrix((data, (row, col)), shape=(len(row2id), len(column2id))))
    if m.mat.nnz != i:
        warn("Found 0-counts or duplicate row,column pairs. (Duplicate entries are summed up.)")

    return m