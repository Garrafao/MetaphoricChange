"""
Part of the code in this file is based on the code developed in

    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
"""

import sys
sys.path.append('./modules/')

import os
import random
from os.path import basename
from docopt import docopt
from xpermutations import xuniqueCombinations
from composes.semantic_space.space import Space 
from collections import defaultdict
from statistics import mean, StatisticsError

from common import *
from slqs_module import *
from slqs_diac_module import *
from dsm_module import *

        
def main():
    """
    Compute row entropy (H_MON) from exemplar matrices as described in:
    
        Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
            2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
            CoNLL 2017. Vancouver, Canada.
    """

    # Get the arguments
    args = docopt("""Compute row entropy (H_MON) from exemplar matrices.

    Usage:
        H_MON.py <testset_file> <exemplar_matrix> <time_boundary_1-time_boundary_2> <window_size> <exemplar_number> <vector_number> <output_dir>
        
    Arguments:
        <testset_file> = a file containing term-pairs corresponding to developments with their
                           starting points and the type of the development
        <exemplar_matrix> = the pkl file of the source exemplar matrix
        <time_boundary_1-time_boundary_2> = the time boundaries to slice up the corpus
        <window_size> = size of sliding time interval
        <exemplar_number> = number of exemplars to construct one target vector
        <vector_number> = number of target vectors to average over
        <output_dir> = where to save the results
        
    """)
    
    matrice1_pkl = args['<exemplar_matrix>']
    testset_file = args['<testset_file>']
    output_dir = args['<output_dir>']
    time_boundaries_str = args['<time_boundary_1-time_boundary_2>']
    global_lower_bound, global_upper_bound = int(time_boundaries_str.split("-")[0]), int(time_boundaries_str.split("-")[1])
    window_size = int(args['<window_size>'])
    N = int(args['<exemplar_number>'])
    V = int(args['<vector_number>'])
    join_sign = ':'
    non_values = [-999.0, -888.0]


    # Generate cut-points
    cut_points = [i for i in range(global_lower_bound,global_upper_bound)]
    
    # Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)


    # Process matrice   
   
    matrice1_name = os.path.splitext(basename(matrice1_pkl))[0]
    matrice1_folder = os.path.dirname(matrice1_pkl) + "/"

    # Receive a .pkl file
    exemplar_space, _, vocab_map1, vocab_size1, column_map1, id2column_map1 = get_space(matrice1_folder, matrice1_name, False, False, False)

    for cut_point in cut_points:        
        
        print cut_point
        
        target_values = {}
        
        current_lower_bound, current_upper_bound = cut_point-window_size, cut_point+window_size
        
        for target in targets:
            
            print target
                        
            values = []
            
            exem_list = [(exem.split(join_sign)[0], exem.split(join_sign)[1], exem.split(join_sign)[2], exem.split(join_sign)[3]) for exem in vocab_map1 if exem.split(join_sign)[0] + join_sign + exem.split(join_sign)[1] == target]
            exem_dict = dict([((lemma, pos, int(date), int(identifier)), int(date)) for (lemma, pos, date, identifier) in exem_list])            
            
            # Get contexts in window
            window = [(lemma, pos, date, identifier) for (lemma, pos, date, identifier) in exem_dict if current_lower_bound <= date <= current_upper_bound]
            print 'Window size is: %d' % (len(window))
            random.shuffle(window)
            
            # Get combinations of exemplars of size N            
            exem_combos = xuniqueCombinations(window, N)            
            
            for i, combo in enumerate(exem_combos):
                
                if i >= V:
                    break
                
                print 'Calculating combination %d of %d...' % (i, V)
                
                # Initialize sparse matrix
                sparse_mat_dict = defaultdict(lambda: defaultdict(lambda: 0))                
                
                cols = []
                for (lemma, pos, date, identifier) in combo:
                         
                    exem_tar = join_sign.join((lemma, pos, str(date), str(identifier)))
                    
                    row = exemplar_space.get_row(exem_tar)
                                        
                    data = row.get_mat().data
                    indices = row.get_mat().indices
                      
                    for i, key in enumerate(data):
                        
                        cxt = id2column_map1[indices[i]]
                        cols.append(cxt)
                        sparse_mat_dict[target][cxt] = sparse_mat_dict[target][cxt] + key
        
                # Bring to sparse matrix format
                rows = set([target])
                cols = set(cols)
                sparse_mat_tup = [(key, context, sparse_mat_dict[key][context]) for key in sparse_mat_dict for context in sparse_mat_dict[key]]        

                [id2row], [row2id] = extract_indexing_structs_mod(rows, [0])
                [id2column], [column2id] = extract_indexing_structs_mod(cols, [0])
                sparse_mat = read_sparse_space_data_mod(sparse_mat_tup, row2id, column2id)
        
                # Create a space from co-occurrence counts in sparse format
                sparse_space = Space(sparse_mat, id2row, id2column, row2id, column2id)
                sparse_space.__class__ = Space_extension
                                
                # Get all row entropies    
                r_entropies_dict1, r_entr_ranked1 = get_r_entropies(row2id, sparse_space, None, output_dir, True)

                values.append(r_entropies_dict1[target])
                
            # Computing mean of values
            print 'Computing mean of values...'
            print 'Averaging over %d values' % (len(values))
            try:
                target_values[target] = mean(values)
            except StatisticsError:
                target_values[target] = non_values[1]

        # Make output
        unscored_output = []
        for (x, y, label, relation) in test_set:
            unscored_output.append((x, y, label, relation, target_values[x]))
            
        # Save results
        save_results(unscored_output, output_dir + 'row_entropy_exem_' + str(current_lower_bound) + '-' + str(current_upper_bound) + '_' + str(N) + 's' + '.txt' )
    

if __name__ == '__main__':
    main()