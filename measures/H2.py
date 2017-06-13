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
from os.path import basename
from docopt import docopt

from common import *
from slqs_module import *
from slqs_diac_module import *
        
            
def main():
    """
    SLQS_Sub (H2) as described in:
    
        Shwartz, Vered; Santus, Enrico; Schlechtweg, Dominik. 2016. Hypernymy under Siege: 
            Linguistically-motivated Artillery for Hypernymy Detection. In Proceedings of EACL 2017.
    
    The script was adapted to take two, possibly different, matrices as input.
    """

    # Get the arguments
    args = docopt("""Compute SLQS_Sub (H2) for a list of (x, y) pairs for two diachronic vector spaces and save their scores.
                        Each word's entropy corresponds to the median of the entropies of its most-associated columns.


    Usage:
        H2.py [-x] (-p | -l) (-f | -w) (-a | -m) <testset_file> <model1> <model2> <N> <context_entropy_file1> <context_entropy_file2> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs corresponding to developments with their
                           starting points and the type of the development
        <model1> = the pkl file for the earlier vector space
        <model2> = the pkl file for the later vector space
        <N> = for a target word the entropy of the N most associated-contexts will be computed
        <output_file> = where to save the results
        <context_entropy_file1> = where to find and/or save the context_entropies for <model1>, will be overwritten
        <context_entropy_file2> = where to find and/or save the context_entropies for <model2>, will be overwritten

    Options:
        -x, --minmax  normalize values to a value between 0 and 1 via minmax
        -p, --ppmi  weight matrice with Ppmi
        -l, --plmi  weight matrice with Plmi
        -f, --freq  calculate context entropies from frequency matrice
        -w, --weighted  calculate context entropies from weighted matrice (with Ppmi, Plmi)
        -a, --average  calculate average of context entropies for target entropy
        -m, --median   calculate median of context entropies for target entropy
        
    """)
    
    matrice1_pkl = args['<model1>']
    matrice2_pkl = args['<model2>']
    testset_file = args['<testset_file>']
    N = int(args['<N>'])
    output_file = args['<output_file>']
    context_entropy_file1 = args['<context_entropy_file1>']
    context_entropy_file2 = args['<context_entropy_file2>']
    is_freq = args['--freq']
    is_weighted = args['--weighted']    
    is_minmax = args['--minmax']
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_average = args['--average']
    is_median = args['--median']
    is_save_weighted = False

# Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)
    

# Process matrice1
   
    matrice1_name = os.path.splitext(basename(matrice1_pkl))[0]
    matrice1_folder = os.path.dirname(matrice1_pkl) + "/"

    # Receive a .pkl file
    cooc_space1, mi_space1, vocab_map1, vocab_size1, column_map1, id2column_map1 = get_space(matrice1_folder, matrice1_name, is_pmi, is_lmi, is_save_weighted)
    
    # Get most associated columns for all targets
    most_associated_cols_dict1, union_m_a_c1 = get_all_most_assoc_cols(mi_space1, targets, vocab_map1, N)

    # Assign context entropy file
    c_entrop_file1 = context_entropy_file1

    # Get context entropies
    c_entropies_dict1, c_entr_ranked1 = get_c_entropies(vocab_map1, cooc_space1, mi_space1, N, c_entrop_file1, vocab_map1, id2column_map1, most_associated_cols_dict1, union_m_a_c1, is_freq, is_weighted)

    c_entropies1 = c_entropies_dict1

    if is_minmax:
        # Scale values to a value between 0 and 1
        c_entropies1 = Min_max_scaling().scale(c_entropies1)

    args['most_associated_cols_dict1'] = most_associated_cols_dict1
    args['id2column_map1'] = id2column_map1
    args['vocab_map1'] = vocab_map1
    args['c_entropies1'] = c_entropies1
     


# Process matrice2    
        
    matrice2_name = os.path.splitext(basename(matrice2_pkl))[0]
    matrice2_folder = os.path.dirname(matrice2_pkl) + "/"

    # Receive a .pkl file
    cooc_space2, mi_space2, vocab_map2, vocab_size2, column_map2, id2column_map2 = get_space(matrice2_folder, matrice2_name, is_pmi, is_lmi, is_save_weighted)
    
    # Get most associated columns for all targets
    most_associated_cols_dict2, union_m_a_c2 = get_all_most_assoc_cols(mi_space2, targets, vocab_map2, N)

    # Assign context entropy file
    c_entrop_file2 = context_entropy_file2

    # Get context entropies
    c_entropies_dict2, c_entr_ranked2 = get_c_entropies(vocab_map2, cooc_space2, mi_space2, N, c_entrop_file2, vocab_map2, id2column_map2, most_associated_cols_dict2, union_m_a_c2, is_freq, is_weighted)

    c_entropies2 = c_entropies_dict2

    if is_minmax:
        # Scale values to a value between 0 and 1
        c_entropies2 = Min_max_scaling().scale(c_entropies2)
    
    args['most_associated_cols_dict2'] = most_associated_cols_dict2
    args['id2column_map2'] = id2column_map2
    args['vocab_map2'] = vocab_map2
    args['c_entropies2'] = c_entropies2



# Compute target entropies and finally SLQS

    # Make relative target entropies
    unscored_output = make_relative_target_entropies_diac(args, test_set, N, is_average, is_median)
    
    # Compute target SLQS for test tuples
    scored_output = score_slqs_sub(unscored_output)

    # Save results
    save_results(scored_output, output_file)
 

if __name__ == '__main__':
    main()