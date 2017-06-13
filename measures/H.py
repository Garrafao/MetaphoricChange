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
    SLQS_Row_Sub (H) as described in:
    
        Shwartz, Vered; Santus, Enrico; Schlechtweg, Dominik. 2016. Hypernymy under Siege: 
            Linguistically-motivated Artillery for Hypernymy Detection. In Proceedings of EACL 2017.
    
    The script was adapted to take two, possibly different, matrices as input.
    """

    # Get the arguments
    args = docopt("""Compute SLQS_Sub (H) for a list of (x, y) pairs for two vector spaces 
                        and save their scores. Each word's entropy corresponds to the entropy 
                        of its row.

    Usage:
        H.py [-x] (-f | -p | -l) <testset_file> <model1> <model2> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs corresponding to developments with their
                           starting points and the type of the development
        <model1> = the pkl file for the earlier vector space
        <model2> = the pkl file for the later vector space
        <output_file> = where to save the results
        
    Options:
        -f, --freq  calculate row entropies from frequency matrice
        -p, --ppmi  calculate row entropies from PPMI weighted matrice
        -l, --plmi  calculate row entropies from PLMI weighted matrice
        -x, --minmax  scale values to a value between 0 and 1 via minmax

    """)
    
    matrice1_pkl = args['<model1>']
    matrice2_pkl = args['<model2>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_minmax = args['--minmax']
    is_save_weighted = False

    
# Set target output
    target_output = output_file  
    
# Load the term-pairs
    targets, test_set = load_test_pairs(testset_file)


# Process matrice1   
   
    matrice1_name = os.path.splitext(basename(matrice1_pkl))[0]
    matrice1_folder = os.path.dirname(matrice1_pkl) + "/"

    # Receive a .pkl file
    cooc_space1, mi_space1, vocab_map1, vocab_size1, column_map1, id2column_map1 = get_space(matrice1_folder, matrice1_name, is_pmi, is_lmi, is_save_weighted)
    
    # Get all row entropies    
    r_entropies_dict1, r_entr_ranked1 = get_r_entropies(vocab_map1, cooc_space1, mi_space1, target_output, is_freq)

    r_entropies1 = r_entropies_dict1
               
    if is_minmax:
        # Scale values to a value between 0 and 1 
        r_entropies1 = Min_max_scaling().scale(r_entropies1)
     
    # Get dict of only targets
    r_entropies1 = prune_dict(r_entropies1, targets)


# Process matrice2    
        
    matrice2_name = os.path.splitext(basename(matrice2_pkl))[0]
    matrice2_folder = os.path.dirname(matrice2_pkl) + "/"

    # Receive a .pkl file
    cooc_space2, mi_space2, vocab_map2, vocab_size2, column_map2, id2column_map2 = get_space(matrice2_folder, matrice2_name, is_pmi, is_lmi, is_save_weighted)
    
    # Get all row entropies    
    r_entropies_dict2, r_entr_ranked2 = get_r_entropies(vocab_map2, cooc_space2, mi_space2, target_output, is_freq)

    r_entropies2 = r_entropies_dict2
               
    if is_minmax:
        # Scale values to a value between 0 and 1 
        r_entropies2 = Min_max_scaling().scale(r_entropies2)
     
    # Get dict of only targets
    r_entropies2 = prune_dict(r_entropies2, targets)
    

# Compute SLQS   

    # Make unscored output
    unscored_output = make_unscored_output_diac(r_entropies1, r_entropies2, test_set, vocab_map1, vocab_map2)
    
    # Compute target SLQS for test tuples
    scored_output = score_slqs_sub(unscored_output)

    # Save results
    save_results(scored_output, output_file)
    

if __name__ == '__main__':
    main()