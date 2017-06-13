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
  
            
def main():
    """
    Modification of SLQS - as described in:
    
        Santus, Enrico; Lu, Qin; Lenci, Alessandro; Schulte im Walde, Sabine. 2014. Chasing Hypernyms in Vector Spaces with Entropy. 
            Proceedings of the 14th Conference of the European Chapter of the Association of Computational Linguistics. 38-42.
      """

    # Get the arguments
    args = docopt("""Compute first order entropies (H) for all rows of a vector space and save their scores.

    Usage:
        H_rank.py (-f | -p | -l) [-n] <model> <output_file> 
        
    Arguments:
        <model> = the pkl file for the vector space
        <output_file> = where to save the results: a tab separated file with x\ty\tscore, where the
                        score is SLQS (for y as the hypernym of x).
        
    Options:
        -f, --freq  calculate row entropies from frequency matrice
        -p, --ppmi  calculate row entropies from PPMI weighted matrice
        -l, --plmi  calculate row entropies from PLMI weighted matrice
        -n, --normalize  whether to normalize the entropies to avalue between 0 and 1 [default: False]
        
    """)
    
    matrice_pkl = args['<model>']
    output_file = args['<output_file>']
    is_freq = args['--freq']
    is_pmi = args['--ppmi']
    is_lmi = args['--plmi']
    is_norm = args['--normalize']
    is_save_weighted = False

        
    matrice_name = os.path.splitext(basename(matrice_pkl))[0]
    matrice_folder = os.path.dirname(matrice_pkl) + "/"
    

    # Receive a .pkl file
    cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map = get_space(matrice_folder, matrice_name, is_pmi, is_lmi, is_save_weighted)

    # Load the term-pairs
    targets = vocab_map

    # Get row entropies    
    r_entropies_dict, r_entr_ranked = get_r_entropies(targets, cooc_space, mi_space, output_file, is_freq)

    if is_norm:
        # Normalize the target entropy values to a value between 0 and 1 
        r_entropies = Min_max_scaling().scale(r_entropies_dict)
    else:
        # Do not normalize
        r_entropies = r_entropies_dict    

    # Save the row entropies      
    save_entropies(r_entr_ranked, r_entropies, output_file)
    

if __name__ == '__main__':
    main()