"""
Part of the code in this file is based on the code developed in

    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
        
--------------        
This module is a collection of classes and methods useful for calculating, saving and
evaluating information-theoretic measures such as entropy (SLQS and variations) for diachronic
purposes as described in:
    
    Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
        2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
        CoNLL 2017. Vancouver, Canada.
"""

import sys
sys.path.append('../')

from statistics import median, mean

from common import *


def make_relative_target_entropies_diac(args, test_set, N, is_average, is_median):
    
    most_associated_cols_dict1 = args['most_associated_cols_dict1']
    id2column_map1 = args['id2column_map1']
    vocab_map1 = args['vocab_map1']
    c_entropies1 = args['c_entropies1']
    most_associated_cols_dict2 = args['most_associated_cols_dict2']
    id2column_map2 = args['id2column_map2']
    vocab_map2 = args['vocab_map2']
    c_entropies2 = args['c_entropies2']    
    
    """
    Get relative entropy values for x and y for each test pair (x,y) of each test item.
    :param args: dictionary, mapping to most-associated columns map, id2column map, 
                    vocab map and context entropies
    :param most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :param vocab_map: dictionary that maps row strings to integer ids
    :param id2column_map: list of strings, the column elements
    :param c_entropies: dictionary, mapping contexts to entropy values
    :param test_set: list of tuples, each tuple a test item
    :param N: int, number of columns to extract
    :param is_average: boolean, whether to calculate average of context entropies
    :param is_median: boolean, whether to calculate median of context entropies
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y)
    """
    
    unscored_output = []   
    for (x, y, label, relation) in test_set:
        if x not in vocab_map1 or y not in vocab_map2:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print x, y
        
        # Get smaller number M of associated columns
        M = min([len(most_associated_cols_dict1[x]), len(most_associated_cols_dict2[y])])
        
        if M == 0:
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
                 
                 
        # Compute Generality Index for x
        
        target_entropies1 = {}

        m_most_assoc_cs1 = {}

        # M Most associated contexts of x
        m_most_assoc_cs1[x] = [id2column_map1[mapping[0]] for mapping in most_associated_cols_dict1[x][:M]]
                                 
        entr_of_m_most_assoc_cs1 = {}    

        # Get the M entropies of the most associated contexts of x
        entr_of_m_most_assoc_cs1[x] = [float(c_entropies1[context]) for context in m_most_assoc_cs1[x]]
        
        # Compute the average or median of the entropies of the M most associated contexts (target entropy)
        if is_average:
            target_entropies1[x] = float(mean(entr_of_m_most_assoc_cs1[x]))
        elif is_median:
            target_entropies1[x] = float(median(entr_of_m_most_assoc_cs1[x]))
            
        print float(median(entr_of_m_most_assoc_cs1[x]))
        
        
        # Compute Generality Index for y
        
        target_entropies2 = {}

        m_most_assoc_cs2 = {}

        # M Most associated contexts of y
        m_most_assoc_cs2[y] = [id2column_map2[mapping[0]] for mapping in most_associated_cols_dict2[y][:M]]
                                 
        entr_of_m_most_assoc_cs2 = {}    

        # Get the M entropies of the most associated contexts of y
        entr_of_m_most_assoc_cs2[y] = [float(c_entropies2[context]) for context in m_most_assoc_cs2[y]]
        
        # Compute the average or median of the entropies of the M most associated contexts (target entropy)
        if is_average:
            target_entropies2[y] = float(mean(entr_of_m_most_assoc_cs2[y]))
        elif is_median:
            target_entropies2[y] = float(median(entr_of_m_most_assoc_cs2[y]))
            
        print float(median(entr_of_m_most_assoc_cs2[y]))
                        
        unscored_output.append((x, y, label, relation, target_entropies1[x], target_entropies2[y]))
    
    return unscored_output
    

def make_unscored_output_diac(x_entropies, y_entropies, test_set, vocab_map1, vocab_map2):
    """
    Append test items and their entropy values in two vector spaces.
    :param x_entropies: dictionary, mapping targets to entropy values
    :param y_entropies: dictionary, mapping targets to entropy values
    :param test_set: list of tuples, each tuple a test item
    :param vocab_map1: dictionary that maps row strings to integer ids
    :param vocab_map2: dictionary that maps row strings to integer ids
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y)
    """
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map1 or y not in vocab_map2:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print (x, y, label, relation, x_entropies[x], y_entropies[y])
    
        unscored_output.append((x, y, label, relation, x_entropies[x], y_entropies[y]))
    
    return unscored_output