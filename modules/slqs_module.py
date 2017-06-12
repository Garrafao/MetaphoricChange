"""
Part of the code in this file is based on the code developed in

    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
        
--------------        
This module is a collection of classes and methods useful for calculating, saving and
evaluating information-theoretic measures such as entropy (SLQS and variations) as 
described in:
    
    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
"""

import sys
sys.path.append('./modules/')

import codecs
import math
import scipy.stats as sc
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
from statistics import median, mean

from common import *


class Normalization(object):
    """
    This class implements normalization methods.
    """
    
    def dict_normalize(self, raw_dict, norm_dict):
        """
        Normalizes every value of a dictionnary by division.
        :param raw_dict: the dictionary
        :param norm_dict: the dictionary with normalization factors
        :return: normalized dictionary
        """
        
        print "Normalizing values..."
        
        # Create the normalized dictionary
        normalized_dict = {key : value/norm_dict[key] for key, value in raw_dict.items()}

        return normalized_dict
        
       
class Min_max_scaling(object):
    """
    This class implements Min-Max-Scaling.
    """
    
    def scale(self, raw_dict):
        """
        Scales values of a dictionary using Min-Max-Scaling.
        :param raw_dict: the dictionary
        :return: scaled dictionary
        """

        def min_max(minimum, maximum, X):
            """
            Scales value using Min-Max-Scaling
            :param minimum: scale minimum
            :param maximum: scale maximum
            :param X: value
            :return: scaled value
            """
            X_norm = (X - minimum) / (maximum - minimum)
            return X_norm
        
        print "Scaling values..."
        
        # Get the minimum and maximum value
        min_key, min_value = min(raw_dict.iteritems(), key=lambda x:x[1])
        max_key, max_value = max(raw_dict.iteritems(), key=lambda x:x[1])

        # Create the scaled dictionary
        scaled_dict = {key : min_max(min_value, max_value, value) for key, value in raw_dict.items()}

        return scaled_dict 


class Space_extension(Space):
    """
    This class extends the Space class implementing semantic spaces.
    
    The extension mainly consists in methods useful for calculating
    information-theoretic measures such as entropy (SLQS and variations).
    """
    
    def get_vocab(self):
        """
        Gets the mapping of row strings to integer ids for a semantic space.
        :param self: a semantic space
        :return: dictionary that maps row strings to integer ids
        """
        
        vocab_map = self.get_row2id()
            
        return vocab_map
        
        
    def get_columns(self):
        """
        Gets the mapping of column strings to integer ids for a semantic space.
        :param self: a semantic space
        :return: dictionary that maps column strings to integer ids
        """
        
        column_map = self.get_column2id()
            
        return column_map
        
        
    def get_column_numbers(self, targets):
        """
        Gets the number of non-zero columns for specific rows.
        :param self: a semantic space
        :param targets: a list of target row strings
        :return: dictionary, mapping targets to their values
        """
        
        col_num_map = {}
        
        vocab_map = self.get_row2id()        
        targets_size = len(targets)
        
        # Iterate over rows
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get number of non-zero columns
            r_types = row.get_mat().getnnz()

            col_num_map[w] = r_types
        
        return col_num_map
        

    def get_token_numbers(self, targets):
        """
        Sums the values of specific rows.
        :param self: a semantic space
        :param targets: a list of target row strings
        :return: dictionary, mapping targets to sums
        """
        
        tok_num_map = {}        

        vocab_map = self.get_row2id()
        targets_size = len(targets)
        
        # Iterate over rows
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get sum of row
            r_total_freq = row.get_mat().sum()

            tok_num_map[w] = r_total_freq
        
        return tok_num_map

        
    def get_id2column_map(self):
        """
        Gets the column elements for a semantic space.
        :param self: a semantic space
        :return: list of strings, the column elements
        """
        
        id_map = self.get_id2column()
            
        return id_map
        

    def get_most_associated_cols(self, target, N):
        """
        Gets N columns with highest values for a row string (target).
        :param self: a semantic space
        :param target: target row string
        :param N: int, number of columns to extract
        :return: dictionary, mapping column ids to their values
        """
        
        row = self.get_row(target)
        
        # Data returns the non-zero elements in the row and indices returns the indices of the non-zero elements
        data = row.get_mat().data
        indices = row.get_mat().indices

        most_associated_cols_indices = data.argsort()[-N:]
        most_associated_cols = { indices[index] : data[index] for index in most_associated_cols_indices }

        return most_associated_cols
        

    def compute_row_entropies(self, targets):
        """
        Computes row entropy for a list of target row strings.
        :param self: a semantic space
        :param targets: list of strings, the targets
        :return: dictionary, mapping targets to entropy values
        """

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        r_entropies_dict = {}
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get all counts in row (non-zero elements)
            counts = row.get_mat().data

            # Get sum of row (total count of context)
            r_total_freq = row.get_mat().sum()

            # Compute entropy of row
            H = -sum([((count/r_total_freq) * math.log((count/r_total_freq),2)) for count in counts])

            r_entropies_dict[w] = H
        
        return r_entropies_dict


    def compute_row_kldivs(self, targets):

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        r_dict = {}
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get all counts in row (non-zero elements)
            counts = row.get_mat().data

            # Get sum of row
            r_total_freq = row.get_mat().sum()

            # Normalize row vector
            counts_norm = [(count/r_total_freq) for count in counts]    
            
            #Create a uniform pdf over the range [0,1]       
            uni = [1.0/len(counts) for count in counts]    
          
            #print uni
            #print counts_norm
            
            #x = np.linspace(0,1,len(counts_norm))
            #uni = sc.uniform.pdf(x) # Create a uniform pdf over the range [0,1]

            #custm = sc.rv_discrete(name='custm', values=(counts, uni))
            #custm = custm.pmf()

            div = sc.entropy(uni, counts_norm)
            
            #print sc.entropy([0.5, 0.5], [0.3, 0.7])
            #print sc.entropy([0.5, 0.5, 0.5], [0.2, 0.6, 0.2])

            #print w, div, len(counts)

            r_dict[w] = div
        
        return r_dict
        
        
    def compute_row_frequencies(self, targets):

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        r_frequencies_dict = {}
        
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get sum of row
            r_total_freq = row.get_mat().sum()

            r_frequencies_dict[w] = r_total_freq
        
        return r_frequencies_dict
        
        
    def compute_row_typtoks(self, targets):

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        r_typtoks_dict = {}
        
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)

            # Get sum of row
            r_total_freq = row.get_mat().sum()
            r_types = row.get_mat().getnnz()

            r_typtoks_dict[w] = r_types/r_total_freq
        
        return r_typtoks_dict
        

    def compute_context_entropies(self, union_m_a_c):
        """
        Computes entropies for a set of column strings.
        :param self: a semantic space
        :param union_m_a_c: set of column ids
        :return: dictionary, mapping column strings to entropy values
        """

        union_m_a_c_size = len(union_m_a_c) 
        id2row = self.id2row
        id2column = self.id2column

        # Transpose matrix to iterate over columns
        print "Transposing the matrix..."
        matrix_transposed = self.cooccurrence_matrix.transpose()
        # Instantiate a new space from transposed matrix
        space_transposed = Space_extension(matrix_transposed, id2column, id2row)

        
        c_entropies_dict = {} 
        # Iterate over columns (contexts) 
        print "Iterating over columns..."
        for j, column_id in enumerate(union_m_a_c):
            context = id2column[column_id]

            print "%d%% done..." % (j*100/union_m_a_c_size)            
            col = space_transposed.get_row(context)

            # Get all counts in column (non-zero elements)
            counts = col.get_mat().data

            # Get sum of column (total count of context)
            c_total_freq = col.get_mat().sum()

            # Compute entropy of context
            H = -sum([((count/c_total_freq) * math.log((count/c_total_freq),2)) for count in counts])

            c_entropies_dict[context] = H
        
        return c_entropies_dict


    def get_r_asoc_vecs(self, N, targets, most_associated_cols_dict):

        targets_size = len(targets)
        vocab_map = self.get_row2id()
        
        # Iterate over rows
        vecs_dict = {}
        
        print "Iterating over rows..."
        for j, w in enumerate(targets):
            
            if w not in vocab_map:
                continue

            print "%d%% done..." % (j*100/targets_size)            
            row = self.get_row(w)
            
            # Data returns the non-zero elements in the row and indices returns the indices of the non-zero elements
            data = row.get_mat().data
            indices = row.get_mat().indices
            
            indices_rev_dict = { index : i for i, index in enumerate(indices) }
                        
            #print data
            #print indices
            #print indices_rev_dict                
            #print most_associated_cols_dict[w]
    
            reduced_row = [ data[indices_rev_dict[index[0]]] for index in most_associated_cols_dict[w] ]

            vecs_dict[w] = reduced_row  
            
            #print reduced_row
            
        return vecs_dict
                
        
    def make_absolute_target_entropies(self, args, targets, most_associated_cols_dict, c_entropies, test_set):
    
        N = int(args['<N>'])    
    
        is_average = args['--average']
        is_median = args['--median']         
        target_output = args['target_output']
        
        targets_size = len(targets)     
        vocab_map = self.get_vocab()
        id2column_map = self.get_id2column_map()
        
        print "Computing target entropies..."
        i = 0
        target_entropies = {}    
        # Compute target entropies for all target rows
        for target in targets:
            if target not in vocab_map:
                continue
            
            print "%d%% done..." % (i*100/targets_size)
            
            # N most associated contexts of target
            most_associated_cs = [id2column_map[mapping[0]] for mapping in most_associated_cols_dict[target][:N]]
                                      
            #print "Most associated contexts of " + target
            #print most_associated_cs
            
            if not len(most_associated_cs) > 0:
                target_entropies[target] = -999.0
                continue
    
            # Get the entropies of the most associated contexts
            entr_of_most_assoc_cs = [float(c_entropies[context]) for context in most_associated_cs]
            
            # Compute the average or median of the entropies of the most associated contexts (target entropy)
            if is_average:
                target_entropy = float(mean(entr_of_most_assoc_cs))
            elif is_median:
                target_entropy = float(median(entr_of_most_assoc_cs))
                   
            target_entropies[target] = target_entropy
            
            i += 1
        
        # Rank the target entropies
        target_entropies_ranked = sorted(target_entropies, key=lambda x: -(target_entropies[x]))

        #TODO: make target entropy file an argument of the script
        # Save the target entropies for maximal number of associated columns <= N
        print "Writing target entropies to %s..." % target_output
        with open(target_output, 'w') as f_out:
            for target in target_entropies_ranked:
                H = target_entropies[target]
                print >> f_out, "\t".join((target, str(H)))
        

        # Prepare output            
        unscored_output = []        
        for (x, y, label, relation) in test_set:
            if x not in vocab_map or y not in vocab_map:
                # Assign a special score to out-of-vocab pairs
                unscored_output.append((x, y, label, relation, -999.0, -999.0))
                continue 
        
            unscored_output.append((x, y, label, relation, target_entropies[x], target_entropies[y]))
            
                    
        return unscored_output
         
         
def slqs(x_entr, y_entr):
    """
    Computes SLQS score from two entropy values.
    :param x_entr: entropy value of x
    :param y_entr: entropy value of y
    :return: SLQS score
    """
    
    score = 1 - (x_entr/y_entr) if y_entr != 0.0 else -1.0  
    
    return score


def slqs_sub(x_entr, y_entr):
    """
    Computes SLQS Sub score from two entropy values.
    :param x_entr: entropy value of x
    :param y_entr: entropy value of y
    :return: SLQS Sub score
    """
    
    score = y_entr - x_entr
    
    return score
    
    
def load_test_pairs(testset_file):
    """
    Loads target tuples from a test file.
    :param testset_file: test file with each line a test item
    :return targets: set of target strings
    :return test_set: list of tuples, each tuple a test item
    """
    
    print "Loading test pairs..."
    with codecs.open(testset_file) as f_in:
        test_set = [tuple(line.strip().split('\t')) for line in f_in]
        
    targets = get_targets(test_set)
    
    return targets, test_set
    

def get_targets(test_set):
    """
    Get set of elements in index position 0 and 1 from each tuple from list.
    :param test_set: list of tuples, each tuple a test item
    :return union: set, union of targets in index position 0 and 1
    """
    
    xs, ys, labels, relation = zip(*test_set)
    union = set(xs) | set(ys)
    return union
    

def get_space(matrice_folder, matrice_name, is_pmi, is_lmi, is_save_weighted):
    """
    Loads semantic space from matrix file.
    :param matrice_folder: string, path of matrice folder
    :param matrice_name: string, name of matrice file
    :param is_pmi: boolean, whether to weight matrice with PPMI values
    :param is_lmi: boolean, whether to weight matrice with PLMI values
    :param is_save_weighted: boolean, whether to save weighted matrice
    :return cooc_space: unweighted semantic space
    :return mi_space: weighted semantic space
    :return vocab_map: dictionary that maps row strings to integer ids
    :return vocab_size: int, number of rows
    :return column_map: dictionary that maps column strings to integer ids
    :return id2column_map: list of strings, the column elements
    """
    
    try:
        print "Loading frequency matrice..."
        cooc_space = load_pkl_files(matrice_folder + matrice_name)
        cooc_space.__class__ = Space_extension
    except IOError:
        print "Format not suitable or file does not exist: " + matrice_folder + matrice_name
       
    mi_space = []   
       
    if is_pmi:
        try:
            mi_space = load_pkl_files(matrice_folder + matrice_name + "_ppmi")
            print "Found Ppmi weighted matrice."
        except:
            print "No Ppmi weighted matrice found."
            print "Building Ppmi weighted matrice..."
            mi_space = cooc_space.apply(PpmiWeighting())
            if is_save_weighted:
                print "Saving Ppmi weighted matrice..."
                save_pkl_files(matrice_folder + matrice_name + "_ppmi", mi_space, False)
            
        mi_space.__class__ = Space_extension

                
    if is_lmi:
        try:
            mi_space = load_pkl_files(matrice_folder + matrice_name + "_plmi")
            print "Found Plmi weighted matrice."
        except:
            print "No Plmi weighted matrice found."
            print "Building Plmi weighted matrice..."
            mi_space = cooc_space.apply(PlmiWeighting())
            if is_save_weighted:
                print "Saving Plmi weighted matrice..."
                save_pkl_files(matrice_folder + matrice_name + "_plmi", mi_space, False)
        
        mi_space.__class__ = Space_extension

    
    vocab_map = cooc_space.get_vocab()
    vocab_size = len(vocab_map)
    column_map = cooc_space.get_columns()
    id2column_map = cooc_space.get_id2column_map()
    
    print "The vocabulary has size: " + str(vocab_size)
    
    return cooc_space, mi_space, vocab_map, vocab_size, column_map, id2column_map
    
    
def get_all_most_assoc_cols(mi_space, targets, vocab_map, N):
    """
    Gets columns with highest values for each target row.
    :param mi_space: a semantic space
    :param targets: set of target strings
    :param vocab_map: dictionary that maps row strings to integer ids
    :param N: int, number of columns to extract
    :return most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :return union_m_a_c: set, union of column ids with highest values
    """
    
    print "Getting most associated columns for all targets..."
    most_associated_cols_dict = {}
    targets_size = len(targets)
    union_m_a_c = set()
    i = 0
    for target in targets:
        if target not in vocab_map:
            continue
        
        print "%d%% done..." % (i*100/targets_size)        
        
        # Get most associated columns for target
        most_associated_cols = mi_space.get_most_associated_cols(target, N) 
        union_m_a_c = union_m_a_c | set(most_associated_cols)
        most_associated_cols_sorted = sorted(most_associated_cols.iteritems(), key=lambda (k,v): (v,k), reverse=True)
        most_associated_cols_dict[target] = most_associated_cols_sorted
        
        i = i + 1
    
    return most_associated_cols_dict, union_m_a_c
    
    
def assign_c_entr_file(matrice_name, is_pmi, is_lmi, is_weighted):
    """
    Assigns the path of the context entropy file.
    :param matrice_name: string, name of matrice file
    :param is_pmi: boolean, whether matrice is weighted with PPMI values
    :param is_lmi: boolean, whether matrice is weighted with PLMI values
    :param is_weighted: boolean, whether matrice is weighted in the first place
    :return c_entrop_file: string, path of context_entropy file
    """
    
    if is_weighted:
        if is_pmi:
            c_entrop_file = "measures/entropies/context_entropies/" + matrice_name + "_ppmi" + "_context_entropies" + ".txt" 
        if is_lmi:
            c_entrop_file = "measures/entropies/context_entropies/" + matrice_name + "_plmi" + "_context_entropies" + ".txt" 
    if not is_weighted:
        c_entrop_file = "measures/entropies/context_entropies/" + matrice_name  + "_freq" + "_context_entropies" + ".txt"

    return c_entrop_file
    

def save_entropies(entr_ranked, entropies_dict, entrop_file):
    """
    Saves entropy rank to disk.
    :param entr_ranked: list of floats, ranked values
    :param entropies_dict: dictionary, mapping strings to values
    :param entrop_file: string, path of output file
    """
    
    print "Writing values to %s..." % entrop_file                
    with open(entrop_file, 'w') as f_out:
        for context in entr_ranked:
            H = entropies_dict[context]
            print >> f_out, "\t".join((context, str(H)))
                
                
def get_r_entropies(targets, cooc_space, mi_space, target_output, is_freq):
    """
    Get row entropies from unweighted or weighted space for a set of targets.
    :param targets: set of target strings
    :return cooc_space: unweighted semantic space
    :param mi_space: weighted semantic space
    :param is_freq: boolean, whether to compute row entropies from unweighted 
                        or weighted space    
    :return r_entropies_dict: dictionary, mapping targets to entropy values
    :return r_entr_ranked: list of ranked entropy values
    """
    
    # Get row entropies
    if is_freq:
        print "Computing row entropies from co-occurence matrice..."
        r_entropies_dict = cooc_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
    else:
        print "Computing row entropies from weighted matrice..."
        r_entropies_dict = mi_space.compute_row_entropies(targets)
        print "Calculated entropies for %d rows." % len(r_entropies_dict)
            
    # Rank the row entropies
    r_entr_ranked = sorted(r_entropies_dict, key=lambda x: -(float(r_entropies_dict[x])))
         
    return r_entropies_dict, r_entr_ranked
    
    
def get_r_frequencies(targets, cooc_space, mi_space, target_output, is_freq):
    
    # Get row frequencies
    if is_freq:
        print "Computing row frequencies from co-occurence matrice..."
        r_frequencies_dict = cooc_space.compute_row_frequencies(targets)
        print "Calculated frequencies for %d rows." % len(r_frequencies_dict)
    else:
        print "Computing row frequencies from weighted matrice..."
        r_frequencies_dict = mi_space.compute_row_frequencies(targets)
        print "Calculated frequencies for %d rows." % len(r_frequencies_dict)
            
    # Rank the row frequencies
    r_freq_ranked = sorted(r_frequencies_dict, key=lambda x: -(float(r_frequencies_dict[x])))
    
#    # Save the row frequencies      
#    save_entropies(r_freq_ranked, r_frequencies_dict, target_output)
            
    return r_frequencies_dict, r_freq_ranked
    

def get_r_typtoks(targets, cooc_space, mi_space, target_output, is_freq):
    
    # Get row types per token
    if is_freq:
        print "Computing row types per token from co-occurence matrice..."
        r_typtoks_dict = cooc_space.compute_row_typtoks(targets)
        print "Calculated types per token for %d rows." % len(r_typtoks_dict)
    else:
        print "Computing row types per token from weighted matrice..."
        r_typtoks_dict = mi_space.compute_row_typtoks(targets)
        print "Calculated types per token for %d rows." % len(r_typtoks_dict)
            
    # Rank the row types per token
    r_typtok_ranked = sorted(r_typtoks_dict, key=lambda x: -(float(r_typtoks_dict[x])))
    
#    # Save the row types per token      
#    save_entropies(r_typtok_ranked, r_typtoks_dict, target_output)
            
    return r_typtoks_dict, r_typtok_ranked
    

def get_r_kldivs(targets, cooc_space, mi_space, target_output, is_freq):
    
    # Get row KL-div
    if is_freq:
        print "Computing row KL-div from co-occurence matrice..."
        r_kldiv_dict = cooc_space.compute_row_kldivs(targets)
        print "Calculated KL-div for %d rows." % len(r_kldiv_dict)
    else:
        print "Computing row KL-div from weighted matrice..."
        r_kldiv_dict = mi_space.compute_row_kldivs(targets)
        print "Calculated KL-div for %d rows." % len(r_kldiv_dict)
            
    # Rank the row KL-div
    r_kldiv_ranked = sorted(r_kldiv_dict, key=lambda x: -(float(r_kldiv_dict[x])))
    
#    # Save the values      
#    save_entropies(r_kldiv_ranked, r_kldiv_dict, target_output)
            
    return r_kldiv_dict, r_kldiv_ranked


def get_c_entropies(targets, cooc_space, mi_space, N, c_entrop_file, vocab_map, id2column_map, most_associated_cols_dict, union_m_a_c, is_freq, is_weighted): 
    """
    Get context entropies from unweighted or weighted space for a set of targets.
    :param targets: set of target strings
    :param cooc_space: unweighted semantic space
    :param mi_space: weighted semantic space
    :param N: int, number of columns to extract
    :param c_entrop_file: string, path of context_entropy file
    :param vocab_map: dictionary that maps row strings to integer ids
    :param id2column_map: list of strings, the column elements
    :param most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :param union_m_a_c: set, union of column ids with highest values
    :param is_freq: boolean, whether to compute entropies from unweighted space
    :param is_weighted: boolean, whether to compute entropies from weighted space
    :return c_entropies_dict: dictionary, mapping contexts to entropy values
    :return c_entr_ranked: list of ranked entropy values
    """
    
    # Try to get context entropy file
    try:
        with open(c_entrop_file) as f_in:
            c_entropies_dict = dict([[line.strip().split("\t")[0], float(line.strip().split("\t")[1])] for line in f_in])
            print "Found context entropy file: " + c_entrop_file

            # Get new contexts
            new_union_m_a_c = set()
            for target in targets:
                if target not in vocab_map:
                    continue
                for mapping in most_associated_cols_dict[target]:
                    col_id = int(mapping[0]) 
                    context = id2column_map[col_id]
                    if not context in c_entropies_dict:
                        new_union_m_a_c = new_union_m_a_c | set([col_id])
                         
            if len(new_union_m_a_c) > 0:
                if is_freq:
                    print "Computing new context entropies from co-occurence matrice..."
                    new_c_entropies_dict = cooc_space.compute_context_entropies(new_union_m_a_c)
                elif is_weighted:
                    print "Computing new context entropies from weighted matrice..."
                    new_c_entropies_dict = mi_space.compute_context_entropies(new_union_m_a_c)               
                # Add the new context entropies to the old ones
                print "Calculated entropies for %d new contexts." % len(new_c_entropies_dict)
                c_entropies_dict.update(new_c_entropies_dict)  
                
    except IOError:
        print "No context entropy file found."
        # Build context entropy file if non-existent
        if is_freq:
            print "Computing context entropies instead from co-occurence matrice..."
            c_entropies_dict = cooc_space.compute_context_entropies(union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
        elif is_weighted:
            print "Computing context entropies instead from weighted matrice..."
            c_entropies_dict = mi_space.compute_context_entropies(union_m_a_c)
            print "Calculated entropies for %d contexts." % len(c_entropies_dict)
            
    # Rank the context entropies
    c_entr_ranked = sorted(c_entropies_dict, key=lambda x: -(float(c_entropies_dict[x])))
    
    # Save the (updated) context entropies      
    save_entropies(c_entr_ranked, c_entropies_dict, c_entrop_file)
            
    return c_entropies_dict, c_entr_ranked
   

def get_asoc_vecs(targets, cooc_space, mi_space, N, most_associated_cols_dict, is_freq, is_weighted):  

    if is_freq:
        print "Getting vectors from co-occurence matrice..."
        vecs_dict = cooc_space.get_r_asoc_vecs(N, targets, most_associated_cols_dict)
    elif is_weighted:
        print "Getting vectors from weighted matrice..."
        vecs_dict = mi_space.get_r_asoc_vecs(N, targets, most_associated_cols_dict)
            
    return vecs_dict   
   
   
def convert_neg(dictionary):
    
    print "Converting values to negative..."    
    
    dictionary_neg = dict([[key, -dictionary[key]] for key in dictionary])

    return dictionary_neg


def prune_dict(dict_, list_):
    
    pruned_dict = {}

    print "Pruning dictionary..."    
    
    for key in list_:
        if key in dict_:
            pruned_dict[key] = dict_[key]
    
    return pruned_dict


def make_relative_target_entropies(output_file, vocab_map, id2column_map, test_set, most_associated_cols_dict, c_entropies, N, is_average, is_median):
    """
    Get relative entropy values for x and y for each test pair (x,y) of each test item.
    :param output_file: string, path of output file
    :param vocab_map: dictionary that maps row strings to integer ids
    :param id2column_map: list of strings, the column elements
    :param test_set: list of tuples, each tuple a test item
    :param most_associated_cols_dict: dictionary, mapping target strings to
                                        column ids
    :param c_entropies: dictionary, mapping contexts to entropy values
    :param N: int, number of columns to extract

    :param is_average: boolean, whether to calculate average of context entropies
    :param is_median: boolean, whether to calculate median of context entropies
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).
    """
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map or y not in vocab_map:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print x, y
        
        # Get smaller number M of associated columns
        M = min([len(most_associated_cols_dict[x]), len(most_associated_cols_dict[y])])
        
        if M == 0:
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
 
        target_entropies = {}
        
        # Compute Generality Index for x and y
        for var in (x, y): 

            m_most_assoc_cs = {}

            # M Most associated contexts of x and y
            m_most_assoc_cs[var] = [id2column_map[mapping[0]] for mapping in most_associated_cols_dict[var][:M]]
                                      
            #print "M Most associated contexts of " + var
            #print m_most_assoc_cs[var]

            entr_of_m_most_assoc_cs = {}    

            # Get the M entropies of the most associated contexts of x and y
            entr_of_m_most_assoc_cs[var] = [float(c_entropies[context]) for context in m_most_assoc_cs[var]]
            
            # Compute the average or median of the entropies of the M most associated contexts (target entropy)
            if is_average:
                target_entropies[var] = float(mean(entr_of_m_most_assoc_cs[var]))
            elif is_median:
                target_entropies[var] = float(median(entr_of_m_most_assoc_cs[var]))
                
            print float(median(entr_of_m_most_assoc_cs[var]))
        
        unscored_output.append((x, y, label, relation, target_entropies[x], target_entropies[y]))
    
    return unscored_output


def make_unscored_output(entropies, test_set, vocab_map):
    """
    Append test items and their entropy values.
    :param entropies: dictionary, mapping targets to entropy values
    :param test_set: list of tuples, each tuple a test item
    :param vocab_map: dictionary that maps row strings to integer ids
    :return unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).
    """
    
    unscored_output = []    
    
    for (x, y, label, relation) in test_set:
        if x not in vocab_map or y not in vocab_map:
            # Assign a special score to out-of-vocab pairs
            unscored_output.append((x, y, label, relation, -999.0, -999.0))
            continue
        
        print (x, y, label, relation, entropies[x], entropies[y])
    
        unscored_output.append((x, y, label, relation, entropies[x], entropies[y]))
    
    return unscored_output


def score_slqs(rel_tar_entrs):
    """
    Make scored SLQS output from individual values for test pairs.
    :param unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).    
    :return scored_output: list of tuples of test items plus their SLQS score
                                for x and y for each test pair (x,y).    
    """
    
    scored_output = []

    print "Computing target SLQS for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in rel_tar_entrs:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        # Compute slqs for y being the hypernym of x
        score = slqs(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def score_slqs_sub(rel_tar_entrs):
    """
    Make scored SLQS Sub output from individual values for test pairs.
    :param unscored_output: list of tuples of test items plus their entropy values
                                for x and y for each test pair (x,y).    
    :return scored_output: list of tuples of test items plus their SLQS Sub score
                                for x and y for each test pair (x,y). 
    """

    scored_output = []

    print "Computing difference for test tuples..."
    
    for (x, y, label, relation, xentr, yentr) in rel_tar_entrs:
        
        if xentr == -999.0 or yentr == -999.0:
            scored_output.append((x, y, label, relation, -999.0))
            continue
        
        # Compute slqs_sub for y being the hypernym of x
        score = slqs_sub(xentr, yentr)            
    
        scored_output.append((x, y, label, relation, score))
        
    return scored_output
    

def save_results(scored_output, output_file):
    """
    Saves scored output to disk.
    :param scored_output: list of tuples of test items plus their score
                                for x and y for each test pair (x,y). 
    :param output_file: string, path of output file
    """
    
    with codecs.open(output_file, 'w') as f_out:
            
        for (x, y, label, relation, score) in scored_output:
    
            print >> f_out, '\t'.join((x, y, label, relation, '%.8f' % score))
    
    print "Saved the results to " + output_file
    

def score_f1_binary(scores):
    
    try:
        f1 = float(len([True for x in scores if x == True]))/len(scores)
    except ZeroDivisionError:
        return 0.0
    
    return f1