"""
Part of the code in this file is based on the code developed in

    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.

--------------
This module is a collection of general methods useful for processing corpora and matrices.
"""

import sys
sys.path.append('./modules/')

import os
import pickle
import numpy as np

from composes.semantic_space.space import Space 
from composes.utils import io_utils
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix


def save_pkl_files(dsm_prefix, dsm, save_in_one_file=False):
    """
    Save the space to separate pkl files.
    :param dsm_prefix:
    :param dsm:
    """
    
    # Save in a single file (for small spaces)
    if save_in_one_file:
        io_utils.save(dsm, dsm_prefix + '.pkl')

    # Save in multiple files: npz for the matrix and pkl for the other data members of Space
    else:
        mat = coo_matrix(dsm.cooccurrence_matrix.get_mat())
        np.savez_compressed(dsm_prefix + 'cooc.npz', data=mat.data, row=mat.row, col=mat.col, shape=mat.shape)

        with open(dsm_prefix + '_row2id.pkl', 'wb') as f_out:
            pickle.dump(dsm._row2id, f_out, 2)

        with open(dsm_prefix + '_id2row.pkl', 'wb') as f_out:
            pickle.dump(dsm._id2row, f_out, 2)

        with open(dsm_prefix + '_column2id.pkl', 'wb') as f_out:
            pickle.dump(dsm._column2id, f_out, 2)

        with open(dsm_prefix + '_id2column.pkl', 'wb') as f_out:
            pickle.dump(dsm._id2column, f_out, 2)


def load_pkl_files(dsm_prefix):
    """
    Load the space from either a single pkl file or numerous files.
    :param dsm_prefix:
    :param dsm:
    """
    
    # Check whether there is a single pickle file for the Space object
    if os.path.isfile(dsm_prefix + '.pkl'):
        return io_utils.load(dsm_prefix + '.pkl')

    # Load the multiple files: npz for the matrix and pkl for the other data members of Space
    with np.load(dsm_prefix + 'cooc.npz') as loader:
        coo = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

    cooccurrence_matrix = SparseMatrix(csr_matrix(coo))

    with open(dsm_prefix + '_row2id.pkl', 'rb') as f_in:
        row2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2row.pkl', 'rb') as f_in:
        id2row = pickle.load(f_in)

    with open(dsm_prefix + '_column2id.pkl', 'rb') as f_in:
        column2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2column.pkl', 'rb') as f_in:
        id2column = pickle.load(f_in)

    return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)
