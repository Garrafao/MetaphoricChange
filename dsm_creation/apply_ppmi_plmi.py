"""
Part of the code in this file is based on the code developed in

    Vered Shwartz, Enrico Santus, Dominik Schlechtweg. 2017. Hypernyms under Siege: 
        Linguistically-motivated Artillery for Hypernymy Detection. Proceedings of 
        the 15th Conference of the European Chapter of the Association of Computational 
        Linguistics.
"""

from docopt import docopt
from composes.utils import io_utils
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting


def main():
    """
    Compute the PPMI/PLMI matrix from a co-occurrence matrix, as default pickle the raw matrix.
    """

    # Get the arguments
    args = docopt('''Compute the PPMI/PLMI matrix from a co-occurrence matrix, as default pickle the raw matrix.

    Usage:
        apply_ppmi_plmi.py <dsm_prefix> [-p | -l]

        <dsm_prefix> = the prefix for the input files (.sm for the matrix, .rows and .cols) and output files (.ppmi)
    
    Options:  
    -p, --ppmi  weight the matrice entries via PPMI
    -l, --plmi  weight the matrice entries via PLMI
    
    ''')

    dsm_prefix = args['<dsm_prefix>']
    is_ppmi = args['--ppmi']
    is_plmi = args['--plmi']
    
    postfix = ""


    # Create a space from co-occurrence counts in sparse format
    dsm = Space.build(data=dsm_prefix + '.sm',
                      rows=dsm_prefix + '.rows',
                      cols=dsm_prefix + '.cols',
                      format='sm')

    if is_ppmi:
        # Apply ppmi weighting
        dsm = dsm.apply(PpmiWeighting())
        postfix = "_ppmi"
    elif is_plmi:
        # Apply plmi weighting
        dsm = dsm.apply(PlmiWeighting())
        postfix = "_plmi"

    # Save the Space object in pickle format
    io_utils.save(dsm, dsm_prefix + postfix + '.pkl')


if __name__ == '__main__':
    main()