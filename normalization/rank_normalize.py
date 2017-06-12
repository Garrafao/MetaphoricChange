import sys
sys.path.append('./modules/')

import codecs
from docopt import docopt

from slqs_module import *


def main():
    """
    Normalizes list of ranked values via division and saves to file.
    """

    # Get the arguments
    args = docopt("""Normalizes list of ranked values via division and saves to file.


    Usage:
        rank_normalize.py <divisor> <file> <output_file>

        <divisor> = tag to consider in file2        
        <file> = file with list of targets + values (tab-separated)
        <output_file> = where to save the normalized target-value pairs from file

    """)

    file1 = args['<file>']
    divisor = float(args['<divisor>'])
    output_file = args['<output_file>']
    

    # Open the rank
    with codecs.open(file1) as f_in:
        
        file_dict = dict([tuple(line.strip().split('\t')) for line in f_in])
        
    # Normalize dict
    scored_file_dict = {target : float(file_dict[target])/divisor for target in file_dict}
    
    # Get rank
    file_dict_ranked = sorted(file_dict, key=lambda x: -(float(file_dict[x])))
              
    # Save normalized rank              
    save_entropies(file_dict_ranked, scored_file_dict, output_file)


if __name__ == "__main__":
    main()
