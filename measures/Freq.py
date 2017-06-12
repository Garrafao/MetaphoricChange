import sys
sys.path.append('./modules/')

import codecs
from docopt import docopt

from common import *
from slqs_module import *
from slqs_diac_module import *


def main():
    """
    Make unscored results from frequency rank for targets in test set.
    score_results.py can then be used to get difference between two unscored files.
    """

    # Get the arguments
    args = docopt("""Make unscored results from frequency rank for targets in test set.

    Usage:
        Freq.py <file1> <testset_file> <output_file>


    Arguments:
        <file1> = file with list of targets + values (tab-separated)
        <testset_file> = a file containing term-pairs corresponding to developments with their
                       starting points and the type of the development
        <output_file> = where to save the results

    """)

    file1 = args['<file1>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    non_values = [-999.0, -888.0]

  
    # Load data files
    with codecs.open(file1) as f_in:
        file1_dict = dict([tuple(line.strip().split('\t')) for line in f_in])
         
    word2val_map = {word : float(file1_dict[word]) for word in file1_dict}
   
       
    # Load term-pairs
    targets, test_set = load_test_pairs(testset_file) 
        
    target_values = {}
    for target in targets:
        
        if target not in word2val_map:
            target_values[target] = non_values[0]
            continue

        target_values[target] = word2val_map[target]

    unscored_output = []
    for (x, y, label, relation) in test_set:
        unscored_output.append((x, y, label, relation, target_values[x]))
        
    # Save results
    save_results(unscored_output, output_file)       


if __name__ == "__main__":
    main()
