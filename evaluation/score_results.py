import sys
sys.path.append('./modules/')

from docopt import docopt

from common import *
from slqs_module import *
from slqs_diac_module import *

            
def main():
    """
    Get the difference of values in two individual result files for targets in test set.
    """

    # Get the arguments
    args = docopt("""Get the difference of values in two individual result files for targets in test set.

    Usage:
        score_results.py <testset_file> <file1> <file2> <output_file>
        
    Arguments:
        <testset_file> = a file containing term-pairs corresponding to developments with their
                           starting points and the type of the development
        <file1> = result file 1
        <file2> = result file 2
        <output_file> = where to save the results
        
        
    """)
    
    file1 = args['<file1>']
    file2 = args['<file2>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    non_values = [-999.0, -888.0]
    

    # Load data files
    with codecs.open(file1) as f_in:
        file1_dict = dict([(line.strip().split('\t')[0], line.strip().split('\t')[4]) for line in f_in])
        
    with codecs.open(file2) as f_in:
        file2_dict = dict([(line.strip().split('\t')[0], line.strip().split('\t')[4]) for line in f_in])
        
 
    file1_values = {target : float(file1_dict[target]) for target in file1_dict}

    file2_values = {target : float(file2_dict[target]) for target in file2_dict}

       
    # Load term-pairs
    targets, test_set = load_test_pairs(testset_file) 
    
    
    # Make vocab_maps
    vocab_map1 = [target for target in file1_values if not file1_values[target] == non_values[0] and not file1_values[target] == non_values[1]]
    vocab_map2 = [target for target in file2_values if not file2_values[target] == non_values[0] and not file1_values[target] == non_values[1]]

    # Make unscored output
    unscored_output = make_unscored_output_diac(file1_values, file2_values, test_set, vocab_map1, vocab_map2)
    
    # Score test tuples
    scored_output = score_slqs_sub(unscored_output)

    # Save results
    save_results(scored_output, output_file)
    

if __name__ == '__main__':
    main()