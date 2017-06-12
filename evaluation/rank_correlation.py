import codecs
from docopt import docopt
import numpy as np
from composes.utils import scoring_utils
from composes.utils.scoring_utils import spearman, pearson, auc


def score_mod(gold, prediction, method):
    """
    Computes correlation coefficient for two lists of values.
    :param gold: list of gold values
    :param prediction: list of predicted values
    :param method: string, can be either of "pearson", "spearman" or "auc" (area under curve)
    :return: correlation coefficient and p-value
    """
    
    if len(gold) != len(prediction):
        raise ValueError("The two arrays must have the same length!")

    gold = np.array(gold, dtype=np.double)
    prediction = np.array(prediction, dtype=np.double)

    if method == "pearson":
        return pearson(gold, prediction)
    elif method == "spearman":
        return spearman(gold, prediction)
    elif method == "auc":
        return auc(gold, prediction)
    else:
        raise NotImplementedError("Unknown scoring measure:%s" % method)


def main():
    """
    Calculate spearman correlation coefficient of two lists.
    """

    # Get the arguments
    args = docopt("""Calculate spearman correlation coefficient of two lists.

    Usage:
        rank_correlation.py <file1> <file2> <target_col> <value_col>

        <file1> = file1 with list of targets + values
        <file2> = file2 with list of targets + values
        <target_col> = column number of targets
        <value_col> = column number of values

    """)

    file1 = args['<file1>']
    file2 = args['<file2>']
    target_col = int(args['<target_col>'])
    value_col = int(args['<value_col>'])
    non_values = [-999.0, -888.0]
    

    with codecs.open(file1) as f_in:
        file1_dict = dict([tuple((line.strip().split('\t')[target_col], line.strip().split('\t')[value_col])) for line in f_in])
        
    with codecs.open(file2) as f_in:
        file2_dict = dict([tuple((line.strip().split('\t')[target_col], line.strip().split('\t')[value_col])) for line in f_in])

    file1_entropies = []
    file2_entropies = []    
    for target in file1_dict:
        if target in file2_dict:
            if not float(file1_dict[target]) == non_values[0] and not float(file2_dict[target]) == non_values[0]:
                file1_entropies.append(float(file1_dict[target]))
                file2_entropies.append(float(file2_dict[target]))

    scoring_utils.score = score_mod
    score, p = scoring_utils.score(file1_entropies, file2_entropies, "spearman")

    print 'The spearman correlation coefficient between\n %s \nand \n %s \nis %.5f (p-value: %.5f)' % (file1, file2, score, p)
            

if __name__ == "__main__":
    main()
