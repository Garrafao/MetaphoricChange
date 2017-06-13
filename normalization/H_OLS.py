import sys
sys.path.append('./modules/')

import codecs
from docopt import docopt
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as smf

from common import *
from slqs_module import *
from slqs_diac_module import *
    
    
def get_data_windows(target, window_size, frequencies, entropies):
    """
    Get frequency and entropy data in window around target.
    :param target: string, target word
    :param window_size: int, size of window
    :param frequencies: dictionary, mapping strings to frequency values
    :param entropies: dictionary, mapping strings to entropy values
    :return window_freq: dictionary, mapping strings to frequency values
    :param window_entr: dictionary, mapping strings to entropy values
    """    
        
    window_freq = {data_point : frequencies[data_point] for data_point in frequencies if abs(frequencies[target] - frequencies[data_point]) <= window_size}

    window_entr = {data_point : entropies[data_point] for data_point in window_freq}

    return window_freq, window_entr
	

def main():
    """
    H_OLS - Regression of dependent variable Y (entropy) from independent variable X (frequency) with plots
    as described in:
    
        Dominik Schlechtweg, Stefanie Eckmann, Enrico Santus, Sabine Schulte im Walde and Daniel Hole. 
            2017. German in Flux: Detecting Metaphoric Change via Word Entropy. In Proceedings of 
            CoNLL 2017. Vancouver, Canada.

    Frequency and entropy ranks can be extracted with dsm_creation/get_freqs.py and measures/H_rank.py .
    """

    # Get the arguments
    args = docopt("""H_OLS - Regression of dependent variable Y (entropy) from independent variable X (frequency) 
                        with plots. Takes as main input a rank (or list) of frequencies and a rank (list) of entropies.
                        Frequency and entropy ranks can be extracted with dsm_creation/get_freqs.py and measures/H_rank.py .


    Usage:
        H_OLS.py <file1> <file2> <testset_file> <output_file> <x_interval> <x_size>


    Arguments:
        <file1> = file1 with list of targets + values (tab-separated)
        <file2> = file2 with list of targets + values (tab-separated)
        <testset_file> = a file containing term-pairs corresponding to developments with their
                       starting points and the type of the development
        <output_file> = where to save the results
        <x_interval> = value difference on x-axis to data points for regression
        <x_size> = size of window on x-axis to data points for regression

    """)

    file1 = args['<file1>']
    file2 = args['<file2>']
    testset_file = args['<testset_file>']
    output_file = args['<output_file>']
    x_interval = float(args['<x_interval>'])
    x_size = int(args['<x_size>'])
    non_values = [-999.0, -888.0]

  
    # Load data files
    with codecs.open(file1) as f_in:
        file1_dict = dict([tuple(line.strip().split('\t')) for line in f_in])
        
    with codecs.open(file2) as f_in:
        file2_dict = dict([tuple(line.strip().split('\t')) for line in f_in])
        
 
    entropies = {target : float(file2_dict[target]) for target in file2_dict}
    
    # Exclude values not found in entropy rank
    frequencies = {target : float(file1_dict[target]) for target in file1_dict if target in entropies}
    
       
    # Load term-pairs
    targets, test_set = load_test_pairs(testset_file) 
        
    target_values = {}
    for target in targets:
        
        print 'Regressing ' + target + '...'
        
        if target not in frequencies or target not in entropies:
            target_values[target] = non_values[0]
            continue
            
        # Get Data Windows
        freqs_in_window, entrs_in_window = get_data_windows(target, x_interval, frequencies, entropies)

        # Build joint data structure
        joint_dict = [(target1, freqs_in_window[target1], entrs_in_window[target1]) for target1 in freqs_in_window]
        
        df = DataFrame( joint_dict[0:], index=[x[0] for x in joint_dict], columns=("target", "frequency", "entropy" ))
        df = df.sort(columns="frequency")

        
        start = df.index.get_loc(target) - (x_size/2)
        stop = df.index.get_loc(target) + (x_size/2)
        
        # Slice Data
        targets = df.target[start:stop]        
        baskets = df.frequency[start:stop]
        scaling_factor = df.entropy[start:stop]
                
        try:       
            res = smf.ols(formula='entropy ~ np.log(frequency)', data=df[start:stop]).fit()
        except ValueError as e:
            print e
            target_values[target] = non_values[1]
            continue
            
        
        print(res.summary())
        params = (float(res.params[0]), float(res.params[1]))

        delta = entrs_in_window[target] - res.fittedvalues[target]
        target_values[target] = delta
        
        prstd, iv_l, iv_u = wls_prediction_std(res)
        
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.plot(baskets,scaling_factor, 'o', color='w', label="data" + ", $R^{2}$ = %.2f" % res.rsquared)
        ax.plot(baskets, res.fittedvalues, 'k-', label="OLS fit with\n$\\alpha$ = %.1f, $\\beta$ = %.2f" % params)
        ax.plot(baskets, iv_u, 'k--')
        ax.plot(baskets, iv_l, 'k--')
        ax.plot([freqs_in_window[target]],[entrs_in_window[target]],linestyle='',marker='o', color='k',label=target.decode("utf8").split(':')[0] + ", $\Delta$ = %.3f" % delta)
        ax.legend(loc='best')
        ax.set_title("$\mathrm{curve}_\mathrm{fit}$")
        ax.set_ylabel("entropy")
        ax.set_xlabel("frequency")
        ax.grid()
        
        plt.savefig('frequency_v_entropy-' + target + '.png', fmt='png', dpi=400)

    scored_output = []
    for (x, y, label, relation) in test_set:
        scored_output.append((x, y, label, relation, target_values[x]))
        
    # Save results
    save_results(scored_output, output_file)       


if __name__ == "__main__":
    main()
