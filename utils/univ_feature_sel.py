#-----------------------------------------------------------------------------
# Perform univariate feature selection on the totality of SNPs
# Rank SNPs according to the correlation between their genotype and class
# labels
# 
# Authors: Menno Witteveen
#          Damian Roqueiro
#-----------------------------------------------------------------------------

# Class imports
from classes.dataset import Dataset
from classes.config_state import ConfigState

import logging
import numpy as np
from scipy.stats import pearsonr

# Pipeline auxiliary functions
from generic_functions import find_vec_entries_that_contain

def univ_feature_sel(data, config):
    ''' 
    Do univariate feature selection  
     
    
    '''
    
    # Parameters
    task_name = "univ_feature_sel"
    romans_trn_gold     = config.get_entry(task_name, "golden_romans_used_for_learning")
    romans_trn_silver   = config.get_entry(task_name, "silver_romans_used_for_learning")
    
    # Load the output of the previous task(s)
    soft_labels = config.load_variable("phenotype_imputation", "soft_labels")
    
    # ---------------------------
    feature_ranking = np.ones((data.folds.shape[1], data.genotype.shape[1]))
    feature_pval = np.ones((data.folds.shape[1], data.genotype.shape[1]))    
    
    # Iterate through the folds:
    i = 0
    for fold in data.folds.transpose():
        logging.info("Fold=%d" % (i + 1))
        sel_trn_gold = find_vec_entries_that_contain(fold, romans_trn_gold)
        sel_trn_silver = find_vec_entries_that_contain(fold, romans_trn_silver)
        sel_trn = np.concatenate([sel_trn_gold, sel_trn_silver])
        trn_labels = np.concatenate([data.labels[0,sel_trn_gold], soft_labels[range(len(sel_trn_silver)),i]])
        j = 0
        # ---------------
        for genotype_snp in data.genotype.transpose():
            feature_pval[i,j] = pearsonr(genotype_snp[sel_trn], trn_labels)[1]                   
            j+=1
        # ---------------
        i+=1
        
    feature_ranking = feature_pval.argsort()
    # ---------------------------
    # Save the output of this task
    config.save_variable(task_name, "%d", feature_ranking=feature_ranking)
