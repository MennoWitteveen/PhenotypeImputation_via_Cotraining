#-----------------------------------------------------------------------------
# Create the genotype classifier
# Use a subset of all SNPs (e.g. a set of SNPs provided by the user or 
# obtained from univariate feature selection)
# 
# Authors: Menno Witteveen
#          Damian Roqueiro
#-----------------------------------------------------------------------------

# Class imports
from classes.dataset import Dataset
from classes.config_state import ConfigState

# System libraries
import logging
import numpy as np
import sklearn as skl
from sklearn import metrics #roc_curve, auc
from sklearn import preprocessing
from scipy import interp

# Pipeline auxiliary functions
from generic_functions import find_vec_entries_that_contain
from generic_functions import harden_labels
import IPython as ip

def random_forest(data, config):
    ''' 
    Function to construct a genotype classifier using random forest
    
    Parameters 
    ---------- 
    data : an object of class Dataset that contains: genotypes, covariates, 
        labels and information about random folds 

    config : an object of class ConfigState. It contains the user-entered 
        parameters in a YAML format.
        See the config_file parameter in the main script for more details.
    '''
    
    # Parameters
    task_name    = "random_forest"
    n_estimators = config.get_entry(task_name, "n_estimators")
    criterion    = config.get_entry(task_name, "criterion")
    n_select     = config.get_entry(task_name, "n_select")
    num_folds    = data.num_folds  
    romans_trn_gold     = config.get_entry(task_name, "golden_romans_used_for_learning")
    romans_trn_silver   = config.get_entry(task_name, "silver_romans_used_for_learning")

    # Load the output of the previous task(s)
    soft_labels = config.load_variable("phenotype_imputation", "soft_labels")
    feature_ranking = np.asarray(config.load_variable("univ_feature_sel", "feature_ranking"), dtype='int64')
    
    # Create array that can be filled with results
    results = np.zeros((num_folds, find_vec_entries_that_contain(data.folds[:,0],[3]).shape[0]))    
    
    # Iterate through the folds:  
    i = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    roc_auc = np.zeros(num_folds)
    for fold in data.folds.transpose():
        logging.info("Fold=%d" % (i + 1))
        # Training data:
        sel_trn_gold = find_vec_entries_that_contain(fold, romans_trn_gold)
        sel_trn_silver = find_vec_entries_that_contain(fold, romans_trn_silver)
        sel_trn = np.concatenate([sel_trn_gold, sel_trn_silver])
        
        # Testing data:
        sel_tst = find_vec_entries_that_contain(fold,[3])
        
        # The model used for training:
        model = skl.ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=None, 
            min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, 
            bootstrap=True, oob_score=False, n_jobs=1, random_state=None, 
            verbose=0, min_density=None, compute_importances=None)
        
        # Slicing of the matrix
        genotype_data_filtered = data.genotype[:,feature_ranking[i,0:n_select]].transpose()

        data_filtered = np.concatenate([genotype_data_filtered,
            preprocessing.scale(data.regular_covariate.transpose()).transpose()]).transpose()
        
        # Harden the labels for classification (This could be an un-needed calculation if only gold data is used)
        n_p = np.sum(data.labels[0, sel_trn_gold] == 1)
        n_n = np.sum(data.labels[0, sel_trn_gold] == 0)
        p_class = float(n_p)/float(n_p + n_n)
        
        hard_labels = harden_labels(soft_labels[range(len(sel_trn_silver)),i], p_class)
        trn_labels = np.concatenate([data.labels[0,sel_trn_gold], hard_labels])        
        
        # Fitting of the model
        model.fit(data_filtered[sel_trn,:], trn_labels)
        
        # Generation of the results:
        results[i,:] = model.predict_proba(data_filtered[sel_tst, :])[:, 1] 
        fpr, tpr, _ = metrics.roc_curve(data.labels[0, sel_tst], results[i, :])
        # Accumulate the interpolated tpr
        mean_tpr += interp(mean_fpr, fpr, tpr)
        roc_auc[i] = metrics.auc(fpr, tpr)
        i += 1

    # Compute the mean ROC curve values
    mean_tpr /= num_folds
    mean_tpr[-1] = 1.0
    # Save the mean auc computed in this way (to compare with the other values)
    mean_auc = np.zeros(1)
    mean_auc[0] = metrics.auc(mean_fpr, mean_tpr)
    # Save the output of this task
    config.save_variable(task_name, "%f", results=results, roc_auc=roc_auc, mean_fpr=mean_fpr, mean_tpr=mean_tpr, mean_auc=mean_auc)
