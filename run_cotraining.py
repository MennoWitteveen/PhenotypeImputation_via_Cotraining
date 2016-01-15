#-----------------------------------------------------------------------------
# Master script to run the co-training pipeline
# 
# Augment training dataset by imputing phenotypes
#
# Authors: Menno Witteveen
#          Damian Roqueiro
#-----------------------------------------------------------------------------

# System imports
import yaml
import sys
import logging

# Class imports
from classes.dataset import Dataset
from classes.config_state import ConfigState

# Pipeline modules
from utils.cv_set_creation import *
from utils.phenotype_imputation import *
from utils.univ_feature_sel import *
from utils.random_forest import *

# -----------------------------------------------------------------------------
# Main -
# -----------------------------------------------------------------------------
def run_pipeline(config_file):
    '''
    Main function to execute the entire cotraining pipeline. For each task to be
    executed, a different module is invoked.
    
    Parameters
    ----------
    config_file : The full path to a a YAML file with the parameters to execute
        the pipeline.
    '''

    # Create the configuration object. It will contain all the parameters 
    # entered by the user
    try:
        config = ConfigState(config_file)
    except yaml.YAMLError, exc:
        print "Err: Cannot open configuration file: %s" % config_file, exc
        sys.exit(1)
    
    # Parameter
    output_dir          = config.get_entry("global", "output_dir")
    # Tasks to run
    do_cv_set_creation  = config.get_entry("global", "cv_set_creation")
    do_pheno_imputation = config.get_entry("global", "phenotype_imputation")
    do_univ_feature_sel = config.get_entry("global", "univ_feature_sel")
    do_random_forest    = config.get_entry("global", "random_forest")
    
    # -------------------------------------------------------------------------
    # Create the log file
    logging.basicConfig(filename="%s/exec.log" % output_dir, filemode='w', 
                        level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # -------------------------------------------------------------------------
    # Load dataset in Dataset class object
    logging.info("Loading dataset")
    data = Dataset()
    data.load_dataset(config)
    logging.info("End")

    # -------------------------------------------------------------------------
    # Create the cotraining folds
    # This step creates an index indicating what records are randomly assigned 
    # to sets I, II and III
    if do_cv_set_creation:
        logging.info("Starting task: cv_set_creation")
        cv_set_creation(data.num_samples, config)
        logging.info("End")
    else:
        logging.info("Skipping task: cv_set_creation")

    # -------------------------------------------------------------------------
    # Get the random folds saved by the previous process and add them to
    # the Dataset object
    data.add_fold_information(config)

    # Performing the phenotype imputation
    if do_pheno_imputation:
        logging.info("Starting task: phenotype_imputation")
        phenotype_imputation(data, config)
        logging.info("End")
    else:
        logging.info("Skipping task: phenotype_imputation")
    
    # Performing univariate feature selection
    if do_univ_feature_sel:
        logging.info("Starting task: univ_feature_sel")
        univ_feature_sel(data, config)
        logging.info("End")
    else:
        logging.info("Skipping task: univ_feature_sel")
    
    # Executing random forest
    if do_random_forest:
        logging.info("Starting task: random_forest")
        random_forest(data, config)
        logging.info("End")
    else:
        logging.info("Skipping task: random_forest")
        
if __name__ in "__main__":
   # TODO check command line arguments. Add syntax/usage output
   run_pipeline(sys.argv[1])
