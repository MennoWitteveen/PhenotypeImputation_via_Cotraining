#-----------------------------------------------------------------------------
# Creation of cross-validation datasets
#  
# Authors: Menno Witteveen
#          Damian Roqueiro
#-----------------------------------------------------------------------------

import numpy as np
from math import floor
from sklearn import cross_validation

def cv_set_creation(n, config):   
    '''
    Function to generate the cross-validation data. Samples are randomly 
    assigned to one of three sets: I, II or III. Each assignment of all samples
    is called a fold. Cross-validation is performed by randomly generating many
    folds.
    
    Parameters
    ----------
    n : the number of samples

    config : an object of class ConfigState. It contains the user-entered 
        parameters in a YAML format.
        See the config_file parameter in the main script for more details.
    '''
    
    # Parameters:
    task_name = "cv_set_creation"
    num_folds = config.get_entry(task_name, "num_folds")
    set_sizes = config.get_entry(task_name, "sizes")

    # Get the sizes of the sets
    size_1 = set_sizes["set_I"]
    size_2 = set_sizes["set_II"]
    size_3 = set_sizes["set_III"]

    # Initialize the matrix with folds.
    # Each row is a sample and the columns indicate if the sample is assigned to
    # set I, II or III
    mat_folds = np.zeros((n, num_folds), 'int32')

    # To partition data into 3 sets, perform shuffle split in 3 steps:
    # 1. Split the data into III and <rest_1>, use the indices of III
    # 2. Split <rest_1> into II and <rest_2>, use the indices for II
    # 3. If size of I+II+III is not 100%, then split <rest_2> into I and <rest_3>, use the indices for I
    full_partition = True if (size_1 + size_2 + size_3 == 1.0) else False

    # 1. Split the data into III and <rest_1>, use the indices of III
#    rs1 = cross_validation.ShuffleSplit(n, n_iter=num_folds, test_size=int(floor(size_3 * n)), random_state=0)
    rs1 = cross_validation.ShuffleSplit(n, n_iter=num_folds, test_size=int(floor(size_3 * n)))
    for i, (train_index, test_index) in zip(range(num_folds), rs1):
        # Assign to set III
        mat_folds[test_index, i] = 3

        # 2. Split <rest_1> into II and <rest_2>, use the indices for II
        orig_index = train_index
        rs2 = cross_validation.ShuffleSplit(len(orig_index), n_iter=1, test_size=int(floor(size_2 * n)))
        for train_index, test_index in rs2:
            # Assign to set II
            mat_folds[orig_index[test_index], i] = 2

            # 3. If size of I+II+III is not 100%, then split <rest_2> into I and <rest_3>, use the indices for I
            if full_partition:
                # Assign to set I
                # The data is fully partitioned, take the remaining from rs2
                mat_folds[orig_index[train_index], i] = 1
            else:
                # Not fully partitioned. Split one more time
                orig_index = orig_index[train_index]
                rs3 = cross_validation.ShuffleSplit(len(orig_index), n_iter=1, test_size=int(floor(size_1 * n)))
                for train_index, test_index in rs3:
                    # Assign to set I and the rest remains 0
                    mat_folds[orig_index[test_index], i] = 1

    # Save the output of this task (format = integers)
    config.save_variable(task_name, "%d", folds=mat_folds)
    
