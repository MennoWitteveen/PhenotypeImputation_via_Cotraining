import tables as tb

class Dataset:
    '''
    A class object that contains the entire dataset.

    Characteristics of the dataset: 
    genotype : SNP calls recoded as dominant [0, 1, 2]. No missing values
               are allowed.
    covariate: The clinical data. The genotype and covariate data do not 
               overlap.
    labels   : Phenotype information for each sample.
    '''

    def __init__(self):
        self.genotype = []
        self.covariate = []
        self.labels = []
        self.num_samples = 0 
        self.folds = []
        self.num_folds = 0

    def load_dataset(self, config):
        '''
        Load the data into the class attributes. 

        :param config : A YAML object with user-entered parameters. One of these
                        parameters is the full path to an HDF5 file with the entire
                        dataset to process.
                        See the config_file parameter in the main script for more
                        details.
        '''
        # Get the parameters from the config file
        input_dir  = config.get_entry("global", "input_dir")
        input_file = config.get_entry("global", "input_file")

        # Read the data from HDF5
        hdf = tb.open_file("%s/%s" % (input_dir, input_file), mode='r')
        self.genotype = hdf.root.GTBox.gt[:]
        self.clin_covariate = hdf.root.GTBox.covar[range(1,12),:]
        self.regular_covariate = hdf.root.GTBox.covar[[0,], :]
        self.labels = (hdf.root.GTBox.lbl[:] + 1) / 2
        
        # Get the number of samples
        self.num_samples = self.labels.shape[1]
        

    def add_fold_information(self, config):
        '''
        Load the information about the random folds.

        :param config : A YAML object with user-entered parameters. One of these
                        parameters is the subdirectory where the cross-validation
                        data is stored.
                        See the config_file parameter in the main script for more
                        details.
        '''
        # Add the fold information to the Dataset object
        self.folds = config.load_variable("cv_set_creation", "folds")
        self.num_folds = 1 if (self.folds.ndim == 1) else self.folds.shape[1]

