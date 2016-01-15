import os
import yaml
import numpy as np 

class ConfigState:
    '''
    A class object that contains the parameters to run the pipeline and allows
    to save/load variables that maintain state information.

    Properties:
    config : A configuration object read from a file in YAML format.
    '''

    def __init__(self):
        self.config = []


    def __init__(self, config_file):
        '''
        Constructor when the path to the configuration file is given
        
        Parameters
        ----------
        config_file : full path to the file
        '''
        self.config = yaml.load(file(config_file, 'r'))


    def load_configuration(config_file):
        '''
        Function to read the content of a configuration file in YAML format.
        
        Parameters
        ----------
        config_file : full path to the file
        '''
        self.config = yaml.load(file(config_file, 'r'))


    def get_entry(self, group_name, key_name):
        '''
        Function to retrieve an entry from the configuration file.
        
        Parameters
        ----------
        group_name : In YAML, the higher entry in the hierarchy

        key_name : In YAML, the key within the group
        '''
        return self.config[group_name][key_name]


    def _create_directory(self, path_dir):
        '''
        Function to create a directory if it does not previously exist.
    
        Parameters
        ----------
        path_dir : the full path of the directory
        '''
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)


    def save_variable(self, task_name, format_string=None, **kwargs):
        '''
        Function to save a variable as a text file (tab-separated). The name of the
        file will correspond with the name of the variable + extension. The
        extension is a parameter in the configuration file.
        
        Parameters
        ----------
        task_name : name of the task for which output data needs to be saved

        format_string : name of the task for which output data needs to be saved

        kwargs : keyword argument with the variable name(s) of the variable(s) to 
                 save
        '''
        # Parameters
        # Global output directory and subdirectory for this task
        output_dir = self.config["global"]["output_dir"]
        output_subdir = self.config[task_name]["output_subdir"]
        # Type of file to save
        extension = self.config["global"]["save_option"]    

        # Create the directory, only if necessary
        out_dir = "%s/%s" % (output_dir, output_subdir)
        self._create_directory(out_dir)

        # Set the delimiter
        if extension == "csv":
            delimiter_char = '\t'

        # Iterate through the variables and save them
        for var_obj in kwargs.items():
            # Get the variable name and its contents
            file_name = "%s/%s.%s" % (out_dir, var_obj[0], extension)
            np.savetxt(file_name, var_obj[1], fmt=format_string, delimiter=delimiter_char)


    def load_variable(self, task_name, var_name):
        '''
        Function to load a variable that was saved as a text file (tab-separated).
        The name of the file with the contents of the variable corresponds will correspond with the name of the variable + extension. The
        extension is a parameter in the configuration file.
        
        Parameters
        ----------
        task_name : name of the task for which saved variable needs to be restored

        kwargs : keyword argument with the variable name(s) of the variable(s) to 
                 load
        '''
        # Parameters
        # Global output directory and subdirectory for this task
        input_dir = self.config["global"]["output_dir"]
        input_subdir = self.config[task_name]["output_subdir"]
        # Type of file to save
        extension = self.config["global"]["save_option"]    

        # Create the directory, only if necessary
        in_dir = "%s/%s" % (input_dir, input_subdir)

        # Set the delimiter
        if extension == "csv":
            delimiter_char = '\t'

        # Get the variable name and its contents
        file_name = "%s/%s.%s" % (in_dir, var_name, extension)
        return np.loadtxt(file_name, delimiter=delimiter_char)

