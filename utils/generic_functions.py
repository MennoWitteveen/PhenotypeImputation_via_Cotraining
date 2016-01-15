#-----------------------------------------------------------------------------
# Generic functions, used by other modules
# 
# Authors: Menno Witteveen 
#          Damian Roqueiro
#-----------------------------------------------------------------------------

import numpy as np
      
def harden_labels(soft_labels, p_class):
    
    ranking = (-soft_labels).argsort() # (-).argsort() is a trick to yield a descending sort
    hard_labels = np.zeros(soft_labels.shape)
    hard_labels[ranking[0:round(p_class*float(soft_labels.shape[0]))]] = 1

    return hard_labels
    
def find_vec_entries_that_contain(x,y):
    z = np.zeros(x.shape,dtype=bool)
    for y_scal in y:
        z = (x == y_scal) + z
    entries = np.nonzero(z)[0]
    return entries

