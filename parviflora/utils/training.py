import numpy as np

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])