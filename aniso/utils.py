import numpy as np
from scipy.ndimage.filters import gaussian_filter

# math utilities 
def eigsorted(A):                                                               
    """                                                                         
    Return sorted eigenvalues and eigenvectors.                                 
                                                                                 
    :param A: A must be a symmetric/hermitian matrix                            
    :return: vals - sorted eigenvalues                                          
             vecs - corresponding eigenvectors                                  
    """                                                                         
    vals, vecs = np.linalg.eigh(A)                                              
    order = vals.argsort()[::-1]                                                
    return vals[order], vecs[:, order]


def gridsmooth(Z, span):                                                        
    """ Smooths values on 2D rectangular grid                                   
    """                                                                                                             
    if isinstance(span, list):                                                  
        return gaussian_filter(Z, np.array(span)/2.)                            
    else:                                                                       
        return gaussian_filter(Z, span/2.) 


def as_ndarrays(*args):
    """ Convert a list of array_like objects to numpy arrays.

    Parameters
    ----------
    args: list of array_like objects

    Returns
    -------
    list of ndarrays
    """
    if len(args) == 1:
        return list(map(np.asarray, args))[0]
    else:
        return list(map(np.asarray, args))


def check_2d(*args):
    """ Check that ndarrays are 2-dimensional.
    args: list of ndarrays.
    """
    for i, item in enumerate(args):
        if i == 0:
            ref_shape = item.shape
        if item.ndim != 2:
            raise ValueError('Array must be 2-dimensional.')
        if item.shape != ref_shape:
            raise ValueError('Dimensions mismatch in 2D arrays.')