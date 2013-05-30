
"""
Cython wrapper for CPU scattering.
"""

import numpy as np
cimport numpy as np

from time import time
import os

from odin.refdata import get_cromermann_parameters

def output_sanity_check(intensities):
    """
    Perform a basic sanity check on the intensity array we're about to return.
    """
    
    # check for NaNs in output
    if np.isnan(np.sum(intensities)):
        raise RuntimeError('Fatal error, NaNs detected in scattering output!')
        
    # check for negative values in output
    if len(intensities[intensities < 0.0]) != 0:
        raise RuntimeError('Fatal error, negative intensities detected in scattering output!')
        
    return
    
                    
cdef extern from "conc.h":
    cdef cppclass C_CPUScatter "CPUScatter":
        C_CPUScatter(int    nQ_,
                     float* h_qx_,
                     float* h_qy_,
                     float* h_qz_,
                     int    nAtoms_,
                     float* h_rx_,
                     float* h_ry_,
                     float* h_rz_,
                     int*   h_id_,
                     int    nCM_,
                     float* h_cm_,
                     float* h_outQ_ ) except +

                     
cdef C_CPUScatter * cpu_scatter_obj
                    
                     
def simulate_shot(np.ndarray qxyz, np.ndarray rxyz, np.ndarray atomic_numbers):
    """
    Parameters
    ----------
    qxyz : ndarray, float
        An n x 3 array of the (x,y,z) positions of each q-vector describing
        the detector.
    
    rxyz : ndarray, float
        An m x 3 array of the (x,y,z) positions of each atom in the molecule.
    
    atomic_numbers : ndarray, int
        A 1d array of the atomic numbers of each atom (same len as `rxyz`).
    
        
    Returns
    -------
    self.intensities : ndarray, float
        A flat array of the simulated intensities, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    # A NOTE ABOUT ARRAY ORDERING
    # In what follows, for multi-dimensional arrays I often take the transpose
    # somewhat mysteriously. This is because the way C++ will loop over arrays
    # in c-order, but ODIN's arrays in python land are in "fortran" order
    
    # extract arrays from input
    
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_qxyz
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rxyz
    c_qxyz = np.ascontiguousarray(qxyz.T, dtype=np.float32)
    c_rxyz = np.ascontiguousarray(rxyz.T, dtype=np.float32)
    
    # get the Cromer-Mann parameters
    py_cromermann, py_aid = get_cromermann_parameters(atomic_numbers)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] c_cromermann
    c_cromermann = np.ascontiguousarray(py_cromermann, dtype=np.float32)
    
    cdef int[::1] c_aid = np.ascontiguousarray(py_aid, dtype=np.int32) # memory-view contiguous "C" array
    
    # initialize output array
    cdef np.ndarray[ndim=1, dtype=np.float32_t] h_outQ
    h_outQ = np.zeros(qxyz.shape[0], dtype=np.float32)
    
    # call the actual C++ code
    
    cpu_scatter_obj = new C_CPUScatter(qxyz.shape[0],
                               &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                               rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                               &c_aid[0], len(c_cromermann), &c_cromermann[0],
                               &h_outQ[0])
    del cpu_scatter_obj
                                   

    # deal with the output
    output_sanity_check(h_outQ)
    
    return h_outQ

