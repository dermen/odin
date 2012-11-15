#!/usr/bin/env python

"""
Reference implementation & unit test for the GPU scattering simulation code
(aka gpuscatter, in odin/src/cuda/gpuscatter.*).
"""

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_almost_equal, assert_allclose

import gpuscatter
from odin.data import cromer_mann_params
from odin import xray
from odin.xray import Detector
from odin.structure import rand_rotate_molecule
from odin.testing import skip

from mdtraj import trajectory



# ------------------------------------------------------------------------------
#                        BEGIN REFERENCE IMPLEMENTATION
# ------------------------------------------------------------------------------

def form_factor(qvector, atomz):
    
    mq = np.sum( np.power(qvector, 2) )
    qo = mq / (16.*np.pi*np.pi)
    
    if ( atomz == 1 ):
        fi = 0.493002*np.exp(-10.5109*qo)
        fi+= 0.322912*np.exp(-26.1257*qo)
        fi+= 0.140191*np.exp(-3.14236*qo)
        fi+= 0.040810*np.exp(-57.7997*qo)
        fi+= 0.003038
    
    elif ( atomz == 8):
        fi = 3.04850*np.exp(-13.2771*qo)
        fi+= 2.28680*np.exp(-5.70110*qo)
        fi+= 1.54630*np.exp(-0.323900*qo)
        fi+= 0.867000*np.exp(-32.9089*qo)
        fi+= 0.2508

    elif ( atomz == 26):
        fi = 11.7695*np.exp(-4.7611*qo)
        fi+= 7.35730*np.exp(-0.307200*qo)
        fi+= 3.52220*np.exp(-15.3535*qo)
        fi+= 2.30450*np.exp(-76.8805*qo)
        fi+= 1.03690

    elif ( atomz == 79):
        fi = 16.8819*np.exp(-0.4611*qo)
        fi+= 18.5913*np.exp(-8.6216*qo)
        fi+= 25.5582*np.exp(-1.4826*qo)
        fi+= 5.86*np.exp(-36.3956*qo)
        fi+= 12.0658
        
    # else approximate with Nitrogen
    else:
        fi = 12.2126*np.exp(-0.005700*qo)
        fi+= 3.13220*np.exp(-9.89330*qo)
        fi+= 2.01250*np.exp(-28.9975*qo)
        fi+= 1.16630*np.exp(-0.582600*qo)
        fi+= -11.529
        
    return fi

    
def ref_simulate_shot(xyzlist, atomic_numbers, num_molecules, q_grid, rfloats=None):
    """
    Simulate a single x-ray scattering shot off an ensemble of identical
    molecules.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 2d
        An n x 3 array of each atom's position in space
        
    atomic_numbers : ndarray, float, 1d
        An n-length list of the atomic numbers of each atom
        
    num_molecules : int
        The number of molecules to include in the ensemble.
        
    q_grid : ndarray, float, 2d
        An m x 3 array of the q-vectors corresponding to each detector position.
    
    Optional Parameters
    -------------------
    rfloats : ndarray, float, n x 3
        A bunch of random floats, uniform on [0,1] to be used to seed the 
        quaternion computation.
        
    Returns
    -------
    I : ndarray, float
        An array the same size as the first dimension of `q_grid` that gives
        the value of the measured intensity at each point on the grid.
    """
    
    I = np.zeros(q_grid.shape[0])
    
    for n in range(num_molecules):
        
        if rfloats == None:
            rotated_xyzlist = rand_rotate_molecule(xyzlist)
        else:
            rotated_xyzlist = rand_rotate_molecule(xyzlist, rfloat=rfloats[n,:])
        
        for i,qvector in enumerate(q_grid):

            # compute the molecular form factor F(q)
            F = 0.0
            for j in range(xyzlist.shape[0]):
                fi = form_factor(qvector, atomic_numbers[j])
                r = rotated_xyzlist[j,:]
                F1 = fi * np.exp( 1j * np.dot(qvector, r) )
                x = np.dot(qvector, r)
                F += fi * np.exp( 1j * np.dot(qvector, r) )
    
            I[i] += F.real*F.real + F.imag*F.imag

    return I


# ------------------------------------------------------------------------------
#                           END REFERENCE IMPLEMENTATION
# ------------------------------------------------------------------------------
#                           BEGIN INTERFACE TO gpuscatter
# ------------------------------------------------------------------------------

def call_gpuscatter(xyzlist, atomic_numbers, num_molecules, qgrid, rfloats):
    """
    Calls the GPU version of the scattering code. This is a simplified interface
    with no random elements for unit testing purposes only.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 2d
        An n x 3 array of each atom's position in space
        
    atomic_numbers : ndarray, float, 1d
        An n-length list of the atomic numbers of each atom
        
    num_molecules : int
        The number of molecules to include in the ensemble.
        
    q_grid : ndarray, float, 2d
        An m x 3 array of the q-vectors corresponding to each detector position.

    rfloats : ndarray, float, n x 3
        A bunch of random floats, uniform on [0,1] to be used to seed the 
        quaternion computation.
        
    Returns
    -------
    I : ndarray, float
        An array the same size as the first dimension of `q_grid` that gives
        the value of the measured intensity at each point on the grid.
    """

    device_id = 0

    assert(num_molecules % 512 == 0)
    bpg = int(num_molecules / 512)
    
    # get detector
    qx = qgrid[:,0].astype(np.float32)
    qy = qgrid[:,1].astype(np.float32)
    qz = qgrid[:,2].astype(np.float32)
    num_q = len(qx)
    
    # get atomic positions
    rx = xyzlist[:,0].astype(np.float32)
    ry = xyzlist[:,1].astype(np.float32)
    rz = xyzlist[:,2].astype(np.float32)
    num_atoms = len(rx)
    assert( num_atoms == 512 )
    
    aid = atomic_numbers.astype(np.int32)
    atom_types = np.unique(aid)
    num_atom_types = len(atom_types)
    
    # get cromer-mann parameters for each atom type
    cromermann = np.zeros(9*num_atom_types, dtype=np.float32)
    for i,a in enumerate(atom_types):
        ind = i * 9
        cromermann[ind:ind+9] = cromer_mann_params[(a,0)]
        aid[ aid == a ] = i # make the atom index 0, 1, 2, ...
 
    # get random numbers
    rand1 = rfloats[:,0].astype(np.float32)
    rand2 = rfloats[:,1].astype(np.float32)
    rand3 = rfloats[:,2].astype(np.float32)
    
    # run dat shit
    print "cm", cromermann
    out_obj = gpuscatter.GPUScatter(device_id,
                                    bpg, qx, qy, qz,
                                    rx, ry, rz, aid,
                                    cromermann,
                                    rand1, rand2, rand3, num_q)
    
    output = out_obj.this[1].astype(np.float64)
    return output

# ------------------------------------------------------------------------------
#                           END INTERFACE TO gpuscatter
# ------------------------------------------------------------------------------
#                              BEGIN nosetest CLASS
# ------------------------------------------------------------------------------

    
class TestScatter():
    """ test all the scattering simulation functionality """
    
    def setup(self):
        
        self.refdir = '/home/tjlane/programs/odin/test/reference/' # TJL todo generalize
        self.nq = 1 # number of detector vectors to do
        
    def test_gpu_scatter(self):
        print "testing c code..."
        
        xyzQ = np.loadtxt(self.refdir + '512_atom_benchmark.xyz')
        xyzlist = xyzQ[:,:3]
        atomic_numbers = xyzQ[:,3].flatten()
    
        q_grid = np.loadtxt(self.refdir + '512_q.xyz')[:self.nq]
    
        rfloats = np.loadtxt(self.refdir + '512_x_3_random_floats.txt')
        num_molecules = rfloats.shape[0]
    
        gpu_I = call_gpuscatter(xyzlist, atomic_numbers, num_molecules, q_grid, rfloats)
        self.ref_I = ref_simulate_shot(xyzlist, atomic_numbers, num_molecules, q_grid, rfloats)

        print "GPU", gpu_I
        print "CPU", self.ref_I
        
        assert_allclose(gpu_I, self.ref_I, rtol=1e-03,
                        err_msg='scatter: gpu/cpu reference mismatch')
        
                            
    def test_python_call(self):
        print "testing python wrapper fxn..."
        
        traj = trajectory.load(self.refdir + 'ala2.pdb')
        num_molecules = 512
        detector = Detector.generic()

        py_I = xray.simulate_shot(traj, num_molecules, detector, verbose=True)[:self.nq]
        
        # this won't be equal because it's random....
        assert_allclose(py_I, self.ref_I, rtol=1e1,
                        err_msg='scatter: python interface in xray.py broke')
       

if __name__ == '__main__':
    t = TestScatter()
    t.setup()
    t.test_gpu_scatter()
    t.test_python_call()
