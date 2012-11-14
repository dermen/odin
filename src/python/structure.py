

"""
structure.py

Functions/classes for manipulating structures.
"""

import numpy as np
from scipy.special import cbrt


class quaternion(object):
    """
    Container class for quaternion-based functions. All methods of this class
    are static, and there is no concept of an instance of this class. It is
    really just a namespace deliminator.
    """
    
    @staticmethod
    def random(rfloat=None):
        """
        Compute a quaterion representing a random rotation, uniform on the
        unit sphere.
        
        Optional Parameters
        -------------------
        rfloat : ndarray, float, len 3
            A 3-vector of random numbers in [0,1) that acts as a random seed. If
            not passed, generates new random numbers.
            
        Returns
        -------
        q : ndarray, float, len 4
            A quaternion representing a random rotation uniform on the unit 
            sphere.
        """
    
        if rfloat == None:
            rfloat = np.random.rand(3)
    
        q = np.zeros(4)
    
        s = rfloat[0]
        sig1 = np.sqrt(s)
        sig2 = np.sqrt(1.0 - s)
    
        theta1 = 2.0 * np.pi * rfloat[1]
        theta2 = 2.0 * np.pi * rfloat[2]
    
        w = np.cos(theta2) * sig2
        x = np.sin(theta1) * sig1
        y = np.cos(theta1) * sig1
        z = np.sin(theta2) * sig2
    
        q[0] = w
        q[1] = x
        q[2] = y
        q[3] = z
    
        return q

    @staticmethod
    def prod(q1, q2):
        """
        Perform the Hamiltonian product of two quaternions. Note that this product
        is non-commutative -- this function returns q1 x q2.
    
        Parameters
        ----------
        q1 : ndarray, float, len(4)
            The first quaternion.
    
        q1 : ndarray, float, len(4)
            The first quaternion.
        
        Returns
        -------
        qprod : ndarray, float, len(4)
            The Hamiltonian product q1 x q2.
        """
        
        if (len(q1) != 4) or (len(q2) != 4):
            raise TypeError('Parameters cannot be interpreted as quaternions')
    
        qprod = np.zeros(4)
    
        qprod[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        qprod[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        qprod[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        qprod[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    
        return qprod
    
    @staticmethod
    def conjugate(q):
        """
        Compute the quaternion conjugate of q.
        
        Parameters
        ----------
        q : ndarray, float, len 4
            A quaternion input.
            
        Returns
        qconj : ndarray, float, len 4
            The conjugate of `q`.
        """
        
        if len(q) != 4:
            raise TypeError('Parameter `q` cannot be interpreted as a quaternion')
    
        qconj = np.zeros(4)
        qconj[0] = q[0]
        qconj[1] = -q[1]
        qconj[2] = -q[2]
        qconj[3] = -q[3]
    
        return qconj

    @staticmethod
    def rand_rotate_vector(v):
        """
        Randomly rotate a three-dimensional vector, `v`, uniformly over the unit
        sphere.
    
        Parameters
        ----------
        v : ndarray, float, len 3
            A vector to rotatea 3-vector in x,y,z space (e.g. the atomic 
            positions of an atom)
            
        Returns
        -------
        v_prime : ndarray, float, len 3
            Another 3-vector, which is the rotated version of v.
        """
        
        if len(v) != 4:
            raise TypeError('Parameter `v` must be in R^3')
        
        # generate a quaternion vector, with the first element zero
        # the last there elements are from v
        qv = np.zeros(4)
        qv[1:] = v.copy()
    
        # get a random quaternion vector
        q = quaternion.random()
        qconj = quaterion.conjugate(q)
    
        q_prime = quaternion.prod( quaternion.prod(q, qv), qconj )
    
        v_prime = q_prime[1:].copy() # want the last 3 elements...
    
        return v_prime


def remove_COM(traj):
    """
    Remove the center of mass from all frames in a trajectory.
    
    Parameters
    ----------
    traj : mdtraj.trajectory
        A trajectory object.
        
    Returns
    -------
    centered_traj : mdtraj.trajectory
        A trajectory with the center of mass removed
    """
    
    for i in range(traj.n_frames):
        masses = [ a.element.mass for a in traj.topology.atoms() ]
        traj.xyz[i,:,:] -= np.average(traj.xyz[i,:,:], axis=0, weights=masses)
        
    return traj
    

def rand_rotate_molecule(xyzlist, rfloat=None):
    """
    Randomly rotate the molecule defined by xyzlist.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 3D
        An n x 3 array representing the x,y,z positions of n atoms.
        
    rfloat : ndarray, float, len 3
        A 3-vector of random numbers in [0,1) that acts as a random seed. If
        not passed, generates new random numbers.
        
    Returns
    -------
    rotated_xyzlist : ndarray, float, 3D
        A rotated version of the input `xyzlist`.
    """

    # get a random quaternion vector
    q = quaternion.random(rfloat)
    
    # take the quaternion conjugate
    qconj = quaternion.conjugate(q)
    
    # prepare data structures
    rotated_xyzlist = np.zeros(xyzlist.shape)
    qv = np.zeros(4)
    
    # put each atom through the same rotation
    for i in range(xyzlist.shape[0]):
        qv[1:] = xyzlist[i,:].copy()
        q_prime = quaternion.prod( quaternion.prod(q, qv), qconj )
        rotated_xyzlist[i,:] = q_prime[1:].copy() # want the last 3 elements...
    
    return rotated_xyzlist
    
    
def rand_rotate_traj(traj, remove_COM=False):
    """
    Randomly rotate all the members of a trajectory.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 3D
        An n x 3 array representing the x,y,z positions of n atoms.
    
    Optional Parameters
    -------------------
    remove_COM : bool
        Whether or not to translate the center of mass of the molecule to the
        origin before rotation.
        
    
    """
    
    if remove_COM:
        traj = remove_COM(traj)
        
    for i in range(traj.n_frames):
        traj.xyz = rand_rotate_molecule(traj.xyz)
    
    return traj


def multiply_conformations(traj, num_replicas, density, traj_weights=None):
    """
    Take a structure and generate a system of many conformations, such that they
    are randomly distributed & rotated in space with a given `density`.
    
    This function is useful for approximating a subset of a homogenous solution,
    gas, slurry, etc. composed of the structures indicated by `trajectory`.
    These structures can be given e.g. Boltzmann weights by using the
    `traj_weights` argument.
    
    Parameters
    ----------
    traj : mdtraj
        The structures to use to generate the system. Note this can be a single
        conformation, in which case each molecule in the system is identical,
        just translated & rotated.
    
    num_replicas : int
        The total number of molecules to include in the system. The total volume
        of the system depends on this parameter and `density`.
        
    density : float
        The number density of species, MICROMOLAR UNITS. That is, umol/L. (This
        software was written by a chemist!).
    
    Optional Parameters
    -------------------
    traj_weights : ndarray, float
        The weights at which to include members of trajectory in the final
        system. Default is to assign equal weight to all members of trajectory.
    
    Returns
    -------
    system_structure : mdtraj
        Length-one trajectory representing the coordinates of the molecular
        system.
    """

    # TODO: 
    #   -- remove center of mass from all frames
    
    # read out some stuff from the trajectory
    #       initialize system_structure

    
    # check traj_weights
    if traj_weights != None:
        if len(traj_weights) != traj.n_frames:
            raise ValueError('Length of `traj_weights` is not the same as `traj`')
    else:
        traj_weights = np.ones(traj.n_frames)
    traj_weights /= traj_weights.sum() # normalize
        
    # generate a random ensemble, defined by a list of indices of `traj`
    num_per_shapshot = np.random.multinomial(num_replicas, traj_weights, size=1)
    
    # determine the box size
    boxvol  = (num_replicas * 1.0e24) / (density * 6.02e17) # in nm^3
    boxsize = cbrt(boxvol)            # one dim of a cubic box, in nm

    # find the maximal radius of each snapshot in traj
    max_radius = np.zeros( len(traj) )
    for i,snapshot in enumerate(traj):
        max_radius[i] = np.max( np.sqrt(np.sum(np.power(snapshot.xyz, 2), axis=3 )) )
        
    # place in space
    snapshots = traj[num_per_shapshot]
    centers_of_mass = np.zeros((num_replicas, 3)) # to store these and use later
    
    for i in range(snapshots):
        molecule_overlapping = True # initial cond.
        
        while molecule_overlapping:
        
            # do a random translation & rotation
            R = np.random.uniform(low=0, high=boxsize, size=3)
            centers_of_mass[i,:] = R
            for x in range(3):
                snapshot.xyzlist[i,:] += R[i]
            snapshot = rand_rotate_molecule(snapshot)
            
            # check to see if we're overlapping another molecule already placed
            for j in range(i):
                molec_dist = np.linalg.norm(centers_of_mass[i] - centers_of_mass[j])
                max_allowable_dist = max_radius[i] + max_radius[j]
                if molec_dist > max_allowable_dist:
                    molecule_overlapping = False
                else:
                    molecule_overlapping = True
                    
        system_structure[i] = snapshot

    return
    
