
from odin import xray
from odin.interp import Bcinterp
from odin.testing import skip, ref_file

import numpy as np
from scipy import interpolate
import scipy.ndimage.interpolation as interpolation
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal
 

class TestBcinterp():
    
    def setup(self):
                
        # initialize parameters
        dim1 = 100
        dim2 = 100
        
        self.vals = np.abs(np.random.randn(dim1 * dim2))
        self.x_space = 0.1
        self.y_space = 0.1
        self.Xdim = dim1
        self.Ydim = dim2
        self.x_corner = 0.0
        self.y_corner = 0.0
                
        self.interp = Bcinterp( self.vals, 
                                self.x_space, self.y_space, 
                                self.Xdim, self.Ydim,
                                self.x_corner, self.y_corner )
                                
        # generate a reference -- need better reference
        self.new_x = np.arange(0.0, 8.0, .01) + 0.1
        self.new_y = np.arange(0.0, 8.0, .01) + 0.1


    def test_for_smoke(self):
        ip = self.interp.evaluate(self.new_x, self.new_y)
        if np.all( ip == 0.0 ):
            print "Interpolator not working, likely cause: OMP failure."
            print "Try reinstalling with no OMP: python setup.py install --no-openmp"
            raise Exception()
    
    def test_point_vs_known(self):
        interp = Bcinterp( np.arange(1000**2),
                           0.1, 0.1, 1000, 1000, 0.0, 0.0 )
                           
        # the below were generated from a ref implementation, but can also
        # be evaluated readily by eye
        
        i = interp.evaluate(1.01, 1.01)
        assert_almost_equal(i, 10110.1, decimal=1)
        
        i = interp.evaluate(1.5, 1.5)
        assert_almost_equal(i, 15015.0)
        
        i = interp.evaluate(2.3, 2.0)
        assert_almost_equal(i, 20023.0)
        
        i = interp.evaluate(20.3, 2.0)
        assert_almost_equal(i, 20203.0)
        
    def test_two_dim(self):
        
        xa = np.array([1.5, 20.3])
        ya = np.array([1.5, 2.0])
        
        interp1 = Bcinterp( np.arange(1000**2),
                           0.1, 0.1, 1000, 1000, 0.0, 0.0 )
                           
        interp2 = Bcinterp( np.arange(1000**2).reshape(1000,1000),
                           0.1, 0.1, 1000, 1000, 0.0, 0.0 )
                           
        assert_array_almost_equal(interp1.evaluate(xa, ya), interp2.evaluate(xa, ya))
        
    def test_trivial(self):
        interp = Bcinterp( np.ones(10*10), 1.0, 1.0, 10, 10, 0.0, 0.0 )
        interp_vals = interp.evaluate( np.arange(1,9) + 0.1, np.ones(8) + 0.1 )
        assert_allclose(np.ones(8), interp_vals)
        
    def test_array_vs_known(self):

        interp = Bcinterp( np.arange(1000**2),
                           0.1, 0.1, 1000, 1000, 0.0, 0.0 )
                           
        xa = np.array([1.5, 20.3])
        ya = np.array([1.5, 2.0])
        
        i = interp.evaluate(xa, ya)
        assert_almost_equal( i, np.array([15015.0, 20203.0]) )
        
    def test_against_scipy(self):
        x_size = 1000
        y_size = 1000
        n_evals = 10

        ex = np.random.randint(0, x_size-2, n_evals) + 0.5
        ey = np.random.randint(0, y_size-2, n_evals) + 0.5

        a = np.arange(x_size * y_size).reshape(x_size, y_size).astype(np.float)

        bc = Bcinterp(a.flatten(), 1.0, 1.0, x_size, y_size, 0.0, 0.0) ; out1 = bc.evaluate(ex, ey)
        out2 = interpolation.map_coordinates(a, [ey, ex], order=3)

        assert np.all( (out1-out2) < 1e-8 )
        
