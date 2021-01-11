""" Test module (this is not a package or subpackage) """

import unittest
import sys
import numpy as np
import copy
sys.path.append("..")
from rubble.rubble import Rubble
from rubble.rubble_data import RubbleData

class BasicTestSuite(unittest.TestCase):
    """ Basic test cases. """

    def test_rubble_with_example_paras(self):
        """ simple test """

        # test a quick full model simulation
        alpha = np.logspace(-3, -2, 5)
        u_f = np.logspace(2, 3, 5)
        npts = 5
        Rs_sim = 1.8
        Bs_sim = 1.5
        paras = np.array([
            [1.5, 1.8, 1.973854786578862e-09, 0.09785230336148297, 42928745810.494060, 0.02932592607042491,
             1578.7776283985802, 1116.6342886511695, 11.166342886511696, 0.010391914679865373],
            [1.5, 1.8, 1.973854786578862e-09, 0.09785230336148297, 40527349387.710686, 0.02768545946394155,
             1407.0870424148045, 704.5486055228718, 7.045486055228718, 0.011007674336371951],
            [1.5, 1.8, 1.973854786578862e-09, 0.09785230336148297, 38260284976.507935, 0.026136759121907074,
             1254.067646588032, 444.5401171979344, 4.445401171979344, 0.01165992004634043],
            [1.5, 1.8, 1.973854786578862e-09, 0.09785230336148297, 36120038161.871216, 0.024674691719901685,
             1117.6889665047618, 280.48585186212256, 2.804858518621226, 0.01235081374435545],
            [1.5, 1.8, 1.973854786578862e-09, 0.09785230336148297, 34099514878.577137, 0.023294411087175403,
             996.1413399391055, 176.9746082551897, 1.7697460825518971, 0.013082645467679368]])
        Rs, Bs, Mdot, R_accu, H, HoverR, T, Sigma_g, Sigma_d, a_critD = paras.T

        idx = 0
        u_idx = 3
        r = Rubble(151, 1e-3, 1e4, 3.5, Sigma_d[idx],
                   ranged_dist=[1e-3, 1e2],
                   Sigma_g=Sigma_g[idx], H=H[idx], T=T[idx], alpha=alpha[idx], u_f=u_f[u_idx],
                   Raccu=R_accu[idx], Z=0.01, Mdot=Mdot[idx], a_critD=a_critD[idx], a_max_in=100
                   )

        r.full_St_flag = True
        r.closed_box_flag = False
        r.run(200, 1, 1)

        rd2compare = RubbleData("data2compare.dat")
        rd = RubbleData("rubble_test.dat")
        sigma_ratio = np.divide(rd2compare.sigma, rd.sigma, where=rd.sigma>0)
        sigma_ratio[sigma_ratio == 0] = 1.0

        assert(sigma_ratio.min() > 0.999)
        assert(sigma_ratio.max() < 1.001)

        rd.shrink_data("rubble_test_shrinked.dat", 2, keep_first_n=10)
        rd_shrinked = RubbleData("rubble_test_shrinked.dat")

        assert(rd_shrinked.t.size == 106)
        assert(np.all(rd_shrinked.sigma[10] == rd.sigma[10]))
        assert(np.all(rd_shrinked.sigma[11] == rd.sigma[12]))

        assert(1 > 0)


if __name__ == '__main__':
    unittest.main()
