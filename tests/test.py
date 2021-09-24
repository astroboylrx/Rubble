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

        assert(1 > 0)


if __name__ == '__main__':
    unittest.main()
