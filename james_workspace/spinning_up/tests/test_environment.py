import unittest
import tempfile

from env import CartPoleStandUp

class test_environment(unittest.TestCase):

    outdir = tempfile.TemporaryDirectory()

    def test_cartpole_random_runs(self):
        cart = CartPoleStandUp(
            score_target=195., episodes_threshold=100, reward_on_fail=-10.)

        cart.do_random_runs(1, 100, verbose=True)
