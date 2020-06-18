import unittest
import tempfile

from env import CartPoleStandUp
from models import DQNSolver, VPGSolver, VPGSolverWithMemory

class test_models_run(unittest.TestCase):

    cart = CartPoleStandUp(
        score_target=195., episodes_threshold=100, reward_on_fail=-10.)

    outdir = tempfile.TemporaryDirectory()

    std_args = ("test", cart.observation_space, cart.action_space)


    def test_dqn_training(self):
        agent = VPGSolver(*self.std_args)
        solved = agent.solve(self.cart, 1, verbose=True, render=False)

    def test_vpg_training(self):
        agent = VPGSolverWithMemory(*self.std_args)
        solved = agent.solve(self.cart, 1, verbose=True, render=False)
    
    def test_vpg_with_memory_training(self):
        agent = VPGSolverWithMemory(*self.std_args)
        solved = agent.solve(self.cart, 1, verbose=True, render=False)
