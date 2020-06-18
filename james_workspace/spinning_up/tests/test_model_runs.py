import unittest
import tempfile

from env import CartPoleStandUp
from models import DQNSolver, VPGSolver, VPGSolverWithMemory

class TestModelsRun(unittest.TestCase):

    env = CartPoleStandUp(
        score_target=195., episodes_threshold=100, reward_on_fail=-10.)

    outdir = tempfile.TemporaryDirectory()

    std_agent_args = ("test", env.observation_space, env.action_space)

    def test_dqn_train(self):
        agent = DQNSolver(*self.std_agent_args)
        agent.solve(self.env, 1, verbose=True, render=False)

    def test_dqn_show(self):
        agent = DQNSolver(*self.std_agent_args)
        agent.show(self.env)

    def test_vpg_train(self):
        agent = VPGSolver(*self.std_agent_args)
        agent.solve(self.env, 1, verbose=True, render=False)

    def test_vpg_show(self):
        agent = VPGSolver(*self.std_agent_args)
        agent.show(self.env)

    def test_vpg_with_memory_train(self):
        agent = VPGSolverWithMemory(*self.std_agent_args)
        agent.solve(self.env, 1, verbose=True, render=False)

    def test_vpg_with_memory_show(self):
        agent = VPGSolverWithMemory(*self.std_agent_args)
        agent.show(self.env)
