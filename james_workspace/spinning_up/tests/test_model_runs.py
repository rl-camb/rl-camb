import os
import tempfile
import unittest
import pytest

from env import CartPoleStandUp
from models import DQNSolver, VPGSolver, VPGSolverWithMemory, A2CSolver, PPOSolver

# agents = (DQNSolver, VPGSolver, VPGSolverWithMemory, A2CSolver PPOSolver)


# TODO - add testing that clone(model).save model = load is the same (e.g. saves and loads)
class TestModelsRun(unittest.TestCase):

    env = CartPoleStandUp(
        score_target=195., episodes_threshold=100, reward_on_fail=-10.)

    tmpdir = tempfile.TemporaryDirectory()
    outdir = tmpdir.name
    os.chdir(outdir)

    env.get_spaces(registry=False)

    std_agent_args = (
        "test",
        env.observation_space,
        env.action_space)

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

    def test_a2c_train(self):
        agent = A2CSolver(*self.std_agent_args)
        agent.solve(self.env, 1, verbose=True, render=False)

    @pytest.mark.skip(reason="show uses self.model - a2c doesn't have")
    def test_a2c_show(self):
        agent = A2CSolver(*self.std_agent_args)
        agent.show(self.env)

    # @pytest.mark.skip(reason="Problem with deepcopy, but hoping to change functionality later")
    def test_ppo_train(self):
        agent = PPOSolver(*self.std_agent_args)
        agent.solve(self.env, 1, verbose=True, render=False)

    @pytest.mark.skip(reason="show uses self.model - ppo doesn't have")
    def test_ppo_show(self):
        agent = PPOSolver(*self.std_agent_args)
        agent.show(self.env)


# TODO - complete
class TestAct(unittest.TestCase):

    env = CartPoleStandUp(
        score_target=195., episodes_threshold=100, reward_on_fail=-10.)

    def test_act_random(self):
        # test epsilon == 1 for agent in agents
        # state = env.reset, use state as the input
        pass

    def test_act_determine(self):
        # test epsilon == None
        pass

    def test_act_fails(self):
        # test epsilon < 0 or > 1 and test 
        # Also test a badly shaped state, like a batch or not matching action space
        pass