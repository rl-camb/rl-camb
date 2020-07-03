import copy, os, pickle, pprint
import matplotlib.pyplot as plt
import numpy as np

from env import CartPoleStandUp
from models import (
    DQNSolver, VPGSolver, VPGSolverWithMemory, A2CSolver, PPOSolver,
    DDPGSolver
)
from utils import MyParser


class RepeatExperiment():
    """A handler for a repeat experiment you may want to do"""

    def __init__(self, experiment_location, score_target=195., 
        max_episodes=2000, episodes_threshold=100, verbose=False, 
        render=False):

        self.experiment_location = os.sep.join(
            ("experiments", experiment_location))

        self.max_episodes = max_episodes
        self.score_target = score_target
        self.episodes_threshold = episodes_threshold
        self.verbose = verbose
        self.render = render

    def initialise_experiment(self, experiment_name):

        self.experiment_name = experiment_name

        self.experiment_dir = os.sep.join(
            (self.experiment_location, self.experiment_name))
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.exp_dict_file = os.sep.join(
            (self.experiment_dir, "exp_dict.p"))

        self.exp_dict = self.load_experiment_from_file(self.exp_dict_file)

        if not self.exp_dict:
            self.exp_dict = {k: [] for k in ("solves", "episodes", "times")}
            self.exp_dict["max_episodes"] = self.max_episodes

        if self.max_episodes != self.exp_dict["max_episodes"]:
            self.max_episodes = self.exp_dict["max_episodes"]
            print(f"WARN: Must keep number of episodes same if continuing an "
                  f"experiment. Set to {self.max_episodes} "
                  f"(specified {self.max_episodes}")

    def repeat_experiment(self, env_wrapper, agent_init, repeats=1):
        """
        Repeat training for a given agent and save the results 
        in a reusable format"""
        if repeats < 1:
            self.exp_dict = self.load_experiment_from_file(
                self.exp_dict_file)

        for r in range(repeats):
            print(f"---\nRepeat {r + 1} / {repeats}\n")

            agent = agent_init()

            solved = agent.solve(
                env_wrapper, self.max_episodes,
                verbose=True, render=self.render)

            print("\nSolved:", solved, " after", agent.solved_on, 
                  "- time elapsed:", agent.elapsed_time)

            self.exp_dict["solves"].append(solved)
            self.exp_dict["episodes"].append(agent.solved_on)
            self.exp_dict["times"].append(agent.elapsed_time.seconds)

        print("FINISHED")
        pprint.pprint(self.exp_dict, compact=True)
        print("Writing results to ", self.exp_dict_file)
        with open(self.exp_dict_file, 'wb') as d:
            pickle.dump(self.exp_dict, d)

        return self.exp_dict

    @staticmethod
    def load_experiment_from_file(exp_file):

        if os.path.exists(exp_file):
            print("Loading", exp_file)
            with open(exp_file, 'rb') as d:
                exp_dict = pickle.load(d)
        else:
            print("File", exp_file, "does not exist.")
            exp_dict = {}

        return exp_dict

    def plot_episode_length_comparison(self, compare_dirs):
        """
        Plots a box plot comparison of the number of episodes 
        taken to solve for each of the experimetns specified.

        compare_dirs:
          1. ["all"], then all experiments in self.experiment_location
            are found
          2. Specify a list of experiment dirs to load pickle dicts from,
           and compares plots them against each other (e.g. for subsets)
        """

        fig, ax = plt.subplots()
        ep_lengths_per_solve = []
        time_per_batch = []
        num_solves = []
        titles = []
        pickle_dicts = []

        # Collect the dirs we want to compare
        if compare_dirs == ["all"]:
            print(f"Collecting experiments from {self.experiment_location}")
            exp_dirs = [
                os.sep.join((self.experiment_location, d)) 
                for d in os.listdir(self.experiment_location)
            ]
            exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
        else:
            exp_dirs = compare_dirs

        # Collect exp_dicts
        for d in exp_dirs:
            f = os.sep.join((d.rstrip(os.sep), "exp_dict.p"))
            if os.path.exists(f):
                print(f"Adding file {f} for comparison")
                pickle_dicts.append(f)
            else:
                print(f"Skipping {f} - does not exist.")

        for pickle_dict_file in pickle_dicts:

            titles.append(pickle_dict_file.split(os.sep)[-2])
            exp_dict = self.load_experiment_from_file(pickle_dict_file)
            if not exp_dict:
                print(f"WARN: Could not find {pickle_dict_file}, skipping.")
                continue

            # Read the dicts and extract useful information
            ep_lengths_per_solve.append([
                exp_dict["episodes"][i][0]
                for i in range(len(exp_dict["solves"]))
                if exp_dict["solves"][i]
            ])
            num_batches = [
                exp_dict["episodes"][i][0] if exp_dict["episodes"][i] is not None
                else exp_dict["max_episodes"]
                for i in range(len(exp_dict["solves"]))
            ]
            time_per_batch.append([
                exp_dict["times"][i] / num_batches[i]
                for i in range(len(exp_dict["times"]))
            ])
            num_solves.append(
                (sum(1 if s else 0 for s in exp_dict["solves"]), 
                 len(exp_dict["solves"]))
            )

        ax.set_title(f'Comparing experiments {self.experiment_location}')
        ax.boxplot(ep_lengths_per_solve)
        ax.set_xticklabels(titles, rotation=40)

        # Add upper axis text
        pos = np.arange(len(pickle_dicts)) + 1
        weights = ['bold', 'semibold']
        for tick, label in zip(range(len(pickle_dicts)), 
                               ax.get_xticklabels()):
            k = tick % 2
            label = "{0}\nTime {1:.3f} +/- {2:.3f}\nSolved {3}/{4}".format(
                titles[tick], np.mean(time_per_batch[tick]), 
                np.std(time_per_batch[tick]), num_solves[tick][0], 
                num_solves[tick][1]
            )

            ax.text(pos[tick], .90, label,
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center', size='x-small',
                weight=weights[tick%2]) # , color=box_colors[k])

        plt.savefig(os.sep.join((self.experiment_location, 'comparison.png')))
        plt.show()


def parse_args():

    parser = MyParser()

    parser.add_argument(
        "--location", type=str, required=False, default="latest_experiment",
        help="The top-level directory for the experiment being kicked off")

    parser.add_argument(
        "--max-episodes", type=int, default=2000,
        help="Max episodes for each repeat run in the experiment.")

    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Number of repeat experiments to perform for each "
             "agent in the experiment.")

    parser.add_argument(
        "--compare", type=str, nargs="*", 
        help="Which experiment dictionaries to compare number "
             "of runs for. Options: list of dicts or 'all'.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    cart = CartPoleStandUp(
            score_target=195., 
            episodes_threshold=100, 
            reward_on_fail=-10.)

    experiment = RepeatExperiment(
            experiment_location=args.location,
            max_episodes=args.max_episodes,
            score_target=cart.score_target,
            episodes_threshold=cart.episodes_threshold
    )
    
    # MAKE YOUR EXPERIMENT

    # Configs to iterate over
    agents = {
        "dqn": DQNSolver,
        "vpg": VPGSolver,
        "vpg_batch": VPGSolverWithMemory,
        "a2c": A2CSolver,
        "ppo": PPOSolver,
        "ddpg": DDPGSolver,
    }
    
    # Iterate over the configs
    for i, agent_name in enumerate(agents):
        # Initialise the experiment
        print(f"---\nAgent {agent_name}, ({i+1}/{len(agents)})")
        experiment.initialise_experiment(agent_name)

        # Set the agent to be run (using config params)
        agent_init = lambda : agents[agent_name](
            experiment.experiment_dir, 
            cart.observation_space, 
            cart.action_space, 
            saving=False
        )

        experiment.repeat_experiment(
            cart,
            agent_init,
            repeats=args.repeat)

        print("\nComplete\n")

    if args.compare is not None:
        experiment.plot_episode_length_comparison(args.compare)
