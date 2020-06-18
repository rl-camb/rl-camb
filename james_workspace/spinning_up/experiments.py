import copy, os, pickle, pprint
import matplotlib.pyplot as plt
import numpy as np

from env import CartPoleStandUp
from models import DQNSolver, VPGSolver, VPGSolverWithMemory
from utils import MyParser


class RepeatExperiment():

    def __init__(self, experiment_name, experiment_dir, score_target=195., 
        max_episodes=3000, episodes_threshold=100, verbose=False, 
        render=False):
        
        self.experiment_name = experiment_name
        self.experiment_dir = ("experiments" + os.sep +
            self.experiment_name + "_" + experiment_dir + os.sep)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.exp_dict_file = self.experiment_dir + "exp_dict.p"
        self.max_episodes = max_episodes
        self.score_target = score_target
        self.episodes_threshold = episodes_threshold
        self.verbose = verbose
        self.render = render

        self.exp_dict = self.load_experiment_from_file(self.exp_dict_file)
        if not self.exp_dict:
            self.exp_dict = {k: [] for k in ("solves", "episodes", "times")}
            self.exp_dict["max_episodes"] = self.max_episodes

        if self.max_episodes != self.exp_dict["max_episodes"]:
            self.max_episodes = self.exp_dict["max_episodes"]
            print(f"WARN: Must keep number of episodes same if continuing an "
                  f"experiment. Set to {self.max_episodes} "
                  f"(specified {max_episodes}")

    def repeat_experiment(
        self, env_wrapper, agent_initialiser, repeats=1):

        if repeats < 1:
            self.exp_dict = self.load_experiment_from_file(self.exp_dict_file)

        for _ in range(repeats):
            agent = agent_initialiser(self.experiment_dir)

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
    
    def plot_episode_length_comparison(self, compare_to_dicts):

        fig, ax = plt.subplots()
        ep_lengths_per_solve = []
        time_per_batch = []
        num_solves = []
        titles = []

        if compare_to_dicts == ["all"]:
            experiment_location = os.sep.join(
                self.experiment_dir.split(os.sep)[:-2])
            print("Searching in experiment location:")
            print(experiment_location)
            pickle_dicts = [
                os.sep.join((experiment_location, d, "exp_dict.p")) for d in 
                os.listdir(experiment_location) if self.experiment_name in d
            ]
            print("Found experiments for comparison:")
            pprint.pprint(pickle_dicts)
        else:
            picke_dicts = [self.exp_dict_file] + compare_to_dicts

        for pickle_dict in pickle_dicts:
            
            titles.append(pickle_dict.split(os.sep)[-2])
            
            exp_dict = RepeatExperiment.load_experiment_from_file(pickle_dict)
            if not exp_dict:
                print("WARN: Could not find file " + pickle_dict + 
                      ", skipping.")
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

        ax.set_title('Comparing experiments')
        ax.boxplot(ep_lengths_per_solve)
        ax.set_xticklabels(titles, rotation=40)

        # Add upper axis text
        pos = np.arange(len(pickle_dicts)) + 1
        weights = ['bold', 'semibold']
        for tick, label in zip(range(len(pickle_dicts)), 
                               ax.get_xticklabels()):
            k = tick % 2
            label = ("{0}\nTime {1:.3f} +/- {2:.3f}\nSolved {3}/{4}").format(
                titles[tick], np.mean(time_per_batch[tick]), 
                np.std(time_per_batch[tick]), num_solves[tick][0], 
                num_solves[tick][1]
            )

            ax.text(pos[tick], .90, label,
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center', size='x-small',
                weight=weights[tick%2]) # , color=box_colors[k])

        plt.savefig(self.experiment_dir + "comparison.png")
        plt.show()


def parse_args():

    parser = MyParser()

    parser.add_argument("--outdir", type=str, required=False, default="outdir",
                        help="Suffix for experiment out dir")

    parser.add_argument("--max-episodes", type=int, default=2000,
                        help="Max episodes for the  experiment.")

    parser.add_argument("--repeat", type=int, default=0,
                        help="Number of repeat experiments to perform.")

    parser.add_argument("--compare", type=str, nargs="*", 
                        help="Which experiment dictionaries to compare number "
                             "of runs for. Options: list of dicts or 'all'.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    agents = {
        "dqn": DQNSolver, 
        "vpg": VPGSolver,
        "vpg_batch": VPGSolverWithMemory
    }

    for agent in agents:

        cart = CartPoleStandUp(
            score_target=195., episodes_threshold=100, reward_on_fail=-10.)

        agent_initialiser = lambda exp_dir: agents[agent](
            exp_dir, cart.observation_space, cart.action_space, saving=False)

        experiment = RepeatExperiment(
            experiment_name=args.outdir,
            experiment_dir=agent,
            max_episodes=args.max_episodes,
            score_target=cart.score_target,
            episodes_threshold=cart.episodes_threshold
        )

        if args.repeat > 0:
            experiment.repeat_experiment(
                cart,
                agent_initialiser,
                repeats=args.repeat)

    if args.compare is not None:
        experiment.plot_episode_length_comparison(args.compare)
