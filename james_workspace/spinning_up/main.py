from env import CartPoleStandUp
from models import DQNSolver
from utils.plotting import plot_scores
import gym, time, random, math, pickle, os, sys, copy, argparse

from utils import smooth_over

class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)


def parse_args():

    parser = MyParser()
    
    parser.add_argument("--outdir", type=str, required=True,
                        help="If supplied, model "
                        "checkpoints will be saved so "
                        "training can be restarted later",
                        default=None)

    parser.add_argument("--train", dest="train", 
                        type=int, default=0, 
                        help="number of episodes to train")

    parser.add_argument("--show", action="store_true", 
                        help="Shows a gif of the agent acting under "
                             "its current set of parameters")

    parser.add_argument("--example", action="store_true", 
                        help="Shows an example gif of the environment"
                             "with random actions")
    
    parser.add_argument("--render", action="store_true", 
                        help="Whether to render the env as we go")

    parser.add_argument("--plot", action="store_true", 
                        help="Whether to plot the experiment output")


    # parser.add_argument("--model", type=str, default="default",
    #                     help="The model to be run. Options: "
    #                          "side_camp_dqn, (default)")

    return parser.parse_args()


# TODO clean up the DQN learning processing
# TODO fix up a virtual env to remove the tensorflow errors (start one from scratch, as with gridworlds)


if __name__ == "__main__":

    args = parse_args()

    # ONE - solve the standard cart pole
    cart = CartPoleStandUp(score_target=195., episodes_threshold=100)

    # Look at the obs and action space
    cart.get_spaces(registry=False)

    agent = DQNSolver(args.outdir, cart.observation_space, cart.action_space)

    if args.example:
        agent.do_random_runs(cart, episodes=1, steps=99, verbose=True)
    
    if args.train:
        solved = agent.solve(cart, max_episodes=args.train, verbose=True, render=args.render)
        print("\nSolved:", solved)
    
    if args.show:
        agent.show_example(cart, steps=99)

    if args.plot:
        plot_scores(agent.experiment_dir + "scores.png", agent.scores)
        smoothed_scores = smooth_over(agent.scores, 10)
        plot_scores(agent.experiment_dir + "smooth_scores.png", smoothed_scores)

