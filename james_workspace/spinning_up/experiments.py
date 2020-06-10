import copy, os, pickle
import matplotlib.pyplot as plt
import numpy as np

from env import CartPoleStandUp, CartPoleTravel

class Experiment:

    def __init__(self, score_target=195., skip_dups=False, verbose=False, render=True):
        self.score_target = score_target
        self.skip_dups = skip_dups
        self.verbose = verbose
        self.render = render

    def repeat(self, funct, args, repeats=1):

        if repeats < 1:
            res = self.load_experiment_from_file(funct + self.exp_file_suffix)

        for _ in range(repeats):
            if funct == "from_scratch_":
                res = self.experiment(**args)
            elif funct == "from_ref_":
                res = self.experiment_from_ref(**args)
            else:
                raise Exception("Unrecognised function")

        return res

    def load_experiment_from_file(self, exp_file, loss_angles=[]):

        if os.path.exists(exp_file):
            print("Loading previous experiments", exp_file)
            with open(exp_file, "rb") as ef:
                lines = pickle.load(ef)
        else:
            print("Couldn't find", exp_file)
            print("Making new experiment dict")
            lines = {an: {"episodes": [], "scores": []} for an in loss_angles}

        return lines

    def plot_summary_figure(self, lines, title, add_to_fig=False):
        
        if not add_to_fig:
            plt.figure()
        print("Plotting in plot summary")
        xs = []
        avg_final_episodes, stdev_final_episodes = [], []
    
        for an in lines:
            xs.append(an)
            episodes = lines[an]["episodes"]
            avg_final_episodes.append(np.mean([episode[-1] for episode in episodes]))
            stdev_final_episodes.append(np.std([episode[-1] for episode in episodes]))
        
        print("Legend label", title.lower())
        plt.errorbar(xs, avg_final_episodes, stdev_final_episodes, marker='x', label=title.lower())
    
        plt.xlabel("Target")
        plt.ylabel("Avg episodes to success")
        if add_to_fig:
            title = "Reference cart vs from scratch"
        plt.title(title)
        plt.legend(loc='lower left')
        plt.savefig(title.lower().replace(" ", "_") + "_summary.png")

class AngleExperiment(Experiment):

    def __init__(self, score_target=195., skip_dups=False, verbose=False, render=True, ref_model_name=None):

        Experiment.__init__(self, score_target, skip_dups, verbose, render)

        if ref_model_name:
            self.reference_cart_name = ref_model_name + "_reference_cart_" + str(score_target)

        self.exp_file_suffix = "dict_" + str(int(self.score_target)) + ".pickle"

    def experiment_from_ref(self, max_episodes=2000, episodes_threshold=100):
        
        # Train a reference cart        
        ref_cart = CartPoleStandUp(max_episodes=max_episodes, 
                                   score_target=self.score_target, 
                                   episodes_threshold=episodes_threshold)
        
        # Load or create a reference cart
        if os.path.exists("models/" + self.reference_cart_name + ".h5"):
            print("Loading reference cart", end='')
            ref_cart.dqn_solver.load_model(self.reference_cart_name)
            print("- loaded.")
        else:
            print("Training reference cart", end='')
            ref_result = ref_cart.solve(verbose=self.verbose, render=self.render)
            ref_cart.dqn_solver.save_model(self.reference_cart_name,
                                           ref_cart.angle_threshold, 
                                           ref_cart.x_threshold, 
                                           ref_result, 
                                           ref_cart.max_episode_steps)
            print("- saved")

        return self.experiment(from_ref=True)

    def experiment(self, from_ref=False):
        
        print("Setting up experiment")
        loss_angles = [8., 12., 16., 20., 24.]
        
        if from_ref:
            print("Running from reference")
            title = "episodes from reference cart"
            if not self.reference_cart_name:
                Exception("No reference cart created")
        else:
            print("No refernce cart defined - running from scratch")
            title = "episodes from scratch"

        lines = self.do_angle_experiment(loss_angles, from_ref=from_ref)

        return lines

    def do_angle_experiment(self, loss_angles, from_ref=False):
        
        if from_ref:
            exp_file = "from_ref_" + self.exp_file_suffix
        else:
            exp_file = "from_scratch_" + self.exp_file_suffix

        lines = self.load_experiment_from_file(exp_file, loss_angles)
        
        # Fill in any missing angles
        for an in loss_angles:
            if an not in lines:
                lines[an] = {"episodes": [], "scores": []}
        
        # Do the experiment and append it up
        for angle in loss_angles:
            if self.skip_dups and lines[angle]["episodes"]:
                print("For angle", angle, "got", len(lines[angle]["episodes"]), "previous runs")
                continue
            print("ANGLE", angle)
            cart = CartPoleStandUp(angle_threshold=angle, score_target=self.score_target)
            
            if from_ref:
                if not self.reference_cart_name:
                    raise Exception("Reference cart not defined")
                print("Loading reference model", self.reference_cart_name)
                cart.dqn_solver.load_model(self.reference_cart_name)
    
            episodes, scores = cart.solve(verbose=self.verbose, render=self.render)
            lines[angle]["episodes"].append(episodes)
            lines[angle]["scores"].append(scores)
    
            with open(exp_file, "wb") as ef:
                pickle.dump(lines, ef)
        
        return lines

    # Broken
    def plot_all_figure(self, lines, title):
        raise NotImplementedException("This is currently out of use")
        def plot_general(lines, title, smoothed=False):
            print("Plotting in plot_generate")
            plt.figure()
            leg = []
            for an in lines:
                # TODO - if different lengths (which they are) - can't average
                episodes, scores = np.array(lines[an]["episodes"]), np.array(lines[an]["scores"])
                print(episodes)
                episodes = np.mean(episodes, axis=0)
                scores = np.mean(scores, axis=0)
    
                if smoothed:
                    episodes_old = copy.copy(episodes)
                    # Smooth 20 points to make between episodes.min and episodes.max
                    episodes = np.linspace(np.min(episodes), np.max(episodes), 20)
                    scores = spline(episodes_old, scores, episodes)
    
                plt.plot(episodes, scores)
                leg.append(str(an) + "(" + str(len(lines[an]["episodes"])) + ")")
    
            plt.legend(leg)
            plt.xlabel("episodes")
            plt.ylabel("Score at episode")
            plt.title(title)
            plt.savefig(title.lower().replace(" ", "_") + "_all.png")
        print("Plotting") 
        plot_general(lines, title)
        # TODO - implement
        # plot_general(lines, title.replace("_all.png", "_smoothed_all.png"), smoothed=True)

class TravelExperiment(Experiment):
    
    def __init__(self, score_target=50, skip_dups=False, verbose=False, render=True, ref_model_name=None):

        Experiment.__init__(self, score_target, skip_dups, verbose, render)

        if ref_model_name:
            self.reference_cart_name = ref_model_name + "_reference_cart_TRAVEL_" + str(score_target)

        self.exp_file_suffix = "dict_" + str(self.score_target).replace(".", "-") + "TRAVEL.pickle"

    def experiment(self, from_ref=False):

        print("Setting up experiment")
        positions = [.3, 1., 1.5, 2.4, 3.]
        
        if from_ref:
            raise NotImplementedException("No from ref in travel exp.")
            print("Running from reference")
            title = "episodes from reference cart"
            if not self.reference_cart_name:
                Exception("No reference cart created")
        else:
            print("No refernce cart defined - running from scratch")
            title = "episodes from scratch"

        lines = self.do_position_experiment(positions, from_ref=from_ref)

        return lines

    def do_position_experiment(self, positions, from_ref=False):
        
        if from_ref:
            exp_file = "from_ref_" + self.exp_file_suffix
        else:
            exp_file = "from_scratch_" + self.exp_file_suffix

        lines = self.load_experiment_from_file(exp_file, positions)
        
        # Do the experiment and append it up
        for pos in positions:
            # Fill in any missing angles
            if pos not in lines:
                lines[pos] = {"episodes": [], "scores": []}

            if self.skip_dups and lines[pos]["episodes"]:
                print("For position", pos, "got", len(lines[pos]["episodes"]), "previous runs")
                continue
            
            print("POSITION", pos)
            
            cart = CartPoleTravel(position_target=pos, score_target=self.score_target)
            
            if from_ref:
                raise NotImplementedException("No from ref in position")
                if not self.reference_cart_name:
                    raise Exception("Reference cart not defined")
                print("Loading reference model", self.reference_cart_name)
                cart.dqn_solver.load_model(self.reference_cart_name)
    
            episodes, scores = cart.solve(verbose=self.verbose, render=self.render)
            lines[pos]["episodes"].append(episodes)
            lines[pos]["scores"].append(scores)
    
            with open(exp_file, "wb") as ef:
                pickle.dump(lines, ef)
        
        return lines

if __name__ == "__main__":
    
    # THREE - do the fail angle experiment
    experiment = AngleExperiment(score_target=5., render=False, verbose=False, skip_dups=False, ref_model_name="tanh")
    
    scratch_dict = experiment.repeat("from_scratch_", dict(), repeats=1) 
    experiment.plot_summary_figure(scratch_dict, "From scratch 195 " + str(len(scratch_dict[12.]["episodes"])))

    ref_dict = experiment.repeat("from_ref_", dict(), repeats=1)

    print("Finished experiment")
    experiment.plot_summary_figure(ref_dict, "From ref 195 " + str(len(ref_dict[12.]["episodes"])))
    experiment.plot_summary_figure(scratch_dict, "From scratch 195 " + str(len(scratch_dict[12.]["episodes"])), add_to_fig=True)

    experiment.plot_summary_figure(scratch_dict, "From scratch 195 " + str(len(scratch_dict[12.]["episodes"])), add_to_fig=False)

    # FOUR - do the travelling experiment
    # Question - how does slowly upping the position to reach change the number of episodes required

    experiment = TravelExperiment(score_target=0.1, render=False, verbose=False, skip_dups=False, ref_model_name="tanh")

    scratch_dict = experiment.repeat("from_scratch_", dict(), 1) 

    # print("Finished experiment")
    experiment.plot_summary_figure(scratch_dict, "From scratch 195 TRAVEL " + str(len(scratch_dict[1.]["episodes"])), add_to_fig=False)