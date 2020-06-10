import matplotlib.pyplot as plt


def plot_scores(figure_name, scores):

    plt.figure()

    plt.plot(range(len(scores)), scores)
    plt.xlabel("Epsisode", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Episodic scores")

    plt.savefig(figure_name)
