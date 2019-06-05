import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def pair_plot(features):
    sns.set(style='ticks', color_codes=True)
    g = sns.pairplot(features, hue='label')
    g.axes[1, 0].set_ylim((0, 4e-7))
    g.axes[2, 0].set_ylim((0, 1e-6))
    g.axes[3, 0].set_ylim((0, 2e-7))
    plt.savefig("./fig/pair_plot.png")


def plot_decision_boundary(X, y, classifier, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(0, 40, 100), np.linspace(-90, -50, 100))

    z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # plot the line, points, and nearest vectors to the plane
    axis.contourf(xx, yy, z, alpha=0.75)
    axis.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')

    axis.axis('off')
    axis.set_title(title)
    plt.savefig("./fig/decision_boundary")

def plot_hist(error):
    plt.figure()
    sns.distplot(error)
    plt.title("Error plot")
    plt.savefig("./fig/error_plot")

