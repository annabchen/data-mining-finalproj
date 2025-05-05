from matplotlib import pyplot as plt
import numpy as np


def get_colormap_colors(n, cmap_name='plasma'):
    """
    Generate a list of n colors using a specified colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_linear_with_scatter(x_values, input_values, errors_inputs, file_name="Figure.png",
                             xlabel="Sample Size (%)", ylabel="Metric", scale='linear'):
    """
    Plots a line graph with scatter points for different probabilities (or methods).

    x_values: list of sample sizes.
    input_values: dict, keys are labels (e.g., 'p=0.1') and values are lists of metric values.
    errors_inputs: dict, same keys as input_values, values are lists of error bars.
    """
    labels = list(input_values.keys())
    y_values = list(input_values.values())
    errors = [errors_inputs[label] for label in labels]

    linestyles = ["solid", "dashed", "dotted", "dashdot"] * 3  # cycle styles
    colors = get_colormap_colors(len(y_values), 'viridis')

    plt.figure(figsize=(8, 6))
    for index, (label, y_value_method, errors_method) in enumerate(zip(labels, y_values, errors)):
        plt.scatter(x_values, y_value_method, label=label,
                    color=colors[index], marker='^', alpha=0.7,
                    linewidth=3, zorder=3)
        plt.errorbar(x_values, y_value_method,
                     yerr=errors_method, ecolor=colors[index],
                     linestyle=linestyles[index], linewidth=3,
                     color=colors[index], alpha=0.7, capsize=5, zorder=1)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


#  For Distribution Plotting
def bar_plot(x_values, n_bins=30, labels=None,
             image_name="Figure.png", xlabel="Degree", ylabel="Normalized Distribution"):
    # PLOTS THE DISTRIBUTION OF VALUES IN A GRAPH. E.G. DEGREE DISTRIBUTION
    colors = get_colormap_colors(len(x_values) + 3, 'plasma')
    if labels is None or len(labels) == 0:
        labels = ["Original Graph", "Sample Graph"]
    for i, x_value_col in enumerate(x_values):
        plt.hist(np.array(x_value_col), bins=n_bins, label=labels[i],
                 width=(1/2) / n_bins, alpha=0.9, color=colors[i+1])
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{i}_{image_name}")
        plt.show()


if __name__ == '__main__':

    sample_sizes = [10, 20, 30, 40, 50, 60]
    metrics = {
        'p=0.1': [0.5, 0.7, 0.8, 1.0, 1.1, 1.2],
        'p=0.3': [0.6, 0.8, 0.9, 1.1, 1.2, 1.3],
        'p=0.5': [0.65, 0.85, 1.0, 1.15, 1.25, 1.35],
    }
    errors = {
        'p=0.1': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'p=0.3': [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
        'p=0.5': [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
    }

    plot_linear_with_scatter(sample_sizes, metrics, errors, xlabel="Sample Size (%)",
                             ylabel="Accuracy", file_name="probability_variation.png")

