from matplotlib import pyplot as plt
import numpy as np


def get_colormap_colors(n, cmap_name='plasma'):
    """
    Generate a list of n colors using a specified matplotlib colormap.

    Parameters:
    - n (int): Number of colors to generate.
    - cmap_name (str): Name of the matplotlib colormap to use (e.g., 'viridis', 'plasma', 'cool', etc.)

    Returns:
    - List of RGBA tuples.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_linear_with_scatter(x_values, input_values, errors_inputs, labels=None,
                             file_name="Figure.png",  xlabel="Sample Size (%)",
                             ylabel="Metric", scale='linear'):
    """
    Plots a line graph with scatter points for the given y-values.

    Parameters:
    - y_values (list or array-like): Y-values to plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    """

    colors = get_colormap_colors(20, 'plasma')

    labels = list(input_values.keys())
    y_values = list(input_values.values())
    errors = list()
    for label in labels:
        errors.append(errors_inputs[label])
    print(errors)
    print(labels)
    print(y_values)
    shift = 2

    plt.figure(figsize=(8, 6))
    for index, (y_value_method, errors_method) in enumerate(zip(y_values, errors)):
        plt.scatter(np.array(x_values)+shift * (index - len(labels) // 2),  y_value_method, label=labels[index],
                    color=colors[4*index + 3], marker='o',
                    linewidth=3, zorder=3)
        plt.errorbar(np.array(x_values) + shift * (index - len(labels) // 2), y_value_method, color=colors[4*index + 3],
                     yerr=errors_method, ecolor=colors[4*index + 3],
                     linestyle='--', linewidth=3, alpha=1, capsize=5, zorder=1)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.yscale(scale)
    plt.savefig(file_name)
    plt.show()


#  For Distributions
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
    # Testing
    plot_linear_with_scatter([10, 20, 30, 40, 50, 60],
                             [[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]],
                             [[0.5] * 6]*2)
    # bar_plot([[1,2,2,3,4,4,5], [1,2,3]])
