from matplotlib import pyplot as plt
import numpy as np


def get_colormap_colors(n, cmap_name='plasma'):
    """
    Generate a list of n colors using a specified colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_linear_with_quartiles(x_values, median_values, quartile_bounds, file_name="Figure_Quartiles.png",
                               xlabel="Sample Size (%)", ylabel="Metric", scale='linear'):
    """
    Plots a line graph with scatter points and shaded areas between Q1 and Q3 for each method.

    x_values: list or array of sample sizes.
    median_values: dict, keys are method names, values are lists of median metric values.
    quartile_bounds: dict with keys 'lower' and 'upper' that mirror median_values in structure (Q1 and Q3).
    """
    labels = list(median_values.keys())
    medians = list(median_values.values())
    q1s = [quartile_bounds['lower'][label] for label in labels]
    q3s = [quartile_bounds['upper'][label] for label in labels]

    linestyles = ["solid", "dashed", "dotted", "dashdot"] * 3
    colors = get_colormap_colors(len(medians), 'viridis')

    plt.figure(figsize=(8, 6))
    for index, (label, y_median, y_q1, y_q3) in enumerate(zip(labels, medians, q1s, q3s)):
        color = colors[index]
        plt.plot(x_values, y_median, label=label,
                 linestyle=linestyles[index], color=color,
                 linewidth=2.5, alpha=0.9, zorder=3)
        plt.scatter(x_values, y_median, color=color, marker='^', zorder=4)

        # Shaded area between Q1 and Q3
        plt.fill_between(x_values, y_q1, y_q3, alpha=0.25,
                         color=color, label=f"{label} (Q1-Q3)", zorder=2)

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()



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
    plt.xlim(0,100)
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


#  For Distribution Plotting
def bar_plot(x_values, n_bins=30, labels=None,
             image_name="Figure.png", xlabel="Degree", ylabel="Normalized Distribution"):
    num_plots = 4
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 12), sharex=True)

    # Generate colors
    colors = get_colormap_colors(num_plots + 3, 'plasma')

    # Default labels
    if labels is None or len(labels) == 0:
        labels = [f"Graph {i+1}" for i in range(len(x_values))]

    # Move the first element to the end
    reordered_x_values = x_values[1:] + [x_values[0]]
    reordered_labels = labels[1:] + [labels[0]]

    for i in range(min(num_plots, len(reordered_x_values))):
        ax = axes[i]
        ax.hist(np.array(reordered_x_values[i]), bins=n_bins,
                label=reordered_labels[i], width=(1/2) / n_bins,
                alpha=0.9, color=colors[i+1], density=True)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=10)

    axes[-1].set_xlabel(xlabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(image_name)
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