from matplotlib import pyplot as plt
import numpy as np


def get_colormap_colors(n, cmap_name='plasma'):
    """
    Generate a list of n colors using a specified colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_linear_with_scatter(x_values, input_values, errors_inputs, labels=None,
                             file_name="Figure.png",  xlabel="Sample Size (%)",
                             ylabel="Metric", scale='linear'):
    """
    Plots a line graph with scatter points for the given y-values.
    """


    # Separating inputs into labels and y_values
    labels = list(input_values.keys())
    y_values = list(input_values.values())
    errors = list()

    # Appending the errors
    for label in labels:
        errors.append(errors_inputs[label])
    # Plotting
    shift = 2
    linestyles = ["solid", "solid", "dashed", "dashed", "dashed", "dashed", "dashed", "solid"]
    colors = get_colormap_colors(10, 'viridis')
    print(len(y_values))
    plt.figure(figsize=(8, 6))

    for index, (y_value_method, errors_method) in enumerate(zip(y_values, errors)):
        split_y_values = [y_value_method[i:i + 10] for i in range(10, 90, 8)]
        split_errors = [errors[i:i + 10] for i in range(10, 90, 8)]

        for split_y_value, split_error in zip(split_y_values, split_errors) :
            plt.scatter(np.array(x_values), split_y_value, label=labels[index], # +shift * (index - len(labels) // 2)
                        color=colors[index], marker='^', alpha=0.7,
                        linewidth=3, zorder=3)
            # plt.errorbar(np.array(x_values), split_y_value, color=colors[index], # + shift * (index - len(labels) // 2)
            #              yerr=split_error, ecolor=colors[index],
            #              linestyle=linestyles[index], linewidth=3, alpha=0.7, capsize=5, zorder=1)
            plt.xlabel(xlabel, fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
            plt.legend(fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.yscale(scale)
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
    # Testing
    plot_linear_with_scatter([10, 20, 30, 40, 50, 60],
                             {1:[0, 1, 2, 3, 4, 5],
                               2: [0, 1, 2, 3, 4, 5],
                               3:[0, 1, 2, 3, 4, 5],
                                4:[0, 1, 2, 3, 4, 5]},
                             [[0.5] * 6]*5)
    # bar_plot([[1,2,2,3,4,4,5], [1,2,3]])
