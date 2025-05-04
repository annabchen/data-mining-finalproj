from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import graph_plotter
from graph_reader import read_graph
import graph_savor
from randomedgesampler import RandomEdgeSampler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from random_node_sampler import RandomNodeSampler
from snowball_sampling import SnowballSampler
from wedge_sampling import WedgeSampler
from RandomWalk import RandomWalk
from RandomJump import RandomJump
from forestfiresampler import FFSampler
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import time


def average_shortest_path_length_analysis(orig_graph, sample_graph):
    """COMPUTES THE DIAMETER FOR AN ORIGINAL AND A SAMPLE GRAPH"""
    graph_avg_SP = nx.average_shortest_path_length(orig_graph)
    return graph_avg_SP


def clustering_analysis(orig_graph, sample_graph):
    """COMPUTES THE CLUSTERING COEFFICIENT FOR AN ORIGINAL AND A SAMPLE GRAPH"""
    orig_graph_clustering = nx.average_clustering(orig_graph)
    # print("Original Graph Clustering Coefficient: ", orig_graph_clustering)
    # print("Sample Graph Clustering Coefficient:: ", sample_graph_clustering)
    return orig_graph_clustering


def graph_edit_distance_analysis(orig_graph, sample_graph):
    """COMPUTES THE GED for a graph"""
    distance_iter = nx.optimize_graph_edit_distance(orig_graph, sample_graph)
    try:
        distance = next(distance_iter)
    except StopIteration:
        distance = float('inf')  # or handle differently if no result

    # Normalize by number of nodes
    distance_avg = round(distance / orig_graph.number_of_nodes(), 2)
    return distance_avg


def get_total_nodes(orig_graph, sample_graph):
    """RETURNS THE TOTAL NUMBER OF NODES"""
    return orig_graph.number_of_nodes()


def get_total_edges(orig_graph, sample_graph):
    """RETURN THE TOTAL NUMBER OF EDGES"""
    return orig_graph.number_of_edges()


def avg_degree_analysis(orig_graph, sample_graph):
    """COMPUTES THE DEGREE OF A GRAPH (HOW MANY EDGES DOES EACH NODE HAVE?)"""
    orig_graph_degree = nx.degree(orig_graph)
    # print("Original Graph Degree: ", orig_graph_degree)
    # print("Sample Graph Degree: ", sample_graph_degree)

    # Computing Average Degree within the whole graph
    orig_graph_avg_deg = round(sum(
        dict(orig_graph.degree()).values()) / orig_graph.number_of_nodes(), 2)

    # print("Original Graph Average Degree: ", orig_graph_avg_deg)
    # print("Sample Graph Average Degree: ", sample_graph_avg_deg)
    return orig_graph_avg_deg


def degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH;

    DEGREE CENTRALITY = DEGREE / #NODES """
    orig_graph_centrality = nx.degree_centrality(orig_graph)

    # print("Original Graph Degree Centrality: ", orig_graph_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_centrality)

    # Computing Average Centrality within the whole graph
    orig_graph_avg_centrality = round(sum(
        orig_graph_centrality.values()) / orig_graph.number_of_nodes(), 2)

    return orig_graph_avg_centrality


def in_degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE IN-DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH.
    ONLY WORKS FOR DIRECTED TYPES
    IN-DEGREE CENTRALITY = IN-DEGREE / #NODES"""

    if not orig_graph.is_directed():
        orig_graph = nx.to_directed(orig_graph)
    orig_graph_in_degree_centrality = nx.in_degree_centrality(orig_graph)
    # print("Original Graph Degree Centrality: ", orig_graph_in_degree_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_in_degree_centrality)
    return orig_graph_in_degree_centrality


def out_degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE OUT-DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH.
    ONLY WORKS FOR DIRECTED TYPES
    OUT-DEGREE CENTRALITY = OUT-DEGREE / #NODES"""

    if not orig_graph.is_directed():
        orig_graph = nx.to_directed(orig_graph)
    orig_graph_out_degree_centrality = nx.out_degree_centrality(orig_graph)
    # print("Original Graph Degree Centrality: ", orig_graph_out_degree_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_out_degree_centrality)
    return orig_graph_out_degree_centrality


def draw_graph(graph, name, options=None):
    """DRAWING A GRAPH WITH SELECTED OPTIONS"""
    if options is None:
        options = {
            'node_color': 'blue',
            'node_size': 3,
            'width': 1,
        }
    nx.draw(graph, with_labels=False, font_weight='bold', **options)
    plt.savefig(name)
    plt.show()


def ks_test(orig_graph, sample_graph):
    """Performs a KS test on two graph samples to determine whether there is a statistical difference between them"""
    coeff, p_value = ks_2samp(orig_graph, sample_graph)
    # print("Coefficient:", coeff)
    # print("P-value: ", p_value)  # If above 0.05, then similar!
    return p_value


def mannwhitneyu_test(orig_graph, sample_graph):
    """Performs a KS test on two graph samples to determine whether there is a statistical difference between them"""
    # Extract degree sequences
    degrees_orig_graph = [d for n, d in orig_graph.degree()]
    degrees_sample_graph = [d for n, d in sample_graph.degree()]

    # Run two-sample t-test (Welch's t-test is safer for unequal variances)
    stat, p_value = mannwhitneyu(degrees_orig_graph, degrees_sample_graph, alternative='two-sided')
    return p_value


def t_test(orig_graph, sample_graph):
    """Performs a KS test on two graph samples to determine whether there is a statistical difference between them"""
    # Extract degree sequences
    degrees_orig_graph = [d for n, d in orig_graph.degree()]
    degrees_sample_graph = [d for n, d in sample_graph.degree()]

    # Run two-sample t-test (Welch's t-test is safer for unequal variances)
    stat, p_value = ttest_ind(degrees_orig_graph, degrees_sample_graph)
    return p_value


def density_analysis(orig_graph, sample_graph):
    """OBTAINS A GRAPH DENSITY [0,1]:
     0 -> VERY SPARSE GRAPH (FEW CONNECTIONS);
     1 -> VERY DENSE GRAPH (EVERY NODE CONNECTED)"""
    orig_graph_density = nx.density(orig_graph)
    # print("Original Graph Density: ", orig_graph_density)
    # print("Sample Graph Density: ", sample_graph_density)
    return orig_graph_density


def assortativity_analysis(orig_graph, sample_graph):
    """MEASURE ASSORTATIVITY OF BOTH GRAPHS:
    TENDENCY OF NODES TO CONNECT TO SIMILAR NODES
    E.G. DO HIGH-DEGREE NODES CONNECT TO OTHER HIGH-DEGREE NODES?
    [-1,1] where -1 -> high-degree nodes connected to low-degree only
                  0 -> no correlation
                  1 -> high-degree nodes connected to high-degree only"""
    orig_graph_assortativity = nx.degree_pearson_correlation_coefficient(orig_graph)
    # print("Original Graph Assortativity: ", orig_graph_assortativity)
    # print("Sample Graph Assortativity: ", sample_graph_assortativity)
    return orig_graph_assortativity


def betweenness_centrality(orig_graph, sample_graph):
    """Measures the betweenness centrality of two graphs"""
    orig_graph_betweenness = nx.betweenness_centrality(orig_graph)

    # Computing Average Centrality within the whole graph
    orig_graph_avg_betweenness = round(sum(
        orig_graph_betweenness.values()) / orig_graph.number_of_nodes(), 2)

    return orig_graph_avg_betweenness


def degree_distribution_analysis(graph):
    """ANALYZING DEGREE DISTRIBUTION OF A GRAPH.
    RETURNS A BAR CHART"""
    degrees = [d for n, d in graph.degree()]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    normalized_counts = counts / counts.sum()  # normalized histogram
    return normalized_counts


def test():
    """TESTS DIFFERENT ANALYSIS METHODS"""
    orig_graph = read_graph("CA-GrQc.txt", n_skip_lines=4, directed_graph=False)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    sample = RandomEdgeSampler(orig_graph, 5000, isDirected=orig_graph.is_directed())
    sample_graph = sample.random_sample()
    print()

    print("Original # Nodes:", sample_graph.number_of_nodes())
    print("Original # Edges ", sample_graph.number_of_edges())

    print()
    print("Original Graph isDirected", orig_graph.is_directed())
    print("Sample Graph isDirected", sample_graph.is_directed())

    # print()
    # print("Clustering Analysis")
    # clustering_analysis(orig_graph, sample_graph) # WORKS
    #
    # print()
    # print("Diameter Analysis")
    # diameters = diameter_analysis(orig_graph, sample_graph)  # DOES NOT WORK BECAUSE OUR GRAPHS ARE NOT STRONGLY CONNECTED
    # print()
    #
    # print()
    # print("Average Graph Degree")
    # avg_degree_analysis(orig_graph, sample_graph)
    # print("Degree Centrality")
    # degree_centrality_analysis(orig_graph, sample_graph) # WORKS
    #
    # print("In-degree Centrality") # ONLY WORKS FOR DIRECTED GRAPHS
    # in_degree_centrality_analysis(orig_graph, sample_graph)
    # #
    # print("Out-degree Centrality") # ONLY WORKS FOR DIRECTED GRAPHS
    # out_degree_centrality_analysis(orig_graph, sample_graph)
    # print()
    # #
    # draw_graph(orig_graph, "orig_graph_as_caida.png") # Drawing the original graph
    # #draw_graph(sample_graph, "sample_graph_as_caida.png") # Drawing the sample graph
    # print()
    # print("K-Components Analysis") # TAKES A LONG TIME
    # k_components_analysis(orig_graph, sample_graph)
    # print()
    # print("Graph Edit Distance")
    # graph_edit_distance_analysis(orig_graph, sample_graph) # Outputs NONE
    # print()
    # print("KS-Test")
    # ks_test(orig_graph, sample_graph)
    print()
    print(graph_edit_distance_analysis(orig_graph, sample_graph))


def analyze_distribution(orig_graph, sampling_methods, metric_function, precomputed_graphs,
                         n_sample_sizes=3, n_repetitions=5, image_name="Figure_Distribution.png"):
    """ANALYZES THE DEFINED METRIC FUNCTION USING A SPECIFIED SAMPLING METHOD"""
    start_time = time.time()
    sample_sizes = np.linspace(10, 90, n_sample_sizes)  # samples sizes are evenly distributed
    for i, sampling_method in enumerate(sampling_methods):
        outputs = list()
        labels = list()
        labels.append("Original Graph")
        outputs.append(metric_function(orig_graph))
        for j, sample_size in enumerate(sample_sizes):
            labels.append(f"Sample Graph: {sample_size} %")
            interm_outputs = []
            for k in range(n_repetitions):
                interm_outputs.append(metric_function(precomputed_graphs[j][i][k]))

            # return an average degree distribution
            max_len = max(len(arr) for arr in interm_outputs)
            padded = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                               for arr in interm_outputs])
            outputs.append(np.mean(padded, axis=0))
        graph_plotter.bar_plot(outputs, labels=labels, ylabel="Degree Distribution",
                               image_name=image_name + "_" + sampling_method.get_method_name(), n_bins=30)
        end_time = time.time()
        print("Analysis Time: ", round(end_time - start_time, 2), " seconds")


def sample_one_graph(args):
    """Helper for parallel sampling. Returns a sample graph as an output"""
    orig_graph, sampling_method, sample_size, n_wedges = args
    sample_graph = sampling_method(
        orig_graph,
        final_number_of_nodes=int(sample_size / 100 * orig_graph.number_of_nodes()),
        final_number_of_edges=int(sample_size / 100 * orig_graph.number_of_edges()),
        final_number_of_wedges=int(sample_size / 100 * n_wedges),
        isDirected=orig_graph.is_directed()
    )
    if orig_graph.is_directed():
        return nx.DiGraph(sample_graph.random_sample())
    else:
        return nx.Graph(sample_graph.random_sample())


def compute_sampling_graphs(orig_graph, sampling_methods, n_sample_sizes=10, n_repetitions=3):
    """Precomputes the defined graphs with the specified sample sizes which will later be used for characterization"""
    # Approx. number of wedges:
    # CA-GrQc: 52612; as-caida: 34617
    wedge_sampler = WedgeSampler(orig_graph, 1, 1)
    n_wedges = wedge_sampler.count_total_wedges()

    sample_sizes = np.linspace(10, 90, n_sample_sizes)  # percentage of original size
    jobs = []

    for sample_size in sample_sizes:  # given in percent
        for method in sampling_methods:
            for _ in range(n_repetitions):
                jobs.append((orig_graph, method, sample_size, n_wedges))

    with Pool(cpu_count()) as pool:
        results = pool.map(sample_one_graph, jobs)

    # Rebuild nested list structure: [sample_size][method][repetition]
    graphs = []
    index = 0
    for sample_size in sample_sizes:
        size_output = []
        for method in sampling_methods:
            method_output = []
            for _ in range(n_repetitions):
                method_output.append(results[index])
                index += 1
            size_output.append(method_output)
        graphs.append(size_output)
    return graphs


def analyze_mean(orig_graph, sampling_methods, metric_function, precomputed_graphs,  y_label="Metric",
                 n_sample_sizes=10, n_repetitions=10, image_name="Figure.png", scale='linear'):
    """ANALYZES THE DEFINED METRIC FUNCTION USING A SPECIFIED SAMPLING METHOD. RETURNS A MEAN OUTPUT.
    E.G. MEAN DEGREE"""
    start_time = time.time()
    sample_sizes = np.linspace(10, 100, n_sample_sizes)  # samples sizes are evenly distributed
    y_values_mean = dict()  # orig_graph, sample_size_1, sample_size_2, ..., sample_size_n
    y_values_error = dict()  # orig_graph, sample_size_1, sample_size_2, ..., sample_size_n
    method_names = set()

    for i in range(len(sampling_methods)):
        # Add method_names for labeling in plots and init the dictionaries
        name = sampling_methods[i].get_method_name()
        method_names.add(name)
        y_values_mean[name] = []
        y_values_error[name] = []

    # Adding separately "Original Graph"
    y_values_mean["Original Graph"] = []
    y_values_error["Original Graph"] = []
    method_names.add("Original Graph")

    # Iterate through all the sampling sizes, sampling methods, and the number of iterations
    for i, sample_size in enumerate(sample_sizes):
        for j, sampling_method in enumerate(sampling_methods):
            interm_outputs = list()

            # generate the specified metric outputs for a precomputed graph
            for k in range(n_repetitions):
                interm_outputs.append(metric_function(precomputed_graphs[i][j][k], orig_graph))

            mean = float(np.mean(interm_outputs))  # mean of a specific method for specific sample size
            std_error = float(2.262 * np.std(interm_outputs))  # ~95 % confidence for 9 degrees of freedom
            y_values_mean[sampling_method.get_method_name()].append(mean)
            y_values_error[sampling_method.get_method_name()].append(std_error)

        y_values_mean["Original Graph"].append(metric_function(orig_graph,
                                                               orig_graph))  # ORIGINAL SAMPLE; CHANGE INDEX TO ONE FOR KS-TEST

        y_values_error["Original Graph"].append(0)  # no error in the original sample -> 0
        # blocks of sample means [10%], [20%], -> [[1,2,3], [2,3,4], ...]
        # for different sample sizes

    x_values = sample_sizes
    graph_plotter.plot_linear_with_scatter(x_values, y_values_mean, y_values_error,
                                           labels=list(method_names), ylabel=y_label, file_name=image_name,
                                           scale=scale)
    end_time = time.time()
    print("Analysis Time: ", round(end_time - start_time, 2), " seconds")


if __name__ == '__main__':
    # test()
    # Two graphs used for analysis
    graph_info_ca = ["CA-GrQc.txt", 4, False, "CA"]
    graph_info_as = ["as-caida20071105.txt", 8, True, "AS"]
    N = 2  # Number of graphs
    # SAMPLING METHODS USED
    sampling_methods = [SnowballSampler, WedgeSampler, FFSampler]  # SnowballSampler, WedgeSampler, FFSampler
    graph_infos = [graph_info_as]
    for i in range(N):
        orig_graph = read_graph(graph_infos[i][0], n_skip_lines=graph_infos[i][1], directed_graph=graph_infos[i][2])
        loaded_graphs = graph_savor.load_graphs(graph_infos[i][3] + str(sampling_methods) + ".pkl")
        precomputed_graphs = []
        if loaded_graphs is not None:
            precomputed_graphs = loaded_graphs
        else:
            precomputed_graphs = compute_sampling_graphs(orig_graph, sampling_methods, 10)
            graph_savor.save_graphs(precomputed_graphs, graph_infos[i][3] + str(sampling_methods))

        # # Analyzing Average Degree for all Nodes
        print("Analyzing Average Degree for all Nodes")
        analyze_mean(orig_graph, sampling_methods, avg_degree_analysis, precomputed_graphs, "Average Degree",
                     n_sample_sizes=10, n_repetitions=10, image_name="avg_degree" + graph_infos[i][3] + ".png")
        print()

        # Analyzing # of nodes
        print("Analyzing # of nodes")
        analyze_mean(orig_graph, sampling_methods, get_total_nodes, precomputed_graphs,y_label="Total Nodes",
                     n_sample_sizes=10, n_repetitions=10, image_name="total_nodes" + graph_infos[i][3] + ".png")
        print()

        # Analyzing # of edges
        print("Analyzing # of edges")
        analyze_mean(orig_graph, sampling_methods, get_total_edges, precomputed_graphs,y_label="Total Edges",
                     n_sample_sizes=10, n_repetitions=10, image_name="total_edges" + graph_infos[i][3] + ".png")
        print()

        # Analyzing Degree Centrality of nodes
        print("Analyzing Degree Centrality of nodes")
        analyze_mean(orig_graph, sampling_methods, degree_centrality_analysis, precomputed_graphs,
                     y_label="Degree Centrality",
                     n_sample_sizes=10, n_repetitions=10, image_name="degree_centrality" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # Analyzing Clustering Coefficient of a graph
        print("Analyzing Clustering Coefficient of a graph")
        analyze_mean(orig_graph, sampling_methods, clustering_analysis, precomputed_graphs,
                     y_label="Clustering Coefficient (Mean)",
                     n_sample_sizes=10, n_repetitions=10, image_name="clustering_coeff" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # Analyzing KS-Coefficient
        print("Analyzing KS-Coefficient")
        analyze_mean(orig_graph, sampling_methods, ks_test, precomputed_graphs, y_label="P-value",
                     n_sample_sizes=10, n_repetitions=10, image_name="ks_test_" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # Mann - Whitney U-test
        print("Mann-Whitney U-test")
        analyze_mean(orig_graph, sampling_methods, mannwhitneyu_test, precomputed_graphs, y_label="P-value",
                     n_sample_sizes=10, n_repetitions=10, image_name="mann_whitney_u_test_" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # T-test
        print("T-test")
        analyze_mean(orig_graph, sampling_methods, t_test, precomputed_graphs, y_label="P-value",
                     n_sample_sizes=10, n_repetitions=10, image_name="t_test_" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # # Analyzing Graph Density
        print("Analyzing Graph Density")
        analyze_mean(orig_graph, sampling_methods, density_analysis, precomputed_graphs, y_label="Density",
                     n_sample_sizes=10, n_repetitions=10, image_name="density" + graph_infos[i][3] + ".png",
                     scale='log')
        print()

        # Analyzing Graph Assortativity
        print("Analyzing Graph Assortativity")
        analyze_mean(orig_graph, sampling_methods, assortativity_analysis, precomputed_graphs,
                     y_label="Assortativity Coeff",
                     n_sample_sizes=10, n_repetitions=10, image_name="assortativity" + graph_infos[i][3] + ".png",
                     scale='linear')
        print()

        # Analyzing Graph Degree Distribution
        print("Analyzing Graph Degree Distribution")
        analyze_distribution(orig_graph, sampling_methods, degree_distribution_analysis, precomputed_graphs,
                             n_sample_sizes=3, n_repetitions=10, image_name="Degree Distribition" + graph_infos[i][3])
        print()