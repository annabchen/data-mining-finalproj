import networkx as nx
import numpy as np
import graph_plotter
from graph_reader import read_graph
from randomedgesampler import RandomEdgeSampler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from random_node_sampler import RandomNodeSampler
from snowball_sampling import SnowballSampler
from wedge_sampling import WedgeSampler
import time


def diameter_analysis(orig_graph, sample_graph):
    """COMPUTES THE DIAMETER FOR AN ORIGINAL AND A SAMPLE GRAPH"""
    orig_graph_diameter = nx.diameter(orig_graph)
    sample_graph_diameter = nx.diameter(sample_graph)
    # print("Original Graph Diameter: ", orig_graph_diameter)
    # print("Sample Graph Diameter: ", sample_graph_diameter)
    return orig_graph_diameter, sample_graph_diameter


def clustering_analysis(orig_graph, sample_graph):
    """COMPUTES THE CLUSTERING COEFFICIENT FOR AN ORIGINAL AND A SAMPLE GRAPH"""
    orig_graph_clustering = nx.average_clustering(orig_graph)
    sample_graph_clustering = nx.average_clustering(sample_graph)
    # print("Original Graph Clustering Coefficient: ", orig_graph_clustering)
    # print("Sample Graph Clustering Coefficient:: ", sample_graph_clustering)
    return orig_graph_clustering, sample_graph_clustering


def graph_edit_distance_analysis(orig_graph, sample_graph):
    """COMPUTES THE GED for a graph"""
    distance = nx.graph_edit_distance(orig_graph, sample_graph, timeout=60)
    # print("Graph Edit Distance Analysis: ", distance)
    return distance


def get_total_nodes(orig_graph, sample_graph):
    """RETURNS THE TOTAL NUMBER OF NODES"""
    return orig_graph.number_of_nodes(), sample_graph.number_of_nodes()


def get_total_edges(orig_graph, sample_graph):
    """RETURN THE TOTAL NUMBER OF EDGES"""
    return orig_graph.number_of_edges(), sample_graph.number_of_edges()


def avg_degree_analysis(orig_graph, sample_graph):
    """COMPUTES THE DEGREE OF A GRAPH (HOW MANY EDGES DOES EACH NODE HAVE?)"""
    orig_graph_degree = nx.degree(orig_graph)
    sample_graph_degree = nx.degree(sample_graph)
    # print("Original Graph Degree: ", orig_graph_degree)
    # print("Sample Graph Degree: ", sample_graph_degree)

    # Computing Average Degree within the whole graph
    orig_graph_avg_deg = round(sum(
        dict(orig_graph.degree()).values()) / orig_graph.number_of_nodes(), 2)
    sample_graph_avg_deg = round(sum(
        dict(sample_graph.degree()).values()) / sample_graph.number_of_nodes(), 2)

    # print("Original Graph Average Degree: ", orig_graph_avg_deg)
    # print("Sample Graph Average Degree: ", sample_graph_avg_deg)
    return orig_graph_avg_deg, sample_graph_avg_deg


def degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH;

    DEGREE CENTRALITY = DEGREE / #NODES """
    orig_graph_centrality = nx.degree_centrality(orig_graph)
    sample_graph_centrality = nx.degree_centrality(sample_graph)

    # print("Original Graph Degree Centrality: ", orig_graph_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_centrality)

    # Computing Average Centrality within the whole graph
    orig_graph_avg_centrality = round(sum(
        orig_graph_centrality.values()) / orig_graph.number_of_nodes(), 2)
    sample_graph_avg_centrality = round(sum(
        sample_graph_centrality.values()) / sample_graph.number_of_nodes(), 2)

    return orig_graph_avg_centrality, sample_graph_avg_centrality


def in_degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE IN-DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH.
    ONLY WORKS FOR DIRECTED TYPES
    IN-DEGREE CENTRALITY = IN-DEGREE / #NODES"""

    if not orig_graph.is_directed():
        orig_graph = nx.to_directed(orig_graph)
        sample_graph = nx.to_directed(sample_graph)
    orig_graph_in_degree_centrality = nx.in_degree_centrality(orig_graph)
    sample_graph_in_degree_centrality = nx.in_degree_centrality(sample_graph)
    # print("Original Graph Degree Centrality: ", orig_graph_in_degree_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_in_degree_centrality)
    return orig_graph_in_degree_centrality, sample_graph_in_degree_centrality


def out_degree_centrality_analysis(orig_graph, sample_graph):
    """COMPUTES THE OUT-DEGREE CENTRALITY FOR AN ORIGINAL AND A SAMPLE GRAPH.
    ONLY WORKS FOR DIRECTED TYPES
    OUT-DEGREE CENTRALITY = OUT-DEGREE / #NODES"""

    if not orig_graph.is_directed():
        orig_graph = nx.to_directed(orig_graph)
        sample_graph = nx.to_directed(sample_graph)
    orig_graph_out_degree_centrality = nx.out_degree_centrality(orig_graph)
    sample_graph_out_degree_centrality = nx.out_degree_centrality(sample_graph)
    # print("Original Graph Degree Centrality: ", orig_graph_out_degree_centrality)
    # print("Sample Graph Degree Centrality: ", sample_graph_out_degree_centrality)
    return orig_graph_out_degree_centrality, sample_graph_out_degree_centrality


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


def k_components_analysis(orig_graph, sample_graph):
    """CONDUCTS A K-COMPONENT ANALYSIS OF A GRAPH"""
    orig_graph_k_comp = nx.k_components(orig_graph)
    sample_graph_k_comp = nx.k_components(sample_graph)
    # print("Original Graph K Components: ", orig_graph_k_comp)
    # print("Sample Graph K Components: ", sample_graph_k_comp)
    return orig_graph_k_comp, sample_graph_k_comp


def ks_test(orig_graph, sample_graph):
    """Performs a KS test on two graph samples to determine whether there is a statistical difference between them"""
    coeff, p_value = ks_2samp(orig_graph, sample_graph)
    # print("Coefficient:", coeff)
    # print("P-value: ", p_value)  # If above 0.05, then similar!
    return coeff, p_value


def density_analysis(orig_graph, sample_graph):
    """OBTAINS A GRAPH DENSITY [0,1]:
     0 -> VERY SPARSE GRAPH (FEW CONNECTIONS);
     1 -> VERY DENSE GRAPH (EVERY NODE CONNECTED)"""
    orig_graph_density = nx.density(orig_graph)
    sample_graph_density = nx.density(sample_graph)
    # print("Original Graph Density: ", orig_graph_density)
    # print("Sample Graph Density: ", sample_graph_density)
    return orig_graph_density, sample_graph_density


def assortativity_analysis(orig_graph, sample_graph):
    """MEASURE ASSORTATIVITY OF BOTH GRAPHS:
    TENDENCY OF NODES TO CONNECT TO SIMILAR NODES
    E.G. DO HIGH-DEGREE NODES CONNECT TO OTHER HIGH-DEGREE NODES?
    [-1,1] where -1 -> high-degree nodes connected to low-degree only
                  0 -> no correlation
                  1 -> high-degree nodes connected to high-degree only"""
    orig_graph_assortativity = nx.degree_pearson_correlation_coefficient(orig_graph)
    sample_graph_assortativity = nx.degree_pearson_correlation_coefficient(sample_graph)
    # print("Original Graph Assortativity: ", orig_graph_assortativity)
    # print("Sample Graph Assortativity: ", sample_graph_assortativity)
    return orig_graph_assortativity, sample_graph_assortativity


def degree_distribution_analysis(graph):
    """ANALYZING DEGREE DISTRIBUTION OF A GRAPH"""
    degrees = [d for n, d in graph.degree()]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    normalized_counts = counts / counts.sum()  # normalized histogram
    return normalized_counts


def test():
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


def compute_sampling_graphs(orig_graph, sampling_methods, n_sample_sizes=10, n_repetitions=10):
    """Computes all the graphs with all the sample sizes,
    as it is computationally expensive, especially for SnowBall Sampling"""
    # Approx. number of wedges:
    # CA-GrQc: 52612; as-caida: 34617

    wedge_sampler = WedgeSampler(orig_graph,1, 1)
    n_wedges = wedge_sampler.count_total_wedges()  # Calculating the total number of wedges

    graphs = []
    sample_sizes = np.linspace(10, 100, n_sample_sizes)  # samples sizes are evenly distributed
    for sample_size in sample_sizes:
        interm_output = []
        for sampling_method in sampling_methods:
            prelim_output = []
            for n_repetition in range(n_repetitions):
                sample_graph = sampling_method(orig_graph,
                                               int(sample_size / 100 * orig_graph.number_of_nodes()),
                                               int(sample_size / 100 * orig_graph.number_of_edges()),
                                               int(sample_size / 100 * n_wedges),
                                               isDirected=orig_graph.is_directed())
                prelim_output.append(sample_graph.random_sample())
            interm_output.append(prelim_output)
        graphs.append(interm_output)
    print(graphs)
    return graphs


def analyze_mean(orig_graph, sampling_methods, metric_function, precomputed_graphs,  y_label="Metric",
                 n_sample_sizes=10, n_repetitions=10, image_name="Figure.png", scale='linear'):
    """ANALYZES THE DEFINED METRIC FUNCTION USING A SPECIFIED SAMPLING METHOD"""
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

    for i, sample_size in enumerate(sample_sizes):
        for j, sampling_method in enumerate(sampling_methods):
            interm_outputs = list()

            for k in range(n_repetitions):
                interm_outputs.append(metric_function(orig_graph, precomputed_graphs[i][j][k])[1])

            mean = float(np.mean(interm_outputs))  # mean of a specific method for specific sample size
            std_error = float(np.std(interm_outputs))  # 68.9 % confidence (needs to be adjusted according the t-value)
            y_values_mean[sampling_method.get_method_name()].append(mean)
            y_values_error[sampling_method.get_method_name()].append(std_error)

        y_values_mean["Original Graph"].append(metric_function(orig_graph,
                                                               orig_graph)[
                                                   0])  # ORIGINAL SAMPLE; CHANGE INDEX TO ONE FOR KS-TEST

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
    graph_info_ca = ["CA-GrQc.txt", 4, False, "CA"]
    graph_info_as = ["as-caida20071105.txt", 8, True, "AS"]
    N = 2
    sampling_methods = [RandomEdgeSampler, RandomNodeSampler, SnowballSampler, WedgeSampler]
    graph_infos = [graph_info_ca, graph_info_as]
    for i in range(N):
        orig_graph = read_graph(graph_infos[i][0], n_skip_lines=graph_infos[i][1], directed_graph=graph_infos[i][2])
        precomputed_graphs = compute_sampling_graphs(orig_graph, sampling_methods, 10)
        # # Analyzing Average Degree for all Nodes
        analyze_mean(orig_graph, sampling_methods, avg_degree_analysis, precomputed_graphs, "Average Degree",
                     n_sample_sizes=10, n_repetitions=10, image_name="avg_degree" + graph_infos[i][3] + ".png")
        # # # # Analyzing # of nodes
        analyze_mean(orig_graph, sampling_methods, get_total_nodes, precomputed_graphs,y_label="Total Nodes",
                     n_sample_sizes=10, n_repetitions=10, image_name="total_nodes" + graph_infos[i][3] + ".png")
        # # # Analyzing # of edges
        analyze_mean(orig_graph, sampling_methods, get_total_edges, precomputed_graphs,y_label="Total Edges",
                     n_sample_sizes=10, n_repetitions=10, image_name="total_edges" + graph_infos[i][3] + ".png")
        # # Analyzing Degree Centrality of nodes
        analyze_mean(orig_graph, sampling_methods, degree_centrality_analysis, precomputed_graphs,
                     y_label="Degree Centrality",
                     n_sample_sizes=10, n_repetitions=10, image_name="degree_centrality" + graph_infos[i][3] + ".png",
                     scale='linear')
        # # Analyzing Clustering Coefficient of a graph
        analyze_mean(orig_graph, sampling_methods, clustering_analysis, precomputed_graphs,
                     y_label="Clustering Coefficient (Mean)",
                     n_sample_sizes=10, n_repetitions=10, image_name="clustering_coeff" + graph_infos[i][3] + ".png",
                     scale='linear')
        # # Analyzing KS-Coefficient
        analyze_mean(orig_graph, sampling_methods, ks_test, precomputed_graphs, y_label="KS-Test Coefficient",
                     n_sample_sizes=10, n_repetitions=10, image_name="ks_test_coeff" + graph_infos[i][3] + ".png",
                     scale='linear')
        # # Analyzing Graph Density
        analyze_mean(orig_graph, sampling_methods, density_analysis, precomputed_graphs, y_label="Density",
                     n_sample_sizes=10, n_repetitions=10, image_name="density" + graph_infos[i][3] + ".png",
                     scale='log')
        # # Analyzing Graph Assortativity
        analyze_mean(orig_graph, sampling_methods, assortativity_analysis, precomputed_graphs,
                     y_label="Assortativity Coeff",
                     n_sample_sizes=10, n_repetitions=10, image_name="assortativity" + graph_infos[i][3] + ".png",
                     scale='linear')
        # Analyzing Graph Degree Distribution
        analyze_distribution(orig_graph, sampling_methods, degree_distribution_analysis, precomputed_graphs,
                             n_sample_sizes=3, n_repetitions=10, image_name="Degree Distribition" + graph_infos[i][3])
