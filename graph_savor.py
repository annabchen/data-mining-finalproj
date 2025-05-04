import dill
import os
from graph_reader import read_graph
import graph_sampling_analyzer as gsa
from random_node_sampler import RandomNodeSampler
from randomedgesampler import RandomEdgeSampler


# File path to store the graphs
def save_graphs(graphs, name):
    """SAVES THE GRAPHS USING DILL"""
    filename = name + ".pkl"
    # Save the graphs to a file
    with open(filename, "wb") as f:
        dill.dump(graphs, f)


def load_graphs(filename):
    """LOAD THE GRAPHS FROM A FILE USING DILL"""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            loaded_graphs = dill.load(f)
        print(f"{len(loaded_graphs)} graphs loaded from file.")
        return loaded_graphs
    else:
        print("Graph file not found. You might need to generate and save graphs first.")
        return None


if __name__ == '__main__':
    graph_info_as = ["as-caida20071105.txt", 8, True, "AS"]
    sampling_methods = [RandomEdgeSampler, RandomNodeSampler]
    graph_infos = [graph_info_as]

    orig_graph = read_graph(graph_infos[0][0],
                            n_skip_lines=graph_infos[0][1],
                            directed_graph=graph_infos[0][2])

    loaded_graphs = load_graphs("test.pkl")
    precomputed_graphs = []
    if loaded_graphs is not None:
        precomputed_graphs = loaded_graphs
    else:
        precomputed_graphs = gsa.compute_sampling_graphs(orig_graph, sampling_methods, 10)
        save_graphs(precomputed_graphs, "test")

    print(precomputed_graphs)
    gsa.analyze_mean(orig_graph,
                     sampling_methods,
                     gsa.get_total_nodes,
                     precomputed_graphs,
                     y_label="Total Nodes",
                     n_sample_sizes=10,
                     n_repetitions=10,
                     image_name="total_nodes" + graph_infos[0][3] + ".png")
