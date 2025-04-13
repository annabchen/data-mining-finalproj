import random

from GraphSampler import GraphSampler
from graph_reader import read_graph


class RandomNodeSampler(GraphSampler):

    def __init__(self, graph, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=False):
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)

    def random_sample(self):
        """
        Sampling nodes randomly.
        First, choose # of nodes.
        Then, connect all the nodes that were connected in the original graph.
        """
        sampled_nodes = random.sample(list(self.graph.nodes()), self.final_number_of_nodes)
        induced_subgraph = self.graph.subgraph(sampled_nodes)
        return induced_subgraph

    @staticmethod
    def get_method_name():
        return "Random Node"


if __name__ == '__main__':
    orig_graph = read_graph("as-caida20071105.txt", n_skip_lines=8, directed_graph=True)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    graph_sample = RandomNodeSampler(orig_graph, 10000)
    sample = graph_sample.random_sample()
    print()

    print("Original # Nodes:", sample.number_of_nodes())
    print("Original # Edges ", sample.number_of_edges())

    print()