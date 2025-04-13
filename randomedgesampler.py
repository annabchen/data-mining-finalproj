import random
from GraphSampler import GraphSampler
from graph_reader import read_graph
import networkx as nx


class RandomEdgeSampler(GraphSampler):

    def __init__(self, graph, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=False):
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)

    def random_sample(self):
        """
        Sampling edges randomly.
        """
        # reading the graph
        orig_edges = list(self.graph.edges)

        if not self.isDirected:
            new_graph = nx.Graph()
        else:
            new_graph = nx.DiGraph()
        # Choosing a random number of nodes and edges from the sample
        for i in range(self.final_number_of_edges):
            index = random.randint(0, len(orig_edges) - 1)  # The index of the edge
            while new_graph.has_edge(orig_edges[index][0],
                                     orig_edges[index][1]):  # Only add the edge if it hasn't been added
                index = random.randint(0, len(orig_edges) - 1)
            else:
                new_graph.add_edge(orig_edges[index][0], orig_edges[index][1])

        return new_graph

    @staticmethod
    def get_method_name():
        return "Random Edge"


if __name__ == '__main__':
    orig_graph = read_graph("as-caida20071105.txt", n_skip_lines=8, directed_graph=True)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    graph_sample = RandomEdgeSampler(orig_graph, 300)
    sample = graph_sample.random_sample()
    print()

    print("Original # Nodes:", sample.number_of_nodes())
    print("Original # Edges ", sample.number_of_edges())

    print()
