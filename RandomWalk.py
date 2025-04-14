import networkx as nx
import random
from GraphSampler import GraphSampler
from graph_reader import read_graph


class RandomWalk(GraphSampler):
    ## should be treating as directed graph?

    def __init__(self, graph, flyback=0.9, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=False):
        self.flyback = flyback
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)

    def random_sample(self):
        """ Return sample of larger graph given flyback probability"""

        if not self.isDirected:
            new_graph = nx.Graph()
        else:
            new_graph = nx.DiGraph()

        nodes = list(self.graph.nodes())
        # random start node
        start = random.choice(nodes)
        # nodes.remove(start)
        # print(f"Starting node: {start}")
        curr = start
        visited = {start}

        # select nodes until reached desired size
        while new_graph.number_of_nodes() < self.final_number_of_nodes:
            if random.random() > self.flyback:
                curr = start
            else:
                neighbors = list(self.graph.neighbors(curr))
                if not neighbors:
                    # print("Not enough neighbors")  # deal with this another way
                    curr = random.choice(nodes)
                    while curr in visited:
                        curr = random.choice(nodes)
                    neighbors = list(self.graph.neighbors(curr))
                nex = random.choice(neighbors)
                new_graph.add_edge(curr, nex)
                print(new_graph.number_of_nodes())

                if nex not in visited:
                    visited.add(nex)
                    nodes.remove(nex)
                curr = nex

        # return constucted graph
        return new_graph

    @staticmethod
    def get_method_name():
        return "Random Walk"


if __name__ == '__main__':
    orig_graph = read_graph("CA-GrQc.txt", n_skip_lines=4, directed_graph=False)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    graph_sample = RandomWalk(orig_graph, 0.95, final_number_of_nodes=5200)
    sample = graph_sample.random_sample()
    print()

    print("Original # Nodes:", sample.number_of_nodes())
    print("Original # Edges ", sample.number_of_edges())

    print()
