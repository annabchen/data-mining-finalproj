import networkx as nx
import random
from GraphSampler import GraphSampler
from graph_reader import read_graph


class RandomJump(GraphSampler):
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
                curr = random.choice(nodes)  # this one line differentiates from Random Walk - jump to new point
            elif curr is not None:
                neighbors = list(self.graph.neighbors(curr))
                if not neighbors:
                    # print("Not enough neighbors")  # deal with this another way
                    curr = random.choice(nodes)
                    while curr in visited:
                        curr = random.choice(nodes)
                    neighbors = list(self.graph.neighbors(curr))

                nex = None

                # Fails for the directed graph
                if len(neighbors) > 0:
                    nex = random.choice(neighbors)
                    new_graph.add_edge(curr, nex)
                #  print(new_graph.number_of_nodes())

                if nex not in visited:
                    visited.add(nex)
                curr = nex

        # return constucted graph
        return new_graph

    @staticmethod
    def get_method_name():
        return "Random Jump"


if __name__ == '__main__':
    orig_graph = read_graph("as-caida20071105.txt", n_skip_lines=8, directed_graph=True)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    graph_sample = RandomJump(orig_graph, final_number_of_nodes=26000)
    sample = graph_sample.random_sample()
    print()

    print("Original # Nodes:", sample.number_of_nodes())
    print("Original # Edges ", sample.number_of_edges())

    print()
