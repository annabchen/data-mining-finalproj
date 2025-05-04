import networkx as nx
import random
from GraphSampler import GraphSampler
from graph_reader import read_graph


class RandomWalk(GraphSampler):
    def __init__(self, graph, flyback=0.9, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=False):
        self.flyback = flyback
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)

    def random_sample(self):
        """ Return sample of larger graph given flyback probability"""

        if self.flyback >= 1 or self.final_number_of_nodes > self.graph.number_of_nodes():
            print("Invalid input")
            return

        if not self.isDirected:
            new_graph = nx.Graph()
        else:
            new_graph = nx.DiGraph()

        # random start node
        nodes = list(self.graph.nodes())
        start = random.choice(nodes)
        curr = start
        visited = set({start})
        sample_nodes = 1
        steps = 0
        maxsteps = self.graph.number_of_nodes() # arbitrary value, may want to increase

        # select nodes until reached desired size
        while sample_nodes < self.final_number_of_nodes:
            # chance of flyback
            if random.random() > self.flyback:
                curr = start
            else:
                neighbors = list(self.graph.neighbors(curr))
                while not neighbors or steps > maxsteps:
                    # prevent getting stuck in traversal
                    steps = 0
                    curr = random.choice(nodes)
                    neighbors = list(self.graph.neighbors(curr))
                nex = random.choice(neighbors)
                new_graph.add_edge(curr, nex)
                if nex not in visited:
                    visited.add(nex)
                    sample_nodes += 1
                curr = nex
            steps += 1

        # return constucted graph
        return new_graph

    @staticmethod
    def get_method_name():
        return "Random Walk"


if __name__ == '__main__':
    orig_graph = read_graph("as-caida20071105.txt", n_skip_lines=8, directed_graph=True)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges ", orig_graph.number_of_edges())

    graph_sample = RandomWalk(orig_graph, 0.8, final_number_of_nodes=26104)
    sample = graph_sample.random_sample()
    print()

    print("Original # Nodes:", sample.number_of_nodes())
    print("Original # Edges ", sample.number_of_edges())

    print()