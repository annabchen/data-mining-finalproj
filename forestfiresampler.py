import random
import numpy as np
import networkx as nx
import networkit as nk
from typing import Union
from collections import deque
from graph_reader import read_graph
from GraphSampler import GraphSampler


class FFSampler(GraphSampler):
    def __init__(self, graph, probOut, probIn, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=True):
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)
        self.probIn = probIn
        self.probOut = probOut

    def random_sample(self):
        if not self.isDirected:
            new_graph = nx.Graph()
        else:
            new_graph = nx.DiGraph()

        # Start from a random seed node
        nodes = list(self.graph.nodes)

        seed_node = random.choice(nodes)
        visited = set()  # Stores the visited nodes
        queue = deque([seed_node]) # Stores the next nodes to be visited

        while (new_graph.number_of_nodes() < self.final_number_of_nodes and
               new_graph.number_of_edges() < self.final_number_of_edges):
            current_node = seed_node
            if len(queue) != 0:
                # If there are no more nodes to visit,
                # but the while condition is still true,
                # then select a new seed node
                current_node = queue.popleft()
            else:
                seed_node = random.choice(nodes)
                if seed_node in visited:
                    continue
                queue = deque([seed_node])
            if current_node in visited:
                continue

            visited.add(current_node)
            neighbors = list(self.graph.neighbors(current_node))
            random.shuffle(neighbors)  # Shuffle neighbors to avoid bias

            # Add all neighbors of each node a queue
            for neighbor in neighbors:
                if new_graph.number_of_nodes() >= self.final_number_of_nodes or \
                        new_graph.number_of_edges() >= self.final_number_of_edges:
                    break
                if random.random() >= self.probOut or random.random() >= self.probIn:
                    new_graph.add_edge(current_node, neighbor)

                if neighbor not in visited:
                    queue.append(neighbor)

        return new_graph

    @staticmethod
    def get_method_name():
        return "Forest Fire Sampling"

    def burn(self, graph):
        """
                start w a single random node and burn outgoing edges.
                when the edge is burned, the nodes connected have a chance of catching fire too:
                - selects x nodes
                - select in-links with probability r (backward burnign ratio) times less than out-links (forward burning prob p)
                nodes cannot be visited a second time

        """
        # if seed is null, call starting_set
        # while the

    def starting_set(self, graph):
        """
            Create a starting set of nodes.
        """


if __name__ == '__main__':
    orig_graph = read_graph("CA-GrQc.txt", n_skip_lines=4, directed_graph=False)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges:", orig_graph.number_of_edges())

    graph_sample = FFSampler(orig_graph, 0.5, 0.5, 5000, 5000)

    sample = graph_sample.random_sample()

    print("\nSampled # Nodes:", sample.number_of_nodes())
    print("Sampled # Edges:", sample.number_of_edges())
