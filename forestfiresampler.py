import random
import numpy as np
import networkx as nx
import networkit as nk
from typing import Union
from collections import deque
from graph_reader import read_graph
from GraphSampler import GraphSampler


class FFSampler(GraphSampler):
    def __init__(self, graph, probOut=0.2, probIn=0.2, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100, isDirected=False):
        super().__init__(graph, final_number_of_nodes, final_number_of_edges, final_number_of_wedges, isDirected)
        self.probIn = probIn
        self.probOut = probOut

    def random_sample(self):
        nodes = list(self.graph.nodes)
        if not nodes:
            return nx.Graph() if not self.isDirected else nx.DiGraph()

        new_graph = nx.Graph() if not self.isDirected else nx.DiGraph()
        visited = set()
        seed_node = random.choice(nodes)
        queue = deque([seed_node])

        while new_graph.number_of_nodes() < self.final_number_of_nodes:
            if not queue:
                unvisited = set(nodes) - visited
                if not unvisited:
                    break  # No more nodes to explore
                seed_node = random.choice(list(unvisited))
                queue.append(seed_node)

            current_node = queue.popleft()
            if current_node in visited:
                continue

            visited.add(current_node)

            if self.isDirected:
                neighbors = list(set(
                    [t for _, t in self.graph.out_edges(current_node)] +
                    [s for s, _ in self.graph.in_edges(current_node)]
                ))
            else:
                neighbors = list(self.graph.neighbors(current_node))

            random.shuffle(neighbors)

            for neighbor in neighbors:
                if new_graph.number_of_nodes() >= self.final_number_of_nodes:
                    break
                if neighbor in visited:
                    continue

                prob = random.random()
                add_edge = False

                if not self.isDirected:
                    if prob >= self.probIn:
                        add_edge = True
                else:
                    if self.graph.has_edge(current_node, neighbor) and prob >= self.probOut:
                        add_edge = True
                    elif self.graph.has_edge(neighbor, current_node) and prob >= self.probOut / self.probIn:
                        add_edge = True

                if add_edge:
                    new_graph.add_edge(current_node, neighbor)
                    queue.append(neighbor)

        return new_graph

    @staticmethod
    def get_method_name():
        return "Forest Fire Sampling"


if __name__ == '__main__':
    orig_graph = read_graph("as-caida20071105.txt", n_skip_lines=8, directed_graph=True)
    # orig_graph = read_graph("CA-GrQc.txt", n_skip_lines=4, directed_graph=False)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges:", orig_graph.number_of_edges())

    graph_sample = FFSampler(orig_graph,  final_number_of_nodes=2.7*2600, final_number_of_edges=2.7*4918, probIn=.2)

    sample = graph_sample.random_sample()

    print("\nSampled # Nodes:", sample.number_of_nodes())
    print("Sampled # Edges:", sample.number_of_edges())
