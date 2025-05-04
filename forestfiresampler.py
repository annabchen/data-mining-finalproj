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
        if not self.isDirected:
            # undirected version, selects from all connected nodes with even probability
            new_graph = nx.Graph()
            # Start from a random seed node
            nodes = list(self.graph.nodes)

            seed_node = random.choice(nodes)
            visited = set()  # Stores the visited nodes
            queue = deque([seed_node])  # Stores the next nodes to be visited

            # loop while sampled size is smaller than final desired values
            while new_graph.number_of_nodes() < self.final_number_of_nodes:
                current_node = seed_node
                if len(queue) != 0:
                    # If there are no more nodes to visit,
                    # but the while condition is still true,
                    # then select a new seed node
                    current_node = queue.popleft()
                else:
                    # randomly choose a new seed node
                    seed_node = random.choice(nodes)
                    # if selected seed node has been visited, continue thru the loop
                    if seed_node in visited:
                        allNodes = set(self.graph.nodes)
                        unvisited = allNodes.difference(visited)
                        unvisitedList = list(unvisited)
                        # is no unvisited nodes, return
                        if len(unvisitedList) == 0:
                            return new_graph

                        seed_node = random.choice(unvisitedList)
                    # add selected seed node to queue of nodes to process
                    queue = deque([seed_node])

                # after this loop, current node will have been visited
                visited.add(current_node)
                # find connected nodes
                neighbors = list(self.graph.neighbors(current_node))
                random.shuffle(neighbors)  # Shuffle neighbors to avoid bias

                # Add all neighbors of each node a queue
                for neighbor in neighbors:
                    if new_graph.number_of_nodes() >= self.final_number_of_nodes:
                        # leave loop if at capacity for # of nodes
                        break
                    # if random num is over threshold, add that neighbor to new_graph, can only use nodes which ahvent been visited
                    if random.random() >= self.probIn and neighbor not in visited:
                        new_graph.add_edge(current_node, neighbor)
                        # add neighbor to visited
                        queue.append(neighbor)

        else:
            # directed version- select in-links with probability r (backward burnign ratio) times less than out-links (forward burning prob p)
            new_graph = nx.DiGraph()
            # Start from a random seed node
            nodes = list(self.graph.nodes)

            seed_node = random.choice(nodes)
            visited = set()  # Stores the visited nodes
            queue = deque([seed_node])  # Stores the next nodes to be visited

            # loop while sampled size is smaller than final desired values
            while new_graph.number_of_nodes() < self.final_number_of_nodes:
                current_node = seed_node
                if len(queue) != 0:
                    # If there are no more nodes to visit,
                    # but the while condition is still true,
                    # then select a new seed node
                    current_node = queue.popleft()
                else:
                    # randomly choose a new seed node
                    seed_node = random.choice(nodes)
                    # if selected seed node has been visited, continue thru the loop
                    if seed_node in visited:
                        continue
                    # add selected seed node to queue of nodes to process
                    queue = deque([seed_node])
                # after this loop, current node will have been visited
                visited.add(current_node)
                # find connected nodes
                neighbors = list(self.graph.neighbors(current_node))
                random.shuffle(neighbors)  # Shuffle neighbors to avoid bias
                # generate the in-edges and out-edges of the seed node
                inN = list(self.graph.in_edges(current_node))
                in_neighbors = [source for source, target in inN]
                outN = list(self.graph.out_edges(current_node))
                out_neighbors = [target for source, target in outN]
                # print(outN)
                # Add all neighbors of each node a queue
                for neighbor in neighbors:
                    if new_graph.number_of_nodes() >= self.final_number_of_nodes:
                        # leave loop if at capacity for # of nodes
                        break
                    # if random num is over threshold, add that neighbor to new_graph, can only use nodes which ahvent been visited
                    if neighbor not in visited:
                        prob = random.random()
                        if neighbor in out_neighbors and prob >= self.probOut:
                            new_graph.add_edge(current_node, neighbor)
                            # add neighbor to visited
                            queue.append(neighbor)
                        if neighbor in in_neighbors and prob >= self.probOut / self.probIn:
                            new_graph.add_edge(current_node, neighbor)
                            # add neighbor to visited
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

    graph_sample = FFSampler(orig_graph,  final_number_of_nodes=2000, final_number_of_edges=2000)

    sample = graph_sample.random_sample()

    print("\nSampled # Nodes:", sample.number_of_nodes())
    print("Sampled # Edges:", sample.number_of_edges())
