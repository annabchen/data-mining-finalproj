import networkx as nx
import random

class RandomWalk:

    ## should be treating as directed graph?

    def __init__(self, graph, ssize, flyback):
        self.flyback = flyback
        self.ssize = ssize
        self.G = graph

    def sampler(self):
        """ Return sample of larger graph given flyback probability"""
        
        H = nx.Graph()

        # random start node
        start = random.choice(list(self.G.nodes()))
        print(f"Starting node: {start}")
        curr = start
        visited = set([start])
        s = 1

        # select nodes until reached desired size
        while s < self.ssize:
            if random.random() <= self.flyback:
                curr = start
            else:
                neighbors = list(self.G.neighbors(curr))
                if not neighbors:
                    print("Not enough neighbors") # deal with this another way
                nex = random.choice(neighbors)
                H.add_edge(curr, nex)
                if(nex not in visited):
                    visited.add(nex)
                    s += 1
                curr = nex

        # return constucted graph
        return H

