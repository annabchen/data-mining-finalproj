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
        snodes = 1
        steps = 0
        maxsteps = 100 * self.G.number_of_nodes() # can adjust this

        # select nodes until reached desired size
        while snodes < self.ssize:
            if random.random() <= self.flyback:
                curr = start
            else:
                neighbors = list(self.G.neighbors(curr))
                if not neighbors or steps > maxsteps:
                    steps = 0
                    curr = random.choice(list(self.G.nodes()))
                nex = random.choice(neighbors)
                H.add_edge(curr, nex)
                if(nex not in visited):
                    visited.add(nex)
                    snodes += 1
                curr = nex
            steps += 1

        # return constucted graph
        return H