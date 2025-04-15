import networkx as nx
import sys
from randomwalk import RandomWalk

def reader(input):
    """Generate graph object from input file"""

    G = nx.Graph()
    with open(input, 'r') as file:
        for line in file:
            a, b = map(int, line.strip().split())
            G.add_edge(a, b)
        return G
    
def eval(H):
    pass

def main():
    input = sys.argv[1]
    G = reader(input)
    walk = RandomWalk(G, 200, 0.15)
    H = walk.sampler()
    print(list(H.nodes))
    print(list(H.edges))

if __name__ == "__main__":
    main()
