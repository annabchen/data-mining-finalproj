import random
from GraphSampler import GraphSampler
from graph_reader import read_graph


class WedgeSampler(GraphSampler):
    def __init__(self, graph,  final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=1000, isDirected=False):
        super().__init__(graph, final_number_of_nodes=final_number_of_nodes,
                         final_number_of_edges=final_number_of_edges, isDirected=isDirected)
        self.final_number_of_wedges = final_number_of_wedges

    def random_sample(self):
        """
        Randomly sample wedges from the graph, count how many are closed,
        and return a subgraph induced by the nodes involved in sampled wedges.
        """
        closed = 0
        sampled_wedges = 0
        wedge_nodes = set()

        nodes = list(self.graph.nodes())

        while sampled_wedges < self.final_number_of_wedges and len(wedge_nodes) < self.final_number_of_nodes:
            center = random.choice(nodes)
            neighbors = list(self.graph.neighbors(center))

            if len(neighbors) < 2:
                continue  # Can't form a wedge

            u, v = random.sample(neighbors, 2)

            # Count wedge
            wedge_nodes.update([u, center, v])
            sampled_wedges += 1

            # Check if wedge is closed (i.e., u-v edge exists)
            if self.graph.has_edge(u, v) or (not self.isDirected and self.graph.has_edge(v, u)):
                closed += 1

        # fraction_closed = closed / sampled_wedges if sampled_wedges else 0
        # estimated_triangles = int(fraction_closed * self.count_total_wedges())

        # print(f"Sampled {sampled_wedges} wedges.")
        # print(f"Fraction of closed wedges: {fraction_closed:.4f}")
        # print(f"Estimated number of triangles: {estimated_triangles}")

        induced_subgraph = self.graph.subgraph(wedge_nodes).copy()
        return induced_subgraph

    def count_total_wedges(self):
        """
        Count total number of wedges in the graph.
        C(d, 2) = d * (d - 1) / 2 wedges centered at a node.
        """
        total = 0
        for node in self.graph.nodes():
            d = self.graph.degree(node)
            if d >= 2:
                total += d * (d - 1) // 2
        return total

    @staticmethod
    def get_method_name():
        return "Wedge Sampling"


if __name__ == '__main__':
    # Approx. number of wedges:
    # as-caida: 34617; CA-GrQc: 52612
    orig_graph = read_graph("CA-GrQc.txt", n_skip_lines=4, directed_graph=False)

    print("Original # Nodes:", orig_graph.number_of_nodes())
    print("Original # Edges:", orig_graph.number_of_edges())

    print()

    sampler = WedgeSampler(orig_graph, final_number_of_nodes=1000, final_number_of_wedges=100000)
    sampled_graph = sampler.random_sample()
    print()

    print("Sampled # Nodes:", sampled_graph.number_of_nodes())
    print("Sampled # Edges:", sampled_graph.number_of_edges())

