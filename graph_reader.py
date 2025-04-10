import networkx as nx


def read_graph(graph_file='toyGraph.txt', n_skip_lines=0):
    # Open the file
    with open(graph_file, 'r') as file:
        # Skip the first N lines (e.g., skip first # lines)
        for _ in range(n_skip_lines):
            next(file)

        graph = nx.Graph()

        for line in file:
            arr = line.split("\t")  # Process the line (e.g., print it)
            first_node = int(arr[0].strip())
            second_node = int(arr[1].strip())

            graph.add_node(first_node)
            graph.add_node(second_node)
            graph.add_edge(first_node, second_node)

        return graph

if __name__ == '__main__':
    graph = read_graph()
    print("Nodes:", graph.nodes)
    print("\n\n\n\nEdges ", graph.edges)


