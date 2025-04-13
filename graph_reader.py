import networkx as nx


def read_graph(graph_file='CA-GrQc.txt', n_skip_lines=0, directed_graph=False):
    """READS A GRAPH FROM ITS FILE AND RETURN AS NETWORKX GRAPH"""
    # Open the file
    with open(graph_file, 'r') as file:
        # Skip the first N lines (e.g., skip first # lines)
        for _ in range(n_skip_lines):
            next(file)

        if not directed_graph:
            graph = nx.Graph()
            for line in file:
                arr = line.split("\t")  # Process the line (e.g., print it)
                first_node = int(arr[0])
                second_node = int(arr[1])
                graph.add_edge(first_node, second_node)

            return graph
        else:
            directed_graph = nx.DiGraph()
            for line in file:
                arr = line.split("\t")  # Process the line (e.g., print it)
                first_node = int(arr[0])
                second_node = int(arr[1])
                directionality = int(arr[2])
                if directionality == 1:
                    directed_graph.add_edge(first_node, second_node)
                elif directionality == -1:
                    directed_graph.add_edge(second_node, first_node)

            return directed_graph

if __name__ == '__main__':
    graph = read_graph()
    print("Nodes:", graph.nodes)
    print("\n\n\n\nEdges ", graph.edges)


