import abc


class GraphSampler(abc.ABC):
    """
    Abstract base class for graph samplers.
    All graph sampling strategies should inherit from this class.
    """

    def __init__(self, graph, final_number_of_nodes=100, final_number_of_edges=100,
                 final_number_of_wedges=100,isDirected=False):
        self.graph = graph
        self.final_number_of_nodes = int(final_number_of_nodes)
        self.final_number_of_edges = int(final_number_of_edges)
        self.final_number_of_wedges = int(final_number_of_wedges)
        self.isDirected = isDirected

    @abc.abstractmethod
    def random_sample(self):
        """
        Abstract method that must be implemented by all subclasses.
        Should return a sampled graph.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_method_name():
        """
        Returns the name of the sampling method.
        """
        pass