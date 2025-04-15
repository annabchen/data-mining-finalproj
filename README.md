# data-mining-finalproj
Testing impact of sampling methods on graphs

Project structure: 
- Main: **graph_sampling_analyzer.py**. The "main" part of the script takes different sampling methods and graph files, and analyzes them according to the specified graph metrics, such as the total number of nodes/edges, mean degree, degree distribution, density, and assortativity.
- File Reading: **graph_reader.py**. Takes a graph as .txt file and provides a NetworkX graph object as an output.
- Data Plotting: **graph_plotter.py**. Takes the data from the main file and plots it as a line & scatter plot or a bar chart.
- Abtract Graph Sampler class: **GraphSampler.py**. Serves as an abstract class for other sampler classes.
- Graph Sampling classes: **random_node_sampler.py**,  **randomedgesampler.py**, **RandomJump.py**, **RandomWalk.py**, **forestfiresampler.py**, **snowball_sampling.py**, **wedge_sampling.py**. Sample the specified graph according to their algorithms and return a NetworkX type of graph as an output. 
