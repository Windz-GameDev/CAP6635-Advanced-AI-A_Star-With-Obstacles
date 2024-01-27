import igraph as igraph
import matplotlib.pyplot as pyplot
import math

# Calculate the weight between two points and return it. Uses Euclidean distance formula.
def calculate_edge_weight(start, destination):
   return math.sqrt((destination[0] - start[0])**2 + (destination[1] - start[1])**2)

node_coordinates = {} # Store coordinates for edge weight calculation
edges = [] # Store edges between nodes to be inserted into the graph
seen_edges = set() # We use this to make sure we don't store an edge twice
edge_attributes = {'weight': []} # Store weights for the edges. Will be placed in the same order as edges are calculated.
num_nodes = 0 # Keep track of the number of nodes as we read the number of coordinates

try:
    with open('coords.txt') as coords:
        for entry in coords:
            num_nodes += 1
            vertex_data = entry.split()
            vertex, x_coordinate, y_coordinate = int(vertex_data[0]), float(vertex_data[1]), float(vertex_data[2])
            node_coordinates[vertex - 1] = (x_coordinate, y_coordinate)

    with open('graph.txt') as graph:
        for entry in graph:
            vertex_data = entry.split()
            vertex = int(vertex_data[0]) - 1
            for data in vertex_data[1: len(vertex_data)]:
                destination = int(data) - 1
                edge = tuple(sorted([vertex, destination]))

                if edge not in seen_edges:
                    edges.append(edge)
                    distance = calculate_edge_weight(node_coordinates[vertex], node_coordinates[destination])
                    edge_attributes['weight'].append(distance)
                    seen_edges.add(edge)
except FileNotFoundError:
    print("Ensure both coords.txt and graph.txt are present in your project directory")

print(node_coordinates)
print(edges)
print(edge_attributes)
print(num_nodes)
print(['blue' for node in range(num_nodes)])

g = igraph.Graph(
    n=num_nodes, edges=edges,
    edge_attrs=edge_attributes,
    vertex_attrs={'color': ['blue' for node in range(num_nodes)]},
    directed=False
)

g.vs["label"] = [str(vertex.index + 1) for vertex in g.vs] 

# Set matplotlib as default plotting backend
igraph.config['plotting.backend'] = 'matplotlib'

# Plot the graph
igraph.plot(g)

# Show the graph
pyplot.show()

