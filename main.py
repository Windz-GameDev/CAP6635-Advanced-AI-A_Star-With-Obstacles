import igraph as igraph
import matplotlib.pyplot as pyplot
import math
import heapq
from copy import deepcopy

# Calculate the weight between two points and return it. Uses Euclidean distance formula.
def calculate_euclidean_distance(start, destination):
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
                    distance = calculate_euclidean_distance(node_coordinates[vertex], node_coordinates[destination])
                    edge_attributes['weight'].append(distance)
                    seen_edges.add(edge)
except FileNotFoundError:
    print("Ensure both coords.txt and graph.txt are present in your project directory")

# Debugging statements to ensure graph data is stored correctly

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


'''
    Debugging loop to ensure vertexes are correctly added to graph

    for v in g.vs:
        print(v)
'''

# Set matplotlib as default plotting backend
igraph.config['plotting.backend'] = 'matplotlib'

# Plot the graph
igraph.plot(g, "graph_visualizations/original_graph.png")

print("To see your input graph visualized, please see graph_visualizations/original_graph.png")
print("Find shortest bath between two nodes where certain nodes might be blocked by obstacles.")
print("Please provide the following inputs as instructed.")

while True:
    try:
        starting_node = int(input("Please enter the integer ID of your starting node: ")) - 1
        if starting_node not in range(num_nodes):
            raise ValueError("The starting node ID provided is not in the graph.")
        print(f"Your starting node is {starting_node + 1}")
        break
    except ValueError as exception:
        print(f"Invalid input: {exception}")

while True:
    try:
        goal_node = int(input("Please enter the integer ID of your goal node: ")) - 1
        if goal_node not in range(num_nodes):
            raise ValueError("The goal node ID provided is not in the graph.")
        print(f"Your starting node is {goal_node + 1}")
        break
    except ValueError as exception:
        print(f"Invalid input: {exception}")

blocked_set = set()

while True:
    try:
        blocked_node = int(input("Please enter the integer ID of the node you want to block. Enter -1 to stop blocking nodes: ")) - 1
        if blocked_node == -2:
            break
        elif blocked_node not in range(num_nodes):
            raise ValueError("This node can not be blocked because it does not exist in the graph.")
        elif blocked_node in [starting_node, goal_node]:
            raise ValueError("You can not block a starting node or a goal node.")
        else:
            blocked_set.add(blocked_node)
            print(f"Node {blocked_node + 1} was added to the blocked set, it will not be traveled to in the constructed path.")
    except ValueError as exception:
        print(f"Invalid input: {exception}")
                                  
node_2D_distance_to_goal = {}
goal_node_coordinates = node_coordinates[goal_node]

for vertex in g.vs:
    beginning_node_coordinates = node_coordinates[vertex.index]
    node_2D_distance_to_goal[vertex.index] = calculate_euclidean_distance(beginning_node_coordinates, goal_node_coordinates)

    if vertex.index in blocked_set:
        g.vs[vertex.index]["color"] = "red" # Nodes that can not be traveled to are marked in red 

vertex_edges = g.incident(4)
print(f" Edges for 4 {vertex_edges}") # Get neighbors of this vertex)

g2 = g.copy() # Used for testing official solution, save a copy of the original graph before we change it to be used for igraph solution

class Node:
    def __init__(self, accumulated_distance, heuristic, path):
        
        # Keep track of cost to reach node so far
        self.accumulated_distance = accumulated_distance 
        
        # A node's priority in terms of when it should be expanded is determined by cost to reach that node, and how far the goal is the node from the goal based on some heuristic
        self.priority = accumulated_distance + heuristic 
        
        # Store solution represented by this path
        self.path = path

    # Define less than method for comparison in priority queue
    def __lt__(self, other):
        return self.priority < other.priority # Lower priority means greater potential for shorter path


extended = set()
priority_queue = []

# Starting solution, contains an accumulated distance of zero, and priority in dequeing is determined by node's distance to goal, initial path solution only contains start node so not complete
initial_node_solution = Node(0, node_2D_distance_to_goal[starting_node], [starting_node]) 
heapq.heappush(priority_queue, initial_node_solution) # Add a path with only a starting node to priority queue

# If we find the goal, and our heuristic is admissable, A* will give the optimal solution when the goal is first found. 
# Note that, in addition to the cost to reach that node, we use Euclidean distance or straight line distance for determining a node's future potential. 
min_complete_solution = None

while (priority_queue):
    
    current_node_solution = heapq.heappop(priority_queue) # Enqueue starting solution 
    current_location_in_solution = current_node_solution.path[-1] # Get the last node in the solution
    
    print(f"Current solution : {current_node_solution.path}")
    
    if current_location_in_solution == goal_node: # If the last node's solution represents us having reached the goal vertex at the end, then we have the solution, as our heuristic is admissable since we use Euclidean distance. 
        min_complete_solution = current_node_solution
        break

    # If a vertex has already been extended, that means it was dequeued before at a higher priority (lower cost). 
    # That means we reached it previously with a lower accumlated cost because the heuristic for this vertex is the same.
    # This means that we should not expand it again as any solutions resulting from this vertex again would only have a worse cost.
    if current_location_in_solution not in extended: 
        extended.add(current_location_in_solution) # Add it to our extended set so we don't extend it again
        print(current_location_in_solution)
        vertex_edges = g.incident(current_location_in_solution) # Get neighbors of this vertex
        for edge in vertex_edges: # For each of our neighbors
            
            edge = g.es[edge] # Get access to edge data
            print(f"Checking vertex {edge.target} : edges : {edge.source} {edge.target}")

            if current_location_in_solution == edge.source:
                target = edge.target
            else:
                target = edge.source

            # If target node is blocked, or already in our path, we can't travel to it, move on to next neighor, if any
            if g.vs[target]["color"] == "red" or target in current_node_solution.path: 
                print(f"{target} is blocked")
                continue # It's in our blocked set 

            # Distance to reach neighbor vertex from current vertex     
            accumulated_distance = current_node_solution.accumulated_distance + edge['weight'] 
            
            # Distance the neighbor is from the goal
            heuristic = node_2D_distance_to_goal[target] 

            # New path
            new_path = [element for element in current_node_solution.path]
            new_path.append(target)

            # Store accumulated distance to that node, priority for dequeing it, and the current solution represented by it
            neighbor_node = Node(accumulated_distance, heuristic, new_path) 

            # Store neighbor node so it can be dequeued later
            heapq.heappush(priority_queue, neighbor_node)
    else: 
        continue # See comments above corresponding if statement to understand why we don't dequeue this vertex again


if min_complete_solution == None:
    print("A path could not be found with the existing edges and / or obstacles using this project's solution.")
else:
    print("The following is the optimal path found by this project's implementation, in order of vertexes visited: ")
    print([v + 1 for v in min_complete_solution.path])
    for node in min_complete_solution.path:
        g.vs[node]["color"] = "green" # Set nodes traveled to in solution to green

# Plot the graph with the solution
igraph.plot(g, "graph_visualizations/my_solution.png")

# Essentially block nodes by setting weights connected to them to infinity
for node in blocked_set:
    edges = list(g2.es.select(_source=node)) + list(g2.es.select(_target=node)) # Essentially get all edges where blocked node is a source or target
    for edge in edges:
        edge['weight'] = float('inf')

# Built in iGraph solution using Dijkstra's algorithm to verify results 
library_solution_paths = g2.get_shortest_paths(v=starting_node, to=goal_node, weights=g2.es['weight'], output="vpath")
first_path = [v + 1 for v in library_solution_paths[0]]

valid_path = True
for node in first_path:
    if node - 1 in blocked_set: # Subtract by one to get the actual index since iGraph indexes start at 0
        valid_path = False
if valid_path:
    print(f"The first path provided by the offical igraph shortest paths function: ")
    print(first_path)
    for node in first_path:
            g2.vs[node - 1]["color"] = "green" # Set nodes traveled to in solution to green
else:
    print("Solution could not be found with the existing edges or blocked obstacles using the iGraph solution.")


# Plot the graph with the solution if any, otherwise print original graph with blocked nodes
igraph.plot(g2, "graph_visualizations/official_igraph_solution.png")
