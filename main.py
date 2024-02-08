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
    def __init__(self, edge_weight, accumulated_distance, heuristic, id, parent):
        
        # Edge weight from parent to current node, used for getting final weight in path construction
        self.edge_weight = edge_weight

        # Keep track of cost to reach node so far
        self.accumulated_distance = accumulated_distance 
        
        # A node's priority in terms of when it should be expanded is determined by cost to reach that node, and how far the goal is the node from the goal based on some heuristic
        self.priority = accumulated_distance + heuristic 
       
        # Node Vertex ID
        self.id = id

        # Get parent
        self.parent = parent

    # Define less than method for comparison in priority queue
    def __lt__(self, other):
        return self.priority < other.priority # Lower priority means greater potential for shorter path


extended = set()
priority_queue = []

# Starting solution, contains an accumulated distance of zero, and priority in dequeing is determined by node's distance to goal
initial_node_solution = Node(0, 0, node_2D_distance_to_goal[starting_node], starting_node, None) 
heapq.heappush(priority_queue, initial_node_solution) # Add a path with only a starting node to priority queue

# If we find the goal, and our heuristic is admissable, A* will give the optimal solution when the goal is first found. 
# Note that, in addition to the cost to reach that node, we use Euclidean distance or straight line distance for determining a node's future potential. 
min_complete_solution = None
total_path_cost = 0
final_accumulated_distance = 0
final_priority = 0
while (priority_queue):
    
    current_node = heapq.heappop(priority_queue) # Enqueue starting solution 
    current_location = current_node.id # Get the vertex location represented by popped node
    
    if current_location == goal_node: # If the last node's solution represents us having reached the goal vertex at the end, then we have the solution, as our heuristic is admissable since we use Euclidean distance. 
        
        # We reached the goal so solution is not none
        min_complete_solution = []

        # Recursively construct the path by using recursion, i.e go to node used to reach the goal node, then node used to reach that node, etc, all the way to the beginning.
        def construct_path(node_to_add):    
            # If we can't go further back return, and start constructing the path
            if node_to_add == None:
                return 0

            # Cost so far to connect the previous nodes
            cost_so_far = construct_path(node_to_add.parent)

            # Add node to path
            min_complete_solution.append(node_to_add.id)

            # Return cost so far + cost to reach this node from it's parent
            return cost_so_far + node_to_add.edge_weight
        total_path_cost = construct_path(current_node)
        final_accumulated_distance = current_node.accumulated_distance
        final_priority = current_node.priority

        break

    # If a vertex has already been extended, that means it was dequeued before at a higher priority (lower cost). 
    # That means we reached it previously with a lower accumlated cost because the heuristic for this vertex is the same.
    # This means that we should not expand it again as any solutions resulting from this vertex again would only have a worse cost.
    if current_location not in extended: 
        
        extended.add(current_location) # Add it to our extended set so we don't extend it again
        vertex_edges = g.incident(current_location) # Get neighbors of this vertex
        
        for edge in vertex_edges: # For each of our neighbors    
            edge = g.es[edge] # Get access to edge data
            if current_location == edge.source:
                target = edge.target
            else:
                target = edge.source

            # If target node is blocked we can't travel to it. If it's already been extended, no point in adding it to the priority queue because it won't be extended
            if g.vs[target]["color"] == "red" or target in extended: 
                continue # It's in our blocked set or has already been extended

            # Distance to reach neighbor vertex from current vertex     
            accumulated_distance = current_node.accumulated_distance + edge['weight'] 
            
            # Distance the neighbor is from the goal
            heuristic = node_2D_distance_to_goal[target] 

            # Store weight to reach this node from parent, accumulated distance to this node, heuristic distance from goal, priority for dequeing it, node id, and it's parent
            neighbor_node = Node(edge['weight'], accumulated_distance, heuristic, target, current_node) 

            # Store neighbor node so it can be dequeued later
            heapq.heappush(priority_queue, neighbor_node)
    else: 
        continue # See comments above corresponding if statement to understand why we don't dequeue this vertex again


if min_complete_solution == None:
    print("A path could not be found with the existing edges and / or obstacles using this project's solution.")
else:
    print("The following is the optimal path found by this project's implementation, in order of vertexes visited: ")
    print([v + 1 for v in min_complete_solution])
    for node in min_complete_solution:
        g.vs[node]["color"] = "green" # Set nodes traveled to in solution to green
    print(f"Cost: {total_path_cost}")
    print(f"Accumulated Distance: {final_accumulated_distance}")
    print(f"Final Priority: {final_priority}")

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

# Get cost of starting node to goal node using Dijkstra's algorithm (assuming all weights are non negative)
library_solution_cost = g2.distances(starting_node, goal_node, weights=g2.es['weight'])

# Extract cost from 2D array, which has one row and one column. The element at [0][0] is the distance between the start, and target vertex.
final_cost = library_solution_cost[0][0]

valid_path = True
for node in first_path:
    if node - 1 in blocked_set: # Subtract by one to get the actual index since iGraph indexes start at 0
        valid_path = False
if valid_path:
    print(f"The first path provided by the offical igraph shortest paths function (uses Dijkstra's algorithm for non negative weights): ")
    print(first_path)
    for node in first_path:
            g2.vs[node - 1]["color"] = "green" # Set nodes traveled to in solution to green
    print(f"Cost returned by libary solution: {final_cost}")
else:
    print("Solution could not be found with the existing edges or blocked obstacles using the iGraph solution.")


# Plot the graph with the solution if any, otherwise print original graph with blocked nodes
igraph.plot(g2, "graph_visualizations/official_igraph_solution.png")
