import igraph as igraph
import matplotlib.pyplot as pyplot
import math
import heapq

"""
    This script is for my Advanced AI Class at the University of North Florida. It visualizes a graph based on
    coordinates and edges between vertexes in 'coords.txt' and 'graph.txt'. Additionally, it calculates the shortest 
    path between two vertexes using A*, taking into account that certain vertexes may be blocked by the user.
    The solution is also implemented using iGraph's Dijkstra solution to verify the accuracy of the implementation of
    the A* solution.
"""

# Calculate the weight between two points and return it. Uses Euclidean distance formula.
def calculate_euclidean_distance(start, destination):
   return math.sqrt((destination[0] - start[0])**2 + (destination[1] - start[1])**2)

node_coordinates = {} # Store coordinates for edge weight calculation
edges = [] # Store edges between nodes to be inserted into the graph
seen_edges = set() # We use this to make sure we don't store an edge twice
edge_attributes = {'weight': []} # Store weights for the edges. Will be placed in the same order as edges are calculated.
num_nodes = 0 # Keep track of the number of nodes as we read the number of coordinates

# Read, and store graph + coordinate data. Use data to create edges and weights to be later used as input for the iGraph graph.
try:
    with open('coords.txt') as coords:
        for entry in coords:
            num_nodes += 1 # For every line we have another node and it's coordinates, increment counter for input to iGraph graph.
            vertex_data = entry.split() # Split vertex id, x coordinate, and y coordinate into elements of an array
            vertex, x_coordinate, y_coordinate = int(vertex_data[0]), float(vertex_data[1]), float(vertex_data[2]) # Store them into variables of the correct data type
            node_coordinates[vertex - 1] = (x_coordinate, y_coordinate) # Store this element and it's coordinates in a hash table for future calculations

    with open('graph.txt') as graph:
        for entry in graph: # Get the vertex and connected nodes
            vertex_data = entry.split() # Split the vertex and connected nodes into elements of an array
            vertex = int(vertex_data[0]) - 1 # First element is the vertex, in iGraph we start indexing from 0, however this is abstracted from the user
            for data in vertex_data[1: len(vertex_data)]: # Ensure an edge is created for every corrected node
                destination = int(data) - 1 # Indexes start from 0 in iGraph

                # Edge A <-> B is the same as B <-> A, so we sort the edge and check to see if it's already been added.
                edge = tuple(sorted([vertex, destination]))

                # We do this by maintaining a seen_edges set
                if edge not in seen_edges:
                    edges.append(edge) # If it's an unseen edge, store it in the edges list 
                    distance = calculate_euclidean_distance(node_coordinates[vertex], node_coordinates[destination]) # Calculate edge weight
                    edge_attributes['weight'].append(distance) # Put the weight in the weight list under the weight key in the same order as the weights
                    seen_edges.add(edge) # Add this weight to seen so we don't add it again
except FileNotFoundError:
    print("Ensure both coords.txt and graph.txt are present in your project directory")

# Debugging statements to ensure graph data is stored correctly
'''
    print(node_coordinates)
    print(edges)
    print(edge_attributes)
    print(num_nodes)
    print(['blue' for node in range(num_nodes)])
'''

# Create our iGraph graph for visualization and A* calculation using the input data
# Note all nodes start as blue
g = igraph.Graph(
    n=num_nodes, edges=edges,
    edge_attrs=edge_attributes,
    vertex_attrs={'color': ['blue' for node in range(num_nodes)]},
    directed=False
)

# All nodes should be displayed starting from 1, unlike how they are stored in our data structures
g.vs["label"] = [str(vertex.index + 1) for vertex in g.vs] 


'''
    Debugging loop to ensure vertexes are correctly added to graph

    for v in g.vs:
        print(v)
'''

# Set matplotlib as default plotting backend for easily visualization and setup
igraph.config['plotting.backend'] = 'matplotlib'

# Plot the graph
igraph.plot(g, "graph_visualizations/original_graph.png")

print("To see your input graph visualized, please see graph_visualizations/original_graph.png")
print("Find shortest bath between two nodes where certain nodes might be blocked by obstacles.")
print("Please provide the following inputs as instructed.")

# Self explanatory, get starting node from user, repeat until a valid input is provided. 
# Note: Input is subtracted by 1 to reflect to how it stored in the program
while True:
    try:
        starting_node = int(input("Please enter the integer ID of your starting node (Use IDs as displayed on original_graph.png - Start Counting from 1): ")) - 1 
        if starting_node not in range(num_nodes):
            raise ValueError("The starting node ID provided is not in the graph.")
        print(f"Your starting node is {starting_node + 1}")
        break
    except ValueError as exception:
        print(f"Invalid input: {exception}")

# Same as above, however we are getting the goal node.
while True:
    try:
        goal_node = int(input("Please enter the integer ID of your goal node: ")) - 1
        if goal_node not in range(num_nodes):
            raise ValueError("The goal node ID provided is not in the graph.")
        print(f"Your starting node is {goal_node + 1}")
        break
    except ValueError as exception:
        print(f"Invalid input: {exception}")

blocked_set = set() # Blocked set is used to store which nodes are blocked for the A* execution

# Allow the user to keep blocking nodes until they press -1, note that starting node and end node can't be blocked
# If a solution becomes impossible after enough nodes are blocked, the program will display so later
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

# Store heuristic for each vertex                                  
node_2D_distance_to_goal = {}

# Get coordinates for goal vertex
goal_node_coordinates = node_coordinates[goal_node]

# Calculate heuristic for each vertex, note we set the vertex's color to red to know whether or not we can't visit it in A*
for vertex in g.vs:
    beginning_node_coordinates = node_coordinates[vertex.index]
    node_2D_distance_to_goal[vertex.index] = calculate_euclidean_distance(beginning_node_coordinates, goal_node_coordinates)

    if vertex.index in blocked_set:
        g.vs[vertex.index]["color"] = "red" # Nodes that can not be traveled to are marked in red 

g2 = g.copy() # Used for testing official solution, save a copy of the original graph before we change it to be used for igraph solution

# Data structure used to represent a part of our final solution. Each node contains a single vertex id
# along with the accumulated distance and heuristic cost to add that vertex to our solution. 
# The combination of these costs represent it's priority in being dequeued from a minheap. 
# Once a node is dequeued from the minheap with the goal as it's vertex ID, we can reconstruct 
# the solution path by following the parent recursively.  
class Node:
    def __init__(self, edge_weight, accumulated_distance, heuristic, id, parent):
        
        # Edge weight from parent to current node, used for getting final weight in path construction
        self.edge_weight = edge_weight

        # Keep track of cost to reach node so far
        self.accumulated_distance = accumulated_distance 
        
        # A node's priority in terms of when it should be expanded is determined by cost to reach that node, and how far the node is from the goal based on some heuristic.
        # In this case, we use the Euclidean Distance of a vertex from the goal node to get our heuristic estimate.
        self.priority = accumulated_distance + heuristic 
       
        # Node Vertex ID
        self.id = id

        # Get parent
        self.parent = parent

    # Define less than method for comparison in priority queue
    def __lt__(self, other):
        return self.priority < other.priority # Lower priority means greater potential for shorter path

# Keep track of vertexes which have already been extended. No reason to extend a vertex twice because it's already been reached before with the same or lower cost.
# This means it was reached with a equal, lower accumulated cost and the heuristic will be the same. Therefore, any path that tries to extend a vertex later than another
# that already reached this node can not possible be faster in reaching the goal node.
extended = set()

# The minheap or priority queue where we determine which nodes to check for extension next. 
# Nodes with the highest priority or lowest cost are dequeued first. See node class for better explanation of what constitutes the priority.
priority_queue = []

# Starting solution, contains a edge weight from parent of 0, an accumulated distance of zero, heuristic distance from goal, the starting vertex id, and a parent of None (similiar to NULL in other languages)
initial_node_solution = Node(0, 0, node_2D_distance_to_goal[starting_node], starting_node, None) 

# Add starting node to priority queue
heapq.heappush(priority_queue, initial_node_solution) 

# If we find the goal, and our heuristic is admissable, A* will give the optimal solution when the goal is first found. 
# Note that, in addition to the cost to reach that node, we use Euclidean distance or straight line distance for determining a node's future potential. 

# If we don't have a solution, this will be left as None
min_complete_solution = None

# Total cost, calculated recursively
total_path_cost = 0

# Final accumulated distance, will be same as total cost, used for demonstration
final_accumulated_distance = 0

# Final priority, same as total path cost and final accumulated distance. This is because the goal node has a heuristic cost of 0. Used for demonstration.
final_priority = 0

# Keep going until there are no more nodes left to extend, or the solution has been found (where we break out of this while loop).
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
        
        # Construct path, and get the cost
        total_path_cost = construct_path(current_node)

        # Showcase how accumulate distance of final node is the same 
        final_accumulated_distance = current_node.accumulated_distance

        # Same as above
        final_priority = current_node.priority
        break # A* is complete

    # If a vertex has already been extended, that means it was dequeued before at a higher priority (lower cost). 
    # That means we reached it previously with a lower accumlated cost because the heuristic for this vertex is the same.
    # This means that we should not expand it again as any solutions resulting from extending this vertex again would only have a worse or equal cost.
    if current_location not in extended: 
        extended.add(current_location) # Add it to our extended set so we don't extend it again
        vertex_edges = g.incident(current_location) # Get neighbors of this vertex (i.e, extend it)
        
        for edge in vertex_edges: # For each of our neighbors    
            edge = g.es[edge] # Get access to edge data
            if current_location == edge.source: # Get the correct target based on weather or not the current location is stored as the edge source
                target = edge.target
            else: # If current location is stored as edge.target, then the source of the stored edge should be our target
                target = edge.source

            # If target node is blocked we can't travel to it. If it's already been extended, no point in adding it to the priority queue because it won't be extended later anyway.
            if g.vs[target]["color"] == "red" or target in extended: 
                continue # It's in our blocked set or has already been extended

            # Distance to reach neighbor vertex from current vertex and cost to reach parent node     
            accumulated_distance = current_node.accumulated_distance + edge['weight'] 
            
            # Distance the neighbor is from the goal
            heuristic = node_2D_distance_to_goal[target] 

            # Store weight to reach this node from parent, accumulated distance to this node, heuristic distance from goal, priority for dequeing it, the vertex id, and it's parent node (essential for path reconstruction)
            neighbor_node = Node(edge['weight'], accumulated_distance, heuristic, target, current_node) 

            # Store neighbor node so it can be dequeued later
            heapq.heappush(priority_queue, neighbor_node)
    else: 
        continue # See comments above corresponding if statement to understand why we don't dequeue this vertex again

# Self explanatory, see print statements
if min_complete_solution == None:
    print("A path could not be found with the existing edges and / or obstacles using this project's solution.")
else:
    print("The following is the optimal path found by this project's implementation, in order of vertexes visited: ")
    print([v + 1 for v in min_complete_solution]) # Add 1 back to vertex IDs so we count from 1 in displaying to user
    for node in min_complete_solution:
        g.vs[node]["color"] = "green" # Set nodes traveled to in solution to green for when we display the solution
    print(f"Cost: {total_path_cost}") # Total path cost calculated recursively 
    print(f"Accumulated Distance: {final_accumulated_distance}") # Accumulated distance should be the same
    print(f"Final Priority: {final_priority}") # Same with priority

# Plot the graph with the solution
igraph.plot(g, "graph_visualizations/my_solution.png")

# Essentially block nodes by setting weights connected to them to infinity, we need to do this to block nodes in the library solution without actually deleting vertexes from our graph
for node in blocked_set:
    edges = list(g2.es.select(_source=node)) + list(g2.es.select(_target=node)) # Essentially get all edges where this blocked node is a source or target
    for edge in edges: # Set the weight of each of those edges to infinity
        edge['weight'] = float('inf')

# Built in iGraph solution using Dijkstra's algorithm to verify results 
library_solution_paths = g2.get_shortest_paths(v=starting_node, to=goal_node, weights=g2.es['weight'], output="vpath")
first_path = [v + 1 for v in library_solution_paths[0]] # Get the path. Note: this path may be different from the one provided by A*, however total cost should be identical.

# Get cost of starting node to goal node using Dijkstra's algorithm (assuming all weights are non negative)
library_solution_cost = g2.distances(starting_node, goal_node, weights=g2.es['weight'])

# Extract cost from 2D array, which has one row and one column. The element at [0][0] is the distance between the start, and target vertex.
final_cost = library_solution_cost[0][0]

# Make sure none of the nodes in the final path are in the blocked set, if there is, this means there was no better alternative than the blocked node which had an edge weight of infinity.
# Therefore, Dijkstra could not find a solution from the start to the goal based on current obstacles.
valid_path = True
for node in first_path:
    if node - 1 in blocked_set: # Subtract by one to get the actual index since iGraph indexes start at 0
        valid_path = False # A blocked node was in the solution, set valid to false
if valid_path: # If a valid path was found, print it to the user, along with it's cost
    print(f"The first path provided by the offical igraph shortest paths function (uses Dijkstra's algorithm for non negative weights): ")
    print(first_path)
    for node in first_path:
            g2.vs[node - 1]["color"] = "green" # Set nodes traveled to in solution to green in seperate graph
    print(f"Cost returned by libary solution: {final_cost}")
else: # Otherwise, inform them it couldn't be found
    print("Solution could not be found with the existing edges or blocked obstacles using the iGraph solution.")


# Plot the graph with the solution if any, otherwise print original graph with blocked nodes
igraph.plot(g2, "graph_visualizations/official_igraph_solution.png")
