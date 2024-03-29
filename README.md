This is my implementation for A*, a widely used, informed, pathfinding algorithm which uses a combination of accumulated and heuristic costs to make more intelligent decisions than simple DFS or BFS. The algorithm is particularly useful for robotics and is a cornerstone in the field of Artificial Intelligence. This project was created for the Advanced AI Class I am taking at the University of North Florida while undertaking my Masters in CS.

This Python project does the following
  - Reads the x-y coordinates of the nodes from a text file called "coords.txt".
  - Reads an adjacency list of an undirected graph from a text file called "graph.txt" in the project directory. This is used to create the edges.
    -- Note: If an edge includes a point in graph.txt, coordinates for that point must exist in coords.txt!
  - Uses the iGraph library for Python to create the graph.
  - Node IDs are 0 and positive integers.
  - Edge weights are calculated based on the nodes' coordinates or the Euclidean distance between two nodes.
  - If there is an edge between nodes a and b, an edge is not added for b and a.
  - The program also displays the graph using matplotlib.
  - The program implements the A* algorithm, allowing the user to find the shortest path between two nodes on the graph. Additionally, the user can block nodes to represent real-world obstacles.

Set up
  - Ensure Python 3.x is installed on your system. You can download it from python.org.
  - Clone this repository to your local machine or download the project files.
  - Navigate to the project directory and create a virtual environment:
    - `python -m venv venv`
  - Activate the virtual environment:
    - On Windows Command Prompt:
      - `.\venv\Scripts\activate`
    - On Windows PowerShell
      - `.\venv\Scripts\Activate.ps1`
        - Note: If you run into issues on PowerShell, you may need to run the following command as an administrator to temporarily change your execution policy.
        - `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
        - This will allow you to run scripts for the current PowerShell session only, without permanently changing your execution policy.
        - If you prefer not to do this, simply use the Windows Command Prompt instead.
    - On macOS and Linux:
      - `source venv/bin/activate`
  - Install the required dependencies:
      - `pip install -r requirements.txt`

Preparing Your Data
- Prepare two text files in the project directory: coords.txt for node coordinates and graph.txt, which is formatted as an adjacency list.
- The coords.txt file should list each node's ID followed by its x and y coordinates, separated by spaces. Start counting nodes from 0 or greater.
- The graph.txt file should list the node IDs followed by the IDs of nodes it is connected to, representing the graph's edges.
- If you have an edge from x -> y in graph.txt, you don't need another edge for y -> x
- Note: Vertex IDs must be integers, or you will run into issues. X and Y coordinates may be floats or integers.


Your graph and coords text files should look like the following: 

Note: In the coordinates file, you may start from 0 or any positive integer, as long as Vertex IDs are sequential by line. Additionally, vertexes must be defined in the coords file before an edge can be placed between it and another 
vertex in the graph file.
  
![Coordinate File Example](coords_file_example.png "Coords File Example")
  
![Graph File Example](graph_file_example.png "Graph File Example")

Running the Program
- With your virtual environment activated and the required text files prepared, run the program using:
  - `python main.py`
- See below image for an example of how to use the program
  - ![Advanced AI - Running Example](example_of_running_program.png "Advanced AI - Running Example")

Viewing the Results
- After running the program, check the `graph_visualizations` folder in the project directory for the generated visualization images. These images will show the original graph and the calculated shortest path.
- The images are saved as `original_graph.png` for the initial graph visualization, `my_solution.png` for my A* implementation, and `official_igraph_solution.png` for the path returned back the iGraph Dijkstra implementation. 
- The visited nodes are highlighted in green in the solutions, while the blocked nodes are highlighted in red.
- The exact path order can seen in the lists in the console. 
  - Note: it is possible my path, and the iGraph solution path may be different, however the total cost should be the same.

Troubleshooting
- Ensure that all node IDs in graph.txt have corresponding coordinates in coords.txt.
- If the program cannot find a path, verify that the starting and goal nodes are connected and that not too many nodes are blocked.
- Ensure vertex indexes start counting from 0 or greater in your input files.
- In coords.txt, ensure each Vertex ID is one greater than the vertex ID of the previous line.
- If you reference a vertex in graph.txt, ensure it's location is defined in coords.txt.
- When choosing which nodes to start from, end on, or block, ensure you enter their IDs as displayed in `original_graph.png`.

My Acknowledgements and Gratitude Goes To
- My professor, Dr. Ayan Dutta for introducing me to A* during class, answering my questions, and providing valuable suggestions for optimizing my implementation.
- The following free MIT video on Youtube from Patrick Winston, was extremely helpful for further understanding the logic behind A*, and was used as a reference as I was writing the logic for the A* code.
  - https://www.youtube.com/watch?v=gGQ-vAmdAOI&t=2306s
