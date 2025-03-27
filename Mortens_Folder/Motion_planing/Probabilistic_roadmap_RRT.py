import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.plotting import plot_polygon, plot_line
from shapely.validation import make_valid
import random

class Node:
    """Node class for RRT"""
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []

def build_rrt(environment, bounds, q_s, q_g, max_nodes=10, step_size=0.5, goal_threshold=0.5):
    """Build an RRT from start to goal"""
    # Initialize tree with start node
    start_node = Node(q_s[0], q_s[1])
    nodes = [start_node]
    goal_reached = False
    goal_node = None

    for _ in range(max_nodes):
        # Sample random point (with 10% bias toward goal)
        if random.random() < 0.1:
            rand_point = Point(q_g[0], q_g[1])
        else:
            rand_x = random.uniform(bounds[0], bounds[2])
            rand_y = random.uniform(bounds[1], bounds[3])
            rand_point = Point(rand_x, rand_y)

        # Find nearest node in tree
        nearest_node = None
        min_dist = float('inf')
        for node in nodes:
            dist = np.sqrt((node.x - rand_point.x)**2 + (node.y - rand_point.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # Grow tree toward random point
        angle = np.arctan2(rand_point.y - nearest_node.y, rand_point.x - nearest_node.x)
        new_x = nearest_node.x + step_size * np.cos(angle)
        new_y = nearest_node.y + step_size * np.sin(angle)
        new_point = Point(new_x, new_y)

        # Check if path is collision-free
        line = LineString([(nearest_node.x, nearest_node.y), (new_x, new_y)])
        if environment.contains(line):
            # Create new node
            new_node = Node(new_x, new_y, nearest_node)
            nearest_node.children.append(new_node)
            nodes.append(new_node)

            # Check if we reached the goal
            dist_to_goal = np.sqrt((new_x - q_g[0])**2 + (new_y - q_g[1])**2)
            if dist_to_goal < goal_threshold:
                goal_node = Node(q_g[0], q_g[1], new_node)
                new_node.children.append(goal_node)
                nodes.append(goal_node)
                goal_reached = True
                break

    # Reconstruct path if goal was reached
    path = []
    if goal_reached:
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()
    
    return nodes, path

def plot_rrt(environment, nodes, path, q_s, q_g):
    """Plot the RRT and path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot environment
    plot_polygon(environment, ax=ax, add_points=False, color='lightgray')
    
    # Plot RRT edges
    for node in nodes:
        if node.parent is not None:
            line = LineString([(node.parent.x, node.parent.y), (node.x, node.y)])
            plot_line(line, ax=ax, color='blue', linewidth=0.5)
    
    # Plot start and goal
    ax.plot(q_s[0], q_s[1], 'go', markersize=10, label='Start')
    ax.plot(q_g[0], q_g[1], 'ro', markersize=10, label='Goal')
    
    # Plot path if found
    if path:
        path_line = LineString(path)
        plot_line(path_line, ax=ax, color='green', linewidth=2, label='Path')
    
    ax.set_aspect('equal')
    plt.title('Rapidly-exploring Random Tree (RRT)')
    plt.legend()
    plt.show()

# Define the environment (free space as a polygon)
outer_boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
obstacle1 = [(2, 2), (2, 4), (4, 4), (4, 2)]  # Clockwise for holes
obstacle2 = [(6, 6), (6, 8), (8, 8), (8, 6)]  # Clockwise for holes
obstacle3 = [(3, 7), (3, 9), (7, 9), (7, 7)]  # Clockwise for holes

# Create the polygon and ensure it's valid
environment = Polygon(outer_boundary, [obstacle1, obstacle2, obstacle3])
if not environment.is_valid:
    environment = make_valid(environment)

# Define bounds for sampling [min_x, min_y, max_x, max_y]
bounds = [0, 0, 10, 10]

# Define start and goal points
q_s = (1, 1)  # Start point
q_g = (9, 9)  # Goal point

# Build RRT
nodes, path = build_rrt(environment, bounds, q_s, q_g, max_nodes=1000, step_size=0.5)

# Plot the RRT and path
plot_rrt(environment, nodes, path, q_s, q_g)

if path:
    print("Path found!")
    print("Path points:", path)
else:
    print("No path found within the maximum number of nodes.")