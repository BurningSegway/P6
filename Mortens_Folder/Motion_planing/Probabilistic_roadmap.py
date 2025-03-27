import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.plotting import plot_polygon, plot_line
from shapely.validation import make_valid

def hammersley_sequence(n, k):
    """Generate Hammersley sequence samples"""
    samples = []
    for i in range(n):
        p = 0
        f = 0.5
        j = i
        while j > 0:
            p += (j % 2) * f
            j = j // 2
            f /= 2
        samples.append([(i + 0.5) / n, p])
    return np.array(samples)

def generate_prm(environment, bounds, num_samples=100, k_neighbors=5):
    """Generate a Probabilistic Roadmap"""
    # Generate samples using Hammersley sequence
    samples = hammersley_sequence(num_samples, 2)
    
    # Scale samples to environment bounds
    samples[:, 0] = bounds[0] + samples[:, 0] * (bounds[2] - bounds[0])
    samples[:, 1] = bounds[1] + samples[:, 1] * (bounds[3] - bounds[1])
    
    # Filter samples that are in free space
    nodes = []
    for x, y in samples:
        point = Point(x, y)
        if environment.contains(point):
            nodes.append((x, y))
    
    # Build roadmap (simple version - just connect to nearest neighbors if line is collision-free)
    edges = []
    for i, (x1, y1) in enumerate(nodes):
        # Find k nearest neighbors
        distances = []
        for j, (x2, y2) in enumerate(nodes):
            if i != j:
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distances.append((j, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        for j, dist in distances[:k_neighbors]:
            x2, y2 = nodes[j]
            line = LineString([(x1, y1), (x2, y2)])
            if environment.contains(line):
                edges.append([(x1, y1), (x2, y2)])
    
    return nodes, edges

def plot_environment(environment, nodes=None, edges=None):
    """Plot the environment and PRM"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot environment obstacles (inverted since we're treating the polygon as free space)
    plot_polygon(environment, ax=ax, add_points=False, color='lightgray')
    
    # Plot edges
    if edges:
        for edge in edges:
            plot_line(LineString(edge), ax=ax, color='blue', linewidth=0.5)
    
    # Plot nodes
    if nodes:
        x = [node[0] for node in nodes]
        y = [node[1] for node in nodes]
        ax.scatter(x, y, color='red', s=10)
    
    ax.set_aspect('equal')
    plt.title('Probabilistic Roadmap (PRM) with Hammersley Sampling')
    plt.show()

# Define the environment (free space as a polygon)
# Here we create a simple rectangular environment with some holes
outer_boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
obstacle1 = [(2, 2), (2, 4), (4, 4), (4, 2)]  # Note: clockwise for holes
obstacle2 = [(6, 6), (6, 8), (8, 8), (8, 6)]  # Note: clockwise for holes
obstacle3 = [(3, 7), (3, 9), (7, 9), (7, 7)]  # Note: clockwise for holes

# Create the polygon and ensure it's valid
environment = Polygon(outer_boundary, [obstacle1, obstacle2, obstacle3])
if not environment.is_valid:
    environment = make_valid(environment)

# Define bounds for sampling [min_x, min_y, max_x, max_y]
bounds = [0, 0, 10, 10]

# Generate PRM
nodes, edges = generate_prm(environment, bounds, num_samples=200, k_neighbors=5)

# Plot the environment and PRM
plot_environment(environment, nodes, edges)