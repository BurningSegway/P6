import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point, LineString




# Function to check if a point is in free space (not in obstacles)
def is_free_space(point, obstacles):
    point_geom = Point(point)
    for obs in obstacles:
        if point_geom.intersects(obs):
            return False
    return True

# 1. Probabilistic Roadmap (Random Number Generator)
def prm_random(grid, obstacles, num_samples, All):
    rows, cols = grid.shape
    samples = []
    while len(samples) < num_samples:
        rand_point = (np.random.randint(0, cols), np.random.randint(0, rows))
        # Flip y-coordinate for visualization
        rand_point = (rand_point[0], rows - rand_point[1] - 1)
        if All == True:
            samples.append(rand_point)
        else:
            if is_free_space(rand_point, obstacles):
                samples.append(rand_point)
            



    return samples

# 2. Halton Sequence
def halton_sequence(num_samples, base=[2, 3]):
    def halton(n, base):
        result = 0.0
        f = 1.0 / base
        while n > 0:
            result += (n % base) * f
            n //= base
            f /= base
        return result

    points = []
    for i in range(num_samples):
        points.append([halton(i + 1, b) for b in base])
    return points

# 3. Hammersley Sequence
def hammersley_sequence(num_samples, dimensions=2):
    points = []
    for i in range(num_samples):
        x = (i + 1) / num_samples
        theta = 0
        for j in range(dimensions):
            theta += (i + 1) * (2 ** -(j + 1))
        points.append([x] + [theta])
    return points

# 4. Rapidly Exploring Random Trees (RRT)
def rrt(start, goal, obstacles, max_iter=1000, step_size=1):
    tree = [start]
    for _ in range(max_iter):
        rand_point = np.random.rand(2) * np.array([20, 20])  # Adjust this range as needed
        nearest_point = min(tree, key=lambda p: np.linalg.norm(np.array(p) - np.array(rand_point)))
        
        # Move towards the random point (stepping towards it)
        direction = np.array(rand_point) - np.array(nearest_point)
        step = direction / np.linalg.norm(direction) * step_size
        new_point = np.array(nearest_point) + step
        
        if is_free_space(new_point, obstacles):
            tree.append(tuple(new_point))
        
        if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
            tree.append(goal)
            break
    return tree

# Visualization function for plotting
def plot_samples(samples, obstacles, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot obstacles
    for obs in obstacles:
        patch = PolygonPatch(obs, fc='black', ec='black', alpha=0.8)
        ax.add_patch(patch)
    
    # Plot sampled points
    for sample in samples:
        ax.plot(sample[0], sample[1], 'ro', markersize=5)
    
    ax.set_aspect('equal')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    plt.show()



grid = np.array([

    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],

    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])




# Define cell size (assuming each cell is 1x1)
cell_size = 1

# Create polygons for obstacles
obstacles = []
rows, cols = grid.shape
obstacles = []
for r in range(rows):
    for c in range(cols):
        if grid[r, c] == 1:
            x, y = c * cell_size, (rows - r - 1) * cell_size  # Flip y-axis for visualization
            
            # Create polygon for the obstacle
            obs_poly = Polygon([
                (x, y), (x + cell_size, y), 
                (x + cell_size, y + cell_size), (x, y + cell_size)
            ])
            obstacles.append(obs_poly)




start = (0,0)
goal = (12,19)
num_samples = 100


# Testing the functions
samples_random = prm_random(grid, obstacles, num_samples, True)
samples_halton = halton_sequence(num_samples)
samples_hammersley = hammersley_sequence(num_samples)
samples_rrt = rrt(start, goal, obstacles)

# Plotting the random PRM samples
#plot_samples(samples_random, obstacles)
#plot_samples(samples_halton, obstacles)
#plot_samples(samples_hammersley, obstacles)
#plot_samples(samples_rrt, obstacles)

