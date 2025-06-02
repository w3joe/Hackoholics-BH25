import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mesh_utils 
from collections import deque
from typing import List, Tuple

def bfs_shortest_path_with_walls(grid: np.ndarray, start: Tuple, goal: Tuple) -> List[Tuple[int, int]]:
    """
    Perform BFS on a grid with interior walls encoded in each cell’s bits.
    Bits in each cell (LSB=bit 0) indicate:
      - bit 4 (1<<4 = 16): right wall
      - bit 5 (1<<5 = 32): bottom wall
      - bit 6 (1<<6 = 64): left wall
      - bit 7 (1<<7 = 128): top wall

    Movement from (r,c) to (nr,nc) is allowed only if:
      • There is no wall on the shared edge in the current cell, AND
      • There is no wall on the shared edge in the neighbor cell.

    Parameters:
    -----------
    grid : 2D NumPy array of ints
        Each entry’s bits mark walls around that cell.
    start : tuple (r, c)
        Starting coordinates (row, col).
    goal : tuple (r, c)
        Goal coordinates (row, col).

    Returns:
    --------
    path : list of (r, c) tuples
        Shortest path from start to goal (inclusive), or [] if unreachable.
    """

    rows, cols = grid.shape

    def has_top(r, c):
        return mesh_utils.has_top_wall(grid[r, c])
    def has_bottom(r, c):
        return mesh_utils.has_bottom_wall(grid[r, c])
    def has_left(r, c):
        return mesh_utils.has_left_wall(grid[r, c])
    def has_right(r, c):
        return mesh_utils.has_right_wall(grid[r, c])

    visited = [[False] * cols for _ in range(rows)]
    parent = {}  # parent[(r, c)] = (pr, pc) or None
    q = deque([start])
    visited[start[0]][start[1]] = True
    parent[start] = None

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break

        # Explore neighbors in 4 directions:
        # 1) Up: (r-1, c)
        if r > 0:
            nr, nc = r - 1, c
            if (not visited[nr][nc]
                and not has_top(r, c)        # current has no top wall
                and not has_bottom(nr, nc)):  # neighbor has no bottom wall
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

        # 2) Down: (r+1, c)
        if r < rows - 1:
            nr, nc = r + 1, c
            if (not visited[nr][nc]
                and not has_bottom(r, c)     # current has no bottom wall
                and not has_top(nr, nc)):     # neighbor has no top wall
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

        # 3) Left: (r, c-1)
        if c > 0:
            nr, nc = r, c - 1
            if (not visited[nr][nc]
                and not has_left(r, c)       # current has no left wall
                and not has_right(nr, nc)):   # neighbor has no right wall
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

        # 4) Right: (r, c+1)
        if c < cols - 1:
            nr, nc = r, c + 1
            if (not visited[nr][nc]
                and not has_right(r, c)      # current has no right wall
                and not has_left(nr, nc)):    # neighbor has no left wall
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    # Reconstruct path if goal was reached
    if goal not in parent:
        return []  # no path

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]

def plot_route(map_array: np.ndarray, height_of_map: int, length_of_map: int, route_coordinates: List[Tuple[int, int]]) -> None:
    """
    Visualizes a grid map with interior walls, coordinate labels, and a given route.

    Parameters:
    -----------
    map_array : 2D np.ndarray
        Grid map where each cell contains an integer encoding walls via bit masks.
    height_of_map : int
        Number of rows in the map.
    length_of_map : int
        Number of columns in the map.
    route_coordinates : list of (row, col) tuples
        The ordered sequence of coordinates representing the route to be drawn.

    Returns:
    --------
    None. Saves the rendered image as "bfs_plot.png".
    """

    # Create a square plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid values using pcolormesh with visible cell borders
    mesh = ax.pcolormesh(map_array, cmap='viridis', shading='auto',
                         facecolor='none', edgecolors='k', linewidth=0.5)

    # Add a colorbar to indicate value scaling
    fig.colorbar(mesh, ax=ax, label='Value')

    # Ensure (0,0) appears at the top-left by inverting the y-axis
    ax.set_ylim(height_of_map, 0)

    # Set ticks at the center of each cell for both axes
    centers_x = np.arange(0.5, length_of_map + 0.5, 1.0)
    centers_y = np.arange(0.5, height_of_map + 0.5, 1.0)
    ax.set_xticks(centers_x)
    ax.set_yticks(centers_y)
    ax.set_xticklabels(np.arange(0, length_of_map))
    ax.set_yticklabels(np.arange(0, height_of_map))

    # Force equal aspect ratio so cells are square
    ax.set_aspect('equal', adjustable='box')

    # Iterate over all cells to draw walls and overlay coordinate labels
    for (r, c) in np.ndindex(map_array.shape):
        value = map_array[r, c]

        # Draw walls if present (bitmask-based)
        if mesh_utils.has_right_wall(value):
            mesh_utils.draw_right_wall(ax, c, r)
        if mesh_utils.has_bottom_wall(value):
            mesh_utils.draw_bottom_wall(ax, c, r)
        if mesh_utils.has_left_wall(value):
            mesh_utils.draw_left_wall(ax, c, r)
        if mesh_utils.has_top_wall(value):
            mesh_utils.draw_top_wall(ax, c, r)

        # Optional debug print (can be commented out if not needed)
        print(f"Coordinate ({r}, {c}) → value = {value} "
              f"top: {mesh_utils.has_top_wall(value)} "
              f"bottom: {mesh_utils.has_bottom_wall(value)} "
              f"right: {mesh_utils.has_right_wall(value)}")

        # Overlay coordinate label at the center of each cell
        ax.text(
            c + 0.5,
            r + 0.5,
            f"{r},{c}",
            ha='center',
            va='center',
            fontsize=6,
            color='white',
            bbox=dict(
                facecolor='black',
                alpha=0.3,
                boxstyle='round,pad=0.1'
            )
        )

    # Extract x and y coordinates from the route (adjusted to center of cells)
    x_coords = [c + 0.5 for (r, c) in route_coordinates]
    y_coords = [r + 0.5 for (r, c) in route_coordinates]

    # Draw the route path with red lines and yellow circular markers
    ax.plot(
        x_coords,
        y_coords,
        color='red',
        linewidth=2,
        marker='o',
        markersize=8,
        markerfacecolor='yellow',
        markeredgecolor='black'
    )

    # Label axes and title
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.set_title("Map")

    # Ensure everything fits nicely
    plt.tight_layout()

    # Save the figure to file
    plt.savefig("bfs_plot.png")


# map of novice env
novice_map = np.array([
    [193,  65,  98, 194, 114, 194,  66,  66,  83,  82,  98, 194,  98, 194,  66,  98],
    [129,   1,  35, 146,  83,  34, 130,  51, 195, 114, 146,  50, 130,   2,  34, 162],
    [146,   2,  19,  67,  66,   2,   3,  98, 146,  82,  98, 195,   3,   3,  34, 162],
    [194,  34, 194,   2,   2,   2,   2,   2,  98, 194,  35, 131,  50, 178, 146,  34],
    [131,  18,  51, 130,   3,   3,   3,   2,   2,  34, 178, 146,  98, 194,  83,  50],
    [146,  98, 194,   2,   2,   2,   2,   3,  34, 130,  66,  82,  18,  50, 210,  98],
    [194,  18,  50, 130,   2,   2,   2,   3,  34, 130,   2,  83,  82,  82,  66,  34],
    [162, 194,  98, 130,   2,   2,   3,   2,   2,   2,   2,  66,  66,  66,   2,  34],
    [162, 131,  34, 130,   3,   2,   3,   2,   3,   3,   2,   3,   2,   2,  35, 163],
    [130,  50, 162, 130,   2,   2,   3,  34, 162, 130,   2,   2,   2,  34, 130,  50],
    [162, 210,  34, 146,  18,  18,  18,  19,  50, 130,   2,  18,  18,  51, 146,  98],
    [146,  67,   2,  67,  82,  82,  66,  82,  82,  18,  18,  82,  99, 194,  67,  50],
    [195,  50, 163, 130,  66,  82,  34, 210,  82,  82,  83,  67,  18,  34, 130,  98],
    [162, 226, 162, 147,  50, 194,   2,  83,  66,  82,  66,  18,  82,  18,   2,  35],
    [162, 162, 162, 210,  98, 162, 162, 227, 130,  99, 163, 194,  83,  83,  18,  50],
    [146,  18,  19,  82,  18,  50, 147,  51, 178, 146,  18,  18,  82,  82,  82, 114]
]).astype(int)

height = novice_map.shape[0]
length = novice_map.shape[1]

# transpose (if needed, in this case need. If map looks crazy, comment this line out and see what happens)
novice_map = novice_map.transpose()

# run bfs to find shortest path between two points
route = bfs_shortest_path_with_walls(novice_map, (0,0), (9,15))

# plot route
plot_route(novice_map, height, length, route)
