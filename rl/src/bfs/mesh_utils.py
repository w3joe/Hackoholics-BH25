import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes

# functions for drawing walls
def draw_top_wall(ax: Axes, c: int, r: int) -> None:

    """
    Draws the top wall of a grid cell at position (r, c).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes on which to draw the wall.
    c : int
        The column index of the cell.
    r : int
        The row index of the cell.

    Returns:
    --------
    None
    """

    ax.plot(
        [c, c+1],  # x-coordinates for the line's start/end
        [r, r],      # y-coordinate (same for start/end)
        color='black',
        linewidth=5
    )

def draw_bottom_wall(ax: Axes, c: int, r: int) -> None:

    """
    Draws the bottom wall of a grid cell at position (r, c).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes on which to draw the wall.
    c : int
        The column index of the cell.
    r : int
        The row index of the cell.

    Returns:
    --------
    None
    """

    ax.plot(
        [c, c+1],  # x-coordinates for the line's start/end
        [r+1, r+1],      # y-coordinate (same for start/end)
        color='black',
        linewidth=5
    )

def draw_left_wall(ax: Axes, c: int, r: int) -> None:

    """
    Draws the left wall of a grid cell at position (r, c).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes on which to draw the wall.
    c : int
        The column index of the cell.
    r : int
        The row index of the cell.

    Returns:
    --------
    None
    """
    
    ax.plot(
        [c, c],  # x-coordinates for the line's start/end
        [r, r+1],      # y-coordinate (same for start/end)
        color='black',
        linewidth=5
    )

def draw_right_wall(ax: Axes, c: int, r: int) -> None:

    """
    Draws the right wall of a grid cell at position (r, c).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axes on which to draw the wall.
    c : int
        The column index of the cell.
    r : int
        The row index of the cell.

    Returns:
    --------
    None
    """

    ax.plot(
        [c+1, c+1],  # x-coordinates for the line's start/end
        [r, r+1],      # y-coordinate (same for start/end)
        color='black',
        linewidth=5
    )

# functions for checking grid is what walls
"""
Returns whether the value is a right/bottom/left/top wall.

Bit index (LSB = index 0)
  2 → Scout
  3 → Guard
  4 → Right wall
  5 → Bottom wall
  6 → Left wall (64)
  7 → Top wall (128)

Parameters:
-----------
mask : int
    Grid value

Returns:
--------
bool
    True if the right/bottom/left/top wall is present, False otherwise.
"""

def has_right_wall(mask: int) -> bool:
    return (mask & (1 << 4)) != 0

def has_bottom_wall(mask: int) -> bool:
    return (mask & (1 << 5)) != 0

def has_left_wall(mask: int) -> bool:
    return (mask & (1 << 6)) != 0

def has_top_wall(mask: int) -> bool:
    return (mask & (1 << 7)) != 0
