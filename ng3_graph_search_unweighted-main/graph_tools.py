from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def read_maze_text(maze_file_path: Path | str) -> list:
    """Read text file defining a maze

    Parameters
    ----------
    maze_file_path : Path | str
        String or Path with text file path

    Returns
    -------
    list
        List with each text line, newline characters removed
    """
    with open(maze_file_path) as file:
        lines = []
        for line in file:
            lines.append(line.rstrip())
    return lines


def maze_text_to_graph(maze_text: list) -> dict:
    """_summary_

    Parameters
    ----------
    maze_text : list
        List of maze text lines.
        Walls in the maze are indicated with '#' symbols.
        Any other symbol is interpreted as open space (a node).
        A "wall" of '#'s surrounding the maze is assumed.
        The maze is assumed to be rectangular (each text
        line is assumed to be equally long).

    Returns
    -------
    dict
        A dict containing nodes and edges in the graph.
        The keys in the dict are 2-element tuples (y,x),
        where y and x are integers indicating node position.
        Position follows standard matrix / image indexing,
        with upper left corner being origo, x axis pointing
        from left to right, and y axis pointing from top to bottom.
        The values in the dict are lists with neighbor nodes,
        which are also identified with (y,x) tuples.

    Example
    -------
    maze_text:  ['#####',
                 '#...#',
                 '#.###',
                 '#####']
    graph:     {(1, 1): [(2, 1), (1, 2)],
                (1, 2): [(1, 1), (1, 3)],
                (1, 3): [(1, 2)],
                (2, 1): [(1, 1)]}
    """
    graph = {}
    for i, line in enumerate(maze_text):
        for j, char in enumerate(line):
            if char != "#":  # If maze tile is not blocked
                graph[(i, j)] = []  # Add tile (node) to graph dict
                if maze_text[i][j - 1] != "#":  # If valid node to the west
                    graph[(i, j)].append((i, j - 1))
                if maze_text[i + 1][j] != "#":  # If valid node to the south
                    graph[(i, j)].append((i + 1, j))
                if maze_text[i][j + 1] != "#":  # If valid node to the east
                    graph[(i, j)].append((i, j + 1))
                if maze_text[i - 1][j] != "#":  # If valid node to the north
                    graph[(i, j)].append((i - 1, j))
    return graph


def maze_text_to_matrix(maze_text: list) -> NDArray:
    """_summary_

    Parameters
    ----------
    maze_text : list
        List of maze text lines.

    Returns
    -------
    NDArray
        Maze represented as a 2D NumPy array, with
        0's indicating walls and 1's indicating open space (nodes)
    """
    n_rows = len(maze_text)
    n_cols = len(maze_text[0])
    grid = np.zeros((n_rows, n_cols), dtype=int)
    for i, line in enumerate(maze_text):
        for j, char in enumerate(line):
            if char == "#":
                grid[i, j] = 0
            else:
                grid[i, j] = 1
    return grid


def plot_maze_edges(graph):
    """Plot edges between maze nodes as arrows

    # Arguments:
    edges:      Dict, key = (from) node, value = list of (to) nodes
    """
    for node in graph.keys():
        for neighbor in graph[node]:
            y0, x0 = node
            y, x = neighbor
            dy, dx = y - y0, x - x0
            plt.arrow(x=x0, y=y0, dx=0.8 * dx, dy=0.8 * dy, head_width=0.1)
