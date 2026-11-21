import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from maze_dataset.generation import LatticeMazeGenerators

warnings.filterwarnings("ignore", category=UserWarning)

SIZE = 30
random.seed(42)
np.random.seed(42)

GENERATORS = [
    ("1. DFS (Standard Corridors)", LatticeMazeGenerators.gen_dfs, {}),
    ("2. DFS (No Forks / Single Snake)", LatticeMazeGenerators.gen_dfs, {"do_forks": False}),
    ("3. Prim's (Randomized Stack)", LatticeMazeGenerators.gen_prim, {}),
    ("4. Kruskal's (Uniform Spanning)", LatticeMazeGenerators.gen_kruskal, {}),
    ("5. Wilson's (Unbiased Walk)", LatticeMazeGenerators.gen_wilson, {}),
    ("6. Recursive Division (Chambers)", LatticeMazeGenerators.gen_recursive_division, {}),
    ("7. Percolation (Open Clusters)", LatticeMazeGenerators.gen_percolation, {"p": 0.55}),
    ("8. DFS + Percolation (Looped)", LatticeMazeGenerators.gen_dfs_percolation, {"p": 0.15}),
    ("9. Heavy Loops (p=0.35)", LatticeMazeGenerators.gen_dfs_percolation, {"p": 0.35}),
]


def generate_all_mazes(size=SIZE, iters=100):
    """Generates all maze types and returns a list of (name, grid, start, end)."""

    mazes = []

    for iteration in range(iters):
        for name, gen_fn, kwargs in GENERATORS:
            raw_maze = gen_fn(grid_shape=(size, size), **kwargs)
            pixels = raw_maze.as_pixels()
            grid = (pixels[:, :, 0] == 0).astype(int)

            start = (1, 1)
            end = (grid.shape[0] - 2, grid.shape[1] - 2)

            mazes.append((name, grid, start, end))

    return mazes


if __name__ == "__main__":
    all_mazes = generate_all_mazes(SIZE)

    first_grid = all_mazes[0][1]

    print(
        f"Successfully generated {len(all_mazes)} mazes "
        f"({len(GENERATORS)} types, grid shape: {first_grid.shape})!"
    )

    # Visualize the first 9 generated mazes (one of each type)
    fig, axes = plt.subplots(3, 3, figsize=(16, 8))

    for ax, (name, grid, _, _) in zip(axes.flat, all_mazes[:9]):
        ax.imshow(grid, cmap="binary")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()