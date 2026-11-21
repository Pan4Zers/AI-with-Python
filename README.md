# Maze Generation and AI Search Benchmarking

A Python-based framework for generating procedurally varied grid mazes, validating their structure, and comparing classical search algorithms on a common set of navigation problems.

The project examines how search strategies behave across different maze-generation methods and evaluates them using computational, path-quality, and memory-related measurements.

## Project Structure

| File | Responsibility |
|---|---|
| `maze_generator.py` | Generates maze instances using nine generation configurations. |
| `maze_validator.py` | Measures connectivity, wall density, dead ends, and junctions. |
| `main.py` | Implements search algorithms, runs the benchmark, computes performance statistics, and generates visualisations. |

## Maze Generation

Nine maze configurations are used to produce different structural characteristics:

1. Depth-first search (standard corridors)
2. Depth-first search (no forks / single snake)
3. Randomised Prim's algorithm
4. Kruskal's algorithm
5. Wilson's algorithm
6. Recursive division
7. Percolation
8. DFS with percolation-based loops
9. DFS with a higher percolation probability to produce denser loops

The generators are provided by the `maze_dataset` package. Generated pixel mazes are converted into binary grids, where open cells are represented by `0` and walls by `1`.

The start and goal are placed at the grid coordinates `(1, 1)` and `(rows - 2, columns - 2)`, respectively.

Maze generation accepts an explicit random seed. The benchmark selects a seed and prints it at the end of execution.

## Maze Validation

Each generated maze is evaluated before it is included in the search benchmark.

The validator calculates:

- **Connectivity:** Whether the start and goal are connected through open cells.
- **Wall density:** The fraction of grid cells occupied by walls.
- **Dead ends:** The number of open cells with exactly one open neighbour.
- **Junctions:** The number of open cells with at least three open neighbours.

Connectivity and neighbourhood-based structural measurements use eight-directional adjacency. Diagonal movement is permitted, including movement between diagonally adjacent open cells when both orthogonally adjacent cells are walls.

Only mazes in which the start and goal are connected are passed to the algorithm benchmark. Validation statistics are calculated over the generated set.

## Search Algorithms

Six search algorithms are included:

| Algorithm | Description |
|---|---|
| **Breadth-First Search (BFS)** | Explores the grid level by level and returns a path with the minimum number of moves. |
| **Depth-First Search (DFS)** | Explores one branch at a time and returns the first path it finds. |
| **Dijkstra's Algorithm** | Finds a minimum-cost path using the geometric movement costs. |
| **A\*** | Uses the octile-distance heuristic with accumulated movement cost to guide the search. |
| **Flood Fill** | Computes a distance field from the goal using geometric movement costs, then reconstructs a minimum-cost path from the start. |
| **Bidirectional A\*** | Searches from both the start and goal using octile-distance heuristics. |

### Movement Model

The search algorithms use eight-directional movement.

- Horizontal and vertical moves have a cost of \(1\).
- Diagonal moves have a cost of \(\sqrt{2}\).

Diagonal corner-cutting is allowed: a diagonal move is valid whenever its destination cell is open and within the grid, regardless of the two cells adjacent to the diagonal.

The octile-distance heuristic is used by A* and bidirectional A*. It is appropriate for this movement model because it represents the minimum geometric distance on an obstacle-free eight-connected grid.

Dijkstra's algorithm provides the reference path cost for the benchmark's optimality-ratio calculation.

## Benchmarking and Evaluation

Every connected maze is evaluated using all six algorithms. The benchmark records the following metrics:

| Metric | Interpretation |
|---|---|
| **Execution time** | Time taken by an algorithm to find a path or establish that no path exists, measured in milliseconds. |
| **Nodes explored** | Number of nodes removed from the algorithm's active search structure and processed. |
| **Path length** | Number of cells in the returned path, including the start and goal. |
| **Geometric path cost** | Sum of the movement costs along the returned path. |
| **Optimality ratio** | Returned path cost divided by the Dijkstra path cost. A value of \(1\) indicates a path with the same cost as Dijkstra's result. |
| **Peak memory** | Peak traced memory during an algorithm run, measured using Python's `tracemalloc` module. |

Execution time, nodes explored, path length, and path cost are recorded for each connected maze. Peak memory is sampled separately for up to the configured number of mazes.

The benchmark reports mean, median, and standard deviation for selected performance measurements, along with success rates and average path costs.

## Visualisation

The benchmark produces two groups of visual outputs:

- **Performance plots** in `maze_analysis_plots/`: per-algorithm distributions across mazes and aggregate comparisons between algorithms.
- **Path overlays** in `maze_solution_overlays/`: combined path visualisations for mazes selected as statistical outliers.

The performance plots cover execution time, nodes explored, path length, peak memory, optimality ratio, and geometric path cost.

Outlier selection uses the interquartile range (IQR) rule. For each algorithm and metric, the most extreme lower and upper outliers are selected when present. The corresponding maze is visualised with the paths found by all algorithms, allowing their routes to be compared on the same grid. Shared path segments are distinguished from segments used by individual algorithms.

## Running the Benchmark

Execute `main.py` to run the benchmark with its configured defaults.

The principal benchmark parameters are defined by `run_tests(iter=1000, size=15, memory_sample=1000)` in `main.py`:

- `iter`: controls the number of generation cycles across the nine maze configurations.
- `size`: sets the generated grid dimensions.
- `memory_sample`: maximum number of connected mazes used for memory measurements.

The benchmark creates the visualisation directories in the working directory and saves the generated plots there.

## Dependencies

The project uses:

- NumPy
- Matplotlib
- maze-dataset
To install, run this command:
`pip install numpy matplotlib maze-dataset`
Python standard-library modules are also used for timing, memory measurement, priority queues, filesystem operations, and queue-based traversal.
