import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import heapq
from maze_generator import generate_all_mazes
from collections import deque
from maze_validator import validate_maze
import os
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def get_neighbours(graph, node):
    r, c = node
    rows, cols = graph.shape
    neighbours = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,1),(-1,-1),(1,-1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbours.append((nr,nc))

    neighbours = [i for i in neighbours if graph[i] == 0]

    return neighbours

def movement_cost(a, b): # local calculations
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])

    if dr == 1 and dc == 1:
        return np.sqrt(2)
    return 1

def backtrack(came_from, start, end):
    if end not in came_from:
        return []
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

def bfs(graph, start, end): 
    queue = deque([start])
    came_from = {start: None}
    nodes_explored = 0

    while queue: 
        curr = queue.popleft()
        nodes_explored += 1

        if curr == end: 
            return backtrack(came_from, start, end), nodes_explored

        for nxt in get_neighbours(graph, curr):
            if nxt not in came_from: 
                came_from[nxt] = curr
                queue.append(nxt)

    return [], nodes_explored

def dfs(graph, start, end): 
    stack = [start]
    came_from = {start: None}
    nodes_explored = 0

    while stack: 
        curr = stack.pop()
        nodes_explored += 1

        if curr == end: 
            return backtrack(came_from, start, end), nodes_explored

        for nxt in get_neighbours(graph, curr):
            if nxt not in came_from: 
                came_from[nxt] = curr
                stack.append(nxt)

    return [], nodes_explored

def iddfs(graph, start, end):
    max_depth = graph.shape[0]*graph.shape[1]
    nodes_explored = 0

    def dls(curr, depth, path_set):
        nonlocal nodes_explored
        nodes_explored += 1

        if curr == end:
            return [curr]
        if depth == 0:
            return None

        for nxt in get_neighbours(graph, curr):
            if nxt not in path_set:
                path_set.add(nxt)
                result = dls(nxt, depth - 1, path_set)
                if result is not None:
                    return [curr] + result
                path_set.remove(nxt)  # Backtrack so other branches can use this cell
        return None

    for depth_limit in range(max_depth):
        print(f"Depth limit: {depth_limit}", flush=True)
        found_path = dls(start, depth_limit, {start})
        if found_path is not None:
            return found_path, nodes_explored

    return [], nodes_explored



def manhattan(a, b):
    """Heuristic function for grid movement (4 directions)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def octile(a,b):
    """Heuristic function for grid movement (8 directions)."""
    return max(abs(a[0]-b[0]), abs(a[1] - b[1])) + (np.sqrt(2)-1)*min(abs(a[0]-b[0]), abs(a[1] - b[1]))

def dijkstra(grid, start, end, weights=None):
    # weights can be a 2D array of step costs; defaults to 1 per step if None
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0

    while pq:
        curr_cost, curr = heapq.heappop(pq)
        if curr_cost > cost_so_far.get(curr, float("inf")):
            continue
        nodes_explored += 1

        if curr == end:
            return backtrack(came_from, start, end), nodes_explored

        for nxt in get_neighbours(grid, curr):
            step_cost = movement_cost(curr, nxt)
            if weights is not None:
                step_cost *= weights[nxt]
            new_cost = curr_cost + step_cost
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = curr
                heapq.heappush(pq, (new_cost, nxt))

    return [], nodes_explored

def astar(grid, start, end, weights=None):
    pq = [(octile(start, end), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0

    while pq:
        _, curr_cost, curr = heapq.heappop(pq)
        if curr_cost > cost_so_far.get(curr, float("inf")):
            continue
        nodes_explored += 1

        if curr == end:
            return backtrack(came_from, start, end), nodes_explored

        for nxt in get_neighbours(grid, curr):
            step_cost = movement_cost(curr, nxt)
            if weights is not None:
                step_cost *= weights[nxt]
            new_cost = curr_cost + step_cost
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = curr
                priority = new_cost + octile(nxt, end)
                heapq.heappush(pq, (priority, new_cost, nxt))

    return [], nodes_explored

def floodfill(grid, start, end):
    distances = {end: 0}
    pq = [(0, end)]
    nodes_explored = 0

    # Stage 1: Calculate minimum geometric distances from the end.
    while pq:
        curr_dist, curr = heapq.heappop(pq)

        if curr_dist > distances.get(curr, float("inf")):
            continue

        nodes_explored += 1

        for nxt in get_neighbours(grid, curr):
            new_dist = curr_dist + movement_cost(curr, nxt)

            if new_dist < distances.get(nxt, float("inf")):
                distances[nxt] = new_dist
                heapq.heappush(pq, (new_dist, nxt))

    if start not in distances:
        return [], nodes_explored

    # Stage 2: Reconstruct the minimum-cost path.
    path = [start]
    curr = start

    while curr != end:
        for nxt in get_neighbours(grid, curr):
            if np.isclose(
                distances.get(nxt, float("inf")) + movement_cost(curr, nxt),
                distances[curr]
            ):
                path.append(nxt)
                curr = nxt
                break

    return path, nodes_explored

def bidirectional_astar(grid, start, end):
    if start == end:
        return [start], 1

    f_pq = [(octile(start, end), 0, start)]
    b_pq = [(octile(end, start), 0, end)]

    f_came_from = {start: None}
    b_came_from = {end: None}

    f_cost = {start: 0}
    b_cost = {end: 0}

    best_total_cost = float("inf")
    meeting_node = None
    nodes_explored = 0

    while f_pq and b_pq:
        if f_pq[0][0] + b_pq[0][0] >= best_total_cost and meeting_node is not None:
            break

        if len(f_pq) <= len(b_pq):
            _, g, curr = heapq.heappop(f_pq)
            if g > f_cost.get(curr, float("inf")):
                continue
            nodes_explored += 1

            for nxt in get_neighbours(grid, curr):
                new_g = g + movement_cost(curr, nxt)
                if new_g < f_cost.get(nxt, float("inf")):
                    f_cost[nxt] = new_g
                    f_came_from[nxt] = curr
                    heapq.heappush(f_pq, (new_g + octile(nxt, end), new_g, nxt))

                    if nxt in b_cost and new_g + b_cost[nxt] < best_total_cost:
                        best_total_cost = new_g + b_cost[nxt]
                        meeting_node = nxt
        else:
            _, g, curr = heapq.heappop(b_pq)
            if g > b_cost.get(curr, float("inf")):
                continue
            nodes_explored += 1

            for nxt in get_neighbours(grid, curr):
                new_g = g + movement_cost(curr, nxt)
                if new_g < b_cost.get(nxt, float("inf")):
                    b_cost[nxt] = new_g
                    b_came_from[nxt] = curr
                    heapq.heappush(b_pq, (new_g + octile(nxt, start), new_g, nxt))

                    if nxt in f_cost and new_g + f_cost[nxt] < best_total_cost:
                        best_total_cost = new_g + f_cost[nxt]
                        meeting_node = nxt

    if meeting_node is None:
        return [], nodes_explored

    forward_path = backtrack(f_came_from, start, meeting_node)
    backward_path = []
    curr = b_came_from[meeting_node]
    while curr is not None:
        backward_path.append(curr)
        curr = b_came_from[curr]

    return forward_path + backward_path, nodes_explored

def calculate_path_cost(path):
    return sum(
        movement_cost(path[i], path[i+1])
        for i in range(len(path)-1)
    )

ALGOS = {
    "BFS": bfs,
    "DFS": dfs,
#    "IDDFS": iddfs,
    "DIJKSTRA": dijkstra,
    "A*": astar,
    "FLOOD_FILL": floodfill,
    "Bi-A*": bidirectional_astar
}


def safe_stats(values):
    if not values:
        return float("nan"), float("nan"), float("nan")

    return (
        np.mean(values),
        np.median(values),
        np.std(values)
    )


def plot_comparisons(results):
    metrics = [
        ("time_ms", "Average Execution Time (ms)"),
        ("nodes", "Average Nodes Explored"),
        ("length", "Average Path Length"),
        ("peak_kb", "Average Peak Memory (KB)"),
        ("optimality", "Average Optimality Ratio"),
        ("path_cost", "Average Geometric Path Cost")
    ]

    for metric, title in metrics:
        names = []
        values = []

        for name, data in results.items():
            if data[metric]:
                names.append(name)
                values.append(np.mean(data[metric]))

        if not values:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(names, values)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel("Algorithm")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


def plot_maze(grid, path, start, end):
    plt.figure(figsize=(7, 7))

    plt.imshow(grid, cmap="binary")

    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]

        plt.plot(cols, rows, linewidth=2)

    plt.scatter(start[1], start[0], marker="o", label="Start")
    plt.scatter(end[1], end[0], marker="*", label="End")

    plt.title("Maze and BFS Solution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_combined_paths(grid, paths, start, end, maze_index, output_dir):

    plt.figure(figsize=(9, 9))
    plt.imshow(grid, cmap="binary", interpolation="nearest")

    # Assign a distinct colour to each algorithm.
    cmap = plt.get_cmap("tab10")

    algo_colors = {
        name: cmap(i % 10)
        for i, name in enumerate(paths)
    }

    # Map every traversed edge to the algorithms using it.
    edge_owners = {}

    for name, path in paths.items():

        for a, b in zip(path[:-1], path[1:]):

            edge = tuple(sorted((tuple(a), tuple(b))))

            if edge not in edge_owners:
                edge_owners[edge] = set()

            edge_owners[edge].add(name)

    # Separate individual and shared edges.
    segments = []
    colors = []

    for edge, owners in edge_owners.items():

        a, b = edge

        segments.append([
            (a[1], a[0]),
            (b[1], b[0])
        ])

        if len(owners) > 1:
            colors.append("black")
        else:
            name = next(iter(owners))
            colors.append(algo_colors[name])

    # Draw all path segments.
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=2.5,
        zorder=3
    )

    plt.gca().add_collection(collection)

    # Mark start and end.
    plt.scatter(
        start[1], start[0],
        marker="o",
        color="lime",
        edgecolors="black",
        s=100,
        zorder=4,
        label="Start"
    )

    plt.scatter(
        end[1], end[0],
        marker="*",
        color="red",
        edgecolors="black",
        s=180,
        zorder=4,
        label="End"
    )

    # Construct the legend.
    legend_handles = [
        Line2D(
            [0], [0],
            color=algo_colors[name],
            linewidth=2.5,
            label=name
        )
        for name in paths
    ]

    if any(len(owners) > 1 for owners in edge_owners.values()):
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linewidth=2.5,
                label="Shared Path"
            )
        )

    legend_handles.extend([
        Line2D(
            [0], [0],
            marker="o",
            color="lime",
            markeredgecolor="black",
            linestyle="None",
            label="Start"
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="red",
            markeredgecolor="black",
            linestyle="None",
            label="End"
        )
    ])

    plt.legend(handles=legend_handles)
    plt.title(f"Maze {maze_index}: Algorithm Path Comparison")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, f"maze_{maze_index}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def run_tests(iter=1000, size=15, memory_sample=1000):
    seed = np.random.randint(0, 2**32)
    mazes = generate_all_mazes(size=size, iters=iter//9, seed=seed)

    results = {
        name: {
            "time_ms": [],
            "nodes": [],
            "length": [],
            "peak_kb": [],
            "success": [],
            "optimality": [],
            "path_cost": []
        }
        for name in ALGOS
    }

    validation_results = {
        "connected": [],
        "wall_density": [],
        "dead_ends": [],
        "junctions": []
    }

    valid_mazes = []

    # -----------------------------
    # 1. MAZE VALIDATION
    # -----------------------------

    for name_maze, grid, start, end in mazes:

        validation = validate_maze(grid, start, end)

        for metric in validation_results:
            validation_results[metric].append(validation[metric])

        if validation["connected"]:
            valid_mazes.append((name_maze, grid, start, end))

    print("\nMAZE VALIDATION")
    print("-" * 50)

    for metric, values in validation_results.items():

        mean, median, std = safe_stats(values)

        print(
            f"{metric:<15} | "
            f"Mean: {mean:.3f} | "
            f"Median: {median:.3f} | "
            f"Std: {std:.3f}"
        )

    print(f"\nGenerated mazes: {len(mazes)}")
    print(f"Connected mazes: {len(valid_mazes)}")
    print(f"Rejected mazes: {len(mazes) - len(valid_mazes)}")

    # -----------------------------
    # 2. ALGORITHM BENCHMARKING
    # -----------------------------

    sample_grid = None
    sample_start = None
    sample_end = None
    sample_path = []

    # Store paths for all connected mazes.
    maze_path_records = []

    for index, (name_maze, grid, start, end) in enumerate(valid_mazes):

        print(f"\nMaze {index + 1}/{len(valid_mazes)}")

        maze_runs = {}

        for name, fn in ALGOS.items():

            print(f"{name}: ", end="", flush=True)

            t0 = time.perf_counter()
            path, nodes = fn(grid, start, end)
            dt = (time.perf_counter() - t0) * 1000

            success = len(path) > 0

            results[name]["time_ms"].append(dt)
            results[name]["nodes"].append(nodes)
            results[name]["length"].append(len(path) if success else 0)
            results[name]["success"].append(success)
            results[name]["path_cost"].append(calculate_path_cost(path))

            maze_runs[name] = path

            print("done")

        maze_path_records.append(maze_runs.copy())

        # BFS provides the shortest-path reference.
        dijkstra_cost = results["DIJKSTRA"]["path_cost"][-1]

        for name in ALGOS:

            path = maze_runs[name]

            if path and dijkstra_cost:
                ratio = results[name]["path_cost"][-1] / dijkstra_cost
                results[name]["optimality"].append(ratio)

        if sample_grid is None:
            sample_grid = grid
            sample_start = start
            sample_end = end
            sample_path = maze_runs["BFS"]

    # -----------------------------
    # 3. MEMORY BENCHMARKING
    # -----------------------------

    print("\nMEMORY BENCHMARKING")

    for index, (name_maze, grid, start, end) in enumerate(
        valid_mazes[:memory_sample]
    ):

        print(f"Memory maze {index + 1}/{min(memory_sample, len(valid_mazes))}")

        for name, fn in ALGOS.items():

            tracemalloc.start()

            fn(grid, start, end)

            _, peak = tracemalloc.get_traced_memory()

            tracemalloc.stop()

            results[name]["peak_kb"].append(peak / 1024)

    # -----------------------------
    # 4. STATISTICAL ANALYSIS
    # -----------------------------

    print("\n\nALGORITHM PERFORMANCE")
    print("=" * 130)

    header = (
        f"{'Algorithm':<12} | "
        f"{'Time Mean':>10} | "
        f"{'Time Med':>10} | "
        f"{'Time Std':>10} | "
        f"{'Nodes Mean':>10} | "
        f"{'Path Mean':>10} | "
        f"{'Memory KB':>10} | "
        f"{'Success %':>10} | "
        f"{'Optimality':>10} | "
        f"{'Path Cost':>10}"
    )

    print(header)
    print("-" * 130)

    for name, data in results.items():

        time_mean, time_median, time_std = safe_stats(data["time_ms"])
        nodes_mean, _, _ = safe_stats(data["nodes"])

        path_mean, _, _ = safe_stats(
            [x for x in data["length"] if x > 0]
        )

        memory_mean, _, _ = safe_stats(data["peak_kb"])

        optimality_mean, _, _ = safe_stats(data["optimality"])

        path_cost_mean, _, _ = safe_stats(
            [x for x in data["path_cost"] if x > 0]
        )

        success_rate = (
            100 * np.mean(data["success"])
            if data["success"]
            else 0
        )

        print(
            f"{name:<12} | "
            f"{time_mean:>10.4f} | "
            f"{time_median:>10.4f} | "
            f"{time_std:>10.4f} | "
            f"{nodes_mean:>10.1f} | "
            f"{path_mean:>10.1f} | "
            f"{memory_mean:>10.2f} | "
            f"{success_rate:>9.2f}% | "
            f"{optimality_mean:>10.3f} | "
            f"{path_cost_mean:>10.3f}"
        )

    # -----------------------------
    # 5. ANALYTICAL VISUALISATION
    # -----------------------------

    plot_dir = "maze_analysis_plots"
    os.makedirs(plot_dir, exist_ok=True)

    algorithms = list(results.keys())

    performance_metrics = {
        "Execution Time (ms)": "time_ms",
        "Nodes Explored": "nodes",
        "Path Length": "length",
        "Memory Usage (KB)": "peak_kb",
        "Optimality Ratio": "optimality",
        "Path Cost": "path_cost"
    }

    # Dijkstra provides the reference cost for optimality.
    dijkstra_costs = np.asarray(
        results["DIJKSTRA"]["path_cost"],
        dtype=float
    )

    # -----------------------------
    # 5.1 PREPARE PERFORMANCE DATA
    # -----------------------------

    performance_data = {}

    for algo_name, data in results.items():

        performance_data[algo_name] = {}

        for metric_name, metric_key in performance_metrics.items():

            if metric_key == "optimality":

                costs = np.asarray(
                    data["path_cost"],
                    dtype=float
                )

                values = np.divide(
                    costs,
                    dijkstra_costs,
                    out=np.full(len(costs), np.nan),
                    where=(dijkstra_costs > 0)
                )

                values[costs == 0] = np.nan

            elif metric_key == "peak_kb":

                values = np.asarray(
                    data["peak_kb"],
                    dtype=float
                )

            else:

                values = np.asarray(
                    data[metric_key],
                    dtype=float
                )

                if metric_key in ("length", "path_cost"):

                    success = np.asarray(
                        data["success"],
                        dtype=bool
                    )

                    values = np.where(
                        success,
                        values,
                        np.nan
                    )

            performance_data[algo_name][metric_key] = values

    # -----------------------------
    # 5.2 INDIVIDUAL DISTRIBUTION PLOTS
    # -----------------------------

    for algo_name in algorithms:

        algo_dir = os.path.join(
            plot_dir,
            algo_name
        )

        os.makedirs(algo_dir, exist_ok=True)

        for metric_name, metric_key in performance_metrics.items():

            values = performance_data[algo_name][metric_key]

            maze_numbers = np.arange(1, len(values) + 1)

            plt.figure(figsize=(12, 6))

            plt.plot(
                maze_numbers,
                values,
                linewidth=1.2
            )

            plt.xlabel("Maze Number")
            plt.ylabel(metric_name)

            plt.title(
                f"{algo_name}: {metric_name} Across Mazes"
            )

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f"{metric_key}_distribution.png"

            plt.savefig(
                os.path.join(algo_dir, filename),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    # -----------------------------
    # 5.3 AGGREGATE PERFORMANCE PLOTS
    # -----------------------------

    aggregate_dir = os.path.join(
        plot_dir,
        "aggregates"
    )

    os.makedirs(aggregate_dir, exist_ok=True)

    for metric_name, metric_key in performance_metrics.items():

        means = []

        for algo_name in algorithms:

            values = performance_data[algo_name][metric_key]

            valid_values = values[np.isfinite(values)]

            mean = (
                np.mean(valid_values)
                if len(valid_values) > 0
                else np.nan
            )

            means.append(mean)

        plt.figure(figsize=(10, 6))

        bars = plt.bar(
            algorithms,
            means
        )

        plt.ylabel(f"Mean {metric_name}")

        plt.title(
            f"Average {metric_name} Across All Tested Mazes"
        )

        plt.xticks(rotation=30, ha="right")

        for bar, mean in zip(bars, means):

            if np.isfinite(mean):

                plt.annotate(
                    f"{mean:.3f}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height()
                    ),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        filename = f"{metric_key}_aggregate.png"

        plt.savefig(
            os.path.join(aggregate_dir, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print(f"\nAll analytical plots saved in: {plot_dir}/")

    # -----------------------------
    # 6. OUTLIER MAZE SELECTION
    # -----------------------------

    visual_dir = "maze_solution_overlays"

    lower_outliers = []
    upper_outliers = []

    for algo_name in algorithms:

        for metric_name, metric_key in performance_metrics.items():

            values = performance_data[algo_name][metric_key]

            # Ignore NaN and infinite values.
            valid_indices = np.where(np.isfinite(values))[0]
            valid_values = values[valid_indices]

            if len(valid_values) < 4:
                continue

            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)

            iqr = q3 - q1

            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr

            low_indices = valid_indices[
                valid_values < lower_fence
            ]

            high_indices = valid_indices[
                valid_values > upper_fence
            ]

            # Select the most extreme lower outlier.
            if len(low_indices) > 0:

                index = low_indices[
                    np.argmin(values[low_indices])
                ]

                lower_outliers.append(
                    (index, algo_name, metric_key)
                )

            # Select the most extreme upper outlier.
            if len(high_indices) > 0:

                index = high_indices[
                    np.argmax(values[high_indices])
                ]

                upper_outliers.append(
                    (index, algo_name, metric_key)
                )

    print("\nOUTLIER SELECTION")
    print("-" * 50)

    print(f"Lower outlier selections: {len(lower_outliers)}")
    print(f"Upper outlier selections: {len(upper_outliers)}")

    # -----------------------------
    # 7. OUTLIER PATH VISUALISATIONS
    # -----------------------------

    os.makedirs(visual_dir, exist_ok=True)

    def plot_outlier_group(outliers, direction):

        for index, algo_name, metric_key in outliers:

            name_maze, grid, start, end = valid_mazes[index]

            paths = maze_path_records[index]

            label = (
                f"{direction}_{algo_name}_{metric_key}_"
                f"maze_{index + 1}"
            )

            plot_combined_paths(
                grid,
                paths,
                start,
                end,
                label,
                visual_dir
            )

    plot_outlier_group(
        lower_outliers,
        "LOWER_OUTLIER"
    )

    plot_outlier_group(
        upper_outliers,
        "UPPER_OUTLIER"
    )

    print(
        f"\nOutlier maze visualisations saved in: "
        f"{visual_dir}/"
    )
    print(seed)
    return results, validation_results
run_tests()
