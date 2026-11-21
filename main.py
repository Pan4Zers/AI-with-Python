import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import heapq
from maze_generator import generate_all_mazes
from collections import deque
from maze_validator import validate_maze

def get_neighbours(graph, node):
    r, c = node
    rows, cols = graph.shape
    neighbours = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbours.append((nr,nc))

    neighbours = [i for i in neighbours if graph[i] == 0]

    return neighbours

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
            step_cost = weights[nxt] if weights is not None else 1
            new_cost = curr_cost + step_cost
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = curr
                heapq.heappush(pq, (new_cost, nxt))

    return [], nodes_explored

def astar(grid, start, end, weights=None):
    pq = [(manhattan(start, end), 0, start)]
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
            step_cost = weights[nxt] if weights is not None else 1
            new_cost = curr_cost + step_cost
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = curr
                priority = new_cost + manhattan(nxt, end)
                heapq.heappush(pq, (priority, new_cost, nxt))

    return [], nodes_explored

def floodfill(grid, start, end):
    distances = {end: 0}
    queue = deque([end])
    nodes_explored = 0

    while queue:
        curr = queue.popleft()
        nodes_explored += 1
        if curr == start:
            break  # Stop flooding once the wave reaches start

        for nxt in get_neighbours(grid, curr):
            if nxt not in distances:
                distances[nxt] = distances[curr] + 1
                queue.append(nxt)

    if start not in distances:
        return [], nodes_explored

    path = [start]
    curr = start
    while curr != end:
        for nxt in get_neighbours(grid, curr):
            if distances.get(nxt) == distances[curr] - 1:
                path.append(nxt)
                curr = nxt
                break

    return path, nodes_explored

def bidirectional_astar(grid, start, end):
    if start == end:
        return [start], 1

    f_pq = [(manhattan(start, end), 0, start)]
    b_pq = [(manhattan(end, start), 0, end)]

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
                new_g = g + 1
                if new_g < f_cost.get(nxt, float("inf")):
                    f_cost[nxt] = new_g
                    f_came_from[nxt] = curr
                    heapq.heappush(f_pq, (new_g + manhattan(nxt, end), new_g, nxt))

                    if nxt in b_cost and new_g + b_cost[nxt] < best_total_cost:
                        best_total_cost = new_g + b_cost[nxt]
                        meeting_node = nxt
        else:
            _, g, curr = heapq.heappop(b_pq)
            if g > b_cost.get(curr, float("inf")):
                continue
            nodes_explored += 1

            for nxt in get_neighbours(grid, curr):
                new_g = g + 1
                if new_g < b_cost.get(nxt, float("inf")):
                    b_cost[nxt] = new_g
                    b_came_from[nxt] = curr
                    heapq.heappush(b_pq, (new_g + manhattan(nxt, start), new_g, nxt))

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
        ("optimality", "Average Optimality Ratio")
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


def run_tests(iter=1000, size=15, memory_sample=50):

    mazes = generate_all_mazes(size=size, iters=iter//9)

    results = {
        name: {
            "time_ms": [],
            "nodes": [],
            "length": [],
            "peak_kb": [],
            "success": [],
            "optimality": []
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

    print(f"\nTotal mazes generated: {len(mazes)}")
    print(f"Connected mazes: {len(valid_mazes)}")

    # -----------------------------
    # 2. ALGORITHM BENCHMARKING
    # -----------------------------

    sample_grid = None
    sample_start = None
    sample_end = None
    sample_path = []

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

            maze_runs[name] = path

            print("done")

        # BFS provides the shortest-path reference.
        bfs_length = len(maze_runs["BFS"])

        for name in ALGOS:

            path = maze_runs[name]

            if path and bfs_length:
                ratio = len(path) / bfs_length
                results[name]["optimality"].append(ratio)

        # Keep one maze for visualisation.
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
    print("=" * 115)

    header = (
        f"{'Algorithm':<12} | "
        f"{'Time Mean':>10} | "
        f"{'Time Med':>10} | "
        f"{'Time Std':>10} | "
        f"{'Nodes Mean':>10} | "
        f"{'Path Mean':>10} | "
        f"{'Memory KB':>10} | "
        f"{'Success %':>10} | "
        f"{'Optimality':>10}"
    )

    print(header)
    print("-" * 115)

    for name, data in results.items():

        time_mean, time_median, time_std = safe_stats(data["time_ms"])
        nodes_mean, _, _ = safe_stats(data["nodes"])
        path_mean, _, _ = safe_stats(
            [x for x in data["length"] if x > 0]
        )
        memory_mean, _, _ = safe_stats(data["peak_kb"])
        optimality_mean, _, _ = safe_stats(data["optimality"])

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
            f"{optimality_mean:>10.3f}"
        )

    # -----------------------------
    # 5. VISUALISATION
    # -----------------------------

    plot_comparisons(results)

    if sample_grid is not None:
        plot_maze(
            sample_grid,
            sample_path,
            sample_start,
            sample_end
        )

    return results, validation_results


run_tests()