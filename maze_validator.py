from collections import deque


def get_open_neighbours(grid, node):
    rows, cols = grid.shape
    r, c = node

    candidates = [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1)
    ]

    return [
        (nr, nc)
        for nr, nc in candidates
        if 0 <= nr < rows
        and 0 <= nc < cols
        and grid[nr, nc] == 0
    ]


def check_connectivity(grid, start, end):
    visited = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        if current == end:
            return True

        for neighbour in get_open_neighbours(grid, current):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return False


def calculate_wall_density(grid):
    return (grid == 1).sum() / grid.size


def count_dead_ends(grid):
    rows, cols = grid.shape
    return sum(
        len(get_open_neighbours(grid, (r, c))) == 1
        for r in range(rows)
        for c in range(cols)
        if grid[r, c] == 0
    )


def count_junctions(grid):
    rows, cols = grid.shape
    return sum(
        len(get_open_neighbours(grid, (r, c))) >= 3
        for r in range(rows)
        for c in range(cols)
        if grid[r, c] == 0
    )


def validate_maze(grid, start, end):
    return {
        "connected": check_connectivity(grid, start, end),
        "wall_density": calculate_wall_density(grid),
        "dead_ends": count_dead_ends(grid),
        "junctions": count_junctions(grid)
    }

print("Loaded maze_validator.py")
print("validate_maze exists:", "validate_maze" in globals())