"""
Online Maze Solver for Unknown Environments using AI Search Techniques
Algorithms: BFS, DFS, A* (offline, known maze) and an Online Frontier-based
A* explorer (agent only senses its 4 neighbours and builds the map as it moves).
"""
import random
import heapq
from collections import deque

WALL, FREE = 1, 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ------------------------------------------------------------------ maze
def generate_maze(rows=21, cols=21, seed=7):
    """Perfect maze with recursive-backtracker; rows/cols must be odd."""
    random.seed(seed)
    grid = [[WALL] * cols for _ in range(rows)]
    stack = [(1, 1)]
    grid[1][1] = FREE
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in DIRS:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr][nc] == WALL:
                nbrs.append((nr, nc, dr, dc))
        if nbrs:
            nr, nc, dr, dc = random.choice(nbrs)
            grid[r + dr][c + dc] = FREE
            grid[nr][nc] = FREE
            stack.append((nr, nc))
        else:
            stack.pop()
    # knock out a few walls so more than one route exists (makes A* vs BFS interesting)
    for _ in range(rows * cols // 40):
        r, c = random.randrange(1, rows - 1), random.randrange(1, cols - 1)
        if grid[r][c] == WALL:
            grid[r][c] = FREE
    start, goal = (1, 1), (rows - 2, cols - 2)
    grid[goal[0]][goal[1]] = FREE
    return grid, start, goal


def neighbours(grid, node):
    r, c = node
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == FREE:
            yield (nr, nc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rebuild(parent, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ------------------------------------------------- offline (known maze)
def bfs(grid, start, goal):
    q, parent, order = deque([start]), {start: None}, []
    while q:
        cur = q.popleft()
        order.append(cur)
        if cur == goal:
            return rebuild(parent, goal), order
        for n in neighbours(grid, cur):
            if n not in parent:
                parent[n] = cur
                q.append(n)
    return None, order


def dfs(grid, start, goal):
    stack, parent, order = [start], {start: None}, []
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        if cur == goal:
            return rebuild(parent, goal), order
        for n in neighbours(grid, cur):
            if n not in seen:
                parent.setdefault(n, cur)
                stack.append(n)
    return None, order


def astar(grid, start, goal):
    pq, parent, g, order = [(manhattan(start, goal), 0, start)], {start: None}, {start: 0}, []
    closed = set()
    while pq:
        _, _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            return rebuild(parent, goal), order
        for n in neighbours(grid, cur):
            ng = g[cur] + 1
            if ng < g.get(n, 1e9):
                g[n] = ng
                parent[n] = cur
                heapq.heappush(pq, (ng + manhattan(n, goal), ng, n))
    return None, order


# ------------------------------------------ online (unknown environment)
class OnlineExplorer:
    """
    The agent does NOT see the maze. At every step it senses only its four
    neighbouring cells, adds them to its internal map, then:
      1. finds all frontier cells (known-free cells that touch unknown cells),
      2. scores each by  f = path_cost_from_agent + heuristic_to_goal,
      3. walks (A*-style over the known map) to the best frontier,
      4. repeats until the goal is sensed and reached.
    """

    UNKNOWN = -1

    def __init__(self, world, start, goal):
        self.world, self.start, self.goal = world, start, goal
        R, C = len(world), len(world[0])
        self.known = [[self.UNKNOWN] * C for _ in range(R)]
        self.pos = start
        self.trail = [start]
        self.visited = {start}
        self.sense()

    def sense(self):
        r, c = self.pos
        self.known[r][c] = FREE
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.world) and 0 <= nc < len(self.world[0]):
                self.known[nr][nc] = self.world[nr][nc]

    def known_free_nbrs(self, node):
        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.known) and 0 <= nc < len(self.known[0]) and self.known[nr][nc] == FREE:
                yield (nr, nc)

    def is_frontier(self, node):
        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.known) and 0 <= nc < len(self.known[0]) and self.known[nr][nc] == self.UNKNOWN:
                return True
        return False

    def plan_to_best_frontier(self):
        # Dijkstra over known map from agent position
        dist, parent = {self.pos: 0}, {self.pos: None}
        q = deque([self.pos])
        while q:
            cur = q.popleft()
            for n in self.known_free_nbrs(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    parent[n] = cur
                    q.append(n)
        if self.known[self.goal[0]][self.goal[1]] == FREE and self.goal in dist:
            return rebuild(parent, self.goal)
        best, best_f = None, 1e9
        for node, d in dist.items():
            if node != self.pos and self.is_frontier(node):
                f = d + manhattan(node, self.goal)
                if f < best_f:
                    best, best_f = node, f
        return rebuild(parent, best) if best else None

    def run(self, max_steps=5000):
        while self.pos != self.goal and len(self.trail) < max_steps:
            plan = self.plan_to_best_frontier()
            if not plan or len(plan) < 2:
                return None
            for step in plan[1:]:
                self.pos = step
                self.trail.append(step)
                self.visited.add(step)
                self.sense()
                if self.pos == self.goal:
                    break
                # replan as soon as new info could change the decision
                if self.goal_known_reachable():
                    break
            # loop continues, replanning from new knowledge
        return self.trail if self.pos == self.goal else None

    def goal_known_reachable(self):
        return self.known[self.goal[0]][self.goal[1]] == FREE

    def known_cells(self):
        return sum(1 for row in self.known for v in row if v != self.UNKNOWN)


def shortest_from_trail(trail):
    """Remove loops from the walked trail to give the cleaned route."""
    seen, out = {}, []
    for node in trail:
        if node in seen:
            out = out[: seen[node] + 1]
            seen = {n: i for i, n in enumerate(out)}
        else:
            seen[node] = len(out)
            out.append(node)
    return out


# --------------------------------------------------------------- driver
def solve_all(grid, start, goal):
    results = {}
    for name, fn in (("BFS", bfs), ("DFS", dfs), ("A*", astar)):
        path, order = fn(grid, start, goal)
        results[name] = dict(path=path, explored=order, length=len(path) - 1, nodes=len(order))
    agent = OnlineExplorer(grid, start, goal)
    trail = agent.run()
    results["Online"] = dict(
        path=shortest_from_trail(trail),
        trail=trail,
        explored=list(agent.visited),
        length=len(shortest_from_trail(trail)) - 1,
        nodes=len(agent.visited),
        steps=len(trail) - 1,
        known=agent.known,
        known_cells=agent.known_cells(),
    )
    return results


if __name__ == "__main__":
    grid, s, g = generate_maze()
    res = solve_all(grid, s, g)
    print(f"{'Algorithm':10}{'Path length':>12}{'Cells explored':>16}{'Steps walked':>14}")
    for k, v in res.items():
        print(f"{k:10}{v['length']:>12}{v['nodes']:>16}{v.get('steps', v['nodes']):>14}")
