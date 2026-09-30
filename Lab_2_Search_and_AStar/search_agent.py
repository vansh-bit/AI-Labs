"""
Artificial Intelligence - Laboratory Exercise: Search and A*
Using an LLM as an Engineering Assistant

This program implements:
1. Formulation of the Warehouse Navigation Problem as a graph search problem.
2. A* Search with selectable heuristics (Manhattan, Euclidean, Zero, Overestimated).
3. Breadth-First Search (BFS) for comparative benchmark.
4. Comprehensive test suite:
   - Test 1: Original laboratory warehouse map.
   - Test 2: Trivial case (goal immediately adjacent).
   - Test 3: No solution (goal unreachable).
   - Test 4: Alternative paths (evaluating shortest path selection).
5. Heuristic investigation suite analyzing admissibility and efficiency.
"""

import math
import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict, Set, Callable


# Original Warehouse Map from Section 4 of search_lab_ex.pdf
ORIGINAL_WAREHOUSE_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################"
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####"
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######"
]

ALTERNATIVE_PATHS_MAP = [
    "#########",
    "#S.....G#",
    "#.#####.#",
    "#.......#",
    "#########"
]


class SearchProblem:
    """
    Formal representation of a state-space search problem P = (S, A, T, s0, G, c)
    """
    def __init__(self, ascii_map: List[str]):
        self.grid = [row.strip() for row in ascii_map if row.strip()]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.obstacles: Set[Tuple[int, int]] = set()

        self._parse_map()

    def _parse_map(self):
        for r in range(self.rows):
            for c in range(self.cols):
                char = self.grid[r][c]
                if char == 'S':
                    self.start = (r, c)
                elif char == 'G':
                    self.goal = (r, c)
                elif char == '#':
                    self.obstacles.add((r, c))

        if self.start is None or self.goal is None:
            raise ValueError("Map must contain both 'S' and 'G'.")

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and pos not in self.obstacles

    def get_successors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int], int]]:
        """
        Transition function T(s, a) -> s' with action cost c(s, a, s') = 1.
        Returns list of (ActionName, NextState, StepCost)
        """
        r, c = pos
        actions = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1))
        ]
        return [(act, nxt, 1) for act, nxt in actions if self.is_valid(nxt)]

    def is_goal(self, pos: Tuple[int, int]) -> bool:
        return pos == self.goal


# Heuristic functions
def heuristic_manhattan(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def heuristic_euclidean(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def heuristic_zero(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return 0.0

def heuristic_overestimated(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return 2.0 * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))


class SearchResult:
    def __init__(self, found: bool, path: Optional[List[Tuple[int, int]]], length: int, states_expanded: int):
        self.found = found
        self.path = path
        self.length = length
        self.states_expanded = states_expanded


def a_star_search(problem: SearchProblem, heuristic_fn: Callable = heuristic_manhattan) -> SearchResult:
    """
    A* Search implementation:
    f(n) = g(n) + h(n)
    Uses a priority queue (min-heap) as the frontier.
    """
    start = problem.start
    goal = problem.goal

    # Priority queue entry: (f(n), g(n), tie_breaker_counter, state)
    counter = 0
    frontier = [(heuristic_fn(start, goal), 0, counter, start)]
    g_cost: Dict[Tuple[int, int], int] = {start: 0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed_set: Set[Tuple[int, int]] = set()
    states_expanded = 0

    while frontier:
        f, g, _, current = heapq.heappop(frontier)

        if current in closed_set:
            continue

        closed_set.add(current)
        states_expanded += 1

        if problem.is_goal(current):
            # Reconstruct path
            path = []
            curr = current
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return SearchResult(True, path, len(path) - 1, states_expanded)

        for _, successor, cost in problem.get_successors(current):
            tentative_g = g + cost
            if successor not in g_cost or tentative_g < g_cost[successor]:
                g_cost[successor] = tentative_g
                parent[successor] = current
                counter += 1
                f_score = tentative_g + heuristic_fn(successor, goal)
                heapq.heappush(frontier, (f_score, tentative_g, counter, successor))

    return SearchResult(False, None, 0, states_expanded)


def breadth_first_search(problem: SearchProblem) -> SearchResult:
    """
    Breadth-First Search (BFS) using a FIFO queue.
    """
    start = problem.start
    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    states_expanded = 0

    while queue:
        current = queue.popleft()
        states_expanded += 1

        if problem.is_goal(current):
            path = []
            curr = current
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            return SearchResult(True, path, len(path) - 1, states_expanded)

        for _, successor, _ in problem.get_successors(current):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = current
                queue.append(successor)

    return SearchResult(False, None, 0, states_expanded)


def render_map_with_path(problem: SearchProblem, path: Optional[List[Tuple[int, int]]]) -> str:
    path_set = set(path) if path else set()
    lines = []
    for r in range(problem.rows):
        row_chars = []
        for c in range(problem.cols):
            pos = (r, c)
            if pos == problem.start:
                row_chars.append('S')
            elif pos == problem.goal:
                row_chars.append('G')
            elif pos in path_set:
                row_chars.append('*')
            elif pos in problem.obstacles:
                row_chars.append('#')
            else:
                row_chars.append('.')
        lines.append("".join(row_chars))
    return "\n".join(lines)


def run_all_tests():
    print("=" * 70)
    print("TASK 3: SYSTEMATIC TESTING OF GENERATED SEARCH PROGRAM")
    print("=" * 70)

    # Test 1: Original Warehouse
    print("\n--- TEST 1: Original Warehouse Map ---")
    p1 = SearchProblem(ORIGINAL_WAREHOUSE_MAP)
    res_astar = a_star_search(p1, heuristic_manhattan)
    print(f"Path Found       : {res_astar.found}")
    print(f"Path Length      : {res_astar.length} steps")
    print(f"States Expanded  : {res_astar.states_expanded}")
    print("Rendered Path on Grid:")
    print(render_map_with_path(p1, res_astar.path))

    # Test 2: Trivial Case
    print("\n--- TEST 2: Trivial Case (Goal immediately adjacent) ---")
    p2 = SearchProblem(TRIVIAL_MAP)
    res2 = a_star_search(p2, heuristic_manhattan)
    print(f"Path Found       : {res2.found}")
    print(f"Path Length      : {res2.length} step(s) (Expected: 1)")
    print(f"States Expanded  : {res2.states_expanded}")
    assert res2.length == 1, "Trivial test failed!"

    # Test 3: No Solution
    print("\n--- TEST 3: No Solution (Inaccessible Goal) ---")
    p3 = SearchProblem(NO_SOLUTION_MAP)
    res3 = a_star_search(p3, heuristic_manhattan)
    print(f"Path Found       : {res3.found} (Expected: False)")
    print(f"States Expanded  : {res3.states_expanded}")
    assert not res3.found, "No solution test failed!"

    # Test 4: Alternative Paths
    print("\n--- TEST 4: Alternative Paths (Shortest Path Guarantee) ---")
    p4 = SearchProblem(ALTERNATIVE_PATHS_MAP)
    res4 = a_star_search(p4, heuristic_manhattan)
    res4_bfs = breadth_first_search(p4)
    print(f"A* Path Length   : {res4.length} steps")
    print(f"BFS Path Length  : {res4_bfs.length} steps")
    print(f"A* States Exp.   : {res4.states_expanded}")
    print(f"BFS States Exp.  : {res4_bfs.states_expanded}")
    assert res4.length == res4_bfs.length, "Alternative paths test failed!"

    print("\n" + "=" * 70)
    print("TASK 5: COMPARISON BETWEEN BFS AND A* ON ORIGINAL MAP")
    print("=" * 70)
    res_bfs = breadth_first_search(p1)
    print(f"{'Measure':<20} | {'BFS':<15} | {'A* (Manhattan)':<15}")
    print("-" * 56)
    print(f"{'Solution found':<20} | {str(res_bfs.found):<15} | {str(res_astar.found):<15}")
    print(f"{'Path length':<20} | {res_bfs.length:<15} | {res_astar.length:<15}")
    print(f"{'States expanded':<20} | {res_bfs.states_expanded:<15} | {res_astar.states_expanded:<15}")

    print("\n" + "=" * 70)
    print("TASK 6: HEURISTIC INVESTIGATION")
    print("=" * 70)
    heuristics = [
        ("h(n) = 0 (Uniform Cost)", heuristic_zero),
        ("h(n) = Euclidean", heuristic_euclidean),
        ("h(n) = Manhattan", heuristic_manhattan),
        ("h(n) = 2 * Manhattan", heuristic_overestimated)
    ]
    print(f"{'Heuristic Function':<25} | {'Found':<8} | {'Path Length':<12} | {'States Expanded':<16}")
    print("-" * 68)
    for name, h_fn in heuristics:
        res = a_star_search(p1, h_fn)
        print(f"{name:<25} | {str(res.found):<8} | {res.length:<12} | {res.states_expanded:<16}")


if __name__ == "__main__":
    run_all_tests()
