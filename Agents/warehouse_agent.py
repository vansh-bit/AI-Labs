"""
Artificial Intelligence - Agents: Laboratory Exercise
Goal-Based Agent for the Warehouse Navigation Problem

This program models the warehouse environment, specifies the goal-based agent,
and uses a search algorithm (Breadth-First Search / A* Search) to compute a
collision-free path from start 'S' to destination 'G'.
"""

import sys
from collections import deque
from typing import List, Tuple, Optional, Dict, Set


# Standard Warehouse Map
DEFAULT_WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class WarehouseEnvironment:
    """
    Represents the 2D discrete grid warehouse environment.
    """
    def __init__(self, grid_map: List[str]):
        self.grid = [row.strip() for row in grid_map if row.strip()]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.obstacles: Set[Tuple[int, int]] = set()

        self._parse_grid()

    def _parse_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                char = self.grid[r][c]
                if char == 'S':
                    self.start_pos = (r, c)
                elif char == 'G':
                    self.goal_pos = (r, c)
                elif char == '#':
                    self.obstacles.add((r, c))

        if self.start_pos is None:
            raise ValueError("No start position 'S' found in grid.")
        if self.goal_pos is None:
            raise ValueError("No goal position 'G' found in grid.")

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and pos not in self.obstacles

    def get_actions(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Returns valid actions and resulting states: (ActionName, (new_row, new_col))
        Available actions: Up, Down, Left, Right
        """
        r, c = pos
        candidate_moves = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1))
        ]
        valid_moves = []
        for action_name, new_pos in candidate_moves:
            if self.is_valid_position(new_pos):
                valid_moves.append((action_name, new_pos))
        return valid_moves


class GoalBasedAgent:
    """
    A goal-based intelligent agent that plans actions using search
    to navigate safely to the goal.
    """
    def __init__(self, environment: WarehouseEnvironment):
        self.env = environment
        self.start = environment.start_pos
        self.goal = environment.goal_pos

    def plan_path_bfs(self) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search (BFS) search strategy to find the shortest
        path in an unweighted grid.
        """
        queue = deque([(self.start, [self.start])])
        visited: Set[Tuple[int, int]] = {self.start}
        states_expanded = 0

        while queue:
            current_pos, path = queue.popleft()
            states_expanded += 1

            if current_pos == self.goal:
                self.states_expanded = states_expanded
                return path

            for _, next_pos in self.env.get_actions(current_pos):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        self.states_expanded = states_expanded
        return None

    def render_path(self, path: Optional[List[Tuple[int, int]]]) -> str:
        """
        Visualizes the warehouse grid with the computed trajectory marked with '*'.
        """
        path_set = set(path) if path else set()
        rendered_lines = []

        for r in range(self.env.rows):
            row_chars = []
            for c in range(self.env.cols):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append('S')
                elif pos == self.goal:
                    row_chars.append('G')
                elif pos in path_set:
                    row_chars.append('*')
                elif pos in self.env.obstacles:
                    row_chars.append('#')
                else:
                    row_chars.append('.')
            rendered_lines.append("".join(row_chars))

        return "\n".join(rendered_lines)


def main():
    print("=" * 60)
    print("Goal-Based Agent - Warehouse Navigation Problem")
    print("=" * 60)

    env = WarehouseEnvironment(DEFAULT_WAREHOUSE_MAP)
    print(f"Warehouse Dimensions: {env.rows} rows x {env.cols} columns")
    print(f"Start Position (S) : {env.start_pos}")
    print(f"Goal Position  (G) : {env.goal_pos}")
    print(f"Obstacles Count    : {len(env.obstacles)}")
    print("\nInitial Warehouse Map:")
    for row in DEFAULT_WAREHOUSE_MAP:
        print("  " + row)

    agent = GoalBasedAgent(env)
    print("\nSearching for collision-free path using Goal-Based Agent (BFS)...")
    path = agent.plan_path_bfs()

    if path:
        print(f"\nSUCCESS: Collision-free path found!")
        print(f"Total Steps (Path Length) : {len(path) - 1}")
        print(f"States Expanded           : {agent.states_expanded}")
        print("\nStep-by-step Coordinates:")
        for idx, step in enumerate(path):
            print(f"  Step {idx:02d}: {step}")

        print("\nVisualized Trajectory (* marks agent path):")
        rendered = agent.render_path(path)
        for line in rendered.split("\n"):
            print("  " + line)
    else:
        print("\nFAILURE: No valid path exists from S to G.")


if __name__ == "__main__":
    main()
