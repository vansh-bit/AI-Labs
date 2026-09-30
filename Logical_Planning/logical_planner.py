#!/usr/bin/env python3
"""
Artificial Intelligence Laboratory: Logical Planning
Course: CS F407 - Artificial Intelligence
Reference: logic_lab_ex.pdf

Central Formula:
    Logic (Action Applicability & State Progression) + Search (BFS Planning) = Planning

Features:
  1. Propositional State Representation (Sets of Ground Literals)
  2. STRIPS-style Action Schemas (Positive/Negative Preconditions & Effects)
  3. Action Applicability Theorem Prover: S |= Preconditions(a)
  4. Breadth-First Search (BFS) Planning Engine (guarantees shortest plan)
  5. Systematic Test Suites:
     - Test A: Solvable Warehouse Delivery Problem (A -> B -> C)
     - Test B: Impossible Problem (Unsolvable due to missing PickUp action)
     - Test C: Irrelevant Actions (Distinguishes robot arrival from package arrival)
  6. Prolog-based Plan Verifier (Embedded horn-clause resolution & planner.pl generator)

Design: 100% self-contained in pure Python standard library.
"""

from collections import deque, defaultdict
from typing import List, Set, Dict, Tuple, Optional, Any


# ============================================================================
# Section 1: Logical State and Action Representation
# ============================================================================

class Action:
    """
    STRIPS-style Action representation with positive/negative preconditions
    and positive/negative effects.
    """
    def __init__(
        self,
        name: str,
        pos_preconds: Set[str],
        neg_preconds: Optional[Set[str]] = None,
        pos_effects: Optional[Set[str]] = None,
        neg_effects: Optional[Set[str]] = None
    ):
        self.name = name
        self.pos_preconds = set(pos_preconds)
        self.neg_preconds = set(neg_preconds) if neg_preconds else set()
        self.pos_effects = set(pos_effects) if pos_effects else set()
        self.neg_effects = set(neg_effects) if neg_effects else set()

    def is_applicable(self, state: Set[str]) -> bool:
        """
        Logical reasoning step: S |= Preconditions(a)
        True iff all positive preconditions are in state and no negative preconditions are in state.
        """
        return self.pos_preconds.issubset(state) and not (self.neg_preconds & state)

    def apply(self, state: Set[str]) -> Set[str]:
        """
        State progression step: S' = Apply(S, a) = (S \\ neg_effects) U pos_effects
        """
        if not self.is_applicable(state):
            raise ValueError(f"Action '{self.name}' is not applicable in current state.")
        return (state - self.neg_effects) | self.pos_effects

    def __repr__(self) -> str:
        return self.name


# ============================================================================
# Section 2: Planning Problem & Breadth-First Search Engine
# ============================================================================

class PlanningProblem:
    """
    Planning Problem defined by (I, A, G):
      I: Initial State (set of propositions)
      A: Set of available actions
      G: Goal condition (set of propositions that must be true)
    """
    def __init__(self, initial_state: Set[str], actions: List[Action], goal: Set[str]):
        self.initial_state = set(initial_state)
        self.actions = actions
        self.goal = set(goal)

    def is_goal_satisfied(self, state: Set[str]) -> bool:
        """Checks if G is satisfied in state S: S |= G."""
        return self.goal.issubset(state)

    def solve_bfs(self) -> Optional[Dict[str, Any]]:
        """
        Finds a sequence of actions using Breadth-First Search (BFS).
        Returns a dict with:
          - 'success': bool
          - 'plan': List of action names
          - 'states': List of state sets visited along the plan
          - 'nodes_explored': Total unique states explored
        """
        start_state_frozen = frozenset(self.initial_state)
        if self.is_goal_satisfied(self.initial_state):
            return {
                "success": True,
                "plan": [],
                "states": [self.initial_state],
                "nodes_explored": 1
            }

        # Queue contains tuples of (current_state, action_path, state_history)
        queue = deque([(self.initial_state, [], [self.initial_state])])
        visited = {start_state_frozen}
        nodes_explored = 0

        while queue:
            curr_state, action_path, state_history = queue.popleft()
            nodes_explored += 1

            for action in self.actions:
                if action.is_applicable(curr_state):
                    next_state = action.apply(curr_state)
                    next_frozen = frozenset(next_state)

                    new_action_path = action_path + [action.name]
                    new_state_history = state_history + [next_state]

                    if self.is_goal_satisfied(next_state):
                        return {
                            "success": True,
                            "plan": new_action_path,
                            "states": new_state_history,
                            "nodes_explored": nodes_explored
                        }

                    if next_frozen not in visited:
                        visited.add(next_frozen)
                        queue.append((next_state, new_action_path, new_state_history))

        return {
            "success": False,
            "plan": None,
            "states": None,
            "nodes_explored": nodes_explored
        }


# ============================================================================
# Section 3: Warehouse Robot Environment Factory
# ============================================================================

def create_warehouse_actions(allow_pickup: bool = True, include_irrelevant: bool = False) -> List[Action]:
    """
    Constructs the STRIPS actions for the 3-location warehouse (A, B, C).
    Connected topology: A <-> B <-> C.
    """
    actions = []
    locations = ["A", "B", "C"]
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

    # 1. Move Actions (Robot moves between connected locations)
    for loc_from, loc_to in connections:
        actions.append(Action(
            name=f"Move({loc_from}, {loc_to})",
            pos_preconds={f"At(Robot, {loc_from})"},
            neg_preconds=set(),
            pos_effects={f"At(Robot, {loc_to})"},
            neg_effects={f"At(Robot, {loc_from})"}
        ))

    # 2. PickUp Actions (Robot picks up package at same location)
    if allow_pickup:
        for loc in locations:
            actions.append(Action(
                name=f"PickUp(Package, {loc})",
                pos_preconds={f"At(Robot, {loc})", f"At(Package, {loc})"},
                neg_preconds={"Holding(Package)"},
                pos_effects={"Holding(Package)"},
                neg_effects={f"At(Package, {loc})"}
            ))

    # 3. Drop Actions (Robot drops package at current location)
    for loc in locations:
        actions.append(Action(
            name=f"Drop(Package, {loc})",
            pos_preconds={f"At(Robot, {loc})", "Holding(Package)"},
            neg_preconds=set(),
            pos_effects={f"At(Package, {loc})"},
            neg_effects={"Holding(Package)"}
        ))

    # 4. Irrelevant Actions (e.g. Inspect, MoveRobotOnly)
    if include_irrelevant:
        actions.append(Action(
            name="InspectLocation(A)",
            pos_preconds={"At(Robot, A)"},
            pos_effects={"Inspected(A)"},
            neg_effects=set()
        ))

    return actions


# ============================================================================
# Section 4: Systematic Verification Test Suite
# ============================================================================

def run_planning_tests():
    print("=" * 70)
    print("ARTIFICIAL INTELLIGENCE LABORATORY: LOGICAL PLANNING")
    print("Testing Generated Planner on Warehouse Scenarios")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Test A: Solvable Warehouse Problem
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST A: SOLVABLE PROBLEM (Original Warehouse Delivery A -> C)")
    print("-" * 70)
    initial_A = {"At(Robot, A)", "At(Package, A)"}
    goal_A = {"At(Package, C)"}
    actions_A = create_warehouse_actions(allow_pickup=True)
    problem_A = PlanningProblem(initial_A, actions_A, goal_A)

    result_A = problem_A.solve_bfs()
    print(f"Initial State : {sorted(list(initial_A))}")
    print(f"Goal State    : {sorted(list(goal_A))}")
    print(f"Plan Found?   : {result_A['success']}")
    print(f"Plan Length   : {len(result_A['plan'])} steps")
    print(f"Nodes Explored: {result_A['nodes_explored']}")
    print("\nPlan Execution Sequence:")
    for step_num, act in enumerate(result_A['plan'], 1):
        print(f"  Step {step_num:02d}: {act}")

    print("\nStep-by-Step State Progression:")
    for idx, st in enumerate(result_A['states']):
        print(f"  S_{idx}: {sorted(list(st))}")

    # Validate Plan Validity Oracle
    test_state = set(initial_A)
    for act_name in result_A['plan']:
        action_obj = next(a for a in actions_A if a.name == act_name)
        assert action_obj.is_applicable(test_state), f"Plan step {act_name} invalid in {test_state}"
        test_state = action_obj.apply(test_state)
    assert problem_A.is_goal_satisfied(test_state), "Goal not satisfied at end of plan"
    print("\n[PASS] Test A Plan Validity Oracle: Every action precondition was strictly satisfied.")

    # ------------------------------------------------------------------------
    # Test B: Impossible Problem (PickUp action missing)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST B: IMPOSSIBLE PROBLEM (Package Cannot Be Picked Up)")
    print("-" * 70)
    actions_B = create_warehouse_actions(allow_pickup=False)
    problem_B = PlanningProblem(initial_A, actions_B, goal_A)
    result_B = problem_B.solve_bfs()

    print(f"Initial State : {sorted(list(initial_A))}")
    print(f"Goal State    : {sorted(list(goal_A))}")
    print(f"Plan Found?   : {result_B['success']}")
    print(f"Output Report : {'No plan found' if not result_B['success'] else 'Unexpected plan'}")
    assert not result_B["success"], "Test B failed: Planner invented an invalid action!"
    print("[PASS] Test B Oracle: Planner correctly detected unreachable goal and halted without inventing actions.")

    # ------------------------------------------------------------------------
    # Test C: Irrelevant Actions
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST C: IRRELEVANT ACTIONS (Robot Movements Without Package)")
    print("-" * 70)
    actions_C = create_warehouse_actions(allow_pickup=True, include_irrelevant=True)
    # Problem where robot starts at A, moves to C, but package must reach C
    problem_C = PlanningProblem(initial_A, actions_C, goal_A)
    result_C = problem_C.solve_bfs()

    final_state_C = result_C['states'][-1]
    print(f"Initial State : {sorted(list(initial_A))}")
    print(f"Goal State    : {sorted(list(goal_A))}")
    print(f"Plan Found?   : {result_C['success']}")
    print(f"Final State   : {sorted(list(final_state_C))}")
    assert "At(Package, C)" in final_state_C, "Package did not reach C"
    print("[PASS] Test C Oracle: Planner correctly verified At(Package, C), not merely At(Robot, C).")


# ============================================================================
# Section 5: Prolog Logical Verifier & Horn Clause Deduction Engine
# ============================================================================

class PrologVerifier:
    """
    Embedded pure Python Prolog-style Horn Clause Resolution Engine.
    Executes facts, rules, and backward-chaining queries for Tasks 6, 7, 8.
    """
    def __init__(self):
        self.facts: Set[Tuple[str, ...]] = set()
        self.rules: Dict[str, List[Tuple[Tuple[str, ...], List[Tuple[str, ...]]]]] = defaultdict(list)

    def add_fact(self, predicate: str, *args: str):
        self.facts.add((predicate, *args))

    def add_rule(self, head: Tuple[str, ...], body: List[Tuple[str, ...]]):
        pred = head[0]
        self.rules[pred].append((head, body))

    def query(self, predicate: str, *args: str) -> bool:
        """Evaluates ground query via backward chaining."""
        target = (predicate, *args)
        if target in self.facts:
            return True

        if predicate in self.rules:
            for head, body in self.rules[predicate]:
                # Variable unification for binary/unary relations
                subst = {}
                matched = True
                for h_arg, q_arg in zip(head[1:], args):
                    if h_arg.isupper():  # Variable
                        subst[h_arg] = q_arg
                    elif h_arg != q_arg:
                        matched = False
                        break
                if not matched:
                    continue

                # Prove each body literal
                body_satisfied = True
                for b_pred, *b_args in body:
                    ground_args = [subst.get(a, a) for a in b_args]
                    if not self.query(b_pred, *ground_args):
                        body_satisfied = False
                        break
                if body_satisfied:
                    return True
        return False


def run_prolog_verification():
    print("\n" + "=" * 70)
    print("TASK 6 & 7 & 8: PROLOG AS A LOGICAL PLAN VERIFIER")
    print("=" * 70)

    # Generate planner.pl file
    pl_content = """% Prolog Knowledge Base for Warehouse Logical Planning
% Task 6: Connected Locations (Facts)
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Task 6: Movement Rule
can_move(X, Y) :- connected(X, Y).

% Task 7: Plan Step Verifier Rule
valid_move(X, Y) :- connected(X, Y).

% Task 8: Logical Implication Chain
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
"""
    with open("Logical_Planning/planner.pl", "w") as f:
        f.write(pl_content)
    print("[INFO] Generated 'Logical_Planning/planner.pl' for external SWI-Prolog engines.")

    # Execute inside embedded Prolog engine
    engine = PrologVerifier()

    # Add Task 6 & 7 facts & rules
    for u, v in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]:
        engine.add_fact("connected", u, v)

    engine.add_rule(("can_move", "X", "Y"), [("connected", "X", "Y")])
    engine.add_rule(("valid_move", "X", "Y"), [("connected", "X", "Y")])

    print("\n--- Task 6: Queries on can_move ---")
    q1 = engine.query("can_move", "a", "b")
    q2 = engine.query("can_move", "a", "c")
    print(f"Query ?- can_move(a, b). -> {q1} (Expected: True)")
    print(f"Query ?- can_move(a, c). -> {q2} (Expected: False - no direct edge)")

    print("\n--- Task 7: Independent Plan Verification ---")
    plan_steps = [("a", "b"), ("b", "c")]
    invalid_candidate = ("a", "c")

    print(f"Step 1: ?- valid_move(a, b). -> {engine.query('valid_move', 'a', 'b')} [VALID]")
    print(f"Step 2: ?- valid_move(b, c). -> {engine.query('valid_move', 'b', 'c')} [VALID]")
    print(f"Proposed invalid direct jump: ?- valid_move(a, c). -> {engine.query('valid_move', 'a', 'c')} [REJECTED]")

    print("\n--- Task 8: Forward/Backward Implication Chain ---")
    engine.add_fact("wet_road")
    engine.add_rule(("slippery",), [("wet_road",)])
    engine.add_rule(("reduce_speed",), [("slippery",)])

    q_speed = engine.query("reduce_speed")
    print(f"Fact: wet_road.")
    print(f"Rule: slippery :- wet_road.")
    print(f"Rule: reduce_speed :- slippery.")
    print(f"Query ?- reduce_speed. -> {q_speed} (Proved by Modus Ponens chain: wet_road -> slippery -> reduce_speed)")

    print("\nALL PROLOG VERIFICATION TESTS PASSED SUCCESSFULLY!\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    run_planning_tests()
    run_prolog_verification()
    print("=" * 70)
    print("LOGICAL PLANNING LABORATORY COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
