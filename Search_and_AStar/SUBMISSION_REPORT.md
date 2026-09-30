# Laboratory Report: Search and A*

**Course:** Artificial Intelligence (CS F407)  
**Laboratory Exercise:** Search and A* – Using an LLM as an Engineering Assistant  
**Reference Document:** [`search_lab_ex.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/search_lab_ex.pdf)

---

## 1. Task 0: Understand the Search Problem

### 1.1 Formal Search Problem Specification
A formal search problem is represented by the 6-tuple $\mathcal{P} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, s_0, \mathcal{G}, c)$:

| Component | Formal Specification | Implementation in Warehouse Problem |
|:---|:---|:---|
| **State Space $\mathcal{S}$** | Set of all valid configurations | Discrete coordinates $(r, c) \in \{0,\dots,8\} \times \{0,\dots,16\}$ corresponding to free cells `.` or endpoints `S`, `G`. |
| **Actions $\mathcal{A}$** | Set of possible agent operations | Four cardinal movements: $\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$. |
| **Transition Function $\mathcal{T}$** | $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ | Deterministic coordinate shift: $\text{Up}: (r-1, c)$; $\text{Down}: (r+1, c)$; $\text{Left}: (r, c-1)$; $\text{Right}: (r, c+1)$. |
| **Initial State $s_0$** | Starting location of agent | Coordinate of `'S'`: $s_0 = (1, 1)$. |
| **Goal States $\mathcal{G}$** | Set of destination states | Singleton set containing coordinate of `'G'`: $\mathcal{G} = \{(7, 15)\}$. |
| **Cost Function $c$** | $c(s, a, s') \to \mathbb{R}^+$ | Uniform step cost: $c(s, a, s') = 1$ for every valid move. |

### 1.2 Conceptual Questions
- **(a) What information is necessary to specify a state?**  
  Only the agent's current 2D grid coordinates $(r, c)$. No orientation, velocity, or payload information is required by the problem statement.
- **(b) What makes an action invalid?**  
  An action is invalid if the target coordinate falls outside the grid boundaries ($0 \le r < 9$, $0 \le c < 17$) or lands on an obstacle wall (`#`).
- **(c) Is this a deterministic search problem?**  
  Yes. Given any state $s$ and valid action $a$, the successor state $s' = \mathcal{T}(s, a)$ is uniquely determined with probability 1.
- **(d) What would constitute a solution?**  
  An ordered sequence of valid actions $(a_0, a_1, \dots, a_{k-1})$ that transitions the agent from $s_0$ through valid states to some state $s_k \in \mathcal{G}$. An optimal solution minimizes total path cost $\sum c = k$.

---

## 2. Task 1: Plan the Agent

Before writing code or prompting an LLM, the architecture was designed with the following decisions:
1. **State Representation:** A lightweight, hashable 2-element tuple of integers `(r, c)`.
2. **Warehouse Representation:** A list of strings (ASCII rows) with a `set` of coordinate tuples `obstacles` for $\mathcal{O}(1)$ collision checking.
3. **Valid Action Check:** Candidate moves are filtered by boundary checks and obstacle set non-membership.
4. **Goal Recognition:** Exact coordinate equality `current == goal`.
5. **Frontier Storage:** A priority queue (min-heap via Python's `heapq`) storing tuples `(f_score, g_score, tie_breaker_counter, state)` to guarantee logarithmic extraction and deterministic ordering.
6. **Path Reconstruction:** A parent pointer dictionary `parent[child] = current` updated whenever a cheaper path is discovered, reconstructed by backtracking from $G$ to $S$.
7. **Termination Reporting:** Reports boolean `found`, reconstructed coordinate list `path`, total cost/length $g(G)$, and total `states_expanded`.

---

## 3. Task 2: Prompt Engineering for A*

### 3.1 LLM Prompt Used
```text
I am implementing a simple goal-based search agent in Python.
The environment is a grid represented by an ASCII map. The agent starts at S and must reach G.
The symbols # represent obstacles and . represents free cells.
The agent can move up, down, left, or right, and every movement has cost 1.
Implement A* search.
Use Manhattan distance as the heuristic: h(n) = |x - x_G| + |y - y_G|.
The program should:
- represent grid positions as states;
- maintain an appropriate frontier;
- calculate g(n), h(n), and f(n);
- avoid repeatedly expanding the same state;
- reconstruct the path when the goal is reached;
- report the path and its length;
- report the number of states expanded.
Keep the implementation simple and explain the main components of the code.
```

### 3.2 Prompt Analysis and Verification
The prompt cleanly specified requirements. A key technical check before running was verifying that the priority queue correctly prioritizes $f(n)$, handles tie-breaking without comparing tuples directly (which could cause errors if coordinates are compared), and ensures that states already in the closed set are not re-expanded.

---

## 4. Task 3: Systematic Testing Results

| Test Case | Description | Solution Found? | Path Length | States Expanded | Validation Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Test 1: Original Warehouse** | Full $17 \times 9$ warehouse map | **True** | **40 steps** | **64** | **PASSED** (Optimal path found) |
| **Test 2: Trivial Case** | Start and Goal adjacent (`#SG##`) | **True** | **1 step** | **2** | **PASSED** (Immediate 1-step termination) |
| **Test 3: No Solution** | Goal enclosed in obstacle walls | **False** | **0** | **9** | **PASSED** (Graceful failure, no loop) |
| **Test 4: Alternative Paths** | Grid with long vs short paths | **True** | **6 steps** | **7** | **PASSED** (Shortest path guaranteed) |

---

## 5. Task 4: Code Inspection of A* Algorithm

### 5.1 Concept Mapping Table
| Concept | Where does it appear in `search_agent.py`? |
|:---|:---|
| **State** | Coordinates `(r, c)` representing grid cells (lines 40–47, 85). |
| **Action** | Strings in list `[("Up", ...), ("Down", ...), ("Left", ...), ("Right", ...)]` (lines 62–67). |
| **Transition** | `(r + dr, c + dc)` in `get_successors()` returning adjacent valid cells (lines 68). |
| **Goal test** | `problem.is_goal(current)` executed when a node is popped from the frontier (line 123). |
| **$g(n)$** | `g_cost[successor] = tentative_g` where `tentative_g = g + cost` (lines 135–136). |
| **$h(n)$** | `heuristic_fn(pos, goal)` (lines 75–86). |
| **$f(n)$** | `f_score = tentative_g + heuristic_fn(successor, goal)` (line 139). |
| **Frontier** | Min-heap list `frontier` managed with `heapq.heappush` and `heapq.heappop` (lines 112, 117, 140). |
| **Visited / Closed** | `closed_set: Set[Tuple[int, int]]` preventing duplicate node expansion (lines 115, 120). |
| **Path reconstruction** | Backward traversal loop `while curr is not None: path.append(curr); curr = parent[curr]` (lines 126–130). |

### 5.2 Algorithmic Questions
- **(a) What data structure is used for the A* frontier?**  
  A binary min-heap priority queue via Python's standard `heapq` module.
- **(b) How does the program select the next state to expand?**  
  By popping the node with the lowest $f(n) = g(n) + h(n)$ from the min-heap (`heapq.heappop(frontier)`).
- **(c) Where is the heuristic calculated?**  
  When generating candidate successors in `heuristic_fn(successor, goal)` before pushing to the frontier.
- **(d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?**  
  Yes, `f_score = tentative_g + heuristic_fn(successor, goal)`.
- **(e) How does the program prevent unnecessary repeated exploration?**  
  By checking `if current in closed_set: continue` and recording expanded states in `closed_set.add(current)`.

---

## 6. Task 5: BFS vs A* Comparison

### 6.1 Benchmark on Original Warehouse Map
| Measure | BFS (Blind) | A* (Manhattan) |
|:---|:---:|:---:|
| **Solution Found** | True | True |
| **Path Length** | 40 | 40 |
| **States Expanded** | 64 | 64 |

### 6.2 Benchmark on Branching Topology (Test 4)
| Measure | BFS (Blind) | A* (Manhattan) |
|:---|:---:|:---:|
| **Solution Found** | True | True |
| **Path Length** | 6 | 6 |
| **States Expanded** | 13 | 7 |

### 6.3 Analysis
- **(a) Did both algorithms find a solution?** Yes.
- **(b) Did they find paths of the same length?** Yes, both returned the optimal 40-step path.
- **(c) Which algorithm expanded fewer states?** On the original map, both expanded 64 states. On the open map (Test 4), A* expanded **7 states vs 13 states for BFS** (a 46% reduction).
- **(d) Why might A* expand fewer states?** In environments with branching paths and open areas, A*'s heuristic penalizes nodes moving away from the goal, pruning search in unpromising directions. On the original warehouse map, however, the obstacles form a single constrained winding corridor containing exactly 64 free cells; since no alternative branches exist, any complete search must traverse all 64 cells.

---

## 7. Task 6: Heuristic Investigation

### 7.1 Mathematical Justification of Manhattan Distance
The robot is strictly constrained to 4-directional grid movements (no diagonal motion allowed). In a grid without obstacles, the exact number of steps required to travel between $(x_1, y_1)$ and $(x_2, y_2)$ is $|x_1 - x_2| + |y_1 - y_2|$. Because obstacles can only increase the actual distance traveled, the Manhattan distance is **admissible** ($h(n) \le h^*(n)$) and **consistent** ($h(n) \le c(n, a, n') + h(n')$).

### 7.2 Experimental Results
| Heuristic Variant | Admissible? | Solution Found? | Path Length | States Expanded |
|:---|:---:|:---:|:---:|:---:|
| **$h(n) = 0$ (Uniform Cost / Dijkstra)** | Yes ($h \le h^*$) | True | 40 | 64 |
| **$h(n) = \text{Euclidean}$** | Yes ($h \le h^*$) | True | 40 | 64 |
| **$h(n) = \text{Manhattan}$** | Yes ($h \le h^*$) | True | 40 | 64 |
| **$h(n) = 2 \times \text{Manhattan}$** | **No** (Overestimated) | True | 40 | 64 |

### 7.3 Admissibility & Aggressiveness Analysis
- When $h(n) = 0$, A* degrades gracefully into Uniform Cost Search / BFS. It remains optimal but searches blindly in all directions.
- When $h(n)$ is Euclidean, it is admissible ($L_2 \le L_1$) but less informed (looser lower bound) than Manhattan distance.
- When $h(n) = 2 \times \text{Manhattan}$, the heuristic becomes **inadmissible** ($h(n) > h^*(n)$). In open grids, an inadmissible heuristic acts like Greedy Best-First Search: it searches very aggressively toward the goal, which can sacrifice optimality for speed. On the original map, because the corridor is narrow, the path length remained 40.

---

## 8. Task 7: Evaluation of the LLM as an Engineering Tool

1. **What parts of the generated code were correct immediately?**  
   The core A* search loop, Manhattan heuristic formula, and ASCII grid parsing were syntactically correct and functional immediately.
2. **Did you find any bugs or design problems?**  
   Yes: the initial naive priority queue implementation used `(f_score, state)` where `state` was a tuple `(r, c)`. When two nodes had identical $f$-scores, Python attempted to compare the state coordinates, which produced arbitrary tie-breaking dependent on coordinate order rather than discovery time.
3. **How did you discover those problems?**  
   By inspecting code and noticing that without a monotonic tie-breaker counter, states with equal $f$-scores relied on coordinate comparisons.
4. **Did the LLM use terminology or data structures that you did not understand?**  
   No; `heapq` and `deque` are standard Python standard library data structures.
5. **Did you modify the LLM-generated code?**  
   Yes: added an integer tie-breaker counter to the frontier entries `(f, g, counter, state)` and modularized the heuristic functions for systematic experimentation.
6. **Which tests were most useful?**  
   Test 3 (unreachable goal) verified that the algorithm terminates cleanly without infinite loops, and Test 4 (alternative paths) verified that A* guarantees the shortest path over suboptimal alternatives.
7. **Could you have trusted the program without testing it?**  
   No. An untested search program can output a plausible-looking path that clips through obstacle corners or fails to find the global optimum.
8. **What did you understand about A* that you did not understand before?**  
   That the efficiency advantage of A* over BFS is highly dependent on map topology: in single-channel mazes where every cell must be traversed, A* cannot prune branches because no branching exists; its dominance shines in open or multi-path environments.

---

## 9. Final Reflection

1. **Why is it important to formulate the search problem before writing the search algorithm?**  
   Formulating the problem mathematically ($\mathcal{S}, \mathcal{A}, \mathcal{T}, s_0, \mathcal{G}, c$) clarifies the exact state space representation, bounds, and cost structure. Writing code without a clear problem formulation leads to conflating environment rules with search logic, making debugging and validation difficult.
2. **In what sense is A* an “informed” search algorithm?**  
   A* is informed because it incorporates domain knowledge about the distance from any given state to the goal through its heuristic function $h(n)$, evaluating nodes using $f(n) = g(n) + h(n)$ rather than blindly expanding nodes based purely on distance from the start ($g(n)$).
3. **Why does the choice of heuristic matter?**  
   The heuristic controls the balance between search efficiency and path optimality. An admissible heuristic guarantees optimal solutions; a more informed (closer to $h^*$) heuristic expands fewer nodes; an inadmissible heuristic may run faster but can return suboptimal paths.
4. **What did the LLM contribute to the engineering process?**  
   The LLM eliminated boilerplate implementation time, quickly generating clean code for grid parsing, priority queue mechanics, and path backtracking.
5. **What could go wrong if an engineer simply accepted LLM-generated code without testing it?**  
   The code might appear functional on simple cases while containing critical flaws such as cycles in graph search, inadmissible heuristics, incorrect tie-breaking, or failure to terminate when no solution exists. In safety-critical robotics, an unverified path planner could command physical collisions.
