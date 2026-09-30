# Logical Reasoning for Planning

## Overview
This directory contains the complete laboratory submission for **Logical Reasoning for Planning**, demonstrating how formal propositional logic and state-space search (BFS) combine to solve automated planning problems, complemented by Prolog-based independent plan verification.

## Contents
- [`logic_lab_ex.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/logic_lab_ex.pdf): Laboratory specification document.
- [`logical_planner.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/logical_planner.py): Standalone Python implementation featuring:
  - STRIPS-style `Action` schemas with positive/negative preconditions and add/delete effects.
  - Action applicability theorem prover ($S \models \text{Preconditions}(a)$).
  - Breadth-First Search (BFS) state-space planning engine.
  - Systematic test oracles: Test A (solvable), Test B (impossible), Test C (irrelevant actions).
  - Embedded Prolog-style Horn clause resolution engine.
- [`planner.pl`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/planner.pl): Declarative Prolog knowledge base for external SWI-Prolog engines.
- [`logical_planning_notebook.ipynb`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/logical_planning_notebook.ipynb): Interactive Jupyter notebook reproducing all tasks and test cases.
- [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/SUBMISSION_REPORT.md): Comprehensive academic report answering Tasks 0–5, Reflection Questions 1–7, and Prolog extension Questions.

## Quick Execution
Run all planning tests and Prolog verifications directly from terminal:
```bash
python3 Logical_Planning/logical_planner.py
```
