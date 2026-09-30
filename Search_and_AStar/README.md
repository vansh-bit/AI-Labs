# Search and A*

## Overview
This directory contains the complete submission materials for **AI Laboratory Exercise: Search and A* – Using an LLM as an Engineering Assistant**.

## Contents
- [`search_lab_ex.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/search_lab_ex.pdf): The original laboratory exercise document.
- [`search_agent.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/search_agent.py): Complete Python implementation including `SearchProblem`, A* search with multiple heuristics, BFS benchmark, full test suite (Tests 1 to 4), and heuristic investigation.
- [`search_notebook.ipynb`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/search_notebook.ipynb): Interactive Jupyter notebook for the laboratory.
- [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/SUBMISSION_REPORT.md): Comprehensive submission report fulfilling all required sections: problem formulation, agent planning, prompt engineering, systematic test results, code concept inspection, BFS vs A* comparison, heuristic analysis, and final reflections.

## Quick Start
To run all tests and benchmarks:
```bash
python3 search_agent.py
```
Expected output:
- Original Warehouse Path Length: 40 steps
- BFS States Expanded: 64
- A* (Manhattan) States Expanded: 64
- Alternative Paths Test: A* (7 states) vs BFS (13 states)
