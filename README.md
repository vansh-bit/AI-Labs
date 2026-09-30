# Artificial Intelligence (CS F407) - Laboratory Submissions

This repository contains the complete laboratory submissions and implementations for the **Artificial Intelligence** course, structured into separate self-contained folders for each laboratory exercise.

---

## Directory Structure

```
AI Labs/
├── Docs/                                       # Original laboratory documents and reference PDFs
│   ├── agents_lab.pdf
│   ├── search_lab_ex.pdf
│   ├── neur_models_lab_ex.pdf
│   ├── BN_lab.pdf
│   └── logic_lab_ex.pdf
│
├── Agents/                                     # Goal-Based Agent (Warehouse Navigation)
│   ├── agents_lab.pdf                          # Lab specification document
│   ├── warehouse_agent.py                      # Python solution script (BFS goal-based agent)
│   ├── agents_notebook.ipynb                   # Interactive Jupyter notebook
│   ├── SUBMISSION_REPORT.md                    # Complete lab submission report
│   └── README.md                               # Folder instructions and quick-start
│
├── Search_and_AStar/                           # Search and A* (LLM as Engineering Assistant)
│   ├── search_lab_ex.pdf                       # Lab specification document
│   ├── search_agent.py                         # Complete search engine (A*, BFS, 4 heuristics)
│   ├── search_notebook.ipynb                   # Interactive Jupyter notebook
│   ├── SUBMISSION_REPORT.md                    # Complete lab submission report
│   └── README.md                               # Folder instructions and quick-start
│
├── Neural_Models/                              # Neural Models (Depth, Activations, Losses)
│   ├── neur_models_lab_ex.pdf                  # Lab specification document
│   ├── neural_models.py                        # PyTorch & standalone autograd implementation
│   ├── neural_models_notebook.ipynb            # Interactive Jupyter notebook
│   ├── SUBMISSION_REPORT.md                    # Complete lab submission report
│   └── README.md                               # Folder instructions and quick-start
│
├── Bayesian_Networks/                          # Bayesian Networks & Autoregressive Models
│   ├── BN_lab.pdf                              # Lab specification document
│   ├── autoregressive_bn.py                    # 1st- and 2nd-order Markov language models
│   ├── autoregressive_bn_notebook.ipynb        # Interactive Jupyter notebook
│   ├── SUBMISSION_REPORT.md                    # Complete lab submission report (Q1-Q14)
│   └── README.md                               # Folder instructions and quick-start
│
└── Logical_Planning/                           # Logical Reasoning for Planning
    ├── logic_lab_ex.pdf                        # Lab specification document
    ├── logical_planner.py                      # STRIPS planning agent + BFS + Prolog verifier
    ├── planner.pl                              # Declarative Prolog knowledge base
    ├── logical_planning_notebook.ipynb         # Interactive Jupyter notebook
    ├── SUBMISSION_REPORT.md                    # Complete lab submission report
    └── README.md                               # Folder instructions and quick-start
```

---

## Laboratory Overview

| Folder | Title | Key Topics | Primary Artifacts |
|:---|:---|:---|:---|
| **[`Agents`](file:///Users/vanshsharma/Documents/AI%20Labs/Agents)** | Goal-Based Agent for Warehouse Navigation | Goal-based vs. reflex agents, problem formulation, BFS path planning | [`warehouse_agent.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Agents/warehouse_agent.py), [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Agents/SUBMISSION_REPORT.md) |
| **[`Search_and_AStar`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar)** | Search and A* | $A^*$ search, Manhattan vs. Euclidean vs. zero vs. $2\times$ Manhattan, BFS comparison, test oracles | [`search_agent.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/search_agent.py), [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Search_and_AStar/SUBMISSION_REPORT.md) |
| **[`Neural_Models`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models)** | Neural Models: Depth, Activations, Losses | Linear non-separability of XOR, backpropagation, weight symmetry breaking, activations (Sigmoid, Tanh, ReLU), 3-class softmax | [`neural_models.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/neural_models.py), [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/SUBMISSION_REPORT.md) |
| **[`Bayesian_Networks`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks)** | Bayesian Networks & Autoregressive LMs | Chain-rule factorisation, 1st & 2nd order Markov models, exact CPTs, invariant test oracles, greedy vs sampling | [`autoregressive_bn.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/autoregressive_bn.py), [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/SUBMISSION_REPORT.md) |
| **[`Logical_Planning`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning)** | Logical Reasoning for Planning | Propositional logic, action preconditions & effects, BFS state-space search, Prolog independent plan verification | [`logical_planner.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/logical_planner.py), [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Logical_Planning/SUBMISSION_REPORT.md) |

---

## Quick Execution
Every Python script is self-contained and can be executed directly from terminal:

```bash
python3 Agents/warehouse_agent.py
python3 Search_and_AStar/search_agent.py
python3 Neural_Models/neural_models.py
python3 Bayesian_Networks/autoregressive_bn.py
python3 Logical_Planning/logical_planner.py
```
