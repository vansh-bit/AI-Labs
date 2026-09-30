# Bayesian Networks and Autoregressive Language Models

## Overview
This directory contains the complete laboratory submission for **Bayesian Networks and Autoregressive Language Models**, connecting probabilistic graphical models, chain-rule joint distribution factorisation, first- and second-order Markov models, and modern next-token generation.

## Contents
- [`BN_lab.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/BN_lab.pdf): Laboratory specification document.
- [`autoregressive_bn.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/autoregressive_bn.py): Standalone, zero-dependency Python script implementing:
  - First-order autoregressive Bayesian network ($P(X_t \mid X_{t-1})$).
  - Second-order autoregressive Bayesian network ($P(X_t \mid X_{t-2}, X_{t-1})$).
  - Exact CPT estimation from transition counts.
  - Normalisation invariant verification test oracles.
  - Deterministic (greedy) vs. probabilistic (sampling) text generation.
  - Quantitative comparison of parameters, sparsity, and sentence diversity.
- [`autoregressive_bn_notebook.ipynb`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/autoregressive_bn_notebook.ipynb): Interactive Jupyter notebook reproducing the experiments.
- [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Bayesian_Networks/SUBMISSION_REPORT.md): In-depth academic report answering all 14 questions, derivations, and reflections.

## Quick Execution
Run the complete pipeline from terminal:
```bash
python3 Bayesian_Networks/autoregressive_bn.py
```
