# Neural Models (Learning, Depth, Activations, and Output Layers)

## Overview
This directory contains the complete submission materials for **AI Laboratory Exercise: Neural Models – Learning, Depth, Activations, and Output Layers**.

## Contents
- [`neur_models_lab_ex.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/neur_models_lab_ex.pdf): The original laboratory exercise document.
- [`neural_models.py`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/neural_models.py): Complete implementation including PyTorch models and a zero-dependency autograd fallback engine implementing Tasks 1 to 5.
- [`neural_models_notebook.ipynb`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/neural_models_notebook.ipynb): Interactive Jupyter notebook for the laboratory.
- [`SUBMISSION_REPORT.md`](file:///Users/vanshsharma/Documents/AI%20Labs/Neural_Models/SUBMISSION_REPORT.md): Comprehensive submission report with mathematical proofs (linear separability, $p - y$ gradient, softmax sum), architecture details, experimental tables, symmetry diagnostics, and reflection questions.

## Quick Start
To run all experiments:
```bash
python3 neural_models.py
```
Expected output:
- Basic Learning Check: Initial Loss $\approx 0.72$, Final Loss $< 0.005$, 4/4 correct.
- Symmetry Experiment: Rows remain identical across all steps.
- Activation Experiment: Comparison across Sigmoid, Tanh, and ReLU.
- Three-Class Extension: Validated softmax and logit gradients.
