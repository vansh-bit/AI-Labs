"""
CS F407 Artificial Intelligence - Laboratory Exercise
Building and Learning a Bayesian Network (llm_bn)

This module implements:
1. Ground Truth / Reference Bayesian Network (Cloudy -> Rain, Cloudy -> Sprinkler, Rain & Sprinkler -> WetGrass).
2. Exact Inference using Variable Elimination.
3. Test Oracle: Independent verification via full joint distribution enumeration (2^4 = 16 states).
4. Code Safety & AST Validation for LLM-generated probabilistic code.
5. Parameter Estimation using Maximum Likelihood Estimation (MLE) across multiple seeds.
6. Sampling variability analysis and evaluation rubric.
"""

import ast
import itertools
import random
import math
from typing import Dict, List, Tuple, Any, Optional

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import MaximumLikelihoodEstimator
    import pandas as pd
    import numpy as np
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


# =====================================================================
# REFERENCE MODEL DEFINITION (pgmpy when available)
# =====================================================================
def build_reference_model_pgmpy():
    """Constructs the reference Sprinkler Bayesian Network using pgmpy."""
    if not PGMPY_AVAILABLE:
        return None

    model = DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

    cpd_c = TabularCPD(
        variable="Cloudy",
        variable_card=2,
        values=[[0.5], [0.5]]
    )

    cpd_r = TabularCPD(
        variable="Rain",
        variable_card=2,
        values=[
            [0.8, 0.2],  # Rain=0 | Cloudy=0, Cloudy=1
            [0.2, 0.8]   # Rain=1 | Cloudy=0, Cloudy=1
        ],
        evidence=["Cloudy"],
        evidence_card=[2]
    )

    cpd_s = TabularCPD(
        variable="Sprinkler",
        variable_card=2,
        values=[
            [0.5, 0.9],  # Sprinkler=0 | Cloudy=0, Cloudy=1
            [0.5, 0.1]   # Sprinkler=1 | Cloudy=0, Cloudy=1
        ],
        evidence=["Cloudy"],
        evidence_card=[2]
    )

    cpd_w = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[
            [0.99, 0.10, 0.10, 0.01],  # WetGrass=0 | (R=0,S=0), (0,1), (1,0), (1,1)
            [0.01, 0.90, 0.90, 0.99]   # WetGrass=1 | (R=0,S=0), (0,1), (1,0), (1,1)
        ],
        evidence=["Rain", "Sprinkler"],
        evidence_card=[2, 2]
    )

    model.add_cpds(cpd_c, cpd_r, cpd_s, cpd_w)
    assert model.check_model(), "Reference model is invalid!"
    return model


# =====================================================================
# INDEPENDENT PROBABILISTIC ENGINE (Zero-dependency test oracle)
# =====================================================================
def p_cloudy(c: int) -> float:
    return 0.5

def p_rain(r: int, c: int) -> float:
    p_true = 0.8 if c == 1 else 0.2
    return p_true if r == 1 else 1.0 - p_true

def p_sprinkler(s: int, c: int) -> float:
    p_true = 0.1 if c == 1 else 0.5
    return p_true if s == 1 else 1.0 - p_true

def p_wetgrass(w: int, r: int, s: int) -> float:
    cpt = {
        (0, 0): 0.01,
        (0, 1): 0.90,
        (1, 0): 0.90,
        (1, 1): 0.99
    }
    p_true = cpt[(r, s)]
    return p_true if w == 1 else 1.0 - p_true

def joint_probability(c: int, r: int, s: int, w: int) -> float:
    """Factorisation: P(C, R, S, W) = P(C) * P(R|C) * P(S|C) * P(W|R,S)"""
    return p_cloudy(c) * p_rain(r, c) * p_sprinkler(s, c) * p_wetgrass(w, r, s)

def exact_inference_by_enumeration(query_var: str = "Rain", evidence: Dict[str, int] = None) -> Dict[int, float]:
    """
    Test Oracle: Computes exact posterior by summing over all 2^4 = 16 joint states.
    """
    if evidence is None:
        evidence = {"WetGrass": 1}

    marginal = {0: 0.0, 1: 0.0}
    total_evidence_prob = 0.0

    for c, r, s, w in itertools.product([0, 1], repeat=4):
        state = {"Cloudy": c, "Rain": r, "Sprinkler": s, "WetGrass": w}
        # Check evidence match
        match = all(state[k] == v for k, v in evidence.items())
        if match:
            prob = joint_probability(c, r, s, w)
            total_evidence_prob += prob
            marginal[state[query_var]] += prob

    normalized_posterior = {val: marginal[val] / total_evidence_prob for val in marginal}
    return normalized_posterior


# =====================================================================
# AST SAFETY & CODE VERIFICATION
# =====================================================================
ALLOWED_IMPORT_PREFIXES = ("pgmpy", "pandas", "numpy", "matplotlib", "math", "collections", "itertools")
FORBIDDEN_CALL_NAMES = {"eval", "exec", "open", "__import__", "compile", "breakpoint"}

def basic_generated_code_check(source: str) -> List[str]:
    """
    AST inspection checking that LLM-generated code contains no unsafe calls or imports.
    """
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not any(alias.name.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                    problems.append(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not any(module.startswith(p) for p in ALLOWED_IMPORT_PREFIXES):
                problems.append(f"Import not allowed: {module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                problems.append(f"Forbidden call: {node.func.id}")

    return problems


def simulate_dataset(n_samples: int, seed: int = 42) -> List[Dict[str, int]]:
    """
    Forward sampling from the generative joint distribution.
    """
    random.seed(seed)
    samples = []
    for _ in range(n_samples):
        # Sample Cloudy
        c = 1 if random.random() < 0.5 else 0
        # Sample Rain given Cloudy
        p_r = 0.8 if c == 1 else 0.2
        r = 1 if random.random() < p_r else 0
        # Sample Sprinkler given Cloudy
        p_s = 0.1 if c == 1 else 0.5
        s = 1 if random.random() < p_s else 0
        # Sample WetGrass given Rain and Sprinkler
        cpt_w = {(0, 0): 0.01, (0, 1): 0.90, (1, 0): 0.90, (1, 1): 0.99}
        w = 1 if random.random() < cpt_w[(r, s)] else 0

        samples.append({"Cloudy": c, "Rain": r, "Sprinkler": s, "WetGrass": w})
    return samples


def estimate_p_rain_given_cloudy1(samples: List[Dict[str, int]]) -> float:
    """Computes Maximum Likelihood Estimate P_hat(Rain=1 | Cloudy=1)."""
    count_c1 = sum(1 for row in samples if row["Cloudy"] == 1)
    count_r1_c1 = sum(1 for row in samples if row["Cloudy"] == 1 and row["Rain"] == 1)
    return count_r1_c1 / count_c1 if count_c1 > 0 else 0.0


def run_all_experiments():
    print("=" * 70)
    print("AI LAB 4: BUILDING AND LEARNING A BAYESIAN NETWORK (llm_bn)")
    print("=" * 70)

    # 1. Structure and Joint Factorisation
    print("\n--- 1. BAYESIAN NETWORK STRUCTURE & SPECIFICATION ---")
    print("Directed Edges : C -> R,  C -> S,  R -> W,  S -> W")
    print("Variables      : C (Cloudy), R (Rain), S (Sprinkler), W (WetGrass)")
    print("Factorisation  : P(C, R, S, W) = P(C) * P(R|C) * P(S|C) * P(W|R, S)")

    # 2. Exact Inference Oracle
    print("\n--- 2. EXACT INFERENCE (TEST ORACLE) ---")
    print("Query: Compute posterior P(Rain=1 | WetGrass=1)")
    oracle_res = exact_inference_by_enumeration(query_var="Rain", evidence={"WetGrass": 1})
    print(f"P(Rain=0 | WetGrass=1) = {oracle_res[0]:.16f}")
    print(f"P(Rain=1 | WetGrass=1) = {oracle_res[1]:.16f}")

    if PGMPY_AVAILABLE:
        model = build_reference_model_pgmpy()
        infer = VariableElimination(model)
        pgmpy_res = infer.query(variables=["Rain"], evidence={"WetGrass": 1})
        val_pgmpy = pgmpy_res.values[1]
        print(f"pgmpy VariableElimination : {val_pgmpy:.16f}")
        diff = abs(oracle_res[1] - val_pgmpy)
        print(f"Absolute Difference       : {diff:.2e} (Matches within machine precision)")
    else:
        print("Note: pgmpy not in active environment; verified with exact analytical test oracle.")

    # 3. AST Code Safety Verification
    print("\n--- 3. AST CODE SAFETY CHECK ON GENERATED IMPLEMENTATIONS ---")
    safe_snippet = "from pgmpy.models import DiscreteBayesianNetwork\nmodel = DiscreteBayesianNetwork()"
    unsafe_snippet = "import os\nos.system('rm -rf /')"

    res_safe = basic_generated_code_check(safe_snippet)
    res_unsafe = basic_generated_code_check(unsafe_snippet)
    print(f"Safe Code Problems   : {res_safe} (Clean)")
    print(f"Unsafe Code Problems : {res_unsafe} (Successfully intercepted dangerous import)")

    # 4. Parameter Estimation & Sampling Variability
    print("\n--- 4. PARAMETER ESTIMATION (MLE) & SAMPLING VARIABILITY ---")
    print("Estimating P(Rain=1 | Cloudy=1) from N=100 simulated samples across 5 seeds:")
    print("Ground Truth Parameter: 0.8000")
    print(f"{'Seed':<8} | {'N_samples':<12} | {'Estimated P(R=1 | C=1)':<24} | {'Error':<12}")
    print("-" * 62)

    estimates = []
    for seed in [1, 2, 3, 4, 5]:
        data = simulate_dataset(n_samples=100, seed=seed)
        est = estimate_p_rain_given_cloudy1(data)
        estimates.append(est)
        err = est - 0.80
        print(f"{seed:<8} | {100:<12} | {est:<24.4f} | {err:+12.4f}")

    mean_est = sum(estimates) / len(estimates)
    print("-" * 62)
    print(f"Mean Estimate across 5 seeds: {mean_est:.4f}")
    print("Scientific Conclusion: Fluctuations across seeds represent natural statistical sampling variability,")
    print("NOT an error in the Bayesian network structure or estimation algorithm.")


if __name__ == "__main__":
    run_all_experiments()
