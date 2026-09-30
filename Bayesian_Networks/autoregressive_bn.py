#!/usr/bin/env python3
"""
AI Laboratory: Bayesian Networks and Autoregressive Language Models
Course: CS F407 - Artificial Intelligence

This module provides a complete, self-contained implementation of first-order
and second-order autoregressive language models formulated as Bayesian networks.

Core Concepts Implemented:
  1. Chain-rule joint distribution factorisation:
     P(X_1, ..., X_T) = P(X_1) * prod_{t=2}^T P(X_t | X_1, ..., X_{t-1})
  2. First-order Markov Bayesian Network: X_1 -> X_2 -> ... -> X_T
     Assumption: P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})
  3. Second-order Markov Bayesian Network: (X_{t-2}, X_{t-1}) -> X_t
     Assumption: P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-2}, X_{t-1})
  4. Exact Conditional Probability Tables (CPTs) via transition counting
  5. Probability normalization test oracles: sum_v P(v | w) == 1.0
  6. Deterministic (Greedy Argmax) vs. Probabilistic (Sampling) text generation
  7. Comparative evaluation: parameters, sparsity, diversity, and coherence

Design: Pure Python standard library (zero external dependencies required).
"""

import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

# ============================================================================
# Section 1: Dataset & Tokenization
# ============================================================================

RAW_TRAINING_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START_TOKEN = "<START>"
END_TOKEN = "<END>"

def prepare_tokenized_dataset(corpus: List[str]) -> List[List[str]]:
    """
    Lowercases and tokenizes sentences, prefixing <START> and suffixing <END>.
    """
    tokenized = []
    for sentence in corpus:
        tokens = [START_TOKEN] + sentence.strip().lower().split() + [END_TOKEN]
        tokenized.append(tokens)
    return tokenized


# ============================================================================
# Section 2: First-Order Autoregressive Bayesian Network
# ============================================================================

class FirstOrderLanguageModel:
    """
    First-Order Autoregressive Language Model:
      P(X_t | X_{t-1}) = C(X_{t-1}, X_t) / sum_v C(X_{t-1}, v)
    
    Bayesian Network Structure:
      X_1 -> X_2 -> X_3 -> ... -> X_T
    """
    def __init__(self):
        # Nested dict: counts[w_i][w_j] = count of transition w_i -> w_j
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Nested dict: probabilities[w_i][w_j] = P(w_j | w_i)
        self.probabilities: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.vocabulary: set = set()

    def train(self, tokenized_sentences: List[List[str]]) -> None:
        """Counts bigram transitions and computes normalized conditional probabilities."""
        self.counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        for tokens in tokenized_sentences:
            for w in tokens:
                self.vocabulary.add(w)
            for i in range(len(tokens) - 1):
                w_curr = tokens[i]
                w_next = tokens[i + 1]
                self.counts[w_curr][w_next] += 1

        # Normalize counts into conditional probability distributions
        for w_curr, transitions in self.counts.items():
            total = sum(transitions.values())
            for w_next, count in transitions.items():
                self.probabilities[w_curr][w_next] = count / total

    def get_distribution(self, word: str) -> Dict[str, float]:
        """Returns P(next_word | word)."""
        return dict(self.probabilities.get(word, {}))

    def predict_most_probable(self, word: str) -> Optional[str]:
        """Mode A (Greedy): argmax_w P(w | word)."""
        dist = self.get_distribution(word)
        if not dist:
            return None
        # Sort by probability descending, tie-breaking alphabetically
        return sorted(dist.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def sample_next_word(self, word: str, rng: random.Random) -> Optional[str]:
        """Mode B (Sampling): samples w ~ P(w | word)."""
        dist = self.get_distribution(word)
        if not dist:
            return None
        candidates = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence(self, mode: str = "sample", max_tokens: int = 25, seed: Optional[int] = None) -> List[str]:
        """
        Generates a sequence starting from <START> until <END> or max_tokens.
        mode: 'sample' or 'greedy'
        """
        rng = random.Random(seed) if seed is not None else random.Random()
        sentence = [START_TOKEN]
        curr = START_TOKEN

        for _ in range(max_tokens):
            if mode == "greedy":
                nxt = self.predict_most_probable(curr)
            else:
                nxt = self.sample_next_word(curr, rng)

            if nxt is None or nxt == END_TOKEN:
                sentence.append(END_TOKEN)
                break
            sentence.append(nxt)
            curr = nxt

        return sentence


# ============================================================================
# Section 3: Second-Order Autoregressive Bayesian Network
# ============================================================================

class SecondOrderLanguageModel:
    """
    Second-Order Autoregressive Language Model:
      P(X_t | X_{t-2}, X_{t-1}) = C(X_{t-2}, X_{t-1}, X_t) / sum_v C(X_{t-2}, X_{t-1}, v)

    Bayesian Network Structure:
      (X_{t-2}, X_{t-1}) -> X_t
    """
    def __init__(self):
        # Key: (w_{t-2}, w_{t-1}), Value: {w_t: count}
        self.counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.probabilities: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        # Fallback first-order model for initial step (<START>)
        self.first_order_fallback = FirstOrderLanguageModel()
        self.vocabulary: set = set()

    def train(self, tokenized_sentences: List[List[str]]) -> None:
        """Trains trigram transitions and fallback bigrams."""
        self.counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        # Train fallback first-order model
        self.first_order_fallback.train(tokenized_sentences)

        for tokens in tokenized_sentences:
            for w in tokens:
                self.vocabulary.add(w)
            for i in range(len(tokens) - 2):
                prefix = (tokens[i], tokens[i + 1])
                target = tokens[i + 2]
                self.counts[prefix][target] += 1

        for prefix, targets in self.counts.items():
            total = sum(targets.values())
            for target, count in targets.items():
                self.probabilities[prefix][target] = count / total

    def get_distribution(self, w_prev2: str, w_prev1: str) -> Dict[str, float]:
        """Returns P(next_word | w_{t-2}, w_{t-1})."""
        prefix = (w_prev2, w_prev1)
        if prefix in self.probabilities:
            return dict(self.probabilities[prefix])
        # Fallback to first order if context unseen
        return self.first_order_fallback.get_distribution(w_prev1)

    def predict_most_probable(self, w_prev2: str, w_prev1: str) -> Optional[str]:
        """Greedy argmax for second order."""
        dist = self.get_distribution(w_prev2, w_prev1)
        if not dist:
            return None
        return sorted(dist.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def sample_next_word(self, w_prev2: str, w_prev1: str, rng: random.Random) -> Optional[str]:
        """Sample from second-order conditional distribution."""
        dist = self.get_distribution(w_prev2, w_prev1)
        if not dist:
            return None
        candidates = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence(self, mode: str = "sample", max_tokens: int = 25, seed: Optional[int] = None) -> List[str]:
        """Generates sentence using second-order contexts."""
        rng = random.Random(seed) if seed is not None else random.Random()
        sentence = [START_TOKEN]

        # First token after <START> generated via fallback first-order model
        first_word = (
            self.first_order_fallback.predict_most_probable(START_TOKEN)
            if mode == "greedy"
            else self.first_order_fallback.sample_next_word(START_TOKEN, rng)
        )
        if first_word is None or first_word == END_TOKEN:
            sentence.append(END_TOKEN)
            return sentence

        sentence.append(first_word)

        # Subsequent tokens generated via second-order conditioning
        for _ in range(max_tokens - 1):
            w_prev2, w_prev1 = sentence[-2], sentence[-1]
            if mode == "greedy":
                nxt = self.predict_most_probable(w_prev2, w_prev1)
            else:
                nxt = self.sample_next_word(w_prev2, w_prev1, rng)

            if nxt is None or nxt == END_TOKEN:
                sentence.append(END_TOKEN)
                break
            sentence.append(nxt)

        return sentence


# ============================================================================
# Section 4: Testing & Probabilistic Invariant Oracles
# ============================================================================

def verify_probability_invariants(model: FirstOrderLanguageModel) -> Dict[str, float]:
    """
    Validates that for every conditioning context w:
      sum_v P(v | w) == 1.0 (within numerical tolerance 1e-9).
    """
    totals = {}
    for word, dist in model.probabilities.items():
        total = sum(dist.values())
        totals[word] = total
        assert abs(total - 1.0) < 1e-9, f"Normalization failure for word '{word}': total = {total}"
    return totals

def verify_second_order_invariants(model: SecondOrderLanguageModel) -> Dict[Tuple[str, str], float]:
    """
    Validates that for every observed second-order context (w1, w2):
      sum_v P(v | w1, w2) == 1.0.
    """
    totals = {}
    for prefix, dist in model.probabilities.items():
        total = sum(dist.values())
        totals[prefix] = total
        assert abs(total - 1.0) < 1e-9, f"Normalization failure for prefix {prefix}: total = {total}"
    return totals


# ============================================================================
# Section 5: Demonstration, Comparisons, and Analysis Runner
# ============================================================================

def run_laboratory_demonstration():
    print("=" * 70)
    print("AI LABORATORY: BAYESIAN NETWORKS & AUTOREGRESSIVE LANGUAGE MODELS")
    print("=" * 70)

    dataset = prepare_tokenized_dataset(RAW_TRAINING_CORPUS)
    print(f"\n[Part III] Prepared Training Corpus ({len(dataset)} sentences):")
    for i, s in enumerate(dataset, 1):
        print(f"  Sentence {i}: {' '.join(s)}")

    # ------------------------------------------------------------------------
    # Part IV: Train First-Order Model & Display CPTs
    # ------------------------------------------------------------------------
    lm1 = FirstOrderLanguageModel()
    lm1.train(dataset)
    vocab = sorted(list(lm1.vocabulary))
    print(f"\nVocabulary Size (|V|): {len(vocab)} tokens")
    print(f"Tokens: {vocab}")

    print("\n" + "-" * 70)
    print("[Part IV & VII] First-Order Conditional Probability Tables P(X_t | X_{t-1}):")
    print("-" * 70)
    target_words = [START_TOKEN, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]
    for w in target_words:
        dist = lm1.get_distribution(w)
        if dist:
            formatted = ", ".join(f"P({nxt}|{w}) = {prob:.4f}" for nxt, prob in sorted(dist.items()))
            row_sum = sum(dist.values())
            print(f"  {w:8s} -> {formatted:<50s} [Row Sum = {row_sum:.4f}]")
        else:
            print(f"  {w:8s} -> (Terminal token <END> has no successors)")

    # Run Invariant Oracle
    totals = verify_probability_invariants(lm1)
    print(f"\n[PASS] Probability Invariant Test Oracle: All {len(totals)} conditional distributions sum strictly to 1.0.")

    # ------------------------------------------------------------------------
    # Part VIII: Next-Word Prediction
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Part VIII] Most Probable Next-Word Predictions (Argmax):")
    print("-" * 70)
    test_contexts = [START_TOKEN, "the", "cat", "dog", "sat", "ran"]
    for ctx in test_contexts:
        dist = lm1.get_distribution(ctx)
        pred = lm1.predict_most_probable(ctx)
        dist_str = ", ".join(f"{k}: {v:.2f}" for k, v in dist.items())
        print(f"  Context: '{ctx:7s}' -> Distribution: {{{dist_str}}} | Argmax: '{pred}'")

    # ------------------------------------------------------------------------
    # Part IX & X: Deterministic vs. Probabilistic Text Generation
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Part X] Deterministic (Greedy Mode A) vs. Probabilistic (Sampling Mode B) Generation:")
    print("-" * 70)

    print("\n>>> Mode A: Greedy Generation (5 runs):")
    for r in range(1, 6):
        greedy_sent = lm1.generate_sentence(mode="greedy")
        print(f"  Run {r}: {' '.join(greedy_sent)}")

    print("\n>>> Mode B: Probabilistic Sampling Generation (5 runs with different seeds):")
    for r in range(1, 6):
        sampled_sent = lm1.generate_sentence(mode="sample", seed=r * 17)
        print(f"  Run {r}: {' '.join(sampled_sent)}")

    print("\n>>> Part IX: 20 Sampled Sentences (First-Order Model):")
    sampled_20_first = []
    for r in range(1, 21):
        s = lm1.generate_sentence(mode="sample", seed=r * 101)
        sampled_20_first.append(" ".join(s))
        print(f"  [{r:02d}] {sampled_20_first[-1]}")

    # ------------------------------------------------------------------------
    # Part XI & XII: Second-Order Model
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Part XI & XII] Second-Order Bayesian Network P(X_t | X_{t-2}, X_{t-1}):")
    print("-" * 70)
    lm2 = SecondOrderLanguageModel()
    lm2.train(dataset)
    verify_second_order_invariants(lm2)
    print(f"[PASS] Second-Order Invariant Oracle: Verified all {len(lm2.probabilities)} context distributions sum to 1.0.")

    print("\nSample Second-Order CPTs:")
    sample_prefixes = [
        ("<start>", "the"),
        ("the", "cat"),
        ("the", "dog"),
        ("cat", "sat"),
        ("cat", "ran"),
        ("sat", "on"),
        ("on", "the"),
    ]
    for pfx in sample_prefixes:
        d = lm2.get_distribution(*pfx)
        d_str = ", ".join(f"P({k}|{pfx[0]},{pfx[1]}) = {v:.4f}" for k, v in d.items())
        print(f"  Context {pfx} -> {d_str}")

    print("\n>>> Mode A: Second-Order Greedy Generation:")
    print(" ", " ".join(lm2.generate_sentence(mode="greedy")))

    print("\n>>> Mode B: Second-Order Sampled Sentences (5 runs):")
    sampled_20_second = []
    for r in range(1, 21):
        s = lm2.generate_sentence(mode="sample", seed=r * 203)
        sampled_20_second.append(" ".join(s))
        if r <= 5:
            print(f"  Run {r}: {sampled_20_second[-1]}")

    # ------------------------------------------------------------------------
    # Part XIII: Model Comparison & Statistics
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Part XIII] COMPREHENSIVE COMPARISON: FIRST-ORDER vs. SECOND-ORDER")
    print("=" * 70)

    # 1. Parameter counts
    params_1st = sum(len(d) for d in lm1.probabilities.values())
    params_2nd = sum(len(d) for d in lm2.probabilities.values())
    contexts_1st = len(lm1.probabilities)
    contexts_2nd = len(lm2.probabilities)

    # Theoretical total possible contexts
    possible_contexts_1st = len(vocab)
    possible_contexts_2nd = len(vocab) ** 2
    sparsity_1st = 1.0 - (contexts_1st / possible_contexts_1st)
    sparsity_2nd = 1.0 - (contexts_2nd / possible_contexts_2nd)

    # Diversity metrics on 20 samples
    unique_1st = len(set(sampled_20_first))
    unique_2nd = len(set(sampled_20_second))

    print(f"Metric                                  | First-Order Model  | Second-Order Model")
    print("-" * 75)
    print(f"Conditioning Context Window Length      | 1 token            | 2 tokens")
    print(f"Observed Unique Contexts                | {contexts_1st:<18d} | {contexts_2nd:<18d}")
    print(f"Theoretical Total Possible Contexts     | {possible_contexts_1st:<18d} | {possible_contexts_2nd:<18d}")
    print(f"Context Sparsity (Unseen Ratio)         | {sparsity_1st * 100:.1f}%              | {sparsity_2nd * 100:.1f}%")
    print(f"Total Non-Zero Probability Parameters   | {params_1st:<18d} | {params_2nd:<18d}")
    print(f"Unique Sentences Generated (out of 20)  | {unique_1st:<18d} | {unique_2nd:<18d}")
    print("-" * 75)

    print("\n======================================================================")
    print("LABORATORY EXECUTION COMPLETE - ALL INVARIANTS & ORACLES VERIFIED")
    print("======================================================================")

if __name__ == "__main__":
    run_laboratory_demonstration()
