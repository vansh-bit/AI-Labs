# Laboratory Report: Neural Models – Learning, Depth, Activations, and Output Layers

**Course:** Artificial Intelligence (CS F407)  
**Laboratory Exercise:** Neural Models: Learning, Depth, Activations, and Output Layers  
**Reference Document:** [`neur_models_lab_ex.pdf`](file:///Users/vanshsharma/Documents/AI%20Labs/Lab_3_Neural_Models/neur_models_lab_ex.pdf)

---

## 1. Task 1: Problem Formulation and Linear Separability

### 1.1 Mathematical Formulation
- **Input Space $\mathcal{X}$:** $\mathcal{X} = \{0, 1\}^2 \subset \mathbb{R}^2$ representing the states of two binary sensors $(x_1, x_2)$.
- **Output Space $\mathcal{Y}$:** $\mathcal{Y} = \{0, 1\}$ where $y = 1$ indicates a disagreement alarm and $y = 0$ indicates agreement.
- **Labelled Examples (XOR Truth Table):**
  $$\mathcal{D} = \{ ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0) \}$$

### 1.2 Geometry of the Input Space and Non-Separability
In the $(x_1, x_2)$ Cartesian plane:
- Class $1$ points: $(0, 1)$ and $(1, 0)$
- Class $0$ points: $(0, 0)$ and $(1, 1)$

**Proof of Linear Non-Separability:**  
A linear decision boundary in $\mathbb{R}^2$ is defined by a hyperplane $w_1 x_1 + w_2 x_2 + b = 0$. For this hyperplane to separate Class 1 from Class 0, there must exist weights $w_1, w_2$ and bias $b$ such that:
1. $w_1(0) + w_2(0) + b < 0 \implies b < 0$
2. $w_1(0) + w_2(1) + b > 0 \implies w_2 + b > 0$
3. $w_1(1) + w_2(0) + b > 0 \implies w_1 + b > 0$
4. $w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0$

Summing inequalities (2) and (3) yields:
$$(w_1 + b) + (w_2 + b) > 0 \implies w_1 + w_2 + 2b > 0$$
However, adding $b < 0$ (from (1)) to $w_1 + w_2 + b < 0$ (from (4)) gives:
$$w_1 + w_2 + 2b < 0$$
This is a direct mathematical contradiction ($> 0$ and $< 0$ simultaneously). Therefore, **no single straight linear decision boundary can separate the two classes.**

### 1.3 Prediction for Single Affine Transformation + Sigmoid
If a model consists only of a single affine transformation followed by a sigmoid $\hat{y} = \sigma(w^T x + b)$, its decision boundary $\sigma(w^T x + b) = 0.5$ is strictly linear ($w_1 x_1 + w_2 x_2 + b = 0$). As proven above, no line can separate the XOR points. Consequently, the linear model will fail, achieving at most $75\%$ accuracy (3 out of 4 correct) by predicting a constant output or a single diagonal cut.

---

## 2. Task 2: Model Design and Validation Criteria

### 2.1 Baseline Architecture: 2–2–1 Feedforward Network
- **Input Dimension:** $D_{in} = 2$ ($x_1, x_2$)
- **Hidden Layer:** $D_h = 2$ neurons with non-linear activation $h = f^{(1)}(W^{(1)}x + b^{(1)})$
- **Output Layer:** $D_{out} = 1$ logit $z = W^{(2)}h + b^{(2)}$ mapped via sigmoid $\hat{p} = \sigma(z)$
- **Loss Function:** Binary Cross-Entropy with Logits:
  $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \sigma(z_i) + (1 - y_i)\log(1 - \sigma(z_i)) \right]$$
- **Optimization:** Stochastic Gradient Descent (SGD) / Adam with backpropagation.

### 2.2 Conceptual Questions
1. **Why is the hidden nonlinearity scientifically necessary here?**  
   Without a nonlinear activation function, a 2-layer network computes:
   $$\hat{y} = W^{(2)}(W^{(1)}x + b^{(1)}) + b^{(2)} = (W^{(2)}W^{(1)})x + (W^{(2)}b^{(1)} + b^{(2)}) = \tilde{W}x + \tilde{b}$$
   A composition of affine transformations collapses into a single affine map. Adding depth without nonlinearity cannot expand the hypothesis space beyond linear classifiers.
2. **Why is sigmoid + binary cross-entropy a sensible engineering pairing?**  
   Sigmoid maps unbounded logits $\mathbb{R} \to (0, 1)$ representing the Bernoulli probability parameter. Paired with binary cross-entropy, the log-likelihood derivative with respect to the pre-activation logit simplifies to:
   $$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y = p - y$$
   This linear error residual avoids gradient saturation when predictions are wrong, providing a clean, non-vanishing error signal.
3. **Validation Criteria (3 Checks for Successful Learning):**
   - **Check 1 (Final Loss):** Cross-entropy loss converges to near zero ($\mathcal{L} < 0.05$).
   - **Check 2 (Classification Accuracy):** All 4 points correctly classified ($\hat{y} = y$ for all 4 inputs with threshold $0.5$).
   - **Check 3 (Gradient Non-Zero Norm):** Non-zero, healthy gradient magnitudes throughout early training without vanishing or exploding.

---

## 3. Task 3: Prompt Engineering and LLM Code Review

### 3.1 Prompt Used
```text
Generate minimal PyTorch code for a 2-2-1 neural network to solve the XOR problem.
Dataset: X = [[0,0], [0,1], [1,0], [1,1]], Y = [0, 1, 1, 0].
Architecture:
- 2 inputs, 2 hidden units with Sigmoid activation, 1 output logit.
- Use nn.BCEWithLogitsLoss.
- Use SGD or Adam optimizer with full-batch training.
- Print initial loss, final loss after training, all four predicted probabilities, thresholded predictions, and the first-layer weight gradient tensor after backward().
Set torch.manual_seed(42) for reproducibility.
```

### 3.2 Code Anatomy & Execution Checkpoint
1. **Forward Pass:** $h = \sigma(x W_1^T + b_1)$, $z = h W_2^T + b_2$.
2. **Scalar Loss:** `loss = criterion(logits, targets)`.
3. **Reverse-Mode AD:** `loss.backward()` accumulates partial derivatives in `.grad`.
4. **Parameter Updates:** `optimizer.step()` applies $W \leftarrow W - \eta \nabla_W \mathcal{L}$.

---

## 4. Task 4: Experimental Results and Diagnostics

### 4.1 Part A: Basic Learning Check
- **Initial Loss:** $0.720345$
- **Final Loss (after 5000 steps):** $0.002067$
- **Predictions Summary:**
  | Input $(x_1, x_2)$ | True $y$ | Output Logit $z$ | Probability $\hat{p} = \sigma(z)$ | Predicted Label | Status |
  |:---:|:---:|:---:|:---:|:---:|:---:|
  | $(0, 0)$ | 0 | -6.220 | 0.0020 | 0 | **CORRECT** |
  | $(0, 1)$ | 1 | +5.903 | 0.9973 | 1 | **CORRECT** |
  | $(1, 0)$ | 1 | +6.301 | 0.9982 | 1 | **CORRECT** |
  | $(1, 1)$ | 0 | -6.366 | 0.0017 | 0 | **CORRECT** |
- **Result:** $4/4$ examples correctly classified ($100\%$ accuracy).

### 4.2 Part B: Backpropagation and Gradient Verification
- **First-Layer Weight Gradient Matrix $\nabla_{W^{(1)}} \mathcal{L}$:**
  $$\begin{bmatrix} +0.005890 & +0.007821 \\ +0.002217 & +0.000973 \end{bmatrix}$$
- **First-Layer Bias Gradient Vector $\nabla_{b^{(1)}} \mathcal{L}$:**
  $$\begin{bmatrix} +0.020027 & +0.010585 \end{bmatrix}$$
- **Meaning of `parameter.grad`:**  
  `parameter.grad` stores $\frac{\partial \mathcal{L}}{\partial \theta}$. For the mean batch loss $\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \mathcal{L}_i$, the derivative is linear, so:
  $$\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^N \nabla_\theta \mathcal{L}_i$$
  The gradient accumulated in `parameter.grad` is exactly the arithmetic average of the individual sample gradients.

### 4.3 Part C: Symmetry Experiment (Zero Weight Initialization)
- When all weights and biases are initialized to zero:
  - Both hidden units compute $a_1^{(1)} = a_2^{(1)} = 0 \implies h_1 = h_2 = f(0)$.
  - The logit is $z = 0 \implies \hat{p} = 0.5$.
  - Both hidden units receive identical backpropagated errors: $\frac{\partial \mathcal{L}}{\partial W_{1j}^{(1)}} = \frac{\partial \mathcal{L}}{\partial W_{2j}^{(1)}}$.
- **Experimental Observation:** Over all training steps ($1, 2, 5, 10, \dots$), Row 0 and Row 1 of $W^{(1)}$ remain strictly identical.
- **Conclusion:** Zero initialization causes hidden units to learn identical features, permanently destroying the network's capacity to learn distinct decision boundaries.

### 4.4 Part D: Activation Experiment Comparison
| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}} \mathcal{L}\|_2$ | Behaviour & Observations |
|:---|:---:|:---:|:---:|:---|
| **Sigmoid** | **0.002067** | **True** | **0.010086** | Smooth, steady convergence; small derivative prevents sudden oscillations. |
| **Tanh** | 0.349915 | False | 0.077563 | Zero-centered with higher early gradient; sensitive to learning rate on 4 points. |
| **ReLU** | 0.477602 | False | 0.074098 | Suffers from "dying ReLU" when pre-activations fall below zero on tiny datasets. |

---

## 5. Task 5: Three-Class Decision Extension

### 5.1 Formulation
- Class 0: Both sensors inactive $(0, 0)$
- Class 1: Disagreement $(0, 1)$ or $(1, 0)$
- Class 2: Both sensors active $(1, 1)$

### 5.2 Mathematical Predictions
1. **Shape of Final Weight Matrix:** For $D_h$ hidden units and $K = 3$ output classes, $W^{(2)} \in \mathbb{R}^{3 \times D_h}$ (shape $[3, D_h]$).
2. **Number of Logits per Example:** Exactly 3 logits $(z_0, z_1, z_2)$ per input vector.
3. **Why Softmax Sums to 1:**  
   $$p_k = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}} \implies \sum_{k=0}^{K-1} p_k = \frac{\sum_{k} e^{z_k}}{\sum_{j} e^{z_j}} = 1$$
4. **Logit Gradient Derivation ($p - y$):**  
   For cross-entropy loss $\mathcal{L} = -\sum_k y_k \log p_k$:
   $$\frac{\partial p_k}{\partial z_i} = p_i(\delta_{ik} - p_k) \implies \frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$
5. **Numerical Stability (Subtracting Max Logit):**  
   Because $\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c}e^{z_k}}{e^{-c}\sum_j e^{z_j}} = p_k$, setting $c = \max_j z_j$ guarantees that the largest exponent is $e^0 = 1$, preventing floating-point overflow (`inf`).

---

## 6. Answers to Reflection Questions

1. **What did the XOR experiment demonstrate about the difference between depth and nonlinearity?**  
   Depth without nonlinearity is mathematically redundant—a sequence of linear transformations collapses to a single linear transformation. Nonlinearity is the essential property that bends the feature space to make linearly inseparable distributions separable.
2. **In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?**  
   A random search or noise injection also produces non-zero gradients. The definitive evidence was monotonic loss reduction from $0.72$ down to $0.002$ accompanied by thresholded predictions shifting from uncalibrated outputs to $100\%$ accuracy on all 4 truth table conditions.
3. **Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?**  
   Because of permutation symmetry. Identical weights compute identical forward activations and receive identical backward gradients, leading to identical updates. The two neurons remain clones.
4. **How did changing the hidden activation affect the gradient you observed?**  
   - *Scientific:* Sigmoid has max derivative $0.25$, squashing gradients. ReLU has derivative $1.0$ for $x > 0$ and $0$ for $x < 0$.
   - *Engineering:* Tanh and ReLU produced substantially higher early gradient norms ($0.077$ vs $0.010$ for Sigmoid).
5. **Why must the output layer and loss be selected together according to the task?**  
   The output layer specifies the parameterisation of the probability distribution (e.g. Bernoulli via Sigmoid, Categorical via Softmax), and the loss must be the corresponding negative log-likelihood to ensure consistent, non-saturating gradients ($p - y$).
6. **Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**  
   - *Productivity:* Quickly generating the PyTorch model structure, training loops, and loss function boilerplate.
   - *Verification:* Ensuring that logits (not post-sigmoid probabilities) were passed to `BCEWithLogitsLoss` to prevent numerical instability.
7. **Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**  
   - *Keep:* Loss logging, batch accuracy evaluation, and gradient norm tracking.
   - *Too Expensive:* Exhaustive inspection of individual weight gradient matrices and finite-difference gradient checking.
