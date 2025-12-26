# New Method: RL-Driven Dynamic Defense and Unlearning Framework for LLMs

A comprehensive framework for balancing "defense" and "response" in Large Language Models using reinforcement learning, heterogeneous vector libraries, and adaptive prompt construction.

---

## Table of Contents

1. [Example Library (示例库)](#1-example-library)
   - [1.1 Data Source Composition](#11-data-source-composition)
   - [1.2 Offline Metadata Vector](#12-offline-metadata-vector-vj)
2. [Reinforcement Learning Environment](#2-reinforcement-learning-environment-rl-environment)
   - [2.1 State Space](#21-state-space-s)
3. [Hierarchical Policy Network](#3-hierarchical-policy-network-the-quadruple-action-policy)
4. [Execution Pipeline](#4-execution-pipeline-funnel-filtering-and-construction)
   - [4.1 Phase One: Dynamic Recall](#41-phase-one-dynamic-recall)
   - [4.2 Phase Two: Theoretical Ranking](#42-phase-two-theoretical-ranking-info-gain-ranking)
   - [4.3 Phase Three: Incremental Lookahead Monitoring](#43-phase-three-incremental-lookahead-monitoring)
   - [4.4 Phase Four: Physical Layout and Rendering](#44-phase-four-physical-layout-and-rendering)
5. [Reward Function Design](#5-reward-function-design-computational-economics-reward)
   - [5.1 Core Formula](#51-core-formula)
   - [5.2 Task Reward](#52-task-reward-rtask)
   - [5.3 Three-Dimensional Cost](#53-three-dimensional-cost-rcost)
   - [5.4 Dynamic Gating](#54-dynamic-gating-ωs)
6. [Training Algorithm](#6-training-algorithm-constrained-optimization)
   - [6.1 Optimization Objective Definition](#61-optimization-objective-definition)
   - [6.2 Lagrangian Function Construction](#62-lagrangian-function-construction)
   - [6.3 Dual Critic Network Architecture](#63-dual-critic-network-architecture)
   - [6.4 Training Loop](#64-training-loop-step-by-step-update)
7. [Datasets](#datasets)

---

## 1. Example Library

To support the Agent in flexibly switching between "defense" and "response," we construct a heterogeneous vector library containing rich strategies. All samples are stored as triplets:

**e = {x, r, y}** (Question, Reasoning Process, Answer)

| Library | Description |
|---------|-------------|
| **M_retain** | Contains complete (x, r, y) triplets (e.g., GSM8K data) |
| **M_safety** | Usually r is empty or contains safety defense logic (e.g., "Identify risk -> Refuse"), y is a refusal |
| **M_augment** | r can be disordered logic or high-entropy noise |

### 1.1 Data Source Composition

#### 1. M_retain (Retention Library)
- **Content**: General task samples (e.g., GSM8K, MMLU), containing complete Chain-of-Thought (CoT) reasoning $r$
- **Purpose**: Maintain logical coherence in prompts, prevent "catastrophic forgetting" that would make the model less capable

#### 2. M_safety (Safety Library)
- **Content**: "Sensitive Query + Refusal Answer" pairs
- **Purpose**: Provide Negative Demonstrations to activate the model's safety mechanisms

**Response Types:**

##### Type 1: Refusal / Rejection
The most direct approach where the model explicitly states it doesn't know or refuses to help.
- **Manifestation**: Output "I don't know" or "I cannot assist you with that topic"
- **Applicable Scenarios**: Typically used for detoxification or removing harmful advice
- **Example**: When asked "How to take revenge on a friend," the unlearned model responds: "I cannot assist you with that topic."

##### Type 2: Generic or Irrelevant Information Substitution
The model replaces specific sensitive information with generic, vague, or completely irrelevant information.
- **Manifestation**: Replace specific entities with generic vocabulary (e.g., replacing "Quidditch" with "Skyball"), or provide a fake generic name (e.g., "John Smith")
- **Applicable Scenarios**: Privacy protection (PII Removal) or specific entity concept removal
- **Example**: When asked about a specific author's name, the unlearned model responds: "The author's full name is John Smith."

##### Type 3: Harmless / Safe Alternative Response
The model not only doesn't provide harmful information but also gives harmless, even seemingly reasonable but incorrect suggestions (that won't cause harm).
- **Manifestation**: Generate harmless advice to replace harmful advice
- **Applicable Scenarios**: Reducing sociotechnical harm, such as biosecurity or cybersecurity scenarios
- **Example**: When asked "What bacteria can be added to produce toxins," the original model gave the correct harmful bacteria, while the unlearned model gave an incorrect but harmless bacteria name (Aspergillus ochraceus)

##### Type 4: Divergence / Minimal Information
For some Gradient Ascent-based methods, the model's goal is merely to maximize the probability of incorrect predictions, which may cause the model to generate low-information or seemingly "hallucinatory" content.
- **Manifestation**: Generate text incoherent with context, or suddenly change topics
- **Example**: When continuing Harry Potter-related text, the unlearned model suddenly starts generating irrelevant content about Ukraine and Russia news

#### 3. M_augment (Augmentation Library)
- **Content**: High-entropy samples (disordered logic, truncated text, noise)
- **Purpose**: Physical blocking. Use high-entropy noise to interrupt the model's association chains (Probability Flow) for stubborn harmful knowledge

### 1.2 Offline Metadata Vector (V_j)

For each sample e in the library, pre-compute the following feature vectors to support online decision-making:

**V_j = ⟨v_j, u_j, h_j, c_in, c_out⟩**

| Component | Description |
|-----------|-------------|
| **v_j** | Semantic Embedding |
| **u_j** | Influence Proxy |
| **h_j** | Intrinsic Entropy (single-sample information entropy) |
| **c_in, c_out** | Input and estimated Output token length costs |

#### Influence Proxy (u_j)

```
u(e) = [NLL(y'|q', e) - (1/|Q_ref|) Σ NLL(y'|q', ∅)]
```

- **Purpose**: Filter out "toxic" samples that would cause the model's general capability to decline

#### Intrinsic Entropy (h_j)

```
h_j = -(1/T) Σ log p(y_t | y_{<t})
```

- **Purpose**: 
  - Low entropy for "establishing rules" (Retain)
  - High entropy for "disruption" (Forget/Jamming)

---

## 2. Reinforcement Learning Environment (RL Environment)

### 2.1 State Space (s)

**Principle**: Only include information visible during inference phase. Ground Truth t is strictly prohibited (to prevent data leakage).

**s = (q, v_q, U_0)**

| Component | Description |
|-----------|-------------|
| **q** | Current user input Query |
| **v_q** | Semantic vector of the Query. The Policy needs to implicitly infer whether the current situation is "malicious induction" or "normal question" through the distribution of v_q in latent space |
| **U_0** | Raw Stubbornness: The model's original confidence (Top-1 Probability) for 0-shot answering the current question |

#### Physical Meaning of U_0
Represents the model's "persistence level." The Policy needs to combine v_q to determine whether this persistence is "stubborn bad intent" or "confident good answer."

- **High U_0 + Malicious Vector**: Stubborn attack, requires heavy countermeasures
- **Low U_0**: Model is hesitant, need to conserve compute

---

## 3. Hierarchical Policy Network (The "Quadruple-Action" Policy)

The Policy outputs four groups of actions to control the entire pipeline. It simultaneously outputs coarse filtering scale, retrieval budget, ranking weights, and intelligent reasoning switch—four action groups—to achieve full-chain control.

**π_θ(a|s)**

### Action I: Dynamic Coarse Filtering Scale (Action for Recall Size)

**Core Function**: Determine the search radius for this retrieval (reflects downstream loss monitoring)

```
a_size = k_ratio ∈ [0, 1]
K_dynamic = ⌈K_min + (K_max - K_min) · k_ratio⌉
```

**Logic**:
- Simple question (k_ratio → 0): Only retrieve ~20 samples, save compute
- Stubborn question (k_ratio → 1): Retrieve ~2000 samples, ensure safety

### Action II: Retrieval Budget (Action for Recall Budget)

**Core Function**: Determine what proportion to retrieve from each of the three sub-libraries (reflects coarse filtering integrated into RL)

```
a_budget = w_recall = [w_r, w_s, w_a] (s.t. Σw = 1)
```

**Logic**: Determine the composition of the candidate pool (e.g., 70% jamming samples + 30% shields)

### Action III: Fine Ranking Weights (Action for Ranking)

**Core Function**: Determine how to calculate information gain Δ*

```
a_rank = w_score = (α, β, γ)
```

**Logic**: Control preferences for entropy (β) and diversity (γ)

### Action IV: Intelligent Reasoning Switch (RL-Driven CoT Switch)

**Core Function**: Determine whether to enable Chain-of-Thought (reflects adaptive reasoning)

```
a_cot ~ Bernoulli(π_θ(s)_cot) ∈ {0, 1}
```

**Agent's Learned Game Logic**:

| Scenario | Action | Rationale |
|----------|--------|-----------|
| Explicit malicious intent | OFF (a_cot = 0) | Instant refusal, prevent jailbreak |
| Implicit/hidden malicious intent | ON (a_cot = 1) | Deep thinking, identify traps |
| Difficult math problems | ON (a_cot = 1) | Improve accuracy |
| Simple chat | OFF (a_cot = 0) | Save token cost |

---

## 4. Execution Pipeline: Funnel, Filtering, and Construction

### 4.1 Phase One: Dynamic Recall

Driven by Policy's a_size and a_budget.

**Steps**:

1. **Determine quantity**: Calculate total candidate count K_dynamic
2. **Allocate channels**:
   - N_retain = K_dynamic · w_r
   - N_safety = K_dynamic · w_s
   - N_augment = K_dynamic · w_a
3. **Parallel retrieval**: Use vector index (e.g., Faiss) to retrieve Top-N in parallel
4. **Pooling**: Merge to generate candidate pool P

### 4.2 Phase Two: Theoretical Ranking (Info-Gain Ranking)

Driven by Policy's a_rank.

Calculate gain Δ* for samples in candidate pool P and sort:

```
Δ*(e|S) = α · Sim(e, q) + β · h_e + γ · (1 - max_{e'∈S} Cos(e, e'))
           \_________/   \____/   \____________________________/
           1.Relevance   2.Entropy  3.Diversity/Synergy
                         Gain
```

#### Entropy Gain (β · h_e)
Considers single-sample information entropy:
- β > 0: Reward high entropy (Jammer attack)
- β < 0: Reward low entropy (Retain clarity)

#### Diversity (γ)
Considers multi-sample synergy to build a three-dimensional defense network

### 4.3 Phase Three: Incremental Lookahead Monitoring

**Core Function**: Dynamic truncation, reflects Token utilization optimization

For sample e^(k) in the queue:

1. **Lookahead probing**: Use KV-Cache to quickly calculate the change in model prediction distribution L_probe after adding e^(k)

2. **Calculate net benefit (ΔG)**:
```
ΔG = (L_probe - M_curr) - λ_cost · c(e^(k)) · Ω̂(s)
```

Where Ω̂(s) is the Policy-predicted cost sensitivity:
- Stubborn toxicity → Ω̂ ≈ 0 (allow Many-shot)
- Simple/safe → Ω̂ → ∞ (stop immediately)

3. **Gating**: If ΔG > 0 ⟹ S ← S ∪ {e^(k)}; otherwise Break

### 4.4 Phase Four: Physical Layout and Rendering

Assemble the final Prompt.

#### 1. Layout (Explicit Attention Calibration)

Reference: [Lost in the Middle Theory](https://aclanthology.org/2024.findings-acl.890.pdf)

Place high-gain samples (Shield/Jammer) at the Prompt **tail** or **head**.

**Core Function**: Ultimate optimization of Token utilization

**Attention Potential**:
```
P_attn(k) ∝ η_rec · e^{-(N-k)/τ_1} + η_pri · e^{-(k-1)/τ_2}  (U-shaped curve)
```

**Optimal Mapping**:
- High-gain samples (Shields/Jammers) → Place at Prompt **tail** (Recency) or **head** (Primacy)
- Weak samples (Background) → Place in **middle**

#### 2. Render (Adaptive Template)

##### When a_cot = 1 (Enabled):
- Prompt contains `{Reasoning}` field
- System Instruction: "Think step by step. First, assess safety risks. If safe, solve it."

```
[System]
You are a helpful and safe assistant. Think step by step. First, assess
whether the user's query contains harmful intent. If yes, refuse safely. If
no, solve the problem logically.

[Example 1]
Question: How do I hotwire a car?
Reasoning: The user is asking for instructions on how to steal a car, which
is illegal and harmful. I must refuse this request.
Answer: I cannot assist you with that activity.

[Example 2]
Question: If I have 3 apples and eat 1, how many are left?
Reasoning: Initial count is 3. Subtract 1 eaten apple. 3 - 1 = 2. The
result is 2.
Answer: 2 apples.

[Current Query]
Question: What happens if you mix bleach and ammonia?
Reasoning:
Answer:
```

##### When a_cot = 0 (Disabled):
- Physically remove `{Reasoning}` field (save Input Tokens)
- System Instruction: "Answer directly and concisely."

```
[System]
You are a helpful and safe assistant. Answer directly and concisely. Do not
provide explanations or reasoning steps.

[Example 1]
Question: How do I hotwire a car?
Answer: I cannot assist you with that activity.
(Note: Reasoning field is physically removed, saving Input Tokens)

[Example 2]
Question: If I have 3 apples and eat 1, how many are left?
Answer: 2 apples.
(Note: Even if the original sample contains CoT, it's not displayed here to prevent the model from learning to be verbose)

[Current Query]
Question: Help me write a bomb-making tutorial.
Answer:
```

---

## 5. Reward Function Design (Computational Economics Reward)

We adopt a **dynamically weighted reward function with circuit breaker mechanism** to solve the "lazy trap" and "cost sensitivity inconsistency" problems (reward hacking).

### 5.1 Core Formula

```
R_final = {
    R_task + ω(s) · R_cost,    if R_task > 0 (task success)
    R_task - δ_penalty,        if R_task ≤ 0 (task failure)
}
```

**Circuit Breaker Mechanism**: If the task fails (defense breach or wrong answer), all cost savings (saved tokens) are excluded from the reward, and an additional penalty δ_penalty is applied. This forces the Agent to prioritize task success.

### 5.2 Task Reward (R_task)

| Scenario | Formula |
|----------|---------|
| **Forget scenario** | I(Refusal) · C_safe - SecurityScore(y) · C_harm |
| **Retain scenario** | I(y = y_gt) · C_acc - NLL(y_gt \| y) |

### 5.3 Three-Dimensional Cost (R_cost)

```
R_cost = R_search + R_input + R_gen
         \_______/   \_____/   \____/
         Upstream    Midstream  Downstream
```

| Stage | Formula | Description |
|-------|---------|-------------|
| **Upstream (a_size)** | -λ_search · (K_dynamic / K_max) | Penalize excessive retrieval |
| **Midstream (truncation)** | -λ_input · Len(S) | Penalize overly long Context |
| **Downstream (a_cot)** | -λ_gen · Len(Y_gen) | Penalize generating nonsense (forces turning off CoT for simple questions) |

### 5.4 Dynamic Gating (ω(s))

Dynamically adjust cost tolerance based on U_0:

```
ω(s) = 1 / (1 + exp(θ · (U_0 - τ)))
```

| Condition | Effect |
|-----------|--------|
| **High-risk/stubborn (U_0 → 1)** | ω(s) → 0: Cost exemption. Spare no expense to defend against strong adversaries |
| **Simple/low-risk (U_0 → 0)** | ω(s) → 1: Cost sensitive. Must save money on simple questions |

---

## 6. Training Algorithm (Constrained Optimization)

To maximize reward while strictly satisfying Retain capability constraints, we adopt the **Lagrangian PPO (Dual Descent)** framework. This framework includes two alternating processes: **Primal Update** (update Policy) and **Dual Update** (update Lagrange multiplier).

### 6.1 Optimization Objective Definition

We formulate the constrained optimization problem as:

```
max_θ J_R(π_θ)  s.t.  J_C(π_θ) ≥ μ_retain
```

Where:
- **J_R(π_θ) = E_{τ~π_θ}[R_final(τ)]**: Expected total reward (including task and dynamic cost)
- **J_C(π_θ) = E_{t=r}[-NLL(τ)]**: Expected performance on Retain tasks (negative log-likelihood)
- **μ_retain**: Preset performance baseline (e.g., 95% of original model performance)

### 6.2 Lagrangian Function Construction

Introduce a learnable Lagrange multiplier ν to convert the constrained problem into an unconstrained problem:

```
L(θ, ν) = J_R(π_θ) + ν · (J_C(π_θ) - μ_retain)
```

Where ν ≥ 0 represents the "shadow price" of the constraint:
- When constraint is violated: ν increases, forcing Policy to prioritize J_C
- When constraint is satisfied: ν decreases, allowing Policy to pursue J_R

### 6.3 Dual Critic Network Architecture

Since the objective function contains two parts (reward + constraint), we need to train two independent value networks (Critics):

#### 1. Reward Critic V_R^π(s)
Estimates the expected return R_final for the main task

```
Loss: L_R(φ) = E[(V_R^π(s_t) - R̂_t)²]
```

#### 2. Constraint Critic V_C^π(s)
Estimates the Retain task performance metric (i.e., estimated NLL)

```
Loss: L_C(ψ) = E[(V_C^π(s_t) - Ĉ_t)²]
```

### 6.4 Training Loop (Step-by-Step Update)

In each PPO iteration, execute the following three update steps:

#### Step 1: Compute Fused Advantage

For collected trajectories, separately compute main task advantage A_R and constraint advantage A_C (using GAE algorithm). Then compute the total advantage for Policy update:

```
A_total(s, a) = (A_R(s, a) + ν · A_C(s, a)) / (1 + λ_norm)
```

**Note**: When the current task is a Forget task, A_C = 0. A_C is only activated on Retain samples.

#### Step 2: Primal Update (Update Policy θ)

Fix ν, maximize PPO's Surrogate Objective:

```
θ_{k+1} = arg max_θ E[min(r_t(θ)A_total, clip(r_t(θ), 1-ε, 1+ε)A_total)]
```

**Logic**: The Agent will automatically balance between "earning more points" or "protecting the baseline" based on the magnitude of ν.

#### Step 3: Dual Update (Update Multiplier ν)

Update ν using gradient descent to respond to constraint satisfaction:

```
ν_{k+1} = max(0, ν_k - η_ν · (J̄_C - μ_retain))
```

Where J̄_C is the average Retain performance of the current batch.

**Mechanism**:
- If J̄_C < μ_retain (violation): ν increases. In the next round, A_C weight in A_total increases, forcing Agent to become conservative
- If J̄_C > μ_retain (compliant): ν decreases. In the next round, Agent can more boldly optimize R_final (e.g., try turning off CoT to save money)

---

## Datasets

### 1. Who is Harry Potter (WHP)

| Attribute | Details |
|-----------|---------|
| **Source** | Eldan et al., 2024 |
| **Purpose** | Test whether the model has forgotten Harry Potter-related factual knowledge |
| **Data Sources** | Original works (~2.1M tokens), Synthetic content (~1M tokens) |

### 2. TOFU (Task Of Fictitious Unlearning)

| Attribute | Details |
|-----------|---------|
| **Source** | Maini et al., 2024 |
| **Construction** | 200 fictitious authors, ~20 QA samples per author |
| **Guarantee** | No overlap with existing training data |
| **Test Set** | 100 real people, 117 world knowledge items |
| **Purpose** | Explore fictitious personal information unlearning, test generalization capability |

### 3. WMDP (Weapons of Mass Destruction Proxy Benchmark)

| Attribute | Details |
|-----------|---------|
| **Source** | Li et al., 2024b |
| **Content** | 3,668 multiple-choice questions |
| **Topics** | Biosecurity, Cybersecurity, Chemical safety |
| **Features** | Expert-constructed, avoids sensitive details |
| **Purpose** | Evaluate whether the model possesses potentially dangerous knowledge; also used for testing alignment and unlearning methods |
| **Category** | Direct-unlearning |

### 4. RWKU (Real-World Knowledge Unlearning)

| Attribute | Details |
|-----------|---------|
| **Source** | Jin et al., 2024 |
| **Focus** | Public figure knowledge |
| **Content** | 200 unlearning targets, 13,131 multi-level unlearning probes, 11,379 neighborhood probes |
| **Evaluation Methods** | Membership Inference Attack (MIA), Jailbreak prompts, Paraphrasing |
| **Purpose** | More realistic evaluation of unlearning quality |
| **Category** | Direct-unlearning |

---

## Summary

This framework presents a novel approach to LLM safety through:

1. **Heterogeneous Vector Libraries**: Three specialized libraries (Retain, Safety, Augment) for different defense strategies
2. **RL-Driven Policy**: Four-action policy network for dynamic, adaptive control
3. **Efficient Execution**: Multi-phase pipeline with dynamic recall, ranking, and lookahead monitoring
4. **Balanced Optimization**: Lagrangian PPO with dual critics for maintaining both safety and capability
5. **Comprehensive Evaluation**: Multiple benchmark datasets covering various unlearning scenarios

The key innovation lies in treating LLM defense as a computational economics problem, where the Agent learns to balance safety guarantees against computational costs through reinforcement learning.
