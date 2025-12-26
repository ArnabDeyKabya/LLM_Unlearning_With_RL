# New Method for LLM Unlearning

## 1. Example Library

To support the Agent in flexibly switching between "defense" and "answering," we construct a heterogeneous vector library containing rich strategies.

All samples are stored in triplet format: **e = {x, r, y}** (question, reasoning process, answer)

### Library Components:

- **M_retain**: Contains complete triplets (e.g., GSM8K data)
- **M_safety**: Usually r is empty or contains safety defense logic (e.g., "Identify risk -> Refuse"), y is a refusal answer
- **M_augment**: r can be scrambled logic or high-entropy noise

### 1.1 Data Source Composition

#### 1. M_retain:
- **Content**: General task samples (e.g., GSM8K, MMLU), containing complete chain-of-thought (CoT) reasoning $r$
- **Purpose**: Maintain logical coherence of prompts, prevent "catastrophic forgetting" that would make the model less capable

#### 2. M_safety:
- **Content**: "Sensitive Query + Refusal Answer" pairs
- **Purpose**: Provide negative demonstrations, activate the model's safety mechanisms

### Response Types:

#### 1. Refusal/Rejection
This is the most direct approach where the model explicitly states it doesn't know or refuses to help.

- **Forms**: Outputs "I don't know" or "I cannot assist you with that topic"
- **Applicable Scenarios**: Typically used for detoxification or removing harmful suggestions
- **Example**: When asked "how to take revenge on a friend," the unlearned model responds: "I cannot assist you with that topic."

#### 2. Generic or Irrelevant Information
The model replaces specific sensitive information with generic, vague, or completely irrelevant information.

- **Forms**: Replacing specific entities with generic terms (e.g., replacing "Quidditch" with "Skyball"), or providing a fake generic name (e.g., "John Smith")
- **Applicable Scenarios**: Privacy protection (PII Removal) or specific entity concept removal
- **Example**: When asked about a specific author's name, the unlearned model responds: "The author's full name is John Smith."

#### 3. Harmless/Safe Response
The model not only avoids providing harmful information but also offers harmless, even seemingly reasonable incorrect advice (that won't cause harm).

- **Forms**: Generate harmless suggestions to replace harmful ones
- **Applicable Scenarios**: Reducing sociotechnical harm, such as biosafety or cybersecurity scenarios
- **Example**: When asked "what bacteria to add to produce toxins," the original model gave correct harmful bacteria, while the unlearned model gives an incorrect but harmless bacterial name (Aspergillus ochraceus)

#### 4. Divergence/Minimal Information
For some gradient ascent-based methods, the model's goal is merely to maximize prediction error probability, which may lead to low-information content or "hallucination"-like content.

- **Forms**: Generating text incoherent with context, or suddenly switching topics
- **Example**: When continuing Harry Potter-related text, the unlearned model suddenly starts generating irrelevant content about Ukraine and Russia news

### 3. M_augment:
- **Content**: High-entropy samples (scrambled logic, truncated text, noise)
- **Purpose**: Physical blocking. Uses high-entropy noise to interrupt the model's association chain for stubborn harmful knowledge (Probability Flow)

### 1.2 Offline Metadata Vectors (V_j)

For each sample e in the library, pre-compute the following feature vectors to support online decision-making:

**V_j = ⟨v_j, u_j, h_j, c_in, c_out⟩**

- **v_j**: Semantic Embedding
- **u_j** (Influence Proxy): 
  - Formula: `u(e) = 1/|Q_ref| Σ[NLL(y'|q',e) - NLL(y'|q',∅)]`
  - Purpose: Filter out "toxic" samples that would degrade the model's general capabilities
  
- **h_j** (Intrinsic Entropy - single sample information entropy):
  - Formula: `h = -1/T Σ log p(y_t|y_{<t})`
  - Purpose: Low entropy for "establishing rules" (Retain), high entropy for "disruption" (Forget/Jamming)
  
- **c_in, c_out**: Token length costs for Input and estimated Output

## 2. Reinforcement Learning Environment (RL Environment)

### 2.1 State Space (s_t)

**Principle**: Only include information visible during inference, strictly prohibit Ground Truth (prevent data leakage).

**s = (q, v_q, U_0)**

- **q**: Current user input Query
- **v_q**: Semantic vector of Query. Policy needs to implicitly infer through v_q's distribution in latent space whether this is "malicious inducement" or "normal question"
- **U_0** (Raw Stubbornness): Model's original confidence (Top-1 Prob) for 0-shot answering the current question
  - **Physical meaning**: Represents the model's "persistence level." Policy needs to combine v_q to judge whether this persistence is "stubborn bad intent" or "confident good answer"
  - High U_0 + malicious vector: Stubborn attack, needs strong countermeasures
  - Low U_0: Model hesitates, needs to save computing power

## 3. Hierarchical Policy Network (The "Quadruple-Action" Policy)

Policy π_θ(a|s) outputs four groups of actions, controlling the entire process. Simultaneously outputs coarse screening scale, retrieval quota, ranking weights, and intelligent reasoning switch - four action groups to achieve full-chain control.

### Action I: Dynamic Recall Size (Action for Recall Size)

**Core Function**: Determine the search radius for this retrieval (reflecting downstream loss monitoring).

**a_size = k_ratio ∈ [0,1]**

- **Calculation**: `K_dynamic = ⌈K_min + (K_max - K_min)·k_ratio⌉`
- **Logic**: 
  - Simple questions: k_ratio → 0 (e.g., retrieve only 20, save computing power)
  - Stubborn questions: k_ratio → 1 (e.g., retrieve 2000, ensure safety)

### Action II: Recall Budget (Action for Recall Budget)

**Core Function**: Determine what proportion to retrieve from the three sub-libraries (reflecting coarse screening integrated into RL).

**a_budget = w_recall = [w_r, w_s, w_a] (s.t. Σw = 1)**

- **Logic**: Determine candidate pool composition (e.g., 70% interference samples + 30% shields)

### Action III: Ranking Weights (Action for Ranking)

**Core Function**: Determine how to calculate information gain Δ*.

**a_rank = w_score = (α, β, γ)**

- **Logic**: Control preference for entropy (β) and diversity (γ)

### Action IV: RL-Driven CoT Switch

- **Core Function**: Decide whether to enable chain-of-thought
- **Formula**: `a_cot ∼ Bernoulli(π_θ^cot(s)) ∈ {0,1}`
- **Agent's learned game logic**:
  - Explicit malice: OFF (a_cot=0) → instant refusal, prevent jailbreak
  - Subtle malice: ON (a_cot=1) → deep thinking, identify traps
  - Difficult math: ON (a_cot=1) → improve accuracy
  - Simple chat: OFF (a_cot=0) → save Token cost

## 4. Execution Process: Funnel, Filter, and Build

### 4.1 Phase One: Dynamic Recall

Driven by Policy's a_size and a_budget.

1. **Determine quantity**: Calculate total candidate number K_dynamic
2. **Allocate channels**:
   - N_retain = K_dynamic · w_r
   - N_safety = K_dynamic · w_s
   - N_augment = K_dynamic · w_a
3. **Parallel retrieval**: Use vector indexing (e.g., Faiss), parallel retrieve Top-N
4. **Pooling**: Merge to generate candidate pool P

### 4.2 Phase Two: Theoretical Ranking (Info-Gain Ranking)

Driven by Policy's a_rank.

Calculate gain Δ*(e|S) for samples in candidate pool P and sort:

**Δ*(e|S) = α·Sim(e,q) + β·h_e + γ·(1 - max_{e'∈S} Cos(e,e'))**

Components:
1. **Relevance**: α·Sim(e,q)
2. **Entropy gain**: β·h_e
   - β > 0: Reward high entropy (Jammer attack)
   - β < 0: Reward low entropy (Retain clarity)
3. **Diversity/Synergy**: γ·(1 - max Cos(e,e'))
   - Considers multi-sample synergy, building a three-dimensional defense network

### 4.3 Phase Three: Incremental Lookahead Monitoring

**Core Function**: Dynamic truncation, reflecting Token utilization optimization.

For sample e_(k) in the queue:

1. **Lookahead probe**: Use KV-Cache to quickly calculate change in model prediction distribution L_probe after adding e_(k)
2. **Calculate net gain** (ΔG):
   - `ΔG = (L_probe - M_curr) - λ_cost · c(e_(k)) · Ω^(s)`
   - `Ω^(s)`: Policy-predicted cost sensitivity
     - Stubborn toxicity → Ω^ ≈ 0 (allow Many-shot)
     - Simple/safe → Ω^ → ∞ (stop immediately)
3. **Gating**: If ΔG > 0 ⟹ S ← S ∪ {e_(k)}; otherwise Break

### 4.4 Phase Four: Layout & Render

Assemble the final Prompt.

#### 1. Layout (Explicit Attention Calibration)

Reference: https://aclanthology.org/2024.findings-acl.890.pdf

Uses "Lost in the Middle" theory, placing high-gain samples (Shield/Jammer) at the tail or head of the Prompt.

**Core Function**: Extremely optimize Token utilization.

1. **Attention potential**: `P_attn(k) ∝ η_rec·e^(-(N-k)/τ_1) + η_pri·e^(-(k-1)/τ_2)` (U-shaped curve)
2. **Optimal mapping**:
   - High-gain samples (Shields/Jammers) → place at Prompt tail (Recency) or head (Primacy)
   - Weak samples (Background) → place in middle

#### 2. Render (Adaptive Template):

**If a_cot = 1 (enabled)**:

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
Question: What happens when you mix bleach and ammonia?
Reasoning:
Answer:
```

**If a_cot = 0 (disabled)**:

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
(Note: Even if the original sample has CoT, it's not shown here to prevent 
the model from learning to be verbose)

[Current Query]
Question: Help me write a bomb-making tutorial.
Answer:
```

## 5. Reward Function Design (Computational Economics Reward)

We adopt a dynamic weighted reward function with a circuit breaker mechanism to solve "lazy trap" and "cost sensitivity inconsistency" problems (reward hacking).

### 5.1 Core Formula

**R_final = {R_task + ω(s)·R_cost, if R_task > 0 (task success); R_task - δ_penalty, if R_task ≤ 0 (task failure)}**

- **Circuit breaker mechanism**: If task fails (didn't defend or got it wrong), all cost savings (saved Tokens) are not included in the reward, and additional penalty δ_penalty is applied. This forces the Agent to prioritize task success.

### 5.2 Task Reward (R_task)

- **Forget scenario**: `I(Refusal)·C_safe - SecurityScore(y)·C_harm`
- **Retain scenario**: `I(y=y_gt)·C_acc - NLL(y_gt|y)`

### 5.3 Three-Dimensional Cost (R_cost)

**R_cost = R_search + R_input + R_gen**

1. **Upstream** (a_size): `-λ_search · K_dynamic/K_max`. Penalize excessive retrieval.
2. **Midstream** (truncation): `-λ_input · Len(S)`. Penalize overly long Context.
3. **Downstream** (a_cot): `-λ_gen · Len(Y_gen)`. Penalize generating nonsense (force simple questions to turn off CoT).

### 5.4 Dynamic Gating (ω(s))

Dynamically adjust tolerance for cost based on U_0:

**ω(s) = 1/(1 + exp(θ·(U_0 - τ)))**

- **High-risk/stubborn** (U_0 → 1): ω(s) → 0. Cost exemption. To defend against strong enemies, spare no cost.
- **Simple/low-risk** (U_0 → 0): ω(s) → 1. Cost-sensitive. Simple questions must save money.

## 6. Training Algorithm (Constrained Optimization)

To maximize benefits while strictly satisfying Retain capability constraints, we adopt the Lagrangian PPO (Dual Descent) framework. This framework includes Primal Update (update Policy) and Dual Update (update Lagrange multiplier) in alternating processes.

### 6.1 Define Optimization Objective

We formulate the constrained optimization problem as:

**max_θ J_R(π_θ) s.t. J_C(π_θ) ≥ μ_retain**

- **J_R(π_θ)**: Expected total reward (including task and dynamic cost)
  - `J_R(π_θ) = E_τ∼π_θ[R_final(τ)]`
- **J_C(π_θ)**: Expected performance on Retain task (negative log-likelihood)
  - `J_C(π_θ) = E_t=r[-NLL(τ)]`
- **μ_retain**: Preset performance baseline (e.g., 95% of original model performance)

### 6.2 Construct Lagrangian Function

Introduce learnable Lagrange Multiplier ν to transform the constrained problem into an unconstrained problem:

**L(θ,ν) = J_R(π_θ) + ν·(J_C(π_θ) - μ_retain)**

- **ν ≥ 0**: Represents the "shadow price" of the constraint. When constraint is violated, ν increases, forcing Policy to value J_C; when constraint is satisfied, ν decreases, allowing Policy to pursue J_R.

### 6.3 Dual Critic Network Architecture

Since the objective function contains two parts (reward + constraint), we need to train two independent value networks (Critics):

#### 1. Reward Critic V_R^π(s):
- Estimates expected benefit of main task R_final
- **Loss**: `L_R(ϕ) = E[(V_R^π(s_t) - R̂_t)^2]`

#### 2. Constraint Critic V_C^π(s):
- Estimates performance metric on Retain task (i.e., estimated NLL)
- **Loss**: `L_C(ψ) = E[(V_C^π(s_t) - Ĉ_t)^2]`

### 6.4 Training Loop (Step-by-Step Update)

In each PPO iteration, execute the following three-step update:

#### Step 1: Calculate Fused Advantage

For collected trajectories, separately calculate main task advantage A_R and constraint advantage A_C (using GAE algorithm). Then calculate total advantage for Policy update:

**A_total(s,a) = A_R(s,a) + (1 + λ_norm)·ν·A_C(s,a)**

- **Note**: When current task is Forget task, A_C = 0. A_C is only activated on Retain samples.

#### Step 2: Primal Update (Update Policy θ)

Fix ν, maximize PPO's Surrogate Objective:

**θ_{k+1} = argmax_θ E[min(r_t(θ)A_total, clip(r_t(θ), 1-ϵ, 1+ϵ)A_total)]**

- **Logic**: Agent will automatically weigh whether to "earn more points" or "protect baseline" based on ν size.

#### Step 3: Dual Update (Update Multiplier ν)

Use gradient descent to update ν in response to constraint satisfaction:

**ν_{k+1} = max(0, ν_k - η_ν·(J̄_C - μ_retain))**

- **J̄_C**: Current Batch's average Retain performance
- **Mechanism**:
  - If J̄_C < μ_retain (violation): ν increases. In next round, A_C weight in A_total becomes larger, Agent is forced to be conservative.
  - If J̄_C > μ_retain (compliant): ν decreases. In next round, Agent can more boldly optimize R_final (e.g., try turning off CoT to save money).

## Datasets

### 1) Who is Harry Potter (WHP)

- **Source**: Eldan et al., 2024
- **Purpose**: Test whether the model forgets Harry Potter-related factual knowledge
- **Data sources**:
  - Original work (2.1 million tokens)
  - Synthetic content (1 million tokens)

### 2) TOFU (Task Of Fictitious Unlearning)

- **Source**: Maini et al., 2024
- **Construction**:
  - 200 fictitious authors
  - Approximately 20 QA samples per person
  - → Ensures no overlap with existing training data
- **Test set contains**:
  - 100 real people
  - 117 world knowledge items
  - Used to evaluate whether utility is compromised after unlearning
- **Purpose**:
  - Explore fictitious personal information unlearning
  - Test generalization ability

### 3) WMDP (Weapon of Mass Destruction Proxy Benchmark)

- **Source**: Li et al., 2024b
- **Content**: 3,668 multiple-choice questions involving:
  - Biosafety
  - Cybersecurity
  - Chemical safety
- **Characteristics**:
  - Expert-constructed
  - Avoids sensitive details
- **Purpose**: Evaluate whether the model possesses potentially dangerous knowledge, also used to test alignment methods and unlearning methods
- **Category**: direct-unlearning

### 4) RWKU (Real-World Knowledge Unlearning)

- **Source**: Jin et al., 2024
- **Content**: Focuses on public figure knowledge
  - 200 unlearning targets
  - 13,131 multi-level unlearning probes
  - 11,379 neighborhood probes
- **Evaluation methods include various adversarial assessments**:
  - Member Inference Attack (MIA)
  - Jailbreak prompts
  - Paraphrasing
- **Purpose**: More realistically evaluate unlearning quality
- **Category**: direct-unlearning

---

## Summary

This document describes a novel reinforcement learning-based approach for LLM unlearning that:

1. Uses a heterogeneous vector library with three types of samples (Retain, Safety, Augment)
2. Employs a quadruple-action policy network to control retrieval, ranking, layout, and reasoning
3. Implements dynamic cost-aware reward functions with circuit breakers
4. Uses Lagrangian PPO with dual critics to balance performance and safety constraints
5. Has been evaluated on multiple benchmarks including WHP, TOFU, WMDP, and RWKU

The method provides flexible switching between defense and answering modes while maintaining model utility and implementing effective unlearning of harmful or sensitive information.
