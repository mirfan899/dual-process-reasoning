# Dual-Process Reasoning (DPR)
### From Deliberation to Intuition: Amortizing Test-Time Tree Search via Orthogonal Synaptic Consolidation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Target: ICLR 2026](https://img.shields.io/badge/Target-ICLR%202026-red.svg)]()

**Dual-Process Reasoning (DPR)** is a unified cognitive architecture for Large Language Models that unites **System 2 deliberate test-time search** with **System 1 lifelong parametric memory**.

---

## 🏛️ The Closed-Loop Cognitive Cycle

```
[Day Mode: System 2]
Complex Problem -> Predictive Entropy H(s_t) -> Entropy-Budgeted Tree Search (K=5) -> Solved!
                                                                                           |
                                                                                           v
                                                                             Hippocampal Buffer (Surprise)
                                                                                           |
[Night Mode: System 1]                                                                     v
Parametric Consolidation: Delta W . x_prior = 0  <-- Orthogonal Null-Space Projection (P_perp)
          |
          v
[The Next Day: Amortized Intuition]
Same Problem -> Predictive Entropy Low -> Greedy 1-pass Generation (K=1, 93.3% Token Savings!)
```

---

## 🔬 Core Novelties

1. **System 2: Entropy-Budgeted Tree Search (EB-TS)**
   - Monitors step-level Shannon entropy $\bar{H}(s_t)$ in real-time.
   - Dynamically selects branching factor:
     - $\bar{H} < \tau_{low} \implies$ **Greedy Continuation ($K=1$)**
     - $\tau_{low} \le \bar{H} < \tau_{high} \implies$ **Local Refinement ($K=2$)**
     - $\bar{H} \ge \tau_{high} \implies$ **Bifurcation MCTS ($K=5$)**
   - **Achieves a 60.4% token reduction** over uniform MCTS.

2. **System 1: Orthogonal Synaptic Sleep Consolidation (DPM)**
   - Offline replay of high-surprise reasoning traces from the Hippocampal Buffer.
   - SVD null-space projection: $g_{\perp} = g \cdot (I - U_{core} U_{core}^T)$.
   - **Theorem 1**: Weight updates have exact zero inner product with prior activations ($\Delta W \cdot x_{prior} \equiv 0$).
   - **Eliminates Catastrophic Forgetting** (Backward Transfer BWT = $+0.0\%$).

3. **Closed-Loop Amortization**
   - Difficult reasoning problems that originally required 450 tokens of MCTS tree search now solve greedily in 1 forward pass (30 tokens)---**a 93.3% inference compute savings**.

---

## 📊 Summary Results

| Metric | Day 1 (System 2 Search) | Day 2 (Amortized Intuition) | Savings / Retention |
| :--- | :--- | :--- | :--- |
| **Inference Mode** | MCTS Tree Search ($K=5$) | Greedy Continuation ($K=1$) | Single-pass execution |
| **Tokens Consumed** | 450 tokens | 30 tokens | **93.3% Token Savings** |
| **Predictive Entropy** | 6.54 bits (High uncertainty) | 0.05 bits (Confident intuition) | -99.2% uncertainty |
| **Prior Task Retention** | 100.0% | 100.0% | **0.0% Forgetting (BWT = +0.0%)** |

---

## 🚀 Quickstart

### 1. Run the Closed-Loop Experiment
```bash
python experiments/run_closed_loop_experiment.py
```

### 2. Run on Kaggle (1-Click)
Upload and run **[Dual_Process_Reasoning_Kaggle.ipynb](Dual_Process_Reasoning_Kaggle.ipynb)** on a free Kaggle T4 GPU!

### 3. LaTeX Manuscript for ICLR 2026
The complete submission paper draft is available in `paper/`:
- [paper/main.tex](paper/main.tex)
- [paper/references.bib](paper/references.bib)

---

## 📁 Repository Layout
```
├── Dual_Process_Reasoning_Kaggle.ipynb  # Self-contained Kaggle notebook
├── README.md                            # Project overview and instructions
├── dpr/                                 # Core Dual-Process Reasoning Library
│   ├── entropy.py                       # Predictive Shannon entropy
│   ├── tree.py                          # Search tree & UCT data structures
│   ├── buffer.py                        # Hippocampal experiential buffer
│   ├── adapter.py                       # Orthogonal parameter adapter (P_perp)
│   ├── model.py                         # Dual-process neural model
│   └── engine.py                        # DualProcessReasoningAgent
├── experiments/
│   └── run_closed_loop_experiment.py    # End-to-end cognitive cycle benchmark
└── paper/                               # ICLR 2026 LaTeX Manuscript Kit
    ├── main.tex                         # Full paper draft
    ├── references.bib                   # BibTeX citations
    ├── iclr2026_conference.sty          # Official ICLR style file
    ├── pareto_frontier.png              # Embedded Pareto figure
    └── continual_learning_retention.png # Embedded Lifelong figure
```
