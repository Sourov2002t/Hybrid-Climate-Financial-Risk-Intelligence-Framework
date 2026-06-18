# HCFRI — Hybrid Climate-Financial Risk Intelligence Framework

> **HCFRI v2.0** is a research-grade, 5-layer deep learning pipeline that
> forecasts, maps, and explains how climate hazards propagate into financial
> system risk. It beats 25 state-of-the-art baselines (RMSE 0.0076 vs.
> iTransformer's 0.0127) and produces NGFS/TCFD-aligned regulatory reports.

---

## 🧠 Model Architecture

### Layer 2 — Hierarchical Temporal Fusion Network (HTFN)
Three temporal modules fused via **Bayesian precision weights**:

| Module | Architecture | Horizon |
|--------|-------------|---------|
| Short-term | **Bidirectional GRU** encoder-decoder + temporal attention | 1–30 days |
| Medium-term | **Transformer** (cross-attention: climate × financial) + Positional Encoding | 1–12 months |
| Long-term | **Physics-Informed Neural Network (PINN)** with learnable climate damage parameters (α, β, κ) | 5–30 years |

A **Regime Detector** (MLP classifier) dynamically gates which temporal
module dominates based on current market state.

### Layer 3 — Climate-Stressed Hybrid Core
| Component | Architecture | Role |
|-----------|-------------|------|
| **CNNBiLSTMExtractor** | Multi-scale 1D CNN (3d / 7d / 14d kernels) → BiLSTM → temporal attention | Spatial-temporal climate feature extraction |
| **ClimateStressedGAT** | Graph Attention Network with dynamic climate-modulated edge weights | Sector-to-sector contagion propagation |
| **RiskAggregator** | Transformer aggregator + output MLP | Fuses temporal + graph signals into portfolio risk score |

The GAT's adjacency matrix is **dynamically restructured at inference time**
based on a real-time climate stress vector — not a fixed graph.

---

## 🔬 Explainability Stack (Layer 4)

| Level | Method | Output |
|-------|--------|--------|
| L1 | **Permutation Feature Importance** (90% CI error bars) | Global SHAP-style bar chart |
| L2 | **Integrated Gradients** (instance-level, 50 interpolation steps) | Per-sample attribution heatmap |
| L3 | **Attention Weight Hooks** (all `MultiheadAttention` layers) | Temporal attention heatmaps |
| L4 | **Granger Causality F-test** + lagged cross-correlation + FDR-BH correction | Climate→Finance lead-lag map |

---

## ✅ Validation & Regulatory Alignment (Layer 5)

- **NGFS Stress Tests**: 5 scenarios — Orderly Net Zero 2050, Disorderly
  Delayed Transition, Hot House World (+4°C), Current Policies, Below 2°C
- **TCFD-aligned policy report** with sector exposure analysis
- **25-baseline SOTA comparison** (FEDformer, iTransformer, ASTGNN, ...)
- **OOD generalization test** on structurally distinct 2023–24 regime

---

## 🛠️ Technology Stack

**Core Framework**
- `Python 3.14` · `PyTorch` (nn, DataLoader, autograd)

**Optimisation**
- `AdamW` (weight_decay=1e-4) · `CosineAnnealingLR` · Gradient Clipping · Early Stopping

**Neural Architectures**
- GRU · Bidirectional LSTM · 1D CNN · Transformer · MultiheadAttention
- Graph Attention Network (GAT) · Physics-Informed NN (PINN)

**XAI / Statistics**
- `statsmodels` — Granger causality tests, FDR-BH multiple testing correction
- `scipy` — Spearman correlation, Wilcoxon signed-rank test, ECE calibration
- `scikit-learn` — StandardScaler, regime classification

**Data**
- `yfinance` — live ETF market data (7 sectors)
- `pandas` / `numpy` — feature engineering & time-series processing
- EM-DAT Global Disaster Database (1900–2021)
- CMIP6 downscaled regional climate surrogates
- Daily Delhi Climate Dataset

**Visualisation & Reporting**
- `matplotlib` · `seaborn` — dark-theme 8-panel dashboards
- LaTeX (`sota_table.tex`) · Markdown manuscript · JSON experiment records

---

## 📊 Results

| Metric | Value | Best Baseline |
|--------|-------|--------------|
| RMSE | **0.0076** | iTransformer: 0.0127 |
| OOD R² (2023–24) | **0.883** | — |
| Systemic Risk Capture | **0.9986** | — |
| Sharpe Ratio | **2.27** | — |
| Information Coefficient | **0.967** | — |
| Calmar Ratio | **1.99** | — |

---

## ▶️ Quick Start

```bash
pip install torch numpy pandas matplotlib seaborn statsmodels scipy scikit-learn yfinance psutil

python main.py              # Single run, seed 42
python main.py --quick      # Fast demo (10 epochs)
python main.py --seed 123   # Custom seed
python main.py --multi-seed # 5-seed ensemble for paper statistics
```

---

## 🔖 Topics
`climate-risk` `financial-forecasting` `graph-attention-network` `gru`
`transformer` `pinn` `physics-informed` `deep-learning` `pytorch`
`explainability` `integrated-gradients` `granger-causality` `shap`
`ngfs` `tcfd` `time-series` `systemic-risk` `climate-finance` `xai`
