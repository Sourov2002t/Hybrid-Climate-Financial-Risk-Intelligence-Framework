# HCFRI — Hybrid Climate-Financial Risk Intelligence Framework

HCFRI is a research-grade, end-to-end machine learning framework that quantifies,
forecasts, and explains the financial risk imposed by physical and transition
climate hazards. It moves beyond standard autoregressive models by combining
deep temporal learning with dynamic graph theory to capture systemic contagion
across financial sectors under climate stress.

## Architecture — 5-Layer Pipeline

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | **Data Ingestion** | CMIP6 downscaled climate surrogates + EM-DAT disaster records + macro indicators |
| 2 | **HTFN — Temporal Modeling** | Hierarchical Temporal Fusion Network: GRU (1–30d) · Transformer (1–12mo) · Physics-informed (5–30yr) |
| 3 | **Hybrid Core — Contagion** | CNN-BiLSTM spatial extractor + Climate-Stressed Graph Attention Network (GAT) |
| 4 | **XAI — Explainability** | Integrated Gradients · SHAP attributions · Attention heatmaps · Granger causality |
| 5 | **Validation & Policy** | NGFS stress tests · TCFD-aligned reports · SOTA comparison (25 baselines) · Ablation study |

## Key Results

| Metric | Value |
|--------|-------|
| RMSE (in-distribution) | **0.0076** (vs. best baseline iTransformer: 0.0127) |
| OOD R² (2023–24 regime) | **0.883** |
| Systemic Risk Capture | **0.9986** |
| Sharpe Ratio | **2.27** |
| Information Coefficient | **0.967** |

## Datasets
- `data/raw/climate/` — Daily Delhi Climate (train/test CSVs)
- `data/raw/disaster/` — EM-DAT global disaster records (1900–2021)
- `data/processed/` — Pre-engineered climate, disaster, and financial feature sets

## Outputs
- `outputs/risk_dashboard.png` — 8-panel dark-theme portfolio risk dashboard
- `outputs/paper_visuals/` — Publication-ready figures (SHAP, GAT heatmap, NGFS fan chart, OOD plot, ablation chart)
- `outputs/risk_report.txt` — TCFD/NGFS-aligned policy report
- `outputs/manuscript_draft.md` — Research paper draft

## Quick Start
```bash
python main.py              # Single run (seed 42)
python main.py --quick      # Fast demo (10 epochs)
python main.py --multi-seed # 5-seed ensemble for paper statistics
```
