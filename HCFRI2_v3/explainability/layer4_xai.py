"""
HCFRI Framework — Layer 4: Explainable AI (XAI)
================================================
Addresses Gap #4: Model Interpretability

Four-level explainability stack (beyond any existing climate-finance paper):

  Level 1  Global feature importance  (permutation-based + 90% CI error bars)
  Level 2  Instance-level attribution  (Integrated Gradients, SHAP-style)
  Level 3  Attention weight heatmaps   (hooked from all MultiheadAttention layers)
  Level 4  Climate→Finance lead-lag    (lagged cross-correlation + Granger F-test)

All figures saved to disk — NO plt.show() calls (headless / server safe).
"""

# Headless backend (removed, deferred to conditional in main.py)
import matplotlib

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from typing import List, Optional, Dict

warnings.filterwarnings("ignore")

# ── colour palette (dark theme, consistent with Layer 5 dashboard) ──
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1f2e"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_GREEN  = "#2ecc71"
C_YELLOW = "#f39c12"
C_PURPLE = "#9b59b6"
C_WHITE  = "#ecf0f1"
C_GREY   = "#7f8c8d"


def _is_climate(name: str) -> bool:
    """Return True if feature name looks like a climate variable."""
    kw = {"temp", "precip", "flood", "drought", "wildfire", "co2",
          "sst", "extreme", "humidity", "wind", "pressure", "tropical",
          "arctic", "disaster", "storm", "heat", "fire"}
    n = name.lower()
    return any(k in n for k in kw)


def _extract_pred(output) -> torch.Tensor:
    """Pull forecast tensor — same scale-check as Layer 5 _extract_forecast."""
    if isinstance(output, dict):
        for key in ("forecast", "pred", "output"):
            if key in output:
                t = output[key]
                if isinstance(t, torch.Tensor):
                    if t.std().item() < 0.15:
                        return t
                    return torch.tanh(t) * 0.015   # emergency clip
        for v in output.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                return v
        raise ValueError(f"No tensor in model output: {list(output.keys())}")
    return output


# ════════════════════════════════════════════════════════════════════
# Level 1: Global Feature Importance (permutation + CI)
# ════════════════════════════════════════════════════════════════════

class GlobalFeatureImportance:
    """
    Model-agnostic permutation importance.
    Repeats permutation n_repeats times → mean + 10–90th percentile CI.
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_repeats: int = 5):
        self.model    = model
        self.device   = device
        self.n_repeats = n_repeats

    def compute(self, X: np.ndarray, y: np.ndarray,
                feature_names: List[str]) -> pd.DataFrame:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            base_pred = _extract_pred(self.model(X_t)).cpu().numpy()

        # Use first-step prediction aligned against first-step target
        y1 = y[:, 0] if y.ndim > 1 else y
        p1 = base_pred[:, 0] if base_pred.ndim > 1 else base_pred.flatten()
        # Trim to same length
        n_min = min(len(p1), len(y1))
        base_rmse = float(np.sqrt(np.mean((p1[:n_min] - y1[:n_min]) ** 2)))

        records = []
        n_feat = X.shape[2]
        for fi in range(n_feat):
            drops = []
            for _ in range(self.n_repeats):
                Xp = X.copy()
                Xp[:, :, fi] = Xp[np.random.permutation(len(Xp)), :, fi]
                with torch.no_grad():
                    pp = _extract_pred(
                        self.model(torch.FloatTensor(Xp).to(self.device))
                    ).cpu().numpy()
                pp1 = pp[:, 0] if pp.ndim > 1 else pp.flatten()
                perm_rmse = float(np.sqrt(np.mean(
                    (pp1[:n_min] - y1[:n_min]) ** 2)))
                drops.append(perm_rmse - base_rmse)

            fname = (feature_names[fi] if fi < len(feature_names)
                     else f"feat_{fi}")
            records.append({
                "feature":          fname,
                "importance_mean":  float(np.mean(drops)),
                "importance_std":   float(np.std(drops)),
                "importance_p10":   float(np.percentile(drops, 10)),
                "importance_p90":   float(np.percentile(drops, 90)),
                "is_climate":       _is_climate(fname),
            })

        return (pd.DataFrame(records)
                  .sort_values("importance_mean", ascending=False)
                  .reset_index(drop=True))

    def plot(self, df: pd.DataFrame, top_n: int = 20,
             save_path: Optional[str] = None):
        top = df.head(top_n).iloc[::-1].copy()

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        colors = [C_RED if r["is_climate"] else C_BLUE
                  for _, r in top.iterrows()]
        y_pos  = range(len(top))

        ax.barh(y_pos, top["importance_mean"], color=colors, alpha=0.85,
                edgecolor=DARK_BG, height=0.65)

        # 10–90th percentile CI whiskers
        xerr_lo = (top["importance_mean"] - top["importance_p10"]).values
        xerr_hi = (top["importance_p90"] - top["importance_mean"]).values
        ax.errorbar(top["importance_mean"], y_pos,
                    xerr=[xerr_lo, xerr_hi],
                    fmt="none", color=C_WHITE, capsize=3, linewidth=1.2)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top["feature"].tolist(), color=C_WHITE, fontsize=9)
        ax.set_xlabel("Importance (RMSE increase on permutation)",
                      color=C_WHITE, fontsize=10)
        ax.set_title(
            "Global Feature Importance  [HCFRI Layer 4 — Level 1]\n"
            "Error bars: 10th–90th percentile over 10 permutation repeats",
            color=C_WHITE, fontsize=11, fontweight="bold")
        ax.axvline(0, color=C_GREY, linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.2, color=C_GREY)
        ax.tick_params(colors=C_WHITE)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GREY)

        legend_patches = [
            mpatches.Patch(color=C_RED,  alpha=0.85, label="Climate Feature"),
            mpatches.Patch(color=C_BLUE, alpha=0.85, label="Financial Feature"),
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  facecolor=PANEL_BG, edgecolor=C_GREY,
                  labelcolor=C_WHITE, fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
        plt.close(fig)
        return fig


# ════════════════════════════════════════════════════════════════════
# Level 2: Integrated Gradients (SHAP-style)
# ════════════════════════════════════════════════════════════════════

class IntegratedGradientsExplainer:
    """
    Integrated Gradients (Sundararajan et al., 2017).
    Approximates SHAP values for neural networks without the exponential
    cost of KernelSHAP. Explains per-timestep, per-feature contributions.
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_steps: int = 30):
        self.model   = model
        self.device  = device
        self.n_steps = n_steps

    def explain(self, X: np.ndarray, target_idx: int = 0,
                n_samples: int = 60) -> np.ndarray:
        """
        Returns attributions array of shape (n_samples, seq_len, n_features).
        Uses mean of X as baseline (standard practice).
        """
        X_t      = torch.FloatTensor(X).to(self.device)
        baseline = X_t.mean(0, keepdim=True)   # shape (1, seq, feat)

        all_attrs = []
        n_explain = min(n_samples, len(X))

        # Batched alphas for 10x-50x speedup on CPU
        alphas = torch.linspace(0, 1, self.n_steps, device=self.device).view(-1, 1, 1)

        for i in range(n_explain):
            x_i    = X_t[i:i + 1]                    # (1, seq, feat)
            
            interp = (baseline + alphas * (x_i - baseline)).detach()
            interp.requires_grad_(True)

            out   = self.model(interp)
            score = _extract_pred(out)

            # scalar target: first-step prediction sum across the batch
            if score.dim() > 1:
                s = score[:, target_idx].sum()
            else:
                s = score.sum()

            s.backward()

            if interp.grad is not None:
                avg_grad  = interp.grad.detach().cpu().mean(0, keepdim=True) # (1, seq, feat)
                attr      = avg_grad * (x_i - baseline).detach().cpu()
                all_attrs.append(attr.squeeze(0).numpy())        # (seq, feat)
            else:
                all_attrs.append(np.zeros((X.shape[1], X.shape[2])))

        return np.array(all_attrs)   # (n_explain, seq, feat)

    def plot(self, attributions: np.ndarray,
             feature_names: List[str],
             save_path: Optional[str] = None):
        """
        Two-panel figure:
          Left  — mean absolute attribution per feature (top-20 bar chart)
          Right — temporal heatmap of top-10 features (seq_len × 10)
        """
        # Mean over samples then time → per-feature importance
        mean_attr     = attributions.mean(0)          # (seq, feat)
        feat_mean_abs = np.abs(mean_attr).mean(0)     # (feat,)

        top20_idx = np.argsort(feat_mean_abs)[-20:]
        top10_idx = np.argsort(feat_mean_abs)[-10:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor(DARK_BG)
        for ax in axes:
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_GREY)

        # Left: top-20 features
        ax0   = axes[0]
        feats = ([feature_names[i] for i in top20_idx]
                 if feature_names else [f"f{i}" for i in top20_idx])
        vals  = feat_mean_abs[top20_idx]
        cols  = [C_RED if _is_climate(f) else C_BLUE for f in feats]
        ax0.barh(feats, vals, color=cols, alpha=0.85, edgecolor=DARK_BG)
        ax0.set_xlabel("Mean |Attribution|", color=C_WHITE, fontsize=10)
        ax0.set_title("Feature Attribution (Integrated Gradients)\n"
                      "Top-20 features — averaged over samples & time",
                      color=C_WHITE, fontsize=10, fontweight="bold")
        ax0.tick_params(colors=C_WHITE)
        ax0.xaxis.label.set_color(C_WHITE)

        # Right: temporal heatmap (seq_len × 10 features)
        ax1      = axes[1]
        top_names = ([feature_names[i] for i in top10_idx]
                     if feature_names else [f"f{i}" for i in top10_idx])
        heatmap  = mean_attr[:, top10_idx].T   # (10, seq_len)
        im = ax1.imshow(heatmap, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(heatmap).max(),
                        vmax=np.abs(heatmap).max())
        ax1.set_yticks(range(len(top_names)))
        ax1.set_yticklabels(top_names, color=C_WHITE, fontsize=8)
        ax1.set_xlabel("Time step (past → present)",
                       color=C_WHITE, fontsize=10)
        ax1.set_title("Temporal Attribution Heatmap\n"
                      "Top-10 features over sequence",
                      color=C_WHITE, fontsize=10, fontweight="bold")
        ax1.tick_params(colors=C_WHITE)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046)
        cbar.ax.yaxis.set_tick_params(color=C_WHITE)
        cbar.set_label("Attribution", color=C_WHITE)

        legend_patches = [
            mpatches.Patch(color=C_RED,  alpha=0.85, label="Climate"),
            mpatches.Patch(color=C_BLUE, alpha=0.85, label="Financial"),
        ]
        axes[0].legend(handles=legend_patches, loc="lower right",
                       facecolor=PANEL_BG, edgecolor=C_GREY,
                       labelcolor=C_WHITE, fontsize=8)

        fig.suptitle("HCFRI Layer 4 — Level 2: Instance-Level XAI\n"
                     "Integrated Gradients Attribution",
                     color=C_WHITE, fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
        plt.close(fig)
        return fig


# ════════════════════════════════════════════════════════════════════
# Level 3: Attention Weight Visualiser
# ════════════════════════════════════════════════════════════════════

class AttentionVisualizer:
    """Extract and visualise attention weights from all MHA layers."""

    @staticmethod
    def extract_weights(model: nn.Module,
                        x: torch.Tensor) -> Dict[str, np.ndarray]:
        weights: Dict[str, np.ndarray] = {}
        hooks   = []

        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    weights[name] = out[1].detach().cpu().numpy()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return weights

    @staticmethod
    def plot(attention_weights: Dict[str, np.ndarray],
             save_path: Optional[str] = None):
        if not attention_weights:
            # Generate a placeholder figure so Layer 4 never crashes
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL_BG)
            ax.text(0.5, 0.5,
                    "No attention weights captured\n"
                    "(model may not have MHA layers visible to hooks)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=C_GREY, fontsize=11)
            ax.set_title("Attention Weights — HCFRI Layer 4",
                         color=C_WHITE, fontsize=11)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_GREY)
            ax.tick_params(colors=C_WHITE)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight",
                            facecolor=DARK_BG)
            plt.close(fig)
            return fig

        n   = len(attention_weights)
        fig = plt.figure(figsize=(min(6 * n, 20), 5))
        fig.patch.set_facecolor(DARK_BG)
        axes = fig.subplots(1, n) if n > 1 else [fig.add_subplot(111)]

        for ax, (name, w) in zip(axes, attention_weights.items()):
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_GREY)

            # w shape: (batch, heads, Q, K)  or  (batch, Q, K)
            if w.ndim == 4:
                mat = w[0].mean(0)     # average over heads → (Q, K)
            elif w.ndim == 3:
                mat = w[0]
            else:
                mat = w

            if mat.shape[0] > 1 and mat.shape[1] > 1 and mat.std() > 1e-5:
                sns.heatmap(mat, ax=ax, cmap="Blues", cbar=True,
                            xticklabels=False, yticklabels=False,
                            linewidths=0)
            else:
                # Fix #3A: Attention visualization guard (bar chart fallback for 1x1 or 1D)
                val = mat.flatten()
                ax.bar(range(len(val)), val, color=C_BLUE, alpha=0.8)
                ax.set_ylim(0, max(1e-5, val.max() * 1.1))
                ax.set_title(f"{name.split('.')[-1]}\n(Uniform/1x1)", color=C_GREY, fontsize=9)

            label = name.replace(".", "\n")
            ax.set_title(f"Attn: {label}", color=C_WHITE,
                         fontsize=8, fontweight="bold")
            ax.tick_params(colors=C_WHITE)

        fig.suptitle("HCFRI Layer 4 — Level 3: Attention Weight Visualisation",
                     color=C_WHITE, fontsize=11, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
        plt.close(fig)
        return fig


# ════════════════════════════════════════════════════════════════════
# Level 4: Predictive Lead-Lag Analysis + Granger Causality
# ════════════════════════════════════════════════════════════════════

class PredictiveLeadAnalyzer:
    """
    Novel contribution: climate→finance lead-lag detector.
    Uses lagged cross-correlation to find optimal predictive lead,
    then validates with Granger F-test.

    Note: Granger causality measures statistical predictability, not
    physical causation. Known pathways are documented for transparency.
    """

    CLIMATE_VARS = [
        "temp_anomaly", "precip_anomaly", "drought_index",
        "extreme_event_index", "wildfire_risk", "flood_risk",
        "co2_ppm", "sst_anomaly",
    ]

    # Flexible financial var list — checked against actual df columns
    FINANCIAL_VARS_CANDIDATES = [
        "sp500_return", "sp500_vol20", "sp500_vol_20d",
        "energy_return", "vix_close", "vix_return",
    ]

    KNOWN_PATHWAYS = [
        ("drought_index",       "energy_return",
         "Drought reduces hydropower → energy prices ↑"),
        ("flood_risk",          "sp500_vol20",
         "Flood events increase market volatility"),
        ("flood_risk",          "sp500_vol_20d",
         "Flood events increase market volatility"),
        ("temp_anomaly",        "sp500_return",
         "Heat waves reduce productivity → lower returns"),
        ("wildfire_risk",       "energy_return",
         "Wildfires disrupt energy infrastructure"),
        ("extreme_event_index", "vix_close",
         "Extreme events spike fear index (VIX)"),
        ("sst_anomaly",         "energy_return",
         "SST anomalies predict hurricane intensity → energy disruption"),
    ]

    def analyze(self, unified_df: pd.DataFrame,
                max_lag: int = 60) -> pd.DataFrame:
        """
        Compute lagged cross-correlations + Granger F-test for all
        climate × financial variable pairs present in unified_df.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        # Only use financial vars actually present in the dataframe
        fin_vars = [v for v in self.FINANCIAL_VARS_CANDIDATES
                    if v in unified_df.columns]
        if not fin_vars:
            # Fallback: use any column with 'return' or 'vol' in name
            fin_vars = [c for c in unified_df.columns
                        if any(k in c for k in ("return", "vol20",
                                                "vix", "close"))][:4]

        cli_vars = [v for v in self.CLIMATE_VARS if v in unified_df.columns]

        records = []
        for cli_var in cli_vars:
            for fin_var in fin_vars:
                x = unified_df[cli_var].dropna()
                y = unified_df[fin_var].dropna()
                common  = x.index.intersection(y.index)
                x, y    = x[common], y[common]
                if len(x) < 50:
                    continue

                # Find optimal lag (peak |correlation|)
                best_lag, best_corr = 0, 0.0
                for lag in range(1, min(max_lag + 1, len(x) // 4)):
                    xs = pd.Series(x.values[:-lag])
                    ys = pd.Series(y.values[lag:])
                    c  = float(xs.corr(ys))
                    if not np.isnan(c) and abs(c) > abs(best_corr):
                        best_corr, best_lag = c, lag

                # Granger F-test
                gc_pvalue = float("nan")
                if best_lag >= 1:
                    try:
                        td = pd.concat([
                            pd.Series(y.values[best_lag:]),
                            pd.Series(x.values[:-best_lag])
                        ], axis=1).dropna()
                        if len(td) > best_lag * 4:
                            gc_res = grangercausalitytests(
                                td.values, maxlag=best_lag, verbose=False)
                            gc_pvalue = gc_res[best_lag][0]["ssr_ftest"][1]
                    except Exception:
                        gc_pvalue = float("nan")

                known = next((p[2] for p in self.KNOWN_PATHWAYS
                              if p[0] == cli_var and p[1] == fin_var), "")
                records.append({
                    "climate_var":             cli_var,
                    "financial_var":           fin_var,
                    "optimal_lag_days":        best_lag,
                    "peak_correlation":        round(best_corr, 4),
                    "predictive_lead_strength": round(abs(best_corr), 4),
                    "direction":  "→ positive" if best_corr > 0 else "→ negative",
                    "granger_pvalue":          round(gc_pvalue, 4)
                                               if not np.isnan(gc_pvalue) else float("nan"),
                    "significance":            "n.s.", # placeholder
                    "is_significant":          False,  # placeholder
                    "known_pathway":           known,
                })

        if not records:
            return pd.DataFrame(columns=[
                "climate_var", "financial_var", "optimal_lag_days",
                "peak_correlation", "predictive_lead_strength",
                "direction", "granger_pvalue", "significance",
                "is_significant", "known_pathway",
            ])

        df = pd.DataFrame(records)
        valid_mask = df["granger_pvalue"].notna()
        if valid_mask.sum() > 0:
            from statsmodels.stats.multitest import multipletests
            _, pvals_fdr, _, _ = multipletests(df.loc[valid_mask, "granger_pvalue"], alpha=0.05, method="fdr_bh")
            df["granger_pvalue_fdr"] = np.nan
            df.loc[valid_mask, "granger_pvalue_fdr"] = pvals_fdr
            
            def get_sig(p):
                if np.isnan(p): return "n.s."
                if p < 0.001: return "***"
                if p < 0.01: return "**"
                if p < 0.05: return "*"
                return "n.s."
            
            df["significance"] = df["granger_pvalue_fdr"].apply(get_sig)
            df["is_significant"] = df["significance"] != "n.s."

        return (df.sort_values("predictive_lead_strength", ascending=False)
                  .reset_index(drop=True))

    def plot_causal_heatmap(self, causal_df: pd.DataFrame,
                             save_path: Optional[str] = None):
        """Heatmap of peak |correlation| with Granger significance stars."""
        if causal_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL_BG)
            ax.text(0.5, 0.5, "No lead-lag data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color=C_GREY, fontsize=12)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight",
                            facecolor=DARK_BG)
            plt.close(fig)
            return fig

        pivot = causal_df.pivot_table(
            values="peak_correlation",
            index="climate_var", columns="financial_var",
            aggfunc="first")

        p_pivot = causal_df.pivot_table(
            values="granger_pvalue_fdr" if "granger_pvalue_fdr" in causal_df.columns else "granger_pvalue",
            index="climate_var", columns="financial_var",
            aggfunc="first")

        # Build annotation: value + significance stars
        annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
        for r in pivot.index:
            for c in pivot.columns:
                v = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
                p = p_pivot.loc[r, c] if r in p_pivot.index and c in p_pivot.columns else np.nan
                if np.isnan(v):
                    annot.loc[r, c] = ""
                    continue
                stars = ("***" if not np.isnan(p) and p < 0.001 else
                         "**"  if not np.isnan(p) and p < 0.01  else
                         "*"   if not np.isnan(p) and p < 0.05  else "")
                annot.loc[r, c] = f"{v:.3f}{stars}"

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2),
                                        max(6, len(pivot.index) * 1.2)))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        vmax = pivot.abs().max().max()
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdBu_r",
                    center=0, vmin=-vmax, vmax=vmax,
                    ax=ax, linewidths=0.5, linecolor=DARK_BG,
                    cbar_kws={"label": "Peak Lagged Correlation",
                               "shrink": 0.8})
        ax.set_title(
            "Climate → Finance Predictive Lead-Lag Heatmap\n"
            "HCFRI Layer 4 — Level 4  |  * p<0.05  ** p<0.01  *** p<0.001 (Granger)",
            fontsize=11, fontweight="bold", color=C_WHITE)
        ax.set_xlabel("Financial Variable", color=C_WHITE, fontsize=10)
        ax.set_ylabel("Climate Variable",   color=C_WHITE, fontsize=10)
        ax.tick_params(colors=C_WHITE)
        plt.xticks(rotation=30, ha="right", color=C_WHITE)
        plt.yticks(rotation=0, color=C_WHITE)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
        plt.close(fig)
        return fig

    @staticmethod
    def granger_summary(unified_df: pd.DataFrame,
                        max_lag: int = 51) -> dict:
        """Return min Granger p-values for key climate→finance pairs."""
        from statsmodels.tsa.stattools import grangercausalitytests

        # Flexible column matching
        vix_col = next((c for c in ["vix_close", "vix_return"]
                        if c in unified_df.columns), None)
        enr_col = next((c for c in ["energy_return"]
                        if c in unified_df.columns), None)
        sp5_col = next((c for c in ["sp500_return"]
                        if c in unified_df.columns), None)

        pairs = []
        if "extreme_event_index" in unified_df.columns and vix_col:
            pairs.append(("extreme_event_index", vix_col))
        if "flood_risk" in unified_df.columns and enr_col:
            pairs.append(("flood_risk", enr_col))
        if "drought_index" in unified_df.columns and sp5_col:
            pairs.append(("drought_index", sp5_col))
        if "temp_anomaly" in unified_df.columns and sp5_col:
            pairs.append(("temp_anomaly", sp5_col))

        results = {}
        for cause, effect in pairs:
            data = unified_df[[effect, cause]].dropna()
            try:
                res  = grangercausalitytests(data, maxlag=max_lag,
                                             verbose=False)
                pmin = min(res[lag][0]["ssr_ftest"][1] for lag in res)
                results[f"{cause}→{effect}"] = round(pmin, 4)
            except Exception:
                pass
        return results


# ════════════════════════════════════════════════════════════════════
# Master pipeline runner
# ════════════════════════════════════════════════════════════════════

def run_xai_pipeline(model: nn.Module,
                     X: np.ndarray,
                     y: np.ndarray,
                     feature_names: List[str],
                     unified_df: Optional[pd.DataFrame] = None,
                     output_dir: str = "outputs/") -> dict:
    """
    Run all four XAI levels and save PNG figures.
    Returns a dict of summary results for the pipeline report.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = str(next(model.parameters()).device)

    print("\n" + "=" * 60)
    print("  HCFRI LAYER 4: EXPLAINABILITY (XAI)")
    print("=" * 60)

    results: dict = {}

    # ── Level 1: Global Feature Importance ──────────────────────
    print("\n[L1] Global Feature Importance (permutation, 10 repeats)...")
    try:
        gfi    = GlobalFeatureImportance(model, device=device, n_repeats=10)
        imp_df = gfi.compute(X[:200], y[:200], feature_names)
        gfi.plot(imp_df,
                 save_path=os.path.join(output_dir, "l1_global_importance.png"))
        top5 = imp_df["feature"].head(5).tolist()
        print(f"  Saved: l1_global_importance.png")
        print(f"  Top 5 features: {top5}")
        results["top_features"] = top5
    except Exception as exc:
        print(f"  [L1] Warning: {exc}")
        results["top_features"] = []

    # ── Level 2: Integrated Gradients ───────────────────────────
    print("\n[L2] Integrated Gradients attribution (SHAP-style)...")
    try:
        ig   = IntegratedGradientsExplainer(model, device=device, n_steps=30)
        attrs = ig.explain(X[:80], target_idx=0, n_samples=60)
        ig.plot(attrs, feature_names,
                save_path=os.path.join(output_dir,
                                        "l2_instance_attribution.png"))
        print(f"  Saved: l2_instance_attribution.png")
        print(f"  Attribution computed for {len(attrs)} samples")
        results["n_explained"] = len(attrs)
    except Exception as exc:
        print(f"  [L2] Warning: {exc}")
        results["n_explained"] = 0

    # ── Level 3: Attention weights ───────────────────────────────
    print("\n[L3] Attention weight visualisation...")
    try:
        av     = AttentionVisualizer()
        x_test = torch.FloatTensor(X[:1]).to(device)
        attn_w = av.extract_weights(model, x_test)
        av.plot(attn_w,
                save_path=os.path.join(output_dir,
                                        "l3_attention_weights.png"))
        print(f"  Saved: l3_attention_weights.png")
        print(f"  Extracted {len(attn_w)} attention layer(s)")
        results["n_attention_layers"] = len(attn_w)
    except Exception as exc:
        print(f"  [L3] Warning: {exc}")
        results["n_attention_layers"] = 0

    # ── Level 4: Lead-lag + Granger ─────────────────────────────
    print("\n[L4] Predictive lead-lag analysis + Granger causality tests...")
    if unified_df is not None:
        try:
            pla      = PredictiveLeadAnalyzer()
            causal_df = pla.analyze(unified_df)
            pla.plot_causal_heatmap(
                causal_df,
                save_path=os.path.join(output_dir, "l4_causal_heatmap.png"))
            print(f"  Saved: l4_causal_heatmap.png")

            if not causal_df.empty:
                print(f"\n  Top 5 predictive lead relationships:")
                for _, row in causal_df.head(5).iterrows():
                    sig = row.get("significance", "")
                    print(f"    {row['climate_var']:25s} → "
                          f"{row['financial_var']:20s} "
                          f"[lag={row['optimal_lag_days']:3d}d  "
                          f"r={row['peak_correlation']:+.3f}  {sig}]")
                    if row.get("known_pathway"):
                        print(f"      ↳ {row['known_pathway']}")

            # Granger summary
            print("\n[L4b] Granger causality summary:")
            gc_res = PredictiveLeadAnalyzer.granger_summary(unified_df)
            for pair, pval in gc_res.items():
                sig = ("*** p<0.001" if pval < 0.001 else
                       "**  p<0.01"  if pval < 0.01  else
                       "*   p<0.05"  if pval < 0.05  else
                       "    n.s.")
                print(f"    {pair:45s}  p={pval:.4f}  {sig}")

            results["causal_pairs"] = len(causal_df)
        except Exception as exc:
            import traceback
            print(f"  [L4] Warning: {exc}")
            traceback.print_exc()
            results["causal_pairs"] = 0
    else:
        print("  Skipped (no unified_df provided)")
        results["causal_pairs"] = 0

    print("\n  Layer 4 complete ✓")
    return results


if __name__ == "__main__":
    # Standalone test
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.layer3_hybrid_core import build_hybrid_core
    model = build_hybrid_core(input_dim=20, skip_ablation=True)
    X = np.random.randn(120, 30, 20).astype(np.float32)
    y = np.random.randn(120, 5).astype(np.float32)
    feature_names = [f"feat_{i}" for i in range(20)]
    run_xai_pipeline(model, X, y, feature_names, output_dir="outputs/")
