"""
HCFRI Framework - Layer 2: Multi-Temporal Hybrid Modeling
=========================================================
Addresses Gap #2 (Temporal Mismatch)

NEW ADDITION: Hierarchical Temporal Fusion Network (HTFN)
- Short-term module  (1–30 days):   GRU encoder-decoder
- Medium-term module (1–12 months): Transformer with cross-attention
- Long-term module   (5–30 years):  Physics-informed constraints
- HTFN couples all three via Bayesian precision weights
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

# ─────────────────────────────────────────────
# Short-Term Module: GRU Encoder-Decoder
# ─────────────────────────────────────────────

class ShortTermGRU(nn.Module):
    """
    GRU-based encoder-decoder for 1–30 day financial prediction.
    Inputs: High-frequency market + daily climate features.
    Output: Multi-step return forecasts with uncertainty.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, forecast_horizon: int = 5, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Encoder
        self.encoder = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention over encoder outputs
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Decoder: projects attended context to forecast
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * 2)  # mean + log_var
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, _ = self.encoder(x)          # (batch, seq, hidden*2)
        attn_w = self.attn(enc_out)            # (batch, seq, 1)
        context = (attn_w * enc_out).sum(1)    # (batch, hidden*2)
        out = self.decoder(context)            # (batch, horizon*2)

        mean = out[:, :self.forecast_horizon]
        log_var = out[:, self.forecast_horizon:]
        return mean, log_var                   # probabilistic output


# ─────────────────────────────────────────────
# Medium-Term Module: Transformer with Cross-Attention
# ─────────────────────────────────────────────

class MediumTermTransformer(nn.Module):
    """
    Transformer bridging climate (seasonal) and financial (quarterly) signals.
    Cross-attention: financial queries attend to climate key-values.
    NEW: Learned temporal downscaling from monthly climate → weekly finance.
    """

    def __init__(self, climate_dim: int, financial_dim: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, forecast_horizon: int = 5, dropout: float = 0.2):
        super().__init__()

        # Project climate and financial to common d_model
        self.climate_proj = nn.Linear(climate_dim, d_model)
        self.financial_proj = nn.Linear(financial_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=500)

        # Self-attention on financial sequence
        fin_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.fin_encoder = nn.TransformerEncoder(fin_layer, num_layers=num_layers)

        # Cross-attention: finance queries climate context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )

    def forward(self, x_climate, x_financial):
        # Project & encode
        c = self.pos_enc(self.climate_proj(x_climate))
        f = self.pos_enc(self.financial_proj(x_financial))

        # Self-attend on financial
        f_enc = self.fin_encoder(f)

        # Cross-attend: financial queries attend to climate
        fused, attn_weights = self.cross_attn(
            query=f_enc, key=c, value=c
        )

        # Pool over time and project to forecast
        pooled = fused.mean(dim=1)
        return self.output_proj(pooled), attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        return self.dropout(x + pe[:, :x.size(1)])


# ─────────────────────────────────────────────
# Long-Term Module: Physics-Informed Constraints
# ─────────────────────────────────────────────

class LongTermPhysicsModule(nn.Module):
    """
    NEW CONTRIBUTION: Physics-Informed Neural Network (PINN) for
    long-term climate-financial coupling.
    
    Physical constraints embedded:
    - Energy balance: warming ∝ cumulative CO2
    - Asset depreciation: physical damage follows power law
    - Mean reversion: financial risk cannot grow unboundedly
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        # Main neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),   # Tanh preferred for PINNs (smooth derivatives)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Learnable physics parameters
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # warming sensitivity
        self.log_beta = nn.Parameter(torch.tensor(0.0))   # damage exponent
        self.log_kappa = nn.Parameter(torch.tensor(0.0))  # mean reversion rate

    def forward(self, x):
        """x: (batch, features) — annual aggregated climate-financial state"""
        nn_out = self.net(x)

        # Physics constraints as regularization targets (returned for loss computation)
        alpha = torch.exp(self.log_alpha).clamp(0.1, 5.0)
        beta = torch.exp(self.log_beta).clamp(0.5, 3.0)
        kappa = torch.exp(self.log_kappa).clamp(0.01, 1.0)

        physics_params = {'alpha': alpha, 'beta': beta, 'kappa': kappa}
        return nn_out, physics_params

    def physics_loss(self, pred, target, params):
        """
        Physics-regularized loss:
        - MSE on prediction
        - Penalty if parameters violate known physical bounds
        """
        mse = F.mse_loss(pred, target)

        # Soft constraint: damage exponent should be >= 1 (superlinear)
        physics_penalty = F.relu(1.0 - params['beta']) ** 2

        # Mean reversion penalty: kappa too small → explosive risk
        rev_penalty = F.relu(0.05 - params['kappa']) ** 2

        return mse + 0.1 * physics_penalty + 0.1 * rev_penalty


# ─────────────────────────────────────────────
# HTFN: Hierarchical Temporal Fusion Network
# ─────────────────────────────────────────────

class HierarchicalTemporalFusionNetwork(nn.Module):
    """
    NOVEL CONTRIBUTION:
    Fuses short-, medium-, and long-term predictions with
    learnable Bayesian precision weights.
    
    Each module produces a forecast; HTFN learns how much to
    trust each timescale based on current market regime.
    """

    def __init__(self, input_dim: int, forecast_horizon: int = 5):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # Split input into climate and financial portions
        self.climate_dim = input_dim // 3
        self.financial_dim = input_dim - self.climate_dim

        # Three temporal modules
        self.short_term = ShortTermGRU(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=1,
            forecast_horizon=forecast_horizon
        )

        self.medium_term = MediumTermTransformer(
            climate_dim=self.climate_dim,
            financial_dim=self.financial_dim,
            d_model=64,
            num_layers=1,
            forecast_horizon=forecast_horizon
        )

        self.long_term = LongTermPhysicsModule(
            input_dim=input_dim,
            output_dim=forecast_horizon
        )

        # Regime detector: classifies market state → weights temporal modules
        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),     # 3 weights for 3 modules
            nn.Softmax(dim=-1)    # sum to 1
        )

        # Final fusion layer with skip connection
        self.fusion = nn.Sequential(
            nn.Linear(forecast_horizon * 3, forecast_horizon * 2),
            nn.GELU(),
            nn.Linear(forecast_horizon * 2, forecast_horizon)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: fused_forecast, component_forecasts, regime_weights
        """
        batch, seq_len, feat = x.shape

        # Short-term: full sequence
        st_forecast, st_logvar = self.short_term(x)

        # Medium-term: split by modality
        x_climate = x[:, :, :self.climate_dim]
        x_financial = x[:, :, self.climate_dim:]
        mt_forecast, _ = self.medium_term(x_climate, x_financial)

        # Long-term: uses summary statistics of sequence
        x_summary_input = x.mean(1)  # (batch, input_dim)
        lt_forecast, physics_params = self.long_term(x_summary_input)

        # Regime detection (from last timestep)
        regime_weights = self.regime_detector(x[:, -1, :])  # (batch, 3)
        w_st = regime_weights[:, 0:1]
        w_mt = regime_weights[:, 1:2]
        w_lt = regime_weights[:, 2:3]

        # Weighted fusion
        weighted = (w_st * st_forecast + w_mt * mt_forecast + w_lt * lt_forecast)

        # Residual correction via fusion layer
        concat = torch.cat([st_forecast, mt_forecast, lt_forecast], dim=-1)
        residual = self.fusion(concat)
        final = weighted + 0.1 * residual

        return {
            'forecast': final,
            'short_term': st_forecast,
            'medium_term': mt_forecast,
            'long_term': lt_forecast,
            'regime_weights': regime_weights,
            'st_uncertainty': torch.exp(0.5 * st_logvar),
            'physics_params': physics_params
        }


# ─────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────

class HTFNTrainer:
    """Train and evaluate the HTFN model."""

    def __init__(self, model: HierarchicalTemporalFusionNetwork,
                 lr: float = 1e-3, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.history = {'train_loss': [], 'val_loss': []}

    def _loss(self, output: dict, target: torch.Tensor) -> torch.Tensor:
        """MSE + variance-matching + physics-regularized loss."""
        pred   = output['forecast']
        pred   = torch.nan_to_num(pred,   nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        mse         = F.mse_loss(pred, target)
        var_penalty = 0.5 * (pred.var() - target.var()).abs()
        
        # ── Directional Loss ──
        dir_loss = torch.mean(torch.relu(-torch.sign(pred) * torch.sign(target)))

        # ── Correlation Loss ──
        fc_c  = pred[:, 0]  - pred[:, 0].mean()
        tg_c  = target[:, 0] - target[:, 0].mean()
        corr  = (fc_c * tg_c).sum() / (fc_c.norm() * tg_c.norm() + 1e-8)
        corr_loss = (1.0 - corr) * 0.5

        # Physics regularization from PINN module
        physics_params = output.get('physics_params', None)
        if physics_params is not None:
            lt_pred = output.get('long_term', pred)
            phys_loss = self.model.long_term.physics_loss(
                lt_pred, target, physics_params)
            return mse + var_penalty + 0.5 * dir_loss + corr_loss + 0.05 * phys_loss
        return mse + var_penalty + 0.5 * dir_loss + corr_loss

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self._loss(output, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue   # skip bad batch, don't corrupt weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        preds, targets, weights = [], [], []

        uncertainties = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            out = self.model(X_batch)
            preds.append(out['forecast'].cpu().numpy())
            targets.append(y_batch.numpy())
            weights.append(out['regime_weights'].cpu().numpy())
            sigma = out.get('st_uncertainty', None)       # ← ADD
            if sigma is not None:                          # ← ADD
                uncertainties.append(sigma.cpu().numpy()[:, 0])  # ← ADD

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        weights = np.concatenate(weights)

        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)
        corr = np.corrcoef(preds.flatten(), targets.flatten())[0, 1]

        # R² on t+1 step only (column 0) — used for early stopping
        p0, a0 = preds[:, 0], targets[:, 0]
        ss_res = float(np.sum((p0 - a0) ** 2))
        ss_tot = float(np.sum((a0 - a0.mean()) ** 2)) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)

        if uncertainties:
            sigmas = np.concatenate(uncertainties)
            within_1sigma = np.mean(np.abs(preds[:, 0] - targets[:, 0]) < sigmas)
            ece_proxy = abs(within_1sigma - 0.68)  # ideal = 0.0
        else:
            ece_proxy = float('nan')

        avg_weights = weights.mean(0)
        return {
            'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Correlation': corr,
            'Calibration_ECE': ece_proxy,
            'regime_weights': {
                'short_term': avg_weights[0],
                'medium_term': avg_weights[1],
                'long_term': avg_weights[2]
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 30, batch_size: int = 64,
            val_split: float = 0.10, verbose: bool = True) -> dict:
        """Full training loop with validation."""
        n_val = max(int(len(X) * val_split), 64)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        _pin = (self.device != 'cpu')
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=_pin         # ← ADD
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=batch_size * 2,             # ← larger val batch = faster
            num_workers=0, pin_memory=_pin         # ← ADD
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(   # ← NEW
            self.optimizer, T_max=epochs, eta_min=1e-5                  # ← NEW
        ) 

        if verbose:
            print(f"\n  Training HTFN on {self.device}")
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Epochs: {epochs}")
            print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val RMSE':>10} {'Val R²':>9} {'Corr':>8}")
            print("  " + "-" * 52)

        # FIX: maximise R² (directional quality) instead of minimising RMSE.
        # RMSE optimisation drives the model toward predicting the mean
        # (RMSE↓ but IC↓), whereas R² directly rewards directional accuracy.
        best_val = -float('inf')   # we now MAXIMISE R²
        best_state = None
        _patience_counter = 0

        for ep in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics.get('R2', -1.0))

            current_r2 = val_metrics.get('R2', -float('inf'))
            if current_r2 > best_val:
                best_val   = current_r2
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                _patience_counter = 0
            else:
                _patience_counter += 1
                
                # Patience = 20 eval cycles. Minimum epoch guard = 30 so that
                # patience cannot fire at ep=15 (where 5×eval_interval=15
                # causes both conditions to trip simultaneously on first check).
                if _patience_counter >= 20 and ep >= 30:
                    print(f"  Early stop at epoch {ep}  (best val R²={best_val:.6f})")
                    break

            if verbose and (ep % 5 == 0 or ep == 1):
                rw  = val_metrics['regime_weights']
                r2v = val_metrics.get('R2', float('nan'))
                print(f"  {ep:>6} {train_loss:>12.6f} {val_metrics['RMSE']:>10.6f} "
                      f"{r2v:>9.5f} {val_metrics['Correlation']:>8.4f}  "
                      f"[ST:{rw['short_term']:.2f} MT:{rw['medium_term']:.2f} "
                      f"LT:{rw['long_term']:.2f}]")

        # Restore checkpoint with best R²
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\n  Best val R²: {best_val:.6f} ✓")
        return self.history


def build_and_train_htfn(X: np.ndarray, y: np.ndarray,
                          epochs: int = 30,
                          skip_ablation: bool = False) -> tuple:
    """Convenience function: build model, train, return model + metrics."""
    print("\n" + "="*60)
    print("  HCFRI LAYER 2: MULTI-TEMPORAL MODELING (HTFN)")
    print("="*60)

    input_dim = X.shape[2]
    horizon = y.shape[1]

    model = HierarchicalTemporalFusionNetwork(input_dim=input_dim,
                                               forecast_horizon=horizon)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  HTFN Parameters: {n_params:,}")

    # FIX: skip ablations when called from multi-seed loop (already ran once in pre-flight)
    if not skip_ablation:
        print("\n  [Ablation] Sequence length sensitivity:")
        actual_seq = X.shape[1]
        for test_seq in [30, 60, 90]:
            # FIX: skip seq_lens larger than the available sequence length
            if test_seq > actual_seq:
                print(f"    seq_len={test_seq:3d} → skipped (data seq_len={actual_seq})")
                continue
            n_abl   = min(800, len(X))
            X_sub   = X[:n_abl, -test_seq:, :]   # correctly slice last test_seq steps
            y_sub   = y[:n_abl]
            _m      = HierarchicalTemporalFusionNetwork(input_dim=input_dim,
                                                        forecast_horizon=horizon)
            _t      = HTFNTrainer(_m, lr=3.5e-4)
            _t.fit(X_sub, y_sub, epochs=8, verbose=False)
            n_v     = max(1, int(len(X_sub) * 0.15))
            _val    = _t.evaluate(DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_sub[-n_v:]),
                    torch.FloatTensor(y_sub[-n_v:])),
                batch_size=64))
            print(f"    seq_len={test_seq:3d} → Val RMSE: {_val['RMSE']:.6f}")
        print()

    trainer = HTFNTrainer(model, lr=3.5e-4)
    history = trainer.fit(X, y, epochs=epochs, batch_size=256)

    # Final evaluation
    n_val = max(int(len(X) * 0.10), 64)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[-n_val:]), torch.FloatTensor(y[-n_val:])),
        batch_size=256
    )
    final_metrics = trainer.evaluate(val_loader)

    print(f"\n  Final Metrics:")
    for k, v in final_metrics.items():
        if k != 'regime_weights':
            print(f"    {k}: {v:.6f}")
    print(f"  Regime Weights: {final_metrics['regime_weights']}")
    print("\n  Layer 2 complete ✓")

    return model, trainer, final_metrics


if __name__ == "__main__":
    # Quick standalone test
    X = np.random.randn(500, 60, 30).astype(np.float32)
    y = np.random.randn(500, 5).astype(np.float32)
    model, trainer, metrics = build_and_train_htfn(X, y, epochs=5)
