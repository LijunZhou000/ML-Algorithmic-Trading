"""
feature_selection.py  &  scoring.py
=====================================
Feature selection (SHAP + Permutation Importance) y scoring completo
para los clasificadores LSTM/GRU de futuros.

Dependencias:
    pip install shap torch scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Wrapper PyTorch → interfaz sklearn-like para SHAP
# ─────────────────────────────────────────────────────────────────────────────

class _RNNPredictor:
    """
    Wrapper que SHAP puede llamar como si fuera una función numpy.
    Recibe X con shape (n, lookback * n_features) — aplanado —
    y devuelve probabilidades (n,).
    """
    def __init__(self, model, lookback: int, n_features: int, device: torch.device):
        self.model     = model
        self.lookback  = lookback
        self.n_features= n_features
        self.device    = device

    def __call__(self, X_flat: np.ndarray) -> np.ndarray:
        X_3d = X_flat.reshape(-1, self.lookback, self.n_features).astype(np.float32)
        tensor = torch.from_numpy(X_3d).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor)).cpu().numpy()
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  SHAP sobre un modelo entrenado
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(
    model,
    X_background : np.ndarray,   # (n_bg,  lookback, n_features)
    X_explain    : np.ndarray,   # (n_exp, lookback, n_features)
    feature_names: list[str],
    device       : torch.device,
    n_background : int  = 100,   # muestras de fondo para KernelExplainer
    n_explain    : int  = 200,   # muestras a explicar
    seed         : int  = 42,
) -> pd.DataFrame:
    """
    Calcula importancia media |SHAP| por feature (promediando el eje temporal).

    Returns
    -------
    pd.DataFrame  columnas: ['feature', 'shap_importance']
                  ordenado de mayor a menor
    """
    rng = np.random.default_rng(seed)
    lookback   = X_background.shape[1]
    n_features = X_background.shape[2]

    # Submuestrear para velocidad
    bg_idx  = rng.choice(len(X_background), min(n_background, len(X_background)), replace=False)
    exp_idx = rng.choice(len(X_explain),    min(n_explain,    len(X_explain)),    replace=False)

    X_bg_flat  = X_background[bg_idx].reshape(len(bg_idx), -1)
    X_exp_flat = X_explain[exp_idx].reshape(len(exp_idx), -1)

    predictor = _RNNPredictor(model, lookback, n_features, device)

    print(f"  [SHAP] KernelExplainer: {len(bg_idx)} bg, {len(exp_idx)} samples...")
    explainer   = shap.KernelExplainer(predictor, X_bg_flat)
    shap_values = explainer.shap_values(X_exp_flat, silent=True)  # (n_exp, lookback*n_feat)

    # shap_values tiene shape (n_exp, lookback * n_features)
    # Reshape a (n_exp, lookback, n_features) y promediamos sobre el eje lookback
    shap_3d    = shap_values.reshape(len(exp_idx), lookback, n_features)
    mean_abs   = np.abs(shap_3d).mean(axis=(0, 1))   # (n_features,)

    df_imp = pd.DataFrame({
        "feature"         : feature_names,
        "shap_importance" : mean_abs,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Permutation Importance (más rápido, buena segunda opinión)
# ─────────────────────────────────────────────────────────────────────────────

def compute_permutation_importance(
    model        ,
    X            : np.ndarray,   # (n, lookback, n_features)
    y            : np.ndarray,
    feature_names: list[str],
    device       : torch.device,
    n_repeats    : int  = 10,
    metric       : str  = "auc",   # "auc" | "accuracy" | "f1"
    batch_size   : int  = 256,
    seed         : int  = 42,
) -> pd.DataFrame:
    """
    Para cada feature: permuta sus valores a lo largo del eje temporal,
    mide cuánto cae el metric. Mayor caída = feature más importante.

    Returns
    -------
    pd.DataFrame  columnas: ['feature', 'perm_importance', 'perm_std']
                  ordenado de mayor a menor
    """
    rng = np.random.default_rng(seed)
    model.eval()

    def _score(X_in):
        ds     = TensorDataset(torch.from_numpy(X_in).float())
        loader = DataLoader(ds, batch_size=batch_size)
        probs  = []
        with torch.no_grad():
            for (Xb,) in loader:
                probs.extend(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
        probs = np.array(probs)
        preds = (probs >= 0.5).astype(int)
        if metric == "auc":
            return roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
        elif metric == "f1":
            return f1_score(y, preds, zero_division=0)
        else:
            return accuracy_score(y, preds)

    baseline = _score(X)
    results  = []

    for fi, fname in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, :, fi] = X_perm[perm_idx, :, fi]
            deltas.append(baseline - _score(X_perm))
        results.append({
            "feature"        : fname,
            "perm_importance": np.mean(deltas),
            "perm_std"       : np.std(deltas),
        })

    df_imp = pd.DataFrame(results).sort_values(
        "perm_importance", ascending=False
    ).reset_index(drop=True)

    return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  Pipeline completo de selección
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    model,
    X_train      : np.ndarray,
    y_train      : np.ndarray,
    X_test       : np.ndarray,
    feature_names: list[str],
    device       : torch.device,
    top_n        : int   = 25,
    method       : str   = "both",      # "shap" | "perm" | "both"
    shap_weight  : float = 0.6,         # peso de SHAP en el ranking combinado
    plot         : bool  = True,
    # kwargs SHAP
    n_background : int   = 100,
    n_explain    : int   = 200,
    # kwargs Perm
    n_repeats    : int   = 10,
    perm_metric  : str   = "auc",
) -> dict:
    """
    Wrapper principal de selección de features.

    Returns
    -------
    {
        "top_features"  : list[str],          # top_n seleccionadas
        "ranking"       : pd.DataFrame,       # tabla completa con scores
        "shap_df"       : pd.DataFrame | None,
        "perm_df"       : pd.DataFrame | None,
    }
    """
    shap_df = perm_df = None

    if method in ("shap", "both"):
        print("[Feature Selection] Calculando SHAP importance...")
        shap_df = compute_shap_importance(
            model, X_train, X_test, feature_names, device,
            n_background=n_background, n_explain=n_explain,
        )

    if method in ("perm", "both"):
        print("[Feature Selection] Calculando Permutation importance...")
        perm_df = compute_permutation_importance(
            model, X_test, y_test if 'y_test' in dir() else y_train,
            feature_names, device, n_repeats=n_repeats, metric=perm_metric,
        )

    # ── Ranking combinado ──────────────────────────────────────────────────
    all_features = pd.DataFrame({"feature": feature_names})

    if shap_df is not None:
        # normalizar 0-1
        mx = shap_df["shap_importance"].max() + 1e-12
        shap_df["shap_norm"] = shap_df["shap_importance"] / mx
        all_features = all_features.merge(
            shap_df[["feature", "shap_importance", "shap_norm"]], on="feature", how="left"
        )

    if perm_df is not None:
        mn = perm_df["perm_importance"].min()
        mx = perm_df["perm_importance"].max() - mn + 1e-12
        perm_df["perm_norm"] = (perm_df["perm_importance"] - mn) / mx
        all_features = all_features.merge(
            perm_df[["feature", "perm_importance", "perm_norm"]], on="feature", how="left"
        )

    # Score combinado
    if method == "both":
        all_features["combined_score"] = (
            shap_weight       * all_features["shap_norm"].fillna(0) +
            (1 - shap_weight) * all_features["perm_norm"].fillna(0)
        )
        sort_col = "combined_score"
    elif method == "shap":
        all_features["combined_score"] = all_features["shap_norm"].fillna(0)
        sort_col = "shap_norm"
    else:
        all_features["combined_score"] = all_features["perm_norm"].fillna(0)
        sort_col = "perm_norm"

    ranking      = all_features.sort_values("combined_score", ascending=False).reset_index(drop=True)
    top_features = ranking["feature"].head(top_n).tolist()

    print(f"\n[Feature Selection] Top {top_n} seleccionadas de {len(feature_names)}:")
    print(ranking[["feature", "combined_score"]].head(top_n).to_string(index=False))

    if plot:
        _plot_feature_importance(ranking, top_n, method)

    return {
        "top_features": top_features,
        "ranking"     : ranking,
        "shap_df"     : shap_df,
        "perm_df"     : perm_df,
    }


def _plot_feature_importance(ranking: pd.DataFrame, top_n: int, method: str):
    top = ranking.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))[::-1]
    ax.barh(top["feature"][::-1], top["combined_score"][::-1], color=colors[::-1])
    ax.set_xlabel("Combined Importance Score")
    ax.set_title(f"Top {top_n} Features  [{method.upper()}]")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — SCORING COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def score_predictions(
    y_true       : np.ndarray,
    y_pred       : np.ndarray,    # etiquetas binarias
    y_prob       : np.ndarray,    # probabilidades clase 1
    returns      : Optional[np.ndarray] = None,  # retornos reales (%)
    threshold    : float = 0.5,
    annual_factor: float = 252.0,  # 252 días | 52 semanas | 12 meses
    rf_rate      : float = 0.0,    # tasa libre de riesgo anualizada
    verbose      : bool  = True,
    plot         : bool  = True,
    label        : str   = "Model",
) -> dict:
    """
    Métricas de clasificación + métricas financieras.

    Parámetros
    ----------
    y_true        : etiquetas reales  (0/1)
    y_pred        : predicciones binarias
    y_prob        : probabilidad de clase 1
    returns       : retornos reales del activo por barra (en fracción, ej 0.002)
                    Si se pasan, se calculan métricas de trading aplicando la señal.
                    Si es clasificador de dirección: +1 si y_pred=1, -1 si y_pred=0
                    Si es clasificador de movimiento: no aplica retornos directamente
    threshold     : umbral de clasificación (permite ajustar precision/recall)
    annual_factor : factor de anualización
    rf_rate       : risk-free anualizado para Sharpe
    verbose       : imprime tabla
    plot          : gráfico de curva equity + distribución de retornos

    Returns
    -------
    dict con todas las métricas
    """
    # ── 1. Re-clasificar con threshold personalizado ───────────────────────
    y_pred_thr = (y_prob >= threshold).astype(int)

    # ── 2. Métricas de clasificación ──────────────────────────────────────
    cm      = confusion_matrix(y_true, y_pred_thr)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])

    clf_metrics = {
        "accuracy"   : accuracy_score(y_true, y_pred_thr),
        "f1"         : f1_score(y_true, y_pred_thr, zero_division=0),
        "precision"  : precision_score(y_true, y_pred_thr, zero_division=0),
        "recall"     : recall_score(y_true, y_pred_thr, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "auc_roc"    : roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "log_loss"   : log_loss(y_true, y_prob),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # ── 3. Métricas de trading (si se pasan retornos) ─────────────────────
    trading_metrics = {}
    equity_curve    = None

    if returns is not None:
        returns = np.array(returns)
        # señal: +1 long, -1 short (para clasificador de dirección)
        signal      = np.where(y_pred_thr == 1, 1.0, -1.0)
        strat_ret   = signal * returns

        # win rate
        wins        = (strat_ret > 0).sum()
        losses      = (strat_ret < 0).sum()
        total_trades= wins + losses

        # profit factor
        gross_profit = strat_ret[strat_ret > 0].sum()
        gross_loss   = np.abs(strat_ret[strat_ret < 0].sum()) + 1e-12
        profit_factor= gross_profit / gross_loss

        # Sharpe ratio anualizado
        mean_ret    = strat_ret.mean()
        std_ret     = strat_ret.std() + 1e-12
        rf_per_bar  = rf_rate / annual_factor
        sharpe      = (mean_ret - rf_per_bar) / std_ret * np.sqrt(annual_factor)

        # Sortino ratio (penaliza solo downside)
        downside    = strat_ret[strat_ret < rf_per_bar]
        sortino_std = downside.std() + 1e-12
        sortino     = (mean_ret - rf_per_bar) / sortino_std * np.sqrt(annual_factor)

        # Calmar ratio
        equity_curve = (1 + strat_ret).cumprod()
        running_max  = np.maximum.accumulate(equity_curve)
        drawdown     = (equity_curve - running_max) / running_max
        max_dd       = drawdown.min()
        calmar       = (mean_ret * annual_factor) / (abs(max_dd) + 1e-12)

        # Average win / loss
        avg_win  = strat_ret[strat_ret > 0].mean() if wins > 0 else 0.0
        avg_loss = strat_ret[strat_ret < 0].mean() if losses > 0 else 0.0
        edge_ratio= abs(avg_win / (avg_loss + 1e-12))

        # Expectancy por operación
        win_rate   = wins / (total_trades + 1e-12)
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Max consecutive losses
        is_loss = (strat_ret < 0).astype(int)
        max_consec_loss = _max_consecutive(is_loss)

        trading_metrics = {
            "win_rate"          : win_rate,
            "profit_factor"     : profit_factor,
            "sharpe_ratio"      : sharpe,
            "sortino_ratio"     : sortino,
            "calmar_ratio"      : calmar,
            "max_drawdown_pct"  : max_dd * 100,
            "avg_win_pct"       : avg_win * 100,
            "avg_loss_pct"      : avg_loss * 100,
            "edge_ratio"        : edge_ratio,
            "expectancy_pct"    : expectancy * 100,
            "max_consec_losses" : max_consec_loss,
            "total_trades"      : int(total_trades),
            "total_return_pct"  : (equity_curve[-1] - 1) * 100,
        }

    # ── 4. Output ─────────────────────────────────────────────────────────
    all_metrics = {**clf_metrics, **trading_metrics}

    if verbose:
        _print_score_table(clf_metrics, trading_metrics, label, threshold)

    if plot:
        _plot_scoring(y_true, y_prob, equity_curve, strat_ret if returns is not None else None, label)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _max_consecutive(arr: np.ndarray) -> int:
    max_c = cur = 0
    for v in arr:
        cur = cur + 1 if v else 0
        max_c = max(max_c, cur)
    return max_c


def _print_score_table(clf: dict, trd: dict, label: str, thr: float):
    sep = "─" * 46
    print(f"\n{'═'*46}")
    print(f"  SCORING REPORT  ·  {label}  ·  threshold={thr:.2f}")
    print(f"{'═'*46}")
    print(f"\n  Classification")
    print(sep)
    for k, v in clf.items():
        if k in ("tp","tn","fp","fn"):
            continue
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<20} {vstr:>10}")
    print(f"  Confusion: TP={clf['tp']} TN={clf['tn']} FP={clf['fp']} FN={clf['fn']}")

    if trd:
        print(f"\n  Trading")
        print(sep)
        for k, v in trd.items():
            vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<25} {vstr:>10}")
    print(f"{'═'*46}\n")


def _plot_scoring(y_true, y_prob, equity_curve, strat_ret, label):
    n_plots = 1 + (equity_curve is not None) + (strat_ret is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # ── Calibración de probabilidades ─────────────────────────────────────
    ax = axes[0]
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, fraction_pos, "o-", label="Model", color="#2563eb")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.set_title(f"Calibration Curve  [{label}]")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    ax.grid(alpha=0.3)

    idx = 1
    if equity_curve is not None:
        ax = axes[idx]; idx += 1
        ax.plot(equity_curve, color="#16a34a", lw=1.2)
        ax.axhline(1, color="gray", lw=0.8, ls="--")
        running_max = np.maximum.accumulate(equity_curve)
        ax.fill_between(range(len(equity_curve)), equity_curve, running_max,
                        alpha=0.2, color="#dc2626")
        ax.set_title(f"Equity Curve  [{label}]")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Cumulative Return")
        ax.grid(alpha=0.3)

    if strat_ret is not None:
        ax = axes[idx]
        ax.hist(strat_ret * 100, bins=50, color="#7c3aed", alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(np.mean(strat_ret) * 100, color="#f59e0b", lw=1.5, ls="--", label="Mean")
        ax.set_title(f"Return Distribution  [{label}]")
        ax.set_xlabel("Return per Bar (%)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"scoring_{label.replace(' ','_')}.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Comparador de dos configuraciones (150 features vs top-N)
# ─────────────────────────────────────────────────────────────────────────────

def compare_feature_sets(
    results_full : dict,
    results_top  : dict,
    label_full   : str = "All features",
    label_top    : str = "Top-N features",
) -> pd.DataFrame:
    """
    Compara métricas de dos resultados de walk_forward_train.
    Útil para validar si la selección de features mejora el modelo.
    """
    keys = ["accuracy", "f1", "precision", "recall", "auc"]
    rows = []
    for k in keys:
        v_full = results_full["summary"].get(k, np.nan)
        v_top  = results_top["summary"].get(k, np.nan)
        delta  = v_top - v_full
        rows.append({
            "metric"  : k,
            label_full: round(v_full, 4),
            label_top : round(v_top,  4),
            "delta"   : round(delta,  4),
            "better"  : "✓ top-N" if delta > 0 else ("= tie" if delta == 0 else "✗ full"),
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — EJEMPLO DE USO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Importamos el módulo base (mismo directorio)
    from futures_rnn_classifier import (
        TrainConfig, walk_forward_train, create_lstm_dataset
    )

    # ── Datos sintéticos ──────────────────────────────────────────────────────
    np.random.seed(42)
    N = 6_000
    n_features_total = 40   # simula tus 150 (reducido para el ejemplo)

    feat_names = [f"feat_{i:03d}" for i in range(n_features_total)]
    data       = {f: np.random.randn(N) for f in feat_names}
    data["returns"]   = np.random.randn(N) * 0.005
    data["movement"]  = (np.abs(data["returns"]) > 0.004).astype(int)
    data["direction"] = (data["returns"] > 0).astype(int)
    df = pd.DataFrame(data)

    CFG = TrainConfig(
        rnn_type="lstm", hidden_size=32, num_layers=1,
        epochs=10, batch_size=64, patience=5,
        train_window=1200, val_size=150, test_size=150, step_size=150,
    )

    # ── PASO 1: Entrenar con todas las features ───────────────────────────────
    print("PASO 1 — Entrenamiento con TODAS las features")
    results_full = walk_forward_train(df, feat_names, "movement", CFG, verbose=True)

    # ── PASO 2: Feature selection sobre el último modelo del walk-forward ─────
    print("\nPASO 2 — Feature Selection (último fold)")
    last_model = results_full["models"][-1]
    device     = torch.device("cpu")

    # Preparamos X de los últimos 600 registros para SHAP/Perm
    df_fs = df.iloc[-600:].reset_index(drop=True)
    X_fs, y_fs = create_lstm_dataset(df_fs, feat_names, "movement", lookback=CFG.lookback)

    fs_result = select_features(
        model         = last_model,
        X_train       = X_fs[:300],
        y_train       = y_fs[:300],
        X_test        = X_fs[300:],
        feature_names = feat_names,
        device        = device,
        top_n         = 10,           # → ajusta a 25 con tus 150 features
        method        = "both",       # shap + permutation
        shap_weight   = 0.6,
        n_background  = 50,           # sube a 100 en prod
        n_explain     = 100,          # sube a 200 en prod
        n_repeats     = 5,
        plot          = True,
    )
    top_features = fs_result["top_features"]

    # ── PASO 3: Re-entrenar con top features ─────────────────────────────────
    print(f"\nPASO 3 — Re-entrenamiento con TOP {len(top_features)} features")
    results_top = walk_forward_train(df, top_features, "movement", CFG, verbose=True)

    # ── PASO 4: Comparar ──────────────────────────────────────────────────────
    print("\nPASO 4 — Comparación Full vs Top-N")
    compare_feature_sets(results_full, results_top,
                         label_full=f"All ({len(feat_names)})",
                         label_top=f"Top ({len(top_features)})")

    # ── PASO 5: Scoring financiero ────────────────────────────────────────────
    print("\nPASO 5 — Scoring financiero")
    real_returns = df["returns"].values[-(len(results_top["all_true"])):]

    metrics = score_predictions(
        y_true       = results_top["all_true"],
        y_pred       = results_top["all_preds"],
        y_prob       = results_top["all_probs"],
        returns      = real_returns,
        threshold    = 0.5,
        annual_factor= 252,
        rf_rate      = 0.04,
        verbose      = True,
        plot         = True,
        label        = "Movement LSTM",
    )