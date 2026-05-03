from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import torch

def evaluate_model_classification(full_results):


    df = full_results.copy()
    
    # Aseguramos que los tipos sean correctos
    y_true = df['actual'].astype(int)
    y_pred = df['pred'].astype(int)
    
    # 1. Reporte detallado
    # Target names alineados con tu mapeo: 0: SELL, 1: HOLD, 2: BUY
    print("📊 --- CLASIFICACIÓN POR CLASE ---")
    print(classification_report(y_true, y_pred, target_names=['SELL (0)', 'HOLD (1)', 'BUY (2)'], zero_division=0))

    # 2. Matriz de Confusión Normalizada (Para ver porcentajes de acierto por clase)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true') # % sobre el total de cada clase real

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matriz de valores absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], 
                xticklabels=['SELL', 'HOLD', 'BUY'], yticklabels=['SELL', 'HOLD', 'BUY'])
    ax[0].set_title("Confusion Matrix (Conteos)")
    
    # Matriz normalizada (Importante para ver el sesgo)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax[1],
                xticklabels=['SELL', 'HOLD', 'BUY'], yticklabels=['SELL', 'HOLD', 'BUY'])
    ax[1].set_title("Confusion Matrix (Recall por Clase)")
    
    plt.tight_layout()
    plt.show()

    # 3. F1-Score Macro (Métrica clave para clases desbalanceadas)
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n⭐ Métricas Globales:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}  <-- Si es < 0.33, el modelo es puro azar")

    return {"accuracy": acc, "f1_macro": f1}

def tune_threshold_wf(
    all_step_results,
    move_thresholds=np.arange(0.50, 0.95, 0.05),
    dir_thresholds=list(np.arange(0.50, 0.80, 0.05)) + [None],  # None = ganador por prob
    train_split=0.7,
    metric='f1_macro',   # 'f1_macro' | 'f1_sell' | 'f1_buy' | 'precision_sell'
):
    from sklearn.metrics import f1_score, precision_score
    from itertools import product

    n_folds = len(all_step_results)
    split   = int(n_folds * train_split)

    df_tr = pd.concat(all_step_results[:split]).reset_index(drop=True)
    df_va = pd.concat(all_step_results[split:]).reset_index(drop=True)

    def apply_thresholds(df, tm, td):
        prob_move = df['prob_move'].values
        prob_buy  = df['prob_buy'].values
        prob_sell = df['prob_sell'].values

        preds = np.ones(len(df), dtype=int)   # HOLD por defecto
        move_mask = prob_move > tm

        if td is None:
            dir_pred = np.where(prob_buy > prob_sell, 2, 0)
            preds[move_mask] = dir_pred[move_mask]
        else:
            buy_conf  = prob_buy  > td
            sell_conf = prob_sell > td
            conflict  = buy_conf & sell_conf   # ambos altos → HOLD

            dir_pred = np.full(len(df), -1)
            dir_pred[buy_conf  & ~conflict] = 2
            dir_pred[sell_conf & ~conflict] = 0

            apply_mask = move_mask & (dir_pred != -1)
            preds[apply_mask] = dir_pred[apply_mask]

        return preds

    def score(y_true, y_pred):
        if metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'f1_sell':
            return f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[0]
        elif metric == 'f1_buy':
            return f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[2]
        elif metric == 'precision_sell':
            return precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[0]

    y_tr = df_tr['actual'].astype(int).values
    y_va = df_va['actual'].astype(int).values

    best_score, best_tm, best_td = 0, 0.7, None
    results_grid = []

    for tm, td in product(move_thresholds, dir_thresholds):
        preds_tr = apply_thresholds(df_tr, tm, td)
        s = score(y_tr, preds_tr)
        results_grid.append({'tm': round(float(tm), 2), 'td': td, 'score_train': round(s, 4)})
        if s > best_score:
            best_score, best_tm, best_td = s, tm, td

    # Validar en folds reservados
    preds_va  = apply_thresholds(df_va, best_tm, best_td)
    score_val = score(y_va, preds_va)
    degradation = best_score - score_val

    print(f"Métrica optimizada: {metric}")
    print(f"Threshold óptimo → move={best_tm:.2f}, dir={best_td}")
    print(f"  {metric} train folds: {best_score:.4f}")
    print(f"  {metric} val   folds: {score_val:.4f}")
    print(f"  Degradación:          {degradation:.4f} {'⚠️ posible overfit' if degradation > 0.02 else '✅ estable'}")

    # Top 10 combinaciones en train para inspección
    df_grid = pd.DataFrame(results_grid).sort_values('score_train', ascending=False)
    print(f"\nTop 10 combinaciones (train):\n{df_grid.head(10).to_string(index=False)}")

    return {'threshold_move': best_tm, 'threshold_dir': best_td, 'score_train': best_score, 'score_val': score_val}

def save_trading_model_2level(model_l1, model_l2, scaler, features_list, model_params, best_thresholds=None, path="trading_model_2level"):
    """Guarda los dos modelos del pipeline de dos niveles."""
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Modelos
    torch.save(model_l1.state_dict(), os.path.join(path, "model_l1_weights.pth"))
    torch.save(model_l2.state_dict(), os.path.join(path, "model_l2_weights.pth"))
    
    # Scaler, features y params
    joblib.dump(scaler,        os.path.join(path, "scaler.pkl"))
    joblib.dump(features_list, os.path.join(path, "features.pkl"))
    joblib.dump(model_params,  os.path.join(path, "model_params.pkl"))
    
    # Umbral óptimo
    if best_thresholds is not None:
        joblib.dump(best_thresholds, os.path.join(path, "best_thresholds.pkl"))
    
    print(f"✅ Modelos L1 y L2 guardados en: {path}")


def load_trading_model_2level(model_class_l1, model_class_l2, path="trading_model_2level", device="cpu"):
    """Carga los dos modelos del pipeline de dos niveles."""
    # Metadatos
    model_params  = joblib.load(os.path.join(path, "model_params.pkl"))
    features_list = joblib.load(os.path.join(path, "features.pkl"))
    scaler        = joblib.load(os.path.join(path, "scaler.pkl"))
    best_thresholds = joblib.load(os.path.join(path, "best_thresholds.pkl")) if os.path.exists(os.path.join(path, "best_thresholds.pkl")) else None

    # L1
    model_l1 = model_class_l1(**model_params)
    model_l1.load_state_dict(torch.load(os.path.join(path, "model_l1_weights.pth"), map_location=device))
    model_l1.to(device)
    model_l1.eval()
    
    # L2
    model_l2 = model_class_l2(**model_params)
    model_l2.load_state_dict(torch.load(os.path.join(path, "model_l2_weights.pth"), map_location=device))
    model_l2.to(device)
    model_l2.eval()
    
    print(f"🚀 Pipeline 2-Level listo. Inputs: {model_params['input_dim']} features.")
    return model_l1, model_l2, scaler, features_list, best_thresholds