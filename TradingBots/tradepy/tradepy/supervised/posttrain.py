from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import torch
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error, log_loss
import torch.nn as nn

def evaluate_model_classification(full_results, show=True, save=False, save_path="classification_report.png"):

    df = full_results.copy()
    
    # Aseguramos que los tipos sean correctos
    y_true = df['actual'].astype(int)
    y_pred = df['final_pred'].astype(int)

    # 1. Reporte detallado
    print("📊 --- CLASIFICACIÓN POR CLASE ---")
    print(classification_report(
        y_true, y_pred,
        target_names=['SELL (0)', 'HOLD (1)', 'BUY (2)'],
        zero_division=0
    ))

    # 2. Matrices de confusión
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

    # 3. F1 por clase
    f1_classes = f1_score(y_true, y_pred, average=None)
    f1_matrix = f1_classes.reshape(1, -1)

    # 4. Métricas por clase (para barplot)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics_df = pd.DataFrame(report).T.loc[['0','1','2'], ['precision','recall','f1-score']]
    metrics_df.index = ['SELL','HOLD','BUY']

    # --- FIGURA 2x2 ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # --- Plot 1: Confusion Matrix (conteos)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0,0],
                xticklabels=['SELL','HOLD','BUY'], yticklabels=['SELL','HOLD','BUY'])
    ax[0,0].set_title("Confusion Matrix (Conteos)")

    # --- Plot 2: Confusion Matrix Normalizada
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax[0,1],
                xticklabels=['SELL','HOLD','BUY'], yticklabels=['SELL','HOLD','BUY'])
    ax[0,1].set_title("Confusion Matrix (Recall por Clase)")

    # --- Plot 3: F1 por clase
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='Purples', ax=ax[1,0],
                xticklabels=['SELL','HOLD','BUY'], yticklabels=['F1-score'])
    ax[1,0].set_title("F1-score por Clase")

    # --- Plot 4: Precision / Recall / F1 por clase
    metrics_df.plot(kind='bar', ax=ax[1,1], colormap='viridis')
    ax[1,1].set_ylim(0, 1)
    ax[1,1].set_title("Métricas por Clase")
    ax[1,1].legend(loc='lower right')

    plt.tight_layout()

    # Guardar si save=True
    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")

    # Mostrar si show=True
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 5. Métricas globales
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    print(f"\n⭐ Métricas Globales:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}  <-- Si es < 0.33, el modelo es puro azar")

    # return {"accuracy": acc, "f1_macro": f1_macro}
    return {
        "classification": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_per_class": {
                "SELL": float(f1_classes[0]),
                "HOLD": float(f1_classes[1]),
                "BUY": float(f1_classes[2])
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "per_class_metrics": metrics_df.to_dict()
        }
    }



def evaluate_model_classification_3level(full_results):
    """
    Evalúa el modelo 3-level completo: L1 (movimiento), L2 (dirección), L3 (log_return)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    df = full_results.copy()
    
    y_true = df['actual'].astype(int)
    y_pred = df['pred'].astype(int)
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN COMPLETA DEL MODELO 3-LEVEL")
    print("="*80)
    
    # ===== NIVEL 1: MOVIMIENTO (0/1) =====
    print("\n🔹 NIVEL 1 - MOVIMIENTO (¿Hay acción?)")
    print("-" * 50)
    y_l1_true = (y_true != 1).astype(int)  # Reconstruir: 0=HOLD, 1=ACCIÓN
    y_l1_pred = (y_pred != 1).astype(int)
    
    from sklearn.metrics import precision_score, recall_score
    l1_prec = precision_score(y_l1_true, y_l1_pred, zero_division=0)
    l1_rec = recall_score(y_l1_true, y_l1_pred, zero_division=0)
    l1_f1 = f1_score(y_l1_true, y_l1_pred, zero_division=0)
    
    print(f"Precision (detectar movimiento): {l1_prec:.4f}")
    print(f"Recall (no perder movimientos):  {l1_rec:.4f}")
    print(f"F1-Score:                        {l1_f1:.4f}")
    
    # ===== NIVEL 2: DIRECCIÓN (0/1 = SELL/BUY, solo donde L1=1) =====
    print("\n🔹 NIVEL 2 - DIRECCIÓN (¿SELL o BUY? | Solo si L1=1)")
    print("-" * 50)
    
    # Filtrar solo donde hay movimiento real (y_true != 1)
    mask_l2 = (y_true != 1)
    if mask_l2.sum() > 0:
        y_l2_true_all = np.where(y_true == 2, 1, 0)  # 0=SELL(0), 1=BUY(2)
        y_l2_pred_all = np.where(y_pred == 2, 1, 0)
        
        y_l2_true = y_l2_true_all[mask_l2]
        y_l2_pred = y_l2_pred_all[mask_l2]
        
        l2_prec = precision_score(y_l2_true, y_l2_pred, zero_division=0)
        l2_rec = recall_score(y_l2_true, y_l2_pred, zero_division=0)
        l2_f1 = f1_score(y_l2_true, y_l2_pred, zero_division=0)
        
        print(f"Muestras con movimiento real: {mask_l2.sum()} / {len(y_true)}")
        print(f"Precision (acertar dirección): {l2_prec:.4f}")
        print(f"Recall (no perder BUYs):       {l2_rec:.4f}")
        print(f"F1-Score:                      {l2_f1:.4f}")
    else:
        print("⚠️ No hay movimientos en los datos")
    
    # ===== NIVEL 3: LOG RETURN =====
    if 'pred_logret' in df.columns:
        print("\n🔹 NIVEL 3 - PREDICCIÓN DE LOG RETURN (magnitud)")
        print("-" * 50)
        
        pred_logret = df['pred_logret'].values
        
        # Asumir que tenemos ground truth de log_return si no, usar NA
        if 'actual_logret' in df.columns:
            actual_logret = df['actual_logret'].values
            valid_idx = ~np.isnan(actual_logret)
            
            if valid_idx.sum() > 0:
                mae = mean_absolute_error(actual_logret[valid_idx], pred_logret[valid_idx])
                rmse = np.sqrt(mean_squared_error(actual_logret[valid_idx], pred_logret[valid_idx]))
                
                print(f"Muestras con log_return: {valid_idx.sum()} / {len(df)}")
                print(f"MAE (Log Return):        {mae:.6f}")
                print(f"RMSE (Log Return):       {rmse:.6f}")
                print(f"Media de predicciones:   {pred_logret[valid_idx].mean():.6f}")
                print(f"Std de predicciones:     {pred_logret[valid_idx].std():.6f}")
        else:
            print(f"Predicciones de log_return (media):  {pred_logret.mean():.6f}")
            print(f"Predicciones de log_return (std):    {pred_logret.std():.6f}")
            print(f"Predicciones de log_return (min):    {pred_logret.min():.6f}")
            print(f"Predicciones de log_return (max):    {pred_logret.max():.6f}")
    
    # ===== RESUMEN GLOBAL =====
    print("\n" + "="*80)
    print("⭐ RESUMEN GLOBAL")
    print("="*80)
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy Overall:          {acc:.4f}")
    print(f"F1-Score Macro (0/1/2):    {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], 
                xticklabels=['SELL(0)', 'HOLD(1)', 'BUY(2)'], 
                yticklabels=['SELL(0)', 'HOLD(1)', 'BUY(2)'])
    ax[0].set_title("Confusion Matrix (Conteos Absolutos)")
    ax[0].set_ylabel("Real")
    ax[0].set_xlabel("Predicción")
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax[1],
                xticklabels=['SELL(0)', 'HOLD(1)', 'BUY(2)'], 
                yticklabels=['SELL(0)', 'HOLD(1)', 'BUY(2)'])
    ax[1].set_title("Confusion Matrix (Recall Normalizado)")
    ax[1].set_ylabel("Real")
    ax[1].set_xlabel("Predicción")
    
    plt.tight_layout()
    plt.show()
    
    # Classification Report completo
    print("\n📋 CLASSIFICATION REPORT DETALLADO:")
    print(classification_report(y_true, y_pred, 
                               target_names=['SELL (0)', 'HOLD (1)', 'BUY (2)'], 
                               zero_division=0))
    
    return {
        "accuracy": acc, 
        "f1_macro": f1_macro,
        "l1_f1": l1_f1,
        "l2_f1": l2_f1 if 'l2_f1' in locals() else None,
    }


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


def tune_threshold_wf_3level(
    all_step_results,
    move_thresholds=np.arange(0.50, 0.95, 0.05),
    dir_thresholds=list(np.arange(0.50, 0.80, 0.05)) + [None],
    logret_thresholds=None,  # None = no usar L3 en filtering; list = umbrales de magnitude
    train_split=0.7,
    metric='f1_macro',
    use_logret_weighting=False,  # Si True, ponderar confianza por |log_return|
):
    """
    Búsqueda de thresholds óptimos para 3-level considerando L3 (log_return).
    
    Args:
        all_step_results: List de DataFrames con predictions y prob_*
        move_thresholds: Array de thresholds para L1
        dir_thresholds: Array de thresholds para L2
        logret_thresholds: Array de magnitudes mínimas de log_return (None = ignorar)
        use_logret_weighting: Si True, ponderar pred por |pred_logret|
        metric: 'f1_macro', 'f1_sell', 'f1_buy', 'precision_sell'
    """
    from sklearn.metrics import f1_score, precision_score
    from itertools import product
    
    n_folds = len(all_step_results)
    split = int(n_folds * train_split)
    
    df_tr = pd.concat(all_step_results[:split]).reset_index(drop=True)
    df_va = pd.concat(all_step_results[split:]).reset_index(drop=True)
    
    # Default: no filtrar por log_return
    if logret_thresholds is None:
        logret_thresholds = [0.0]
    
    def apply_thresholds_3level(df, tm, td, tlr, use_weighting=False):
        """
        Aplicar thresholds considerando L3
        tm: threshold movement (L1)
        td: threshold direction (L2)
        tlr: threshold log_return magnitude (L3)
        """
        prob_move = df['prob_move'].values
        prob_buy = df['prob_buy'].values
        prob_sell = df['prob_sell'].values
        
        preds = np.ones(len(df), dtype=int)  # HOLD por defecto
        move_mask = prob_move > tm
        
        # Inicializar dirección predicha
        if td is None:
            dir_pred = np.where(prob_buy > prob_sell, 2, 0)
        else:
            buy_conf = prob_buy > td
            sell_conf = prob_sell > td
            conflict = buy_conf & sell_conf
            
            dir_pred = np.full(len(df), -1)
            dir_pred[buy_conf & ~conflict] = 2
            dir_pred[sell_conf & ~conflict] = 0
        
        # Filtrar por L3 (log_return magnitude)
        if 'pred_logret' in df.columns:
            logret_magnitude = np.abs(df['pred_logret'].values)
            logret_mask = logret_magnitude > tlr
        else:
            logret_mask = np.ones(len(df), dtype=bool)
        
        # Aplicar thresholds
        if td is None:
            apply_mask = move_mask & logret_mask
            preds[apply_mask] = dir_pred[apply_mask]
        else:
            apply_mask = move_mask & (dir_pred != -1) & logret_mask
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
    
    best_score, best_tm, best_td, best_tlr = 0, 0.7, None, 0.0
    results_grid = []
    
    print(f"\n🔍 Búsqueda de thresholds óptimos...")
    print(f"   Combinaciones: {len(move_thresholds)} × {len(dir_thresholds)} × {len(logret_thresholds)} = {len(move_thresholds) * len(dir_thresholds) * len(logret_thresholds)}")
    
    for tm, td, tlr in product(move_thresholds, dir_thresholds, logret_thresholds):
        preds_tr = apply_thresholds_3level(df_tr, tm, td, tlr)
        s = score(y_tr, preds_tr)
        
        results_grid.append({
            'tm': round(float(tm), 2),
            'td': td,
            'tlr': round(float(tlr), 4),
            'score_train': round(s, 4)
        })
        
        if s > best_score:
            best_score, best_tm, best_td, best_tlr = s, tm, td, tlr
    
    # Validar en folds reservados
    preds_va = apply_thresholds_3level(df_va, best_tm, best_td, best_tlr)
    score_val = score(y_va, preds_va)
    degradation = best_score - score_val
    
    print(f"\n✅ Thresholds óptimos encontrados:")
    print(f"   L1 (move):        {best_tm:.2f}")
    print(f"   L2 (direction):   {best_td}")
    print(f"   L3 (logret min):  {best_tlr:.4f}")
    print(f"\n📊 Scores:")
    print(f"   {metric} (train): {best_score:.4f}")
    print(f"   {metric} (val):   {score_val:.4f}")
    print(f"   Degradation:      {degradation:.4f} {'⚠️ posible overfit' if degradation > 0.02 else '✅ estable'}")
    
    # Top 10 en train
    df_grid = pd.DataFrame(results_grid).sort_values('score_train', ascending=False)
    print(f"\n🏆 Top 10 combinaciones (train):")
    print(df_grid.head(10).to_string(index=False))
    
    return {
        'threshold_move': best_tm,
        'threshold_dir': best_td,
        'threshold_logret': best_tlr,
        'score_train': best_score,
        'score_val': score_val,
        'degradation': degradation
    }


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


def save_trading_model_3level(model_l1, model_l2, model_l3, scaler, scaler_l3, features_list, model_params, res, best_thresholds=None, path="trading_model_3level"):
    """Guarda los tres modelos del pipeline de tres niveles."""
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Modelos
    torch.save(model_l1.state_dict(), os.path.join(path, "model_l1_weights.pth"))
    torch.save(model_l2.state_dict(), os.path.join(path, "model_l2_weights.pth"))
    torch.save(model_l3.state_dict(), os.path.join(path, "model_l3_weights.pth"))
    
    # Scaler, features y params
    joblib.dump(scaler,        os.path.join(path, "scaler.pkl"))
    joblib.dump(scaler_l3,     os.path.join(path, "scaler_l3.pkl"))
    joblib.dump(features_list, os.path.join(path, "features.pkl"))
    joblib.dump(model_params,  os.path.join(path, "model_params.pkl"))
    
    # Umbral óptimo
    if best_thresholds is not None:
        joblib.dump(best_thresholds, os.path.join(path, "best_thresholds.pkl"))
    
    parquet_path = os.path.join(path, "res.parquet")
    res.to_parquet(parquet_path, index=False)
    
    print(f"✅ Modelos L1 y L2 guardados en: {path}")


def load_trading_model_3level(
    model_class_l1,
    model_class_l2,
    model_class_l3,
    path="trading_model_3level",
    device="cpu"
):
    """
    Carga un pipeline completo de 3 niveles:
        - L1: Movimiento
        - L2: Dirección
        - L3: Regresión
    Devuelve:
        model_l1, model_l2, model_l3,
        scaler_features, scaler_l3,
        features_list, model_params,
        best_thresholds, full_results
    """

    # -----------------------------
    # 1. Cargar metadatos
    # -----------------------------
    model_params     = joblib.load(os.path.join(path, "model_params.pkl"))
    features_list    = joblib.load(os.path.join(path, "features.pkl"))
    scaler_features  = joblib.load(os.path.join(path, "scaler.pkl"))
    scaler_l3        = joblib.load(os.path.join(path, "scaler_l3.pkl"))

    thresholds_path = os.path.join(path, "best_thresholds.pkl")
    best_thresholds = joblib.load(thresholds_path) if os.path.exists(thresholds_path) else None

    # -----------------------------
    # 2. Cargar modelos
    # -----------------------------
    model_l1 = model_class_l1(**model_params)
    model_l1.load_state_dict(torch.load(os.path.join(path, "model_l1_weights.pth"), map_location=device))
    model_l1.to(device)
    model_l1.eval()

    model_l2 = model_class_l2(**model_params)
    model_l2.load_state_dict(torch.load(os.path.join(path, "model_l2_weights.pth"), map_location=device))
    model_l2.to(device)
    model_l2.eval()

    model_l3 = model_class_l3(input_dim=model_params["input_dim"], output_dim=1)
    model_l3.load_state_dict(torch.load(os.path.join(path, "model_l3_weights.pth"), map_location=device))
    model_l3.to(device)
    model_l3.eval()

    # -----------------------------
    # 3. Cargar resultados parquet
    # -----------------------------
    parquet_path = os.path.join(path, "res.parquet")
    full_results = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else None

    print(f"🚀 Pipeline 3-Level cargado correctamente desde: {path}")
    print(f"📌 Features: {len(features_list)}")
    print(f"📌 Thresholds cargados: {best_thresholds is not None}")
    print(f"📁 Resultados cargados: {full_results is not None}")

    return (
        model_l1,
        model_l2,
        model_l3,
        scaler_features,
        scaler_l3,
        features_list,
        model_params,
        best_thresholds,
        full_results
    )


def get_predict(result, params):
    df = result.copy()
    tm = params['threshold_move']
    td = params['threshold_dir']
    
    prob_move = df['prob_move'].values
    prob_buy  = df['prob_buy'].values
    prob_sell = df['prob_sell'].values
    
    # -----------------------------
    # 1) Movimiento (L1)
    # -----------------------------
    pred_move = (prob_move > tm).astype(int)
    df['pred_move'] = pred_move
    
    # -----------------------------
    # 2) Dirección (L2)
    # -----------------------------
    pred_dir = np.ones(len(df), dtype=int)  # por defecto HOLD (1)

    if td is None:
        # Caso sin threshold_dir: solo comparar BUY vs SELL
        raw_dir = np.where(prob_buy > prob_sell, 2, 0)
        pred_dir[pred_move == 1] = raw_dir[pred_move == 1]
    else:
        # Caso con threshold_dir explícito
        buy_conf  = prob_buy  > td
        sell_conf = prob_sell > td
        conflict  = buy_conf & sell_conf

        raw_dir = np.full(len(df), -1)
        raw_dir[buy_conf  & ~conflict] = 2
        raw_dir[sell_conf & ~conflict] = 0

        mask_apply = (pred_move == 1) & (raw_dir != -1)
        pred_dir[mask_apply] = raw_dir[mask_apply]

    df['pred_dir'] = pred_dir
    
    
    # -----------------------------
    # 3) FINAL PRED (0/1/2)
    # -----------------------------
    final_pred = np.where(pred_move == 0, 1, pred_dir)
    df['final_pred'] = final_pred

    return df

def plot_roc_and_reliability_2x2(
    fpr_L1, tpr_L1, auc_L1,
    fpr_L2, tpr_L2, auc_L2,
    prob_pred_L1, prob_true_L1,
    prob_pred_L2, prob_true_L2,
    show=True, save=False, save_path="roc_reliability_2x2.png"
):

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # --- ROC L1 ---
    ax[0,0].plot(fpr_L1, tpr_L1, label=f"AUC L1 = {auc_L1:.3f}")
    ax[0,0].plot([0,1],[0,1],'--',color='gray')
    ax[0,0].set_title("ROC L1: MOVE vs HOLD")
    ax[0,0].set_xlabel("False Positive Rate")
    ax[0,0].set_ylabel("True Positive Rate")
    ax[0,0].legend()

    # --- ROC L2 ---
    ax[0,1].plot(fpr_L2, tpr_L2, label=f"AUC L2 = {auc_L2:.3f}")
    ax[0,1].plot([0,1],[0,1],'--',color='gray')
    ax[0,1].set_title("ROC L2: BUY vs SELL")
    ax[0,1].set_xlabel("False Positive Rate")
    ax[0,1].set_ylabel("True Positive Rate")
    ax[0,1].legend()

    # --- Reliability L1 ---
    ax[1,0].plot(prob_pred_L1, prob_true_L1, marker='o', label='L1 calibration')
    ax[1,0].plot([0,1],[0,1],'--',color='gray')
    ax[1,0].set_title("Reliability Curve — L1 MOVE vs HOLD")
    ax[1,0].set_xlabel("Predicted probability (MOVE)")
    ax[1,0].set_ylabel("Observed frequency")
    ax[1,0].grid(True)
    ax[1,0].legend()

    # --- Reliability L2 ---
    ax[1,1].plot(prob_pred_L2, prob_true_L2, marker='o', label='L2 calibration')
    ax[1,1].plot([0,1],[0,1],'--',color='gray')
    ax[1,1].set_title("Reliability Curve — L2 BUY vs SELL")
    ax[1,1].set_xlabel("Predicted probability (BUY)")
    ax[1,1].set_ylabel("Observed frequency")
    ax[1,1].grid(True)
    ax[1,1].legend()

    plt.tight_layout()

    # Guardar si save=True
    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")

    # Mostrar si show=True
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return {
    "roc_reliability": {
        "L1": {
            "auc": float(auc_L1),
            "fpr": list(map(float, fpr_L1)),
            "tpr": list(map(float, tpr_L1)),
            "prob_pred": list(map(float, prob_pred_L1)),
            "prob_true": list(map(float, prob_true_L1))
        },
        "L2": {
            "auc": float(auc_L2),
            "fpr": list(map(float, fpr_L2)),
            "tpr": list(map(float, tpr_L2)),
            "prob_pred": list(map(float, prob_pred_L2)),
            "prob_true": list(map(float, prob_true_L2))
        }
    }
}

def plot_regression_2x2(
    results,
    show=True, save=False, save_path="regression_L3_2x2.png"
):
    df = results.copy()
    df = df.dropna(subset=["pred_logret", "real_logret"])

    y_true = df["real_logret"].astype(float).values
    y_pred = df["pred_logret"].astype(float).values

    # -----------------------------
    # 1. Métricas numéricas
    # -----------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    medae = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    errors = y_pred - y_true
    fitted = y_pred

    # Texto común para la leyenda
    legend_text = (
        f"MAE={mae:.4f}\n"
        f"RMSE={rmse:.4f}\n"
        f"R²={r2:.3f}\n"
        f"Corr={corr:.3f}\n"
        f"MedAE={medae:.4f}\n"
        f"MAPE={mape:.2f}%"
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # -----------------------------
    # 1. Scatter Pred vs Real
    # -----------------------------
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.4, ax=ax[0,0])
    ax[0,0].set_title("Scatter Pred vs Real")
    ax[0,0].set_xlabel("Real log-return")
    ax[0,0].set_ylabel("Predicted log-return")
    ax[0,0].axhline(0, color='gray', linestyle='--')
    ax[0,0].axvline(0, color='gray', linestyle='--')
    ax[0,0].grid(alpha=0.3)

    # -----------------------------
    # 2. Histograma del error
    # -----------------------------
    sns.histplot(errors, bins=50, kde=True, ax=ax[0,1])
    ax[0,1].set_title("Distribución del Error (pred - real)")
    ax[0,1].set_xlabel("Error")
    ax[0,1].grid(alpha=0.3)

    # -----------------------------
    # 3. KDE conjunta
    # -----------------------------
    sns.kdeplot(x=y_true, y=y_pred, fill=True, cmap="viridis", thresh=0.05, ax=ax[1,0])
    ax[1,0].set_title("Densidad conjunta Pred vs Real")
    ax[1,0].set_xlabel("Real log-return")
    ax[1,0].set_ylabel("Predicted log-return")
    ax[1,0].grid(alpha=0.3)

    # -----------------------------
    # 4. Residuals vs Fitted
    # -----------------------------
    sns.scatterplot(x=fitted, y=errors, s=10, alpha=0.4, ax=ax[1,1])
    ax[1,1].axhline(0, color='gray', linestyle='--')
    ax[1,1].set_title("Residuals vs Fitted")
    ax[1,1].set_xlabel("Predicted log-return")
    ax[1,1].set_ylabel("Residuals")
    ax[1,1].grid(alpha=0.3)

    # -----------------------------
    # Leyenda común
    # -----------------------------
    # fig.text(0.92, 0.5, legend_text, fontsize=12, va='center')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Guardar
    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")

    # Mostrar
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return {
        "regression": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "corr": float(corr),
            "medae": float(medae),
            "mape": float(mape),
            "errors": errors.tolist(),
            "fitted": fitted.tolist()
        }
    }

def compute_test_losses(res, df_losses, sc_l3, test_size=180):
    """
    Calcula test losses fold-by-fold para L1, L2 y L3 (HuberLoss),
    usando el MISMO scaler que en entrenamiento.
    """

    df = res.copy()
    df = df.dropna(subset=["prob_move", "prob_hold"], how="any")

    lista_no_na = df.fold.unique()
    n_folds = len(df) // test_size

    losses_L1 = []
    losses_L2 = []
    losses_L3 = []

    huber = nn.HuberLoss(delta=1.0)

    for i in range(n_folds):
        start = i * test_size
        end = start + test_size
        fold_df = df.iloc[start:end]

        # -------------------------
        # L1: MOVE vs HOLD
        # -------------------------
        y_true_L1 = (fold_df["actual"] != 0).astype(int)
        y_prob_L1 = fold_df["prob_move"]
        loss_L1 = log_loss(y_true_L1, y_prob_L1)
        losses_L1.append(loss_L1)

        # -------------------------
        # L2: BUY vs SELL
        # -------------------------
        if "prob_buy" in fold_df.columns:
            df_L2 = fold_df[fold_df["actual"] != 0]
            if len(df_L2) > 0:
                y_true_L2 = (df_L2["actual"] == 2).astype(int)
                y_prob_L2 = df_L2["prob_buy"]
                loss_L2 = log_loss(y_true_L2, y_prob_L2)
            else:
                loss_L2 = np.nan
            losses_L2.append(loss_L2)

        # -------------------------
        # L3: REGRESIÓN (HuberLoss con scaler)
        # -------------------------
        if "pred_logret" in fold_df.columns:

            df_L3 = fold_df.dropna(subset=["pred_logret", "real_logret"])

            if len(df_L3) > 0:
                real_scaled = sc_l3.transform(df_L3["real_logret"].values.reshape(-1,1)).flatten()
                pred_scaled = sc_l3.transform(df_L3["pred_logret"].values.reshape(-1,1)).flatten()

                real_t = torch.tensor(real_scaled, dtype=torch.float32)
                pred_t = torch.tensor(pred_scaled, dtype=torch.float32)

                loss_L3 = huber(pred_t, real_t).item()
            else:
                loss_L3 = np.nan

            losses_L3.append(loss_L3)

    # -------------------------
    # Construir df final
    # -------------------------
    df_test = df_losses.groupby("fold").last().reset_index()
    df_test = df_test[df_test.fold.isin(lista_no_na)]

    df_test["loss_l1_move_test"] = losses_L1

    if len(losses_L2) > 0:
        df_test["loss_l2_dir_test"] = losses_L2

    if len(losses_L3) > 0:
        df_test["loss_l3_reg_test"] = losses_L3

    return df_test

def plot_losses_by_fold(
    df_losses,
    df_test_losses=None,
    show=True,
    save=False,
    save_path="losses_by_fold.png"
):
    """
    Dibuja automáticamente los losses disponibles:
    - L1 clasificación (move)
    - L2 clasificación (dir)
    - L3 regresión (reg)
    Y añade un cuarto panel con la diferencia test - train por fold.
    """

    # Detectar qué losses existen
    available_losses = []
    if "loss_l1_move" in df_losses.columns:
        available_losses.append(("L1 Movement", "loss_l1_move", "loss_l1_move_test"))
    if "loss_l2_dir" in df_losses.columns:
        available_losses.append(("L2 Direction", "loss_l2_dir", "loss_l2_dir_test"))
    if "loss_l3_reg" in df_losses.columns:
        available_losses.append(("L3 Regression", "loss_l3_reg", "loss_l3_reg_test"))

    n = len(available_losses)

    # Layout: siempre 2x2 si hay más de 2 losses
    if n <= 2:
        fig, ax = plt.subplots(1, n, figsize=(14, 5))
        if n == 1:
            ax = [ax]
    else:
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax = ax.flatten()

    # Dibujar cada loss
    for i, (title, train_col, test_col) in enumerate(available_losses):

        df_train = df_losses.groupby("fold")[train_col].last().reset_index()

        if df_test_losses is not None and test_col in df_test_losses.columns:
            df_test = df_test_losses[["fold", test_col]]
        else:
            df_test = None

        ax[i].plot(df_train["fold"], df_train[train_col],
                   label=f"Train {title}", marker="o", alpha=0.8)

        if df_test is not None:
            ax[i].plot(df_test["fold"], df_test[test_col],
                       label=f"Test {title}", marker="o", alpha=0.8)

        ax[i].set_title(f"{title} Loss por Fold")
        ax[i].set_xlabel("Fold")
        ax[i].set_ylabel("Loss")
        ax[i].grid(True)
        ax[i].legend()

    # -----------------------------------------
    # PANEL EXTRA: diferencias test - train
    # -----------------------------------------
    if n >= 2:  # solo si hay más de un loss
        idx = 3 if n == 3 else n  # último panel disponible

        ax_extra = ax[idx] if n > 2 else None
        if ax_extra is None:
            fig_extra, ax_extra = plt.subplots(1, 1, figsize=(7, 5))

        for title, train_col, test_col in available_losses:
            if df_test_losses is not None and test_col in df_test_losses.columns:
                df_train = df_losses.groupby("fold")[train_col].last().reset_index()
                df_test = df_test_losses[["fold", test_col]]

                merged = df_train.merge(df_test, on="fold", how="inner")
                merged["diff"] = merged[test_col] - merged[train_col]

                ax_extra.plot(
                    merged["fold"], merged["diff"],
                    marker="o", alpha=0.8, label=f"{title} (Test - Train)"
                )

        ax_extra.axhline(0, color="gray", linestyle="--", alpha=0.6)
        ax_extra.set_title("Diferencia Test - Train por Fold")
        ax_extra.set_xlabel("Fold")
        ax_extra.set_ylabel("Test - Train")
        ax_extra.grid(True)
        ax_extra.legend()

    plt.tight_layout()

    if save:
        fig.savefig(save_path, dpi=300)
        print(f"💾 Figura guardada en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def build_loss_report(df_losses, df_test):
    report = {}

    for name, train_col, test_col in [
        ("L1", "loss_l1_move", "loss_l1_move_test"),
        ("L2", "loss_l2_dir", "loss_l2_dir_test"),
        ("L3", "loss_l3_reg", "loss_l3_reg_test")
    ]:
        if train_col not in df_losses.columns:
            continue

        # Train por fold
        train = df_losses.groupby("fold")[train_col].last().reset_index()

        # Test por fold
        if test_col in df_test.columns:
            test = df_test[["fold", test_col]]
            merged = train.merge(test, on="fold", how="inner")
            gap = merged[test_col] - merged[train_col]
        else:
            merged = train.copy()
            merged[test_col] = None
            gap = None

        report[name] = {
            "train": merged[train_col].tolist(),
            "test": merged[test_col].tolist(),
            "gap": gap.tolist() if gap is not None else None,
            "summary": {
                "train_mean": float(merged[train_col].mean()),
                "train_std": float(merged[train_col].std()),
                "test_mean": float(merged[test_col].mean()) if test_col in df_test.columns else None,
                "test_std": float(merged[test_col].std()) if test_col in df_test.columns else None,
                "gap_mean": float(gap.mean()) if gap is not None else None
            }
        }

    return {"losses": report}
