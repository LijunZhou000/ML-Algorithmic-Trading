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
    y_pred = df['final_pred'].astype(int)
    
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