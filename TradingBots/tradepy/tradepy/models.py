from load import import_dataset, clean, wavelet_denoising, resample_ohlcv, daily_ohlcv_cummulative
from features import generate_features
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm
from torch import amp
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

def prepare_all(asset, minutes, return_horizon_min, json_config_path="features_config.json"):
    df_min, spec = import_dataset(asset=asset)
    df_min_clean = clean(df_min, asset=asset)
    df_min_clean["close"] = wavelet_denoising(df_min_clean['close'].values.copy())
    df_resampled = resample_ohlcv(df_min_clean, period=f"{minutes}min")
    df_cumulative = daily_ohlcv_cummulative(df_resampled)
    with open(json_config_path) as f:
        config = json.load(f)

    config["global"]["sampling_minutes"] = minutes
    config["global"]["return_horizon_min"] = return_horizon_min
    config["global"]["tick_size"] = spec["tick_size"]
    
    df_final = generate_features(df_cumulative, config_json=config, dropna_strategy='any')
    return df_final, spec


def compute_trade_pnl(preds, ret_arr, prob_trade_arr=None, stop_loss_pct=None):
    """Compute per-trade PnL with optional position sizing and stop-loss cap.

    Exposed helper for notebooks/scripts.
    """
    preds = np.array(preds)
    ret_arr = np.array(ret_arr)
    pnl = np.zeros(len(preds), dtype=float)
    pos_size = np.ones(len(preds), dtype=float)
    if prob_trade_arr is not None:
        pos_size = np.clip(np.array(prob_trade_arr, dtype=float), 0.0, 1.0)

    buy_idx = (preds == 2)
    sell_idx = (preds == 0)

    if stop_loss_pct is None:
        pnl[buy_idx] = pos_size[buy_idx] * ret_arr[buy_idx]
        pnl[sell_idx] = pos_size[sell_idx] * (-ret_arr[sell_idx])
    else:
        sl = float(stop_loss_pct)
        pnl[buy_idx] = pos_size[buy_idx] * np.clip(ret_arr[buy_idx], -sl, None)
        pnl[sell_idx] = pos_size[sell_idx] * np.clip(-ret_arr[sell_idx], -sl, None)

    return pnl

def create_lstm_dataset(df, features, target_col, lookback=14):
    """
    Crea el dataset para LSTM usando NumPy strides (ultra rápido).
    """
    # 1. Convertimos a NumPy array (importante para velocidad)
    feature_array = df[features].values
    target_array = df[target_col].values
    
    # 2. Calculamos las dimensiones
    num_samples = len(df) - lookback
    num_features = len(features)
    
    # 3. Magia de NumPy Strides: Creamos ventanas sin bucles
    # (samples, lookback, features)
    shape = (num_samples, lookback, num_features)
    strides = (feature_array.strides[0], feature_array.strides[0], feature_array.strides[1])
    
    X = np.lib.stride_tricks.as_strided(feature_array, shape=shape, strides=strides)
    
    # 4. El target es simplemente el valor DESPUÉS de la ventana
    y = target_array[lookback:]
    
    return X, y

class GoldLSTM_Bin_Default(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 32)
        self.relu = nn.LeakyReLU(0.1)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.bn(hn[-1])
        out = self.relu(self.fc(out))
        return self.out(out)
    
class GoldLSTM_Triple_Default(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.bn(hn[-1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)
    
def run_gold_master_workflow(df, feature_cols, suffix, model_class=None, target_col='target_bin', 
                                  train_size=40000, test_size=10000, step=10000, 
                                  lookback=20, epochs=15, batch_size=256):
    
    device = torch.device("cuda")
    scaler_amp = amp.GradScaler('cuda')
    
    # 1. DETECCIÓN DE MODO Y SELECCIÓN DE MODELO AUTOMÁTICA
    num_classes = df[target_col].nunique()
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else 3
    
    if model_class is None:
        model_class = GoldLSTM_Bin_Default if is_binary else GoldLSTM_Triple_Default
        print(f"🤖 Auto-seleccionado: {'Binario' if is_binary else 'Triple'} Model")

    # Configuración de Loss y Pesos
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        target_dtype = torch.float32
    else:
        # Tus pesos específicos para el Oro
        # weights = torch.tensor([2.5, 1.0, 2.5], dtype=torch.float32).to(device)
        # weights = torch.tensor([1.5, 3.0, 1.5], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss()
        target_dtype = torch.long

    all_predictions = []
    df_wf = df.copy().sort_values('datetime').reset_index(drop=True)
    # suffix = get_return_col_value(target_col)

    # Verificar que exista la columna de retornos esperada (p. ej. target_ret_30m)
    # desired_ret_col = f'target_ret_{suffix}m'
    # if desired_ret_col not in df_wf.columns:
    #     # Buscar columnas candidatas con el patrón target_ret_{N}m
    #     candidates = [c for c in df_wf.columns if re.match(r'target_ret_(\d+)m', c)]
    #     if candidates:
    #         # Elegir la primera candidata disponible como fallback
    #         chosen = candidates[0]
    #         new_suffix = re.search(r'(\d+)', chosen).group(1)
    #         print(f"⚠️ Columna esperada '{desired_ret_col}' no encontrada. Usando '{chosen}' en su lugar.")
    #         suffix = new_suffix
    #     else:
    #         available = [c for c in df_wf.columns if 'target_ret' in c]
    #         raise KeyError(f"Esperada columna '{desired_ret_col}' no encontrada y no hay columnas 'target_ret_*' disponibles. Columnas disponibles: {available}")

    for start in tqdm(range(0, len(df_wf) - train_size - test_size, step), desc="Gold Workflow"):
        # ... [Lógica de Split y Escalado idéntica a las anteriores] ...
        end_train, end_test = start + train_size, start + train_size + test_size
        train_df = df_wf.iloc[start:end_train].copy()
        test_df = df_wf.iloc[end_train:end_test].copy()
        
        sc = RobustScaler()
        train_df[feature_cols] = sc.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = sc.transform(test_df[feature_cols])
        
        X_train, y_train = create_lstm_dataset(train_df, feature_cols, target_col, lookback)
        X_test, y_test = create_lstm_dataset(test_df, feature_cols, target_col, lookback)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=target_dtype).to(device)
        if is_binary:
            y_train_t = y_train_t.unsqueeze(1)
        
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        # Inicializar modelo pasándole el output_dim detectado
        model = model_class(input_dim=len(feature_cols), output_dim=output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        
        # --- Entrenamiento ---
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                with amp.autocast('cuda'):
                    loss = criterion(model(bx), by)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
        
        # --- Predicción ---
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            if is_binary:
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                preds = (probs > 0.5).astype(int)
                res_dict = {'prob_up': probs}
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                res_dict = {'prob_0': probs[:,0], 'prob_1': probs[:,1], 'prob_2': probs[:,2]}
        
        # --- Recopilar ---
        res_df = pd.DataFrame({
            'datetime': df_wf['datetime'].iloc[end_train + lookback : end_test].values,
            'actual': y_test, 'pred': preds, **res_dict,
            'ret_real': df_wf[f'target_ret_{suffix}m'].iloc[end_train + lookback : end_test].values
        })
        all_predictions.append(res_df)
        torch.cuda.empty_cache()

    return pd.concat(all_predictions, ignore_index=True), model, sc

def apply_confidence_filter(df, prob_trade_col="prob_trade", prob_buy_col="prob_buy", prob_sell_col="prob_sell",
                            trade_threshold=0.6, dir_threshold=0.5, buy_label=2, sell_label=0, hold_label=1,
                            verbose=True, fallback_on_no_trades=False, fallback_trade_percentile=80):
    """Apply a combined confidence filter:
    - require prob_trade > trade_threshold to consider a directional signal
    - require directional prob (buy/sell) >= dir_threshold and greater than the opposite

    Returns a copy of df with a new column `pred_filtered` containing labels {buy_label,sell_label,hold_label}.
    """
    df = df.copy()
    df['pred_filtered'] = hold_label

    # Masks
    mask_trade = df[prob_trade_col] > trade_threshold
    buy_mask = (df[prob_buy_col] >= dir_threshold) & (df[prob_buy_col] > df[prob_sell_col])
    sell_mask = (df[prob_sell_col] >= dir_threshold) & (df[prob_sell_col] > df[prob_buy_col])

    df.loc[mask_trade & buy_mask, 'pred_filtered'] = buy_label
    df.loc[mask_trade & sell_mask, 'pred_filtered'] = sell_label

    if verbose:
        n_buy = int(((df['pred_filtered'] == buy_label)).sum())
        n_sell = int(((df['pred_filtered'] == sell_label)).sum())
        n_hold = int(((df['pred_filtered'] == hold_label)).sum())
        print(f"apply_confidence_filter -> buy:{n_buy} sell:{n_sell} hold:{n_hold}")

    # Optional fallback: if no directional trades were selected, optionally assign
    # direction by argmax of directional probabilities for the rows that passed the
    # trade mask. This is useful when directional confidences are weak but we still
    # want to act on high trade-probability signals.
    if fallback_on_no_trades:
        if (df['pred_filtered'] == hold_label).all():
            # Consider only rows that passed the trade mask, then restrict to the top
            # percentile of `prob_trade` among those rows to avoid acting on weak signals.
            idx_trade = df.index[mask_trade]
            if len(idx_trade) > 0:
                trade_probs = df.loc[idx_trade, prob_trade_col].values
                try:
                    cutoff = float(np.percentile(trade_probs, float(fallback_trade_percentile)))
                except Exception:
                    cutoff = float(np.percentile(df[prob_trade_col].values, float(fallback_trade_percentile)))

                idx_top_mask = df.loc[idx_trade, prob_trade_col] >= cutoff
                idx_top = idx_trade[idx_top_mask.values]
                if len(idx_top) == 0:
                    if verbose:
                        print(f"apply_confidence_filter: fallback_on_no_trades enabled but no rows above percentile {fallback_trade_percentile}")
                else:
                    buy_inds = df.loc[idx_top, prob_buy_col] > df.loc[idx_top, prob_sell_col]
                    df.loc[idx_top[buy_inds.values], 'pred_filtered'] = buy_label
                    df.loc[idx_top[~buy_inds.values], 'pred_filtered'] = sell_label
                    if verbose:
                        n_buy2 = int(((df['pred_filtered'] == buy_label)).sum())
                        n_sell2 = int(((df['pred_filtered'] == sell_label)).sum())
                        n_hold2 = int(((df['pred_filtered'] == hold_label)).sum())
                        print(f"apply_confidence_filter (fallback argmax top{fallback_trade_percentile}%) -> buy:{n_buy2} sell:{n_sell2} hold:{n_hold2}")

    return df


def sweep_confidence_thresholds(probs_df, ret_array, trade_percentiles=(60,70,80), dir_thresholds=(0.5,0.6,0.7), verbose=True):
    """Evaluate combinations of trade thresholds (percentiles on prob_trade) and directional thresholds.

    probs_df: DataFrame with columns `prob_trade`, `prob_buy`, `prob_sell`.
    ret_array: 1D array with returns aligned to probs_df (used to compute pnl per signal).

    Returns a DataFrame summarizing (trade_count, winrate, avg_trade, total_return) for each combo.
    Also prints a compact table for quick inspection.
    """
    import pandas as _pd
    results = []
    p_trade = probs_df['prob_trade'].values

    for perc in trade_percentiles:
        tt = float(np.percentile(p_trade, perc))
        for dt in dir_thresholds:
            dfp = apply_confidence_filter(probs_df, trade_threshold=tt, dir_threshold=dt)
            preds = dfp['pred_filtered'].values
            trade_mask = preds != 1
            if trade_mask.sum() == 0:
                results.append({'trade_percentile': perc, 'trade_threshold': tt, 'dir_threshold': dt,
                                'trades': 0, 'winrate': None, 'avg_trade': None, 'total_return': 0.0})
                continue

            # compute pnl with optional sizing/stop-loss if probs_df contains prob_trade
            prob_trade_arr = None
            if 'prob_trade' in dfp.columns:
                prob_trade_arr = dfp['prob_trade'].values

            # look for optional stop-loss parameter passed via probs_df attribute
            # (not all callers will provide stop_loss; default behavior preserved)
            stop_loss = getattr(probs_df, '_stop_loss_pct', None)

            pnl_arr = compute_trade_pnl(preds, np.array(ret_array), prob_trade_arr=prob_trade_arr, stop_loss_pct=stop_loss)
            pnl = _pd.Series(pnl_arr)

            trades = pnl[trade_mask]
            winrate = float((trades > 0).mean()) if len(trades) > 0 else None
            avg_trade = float(trades.mean()) if len(trades) > 0 else None
            total_return = float(trades.sum()) if len(trades) > 0 else 0.0

            results.append({'trade_percentile': perc, 'trade_threshold': tt, 'dir_threshold': dt,
                            'trades': int(trade_mask.sum()), 'winrate': winrate, 'avg_trade': avg_trade, 'total_return': total_return})

    out = _pd.DataFrame(results)
    # Print compact table (only if verbose)
    if verbose:
        print("Sweep results (trade_percentile / dir_threshold -> trades / winrate / avg_trade / total_return):")
        for perc in sorted(set(out['trade_percentile'])):
            row = out[out['trade_percentile'] == perc]
            print(f"Percentile {perc}:")
            for _, r in row.sort_values('dir_threshold').iterrows():
                try:
                    trades_val = int(r['trades']) if not _pd.isna(r['trades']) else 0
                except Exception:
                    trades_val = 0
                win = r['winrate']
                avg = r['avg_trade']
                tot = r['total_return'] if (('total_return' in r.index) and (not _pd.isna(r['total_return']))) else 0.0
                print(f"  dir={r['dir_threshold']:.2f} -> trades={trades_val:6d} win={win} avg={avg} tot={tot:.6f}")

    return out

def filter_high_correlation(df, threshold=0.95):
    corr_matrix = df.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def train_walk_forward(
    df,
    feature_cols,
    target_col,
    model_class,
    train_size=40000,
    test_size=10000,
    lookback=20,
    gap=2,  # IMPORTANTE: gap >= (horizonte_min / sampling_min)
    epochs=15,
    batch_size=256,
    lr=5e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []
    
    # 1. Generamos X e y de forma global con tus Strides (Velocidad Máxima)
    # X shape: (N, lookback, features), y shape: (N,)
    X_all, y_all = create_lstm_dataset(df, feature_cols, target_col, lookback)
    
    # Guardamos los datetimes alineados con 'y' (desde lookback hasta el final)
    dt_all = df['datetime'].values[lookback:]
    
    # 2. Bucle Walk-Forward
    # El índice 'i' se refiere a la posición en los arrays X e y generados
    total_samples = len(X_all)
    for start in tqdm(range(0, total_samples - train_size - test_size - gap, test_size), desc="WF Steps"):
        
        # Definición de índices de corte
        train_end = start + train_size
        test_start = train_end + gap
        test_end = test_start + test_size
        
        # --- SPLIT ---
        X_train, y_train = X_all[start:train_end], y_all[start:train_end]
        X_test, y_test = X_all[test_start:test_end], y_all[test_start:test_end]
        dt_test = dt_all[test_start:test_end]
        
        # --- ESCALADO DINÁMICO (No Leakage) ---
        # RobustScaler no soporta 3D (samples, seq, feat) directamente,
        # así que aplanamos, fiteamos y volvemos a dar forma.
        sc = RobustScaler()
        N_train, L, F = X_train.shape
        X_train_2d = X_train.reshape(-1, F)
        X_train_scaled = sc.fit_transform(X_train_2d).reshape(N_train, L, F)
        
        N_test = X_test.shape[0]
        X_test_scaled = sc.transform(X_test.reshape(-1, F)).reshape(N_test, L, F)
        
        # --- PREPARAR TENSORES ---
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        # --- PESOS DE CLASE (Balanceo) ---
        # class_counts = np.bincount(y_train.astype(int), minlength=3)
        # # Inversa de la frecuencia para penalizar clases mayoritarias (HOLD)
        # weights = 1.0 / (class_counts + 1e-6)
        # weights = weights / weights.sum()
        # weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        # --- PESOS DE CLASE PERSONALIZADOS ---
        class_counts = np.bincount(y_train.astype(int), minlength=3)
        freq = class_counts / len(y_train)
        
        # 1. Base inversa (lo que ya tienes)
        weights = 1.0 / (freq + 1e-6)
        
        # 2. PENALIZACIÓN EXTRA (El ajuste "Alpha")
        # Queremos que al modelo le "duela" más fallar en BUY y SELL que en HOLD.
        # Multiplicamos el peso de las señales (0 y 2) para forzar precisión.
        weights[0] *= 2.0  # Peso extra para captar mejor el SELL
        weights[2] *= 2.5  # Peso extra para captar mejor el BUY (suele ser más difícil)
        weights[1] *= 0.8  # Reducimos un poco la importancia de acertar el HOLD
        
        # Normalizamos para que sumen 1 (opcional pero recomendado)
        weights = weights / weights.sum()
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        
        # --- MODELO Y OPTIMIZADOR ---
        model = model_class(input_dim=len(feature_cols), output_dim=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(loader),
            epochs=epochs,
            pct_start=0.3      # 30% del tiempo subiendo, 70% bajando
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights_t)
        scaler_amp = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        
        # --- TRAINING LOOP ---
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(bx)
                    loss = criterion(logits, by)
                
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()
        
        # --- INFERENCIA ---
        model.eval()
        conf_threshold = 0.55
        with torch.no_grad():
            # test_logits = model(X_test_t)
            # probs = torch.softmax(test_logits, dim=1).cpu().numpy()
            # preds = np.argmax(probs, axis=1)
            test_logits = model(X_test_t)
            probs = torch.softmax(test_logits, dim=1).cpu().numpy()

            # Inicializamos todos como HOLD (1)
            preds = np.ones(len(probs), dtype=int) 

            for i in range(len(probs)):
                prob_sell = probs[i, 0]
                prob_buy = probs[i, 2]

                if prob_buy > conf_threshold:
                    preds[i] = 2
                elif prob_sell > conf_threshold:
                    preds[i] = 0
        
        # --- RECOLECCIÓN DE RESULTADOS ---
        step_res = pd.DataFrame({
            'datetime': dt_test,
            'actual': y_test,
            'pred': preds,
            'prob_0': probs[:, 0], # SELL
            'prob_1': probs[:, 1], # HOLD
            'prob_2': probs[:, 2]  # BUY
        })
        all_results.append(step_res)
        
        # Limpieza de memoria para el siguiente salto
        del y_train_t
        torch.cuda.empty_cache()
    # --- Al finalizar el bucle Walk-Forward ---
    
    # 1. Pasamos a evaluación para fijar Batchnorm/Dropout
    model.eval() 
    
    # 2. TRUCO: Desactivamos cuDNN solo para este bloque
    # Esto evita el error "cudnn RNN backward can only be called in training mode"
    torch.backends.cudnn.enabled = False 
    
    idx_bg = np.random.choice(X_train_t.shape[0], 100, replace=False)
    background = X_train_t[idx_bg]
    test_data = X_test_t[:min(len(X_test_t), 200)]
    
    print(f"🧠 Calculando SHAP (Modo compatibilidad cuDNN: OFF)...")
    
    try:
        # SHAP DeepExplainer necesita que el modelo esté en modo "gradiente"
        # pero con los pesos fijos de evaluación.
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_data, check_additivity=False)
        
    except Exception as e:
        print(f"⚠️ Error persistente en SHAP: {e}")
        shap_values = None
    finally:
        # 3. MUY IMPORTANTE: Volver a activar cuDNN para futuros entrenamientos
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

    return pd.concat(all_results, ignore_index=True), model, sc, shap_values
    # return pd.concat(all_results, ignore_index=True), model, sc, shap_values
    
def calculate_optimal_params(df, sampling_minutes=60, horizon_min=120):
    """
    Calcula parámetros sugeridos para GoldLSTM_Triple_Pro.
    Basado en lógica de microestructura de mercado.
    """
    # 1. VELAS POR DÍA (Asumiendo mercado 24h como el Oro/Forex)
    candles_per_day = (24 * 60) // sampling_minutes
    
    # 2. LOOKBACK (Memoria del modelo)
    # Sugerencia: 1.5 a 2 días de datos para captar el ciclo diario/noticias.
    lookback = int(candles_per_day * 1.5)
    
    # 3. TRAIN SIZE (Ventana de aprendizaje)
    # Sugerencia: 1 a 1.5 años de datos para ver todas las estaciones/regímenes.
    # 252 días de trading al año aprox.
    train_size = int(candles_per_day * 252 * 1.2)
    
    # 4. TEST SIZE (Ventana de validación/re-entrenamiento)
    # Sugerencia: Re-entrenar cada 1 o 2 meses para evitar obsolescencia.
    test_size = int(candles_per_day * 30 * 1.5)
    
    # 5. GAP (Seguridad Triple Barrera)
    # Debe ser al menos igual al horizonte de la barrera vertical.
    gap = int(np.ceil(horizon_min / sampling_minutes))
    
    return {
        "lookback": round(lookback),
        "train_size": round(train_size, -3),
        "test_size": round(test_size, -3),
        "gap": round(gap)
    }
    
def evaluate_model_classification(full_results):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    import seaborn as sns
    import matplotlib.pyplot as plt

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

def get_top_features_from_shap(shap_values, feature_names, top_n=30, plot=True):
    """
    1. Calcula la importancia.
    2. Pinta los gráficos.
    3. Devuelve la lista combinada (Set) de las Top N de cada clase.
    """
    if shap_values is None:
        return feature_names # Si falla, devolvemos todo por seguridad

    def get_importance(shap_array):
        abs_val = np.abs(shap_array)
        # Ajuste dinámico de ejes según la forma del array (muestras, features, tiempo)
        if abs_val.shape[1] == len(feature_names):
            return abs_val.mean(axis=(0, 2)) 
        else:
            return abs_val.mean(axis=(0, 1))

    # 1. Obtener importancia para SELL (0) y BUY (2)
    imp_sell = get_importance(shap_values[0])
    imp_buy = get_importance(shap_values[2])

    # 2. Crear DataFrames y extraer los nombres de las Top N
    top_sell = pd.DataFrame({'f': feature_names, 'i': imp_sell}).sort_values('i', ascending=False).head(top_n)['f'].tolist()
    top_buy = pd.DataFrame({'f': feature_names, 'i': imp_buy}).sort_values('i', ascending=False).head(top_n)['f'].tolist()

    # 3. Combinar listas evitando duplicados (Unión de conjuntos)
    combined_features = list(set(top_sell) | set(top_buy))
    
    print(f"✅ Extracción dinámica completada.")
    print(f"📊 Top {top_n} SELL + Top {top_n} BUY = {len(combined_features)} features únicas.")

    # --- Reutilizamos la lógica del plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(x='i', y='f', data=pd.DataFrame({'f': feature_names, 'i': imp_sell}).sort_values('i', ascending=False).head(top_n), ax=axes[0], palette='Reds_r', hue='f', legend=False)
    axes[0].set_title(f'Top {top_n} SELL')
    sns.barplot(x='i', y='f', data=pd.DataFrame({'f': feature_names, 'i': imp_buy}).sort_values('i', ascending=False).head(top_n), ax=axes[1], palette='Greens_r', hue='f', legend=False)
    axes[1].set_title(f'Top {top_n} BUY')
    plt.tight_layout()
    if plot:
        plt.show()

    return combined_features

def train_walk_forward_2level(
    df,
    feature_cols,
    target_col,           # columna original: 0=SELL, 1=HOLD, 2=BUY
    model_class_l1,       # GoldLSTM_L1_Move
    model_class_l2,       # GoldLSTM_L2_Dir
    train_size=40000,
    test_size=10000,
    gap=2,
    lookback=24,
    epochs=15,
    batch_size=256,
    lr=5e-4,
    threshold_move=0.5,
    threshold_dir=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    # --- Preparar targets derivados en el df ---
    # L1: 0=HOLD, 1=ACCIÓN (cualquier BUY o SELL)
    df = df.copy()
    df['_target_l1'] = (df[target_col] != 1).astype(int)
    # L2: 0=SELL, 2=BUY  (solo filas con movimiento, pero creamos la columna global)
    df['_target_l2'] = (df[target_col] == 2).astype(int)  # HOLD quedará como 0, no importa

    # --- Generar secuencias globales para L1 y L2 ---
    X_all, y_l1_all = create_lstm_dataset(df, feature_cols, '_target_l1', lookback)
    _,     y_l2_all = create_lstm_dataset(df, feature_cols, '_target_l2', lookback)
    dt_all = df['datetime'].values[lookback:]

    total_samples = len(X_all)

    for start in tqdm(range(0, total_samples - train_size - test_size - gap, test_size), desc="WF 2-Level"):

        train_end   = start + train_size
        test_start  = train_end + gap
        test_end    = test_start + test_size

        # --- SPLIT ---
        X_train = X_all[start:train_end]
        X_test  = X_all[test_start:test_end]
        y_l1_train = y_l1_all[start:train_end]
        y_l1_test  = y_l1_all[test_start:test_end]
        y_l2_train = y_l2_all[start:train_end]
        y_l2_test  = y_l2_all[test_start:test_end]
        dt_test    = dt_all[test_start:test_end]

        # --- ESCALADO (fit solo en train) ---
        sc = RobustScaler()
        N_train, L, F = X_train.shape
        X_train_scaled = sc.fit_transform(X_train.reshape(-1, F)).reshape(N_train, L, F)
        N_test = X_test.shape[0]
        X_test_scaled  = sc.transform(X_test.reshape(-1, F)).reshape(N_test, L, F)

        # --- TENSORES ---
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)

        # =====================================================
        # NIVEL 1 — ¿Hay movimiento?
        # =====================================================
        y_l1_train_t = torch.tensor(y_l1_train, dtype=torch.long).to(device)

        # Pesos L1: HOLD es mayoría, penalizamos
        counts_l1 = np.bincount(y_l1_train.astype(int), minlength=2)
        w_l1 = 1.0 / (counts_l1 / len(y_l1_train) + 1e-6)
        w_l1[1] *= 2.0   # penalizar extra fallar en ACCIÓN
        w_l1 = w_l1 / w_l1.sum()
        w_l1_t = torch.tensor(w_l1, dtype=torch.float32).to(device)

        model_l1 = model_class_l1(input_dim=len(feature_cols), output_dim=2).to(device)
        opt_l1   = torch.optim.Adam(model_l1.parameters(), lr=lr)
        crit_l1  = nn.CrossEntropyLoss(weight=w_l1_t)
        sched_l1 = torch.optim.lr_scheduler.OneCycleLR(
            opt_l1, max_lr=lr,
            steps_per_epoch=max(1, (len(X_train) + batch_size - 1) // batch_size),  # ← ceil en vez de floor
            epochs=epochs, pct_start=0.3
        )
        scaler_amp = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        loader_l1  = DataLoader(TensorDataset(X_train_t, y_l1_train_t),
                                batch_size=batch_size, shuffle=True)

        model_l1.train()
        for epoch in range(epochs):
            for bx, by in loader_l1:
                opt_l1.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = crit_l1(model_l1(bx), by)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt_l1)
                nn.utils.clip_grad_norm_(model_l1.parameters(), 1.0)
                scaler_amp.step(opt_l1)
                scaler_amp.update()
                sched_l1.step()

        # =====================================================
        # NIVEL 2 — ¿BUY o SELL? (entrenado solo con filas de movimiento)
        # =====================================================
        mask_move = (y_l1_train == 1)   # solo filas donde hay acción real
        X_l2_train = X_train_scaled[mask_move]
        y_l2_train_filt = y_l2_train[mask_move]

        y_l2_train_t = torch.tensor(y_l2_train_filt, dtype=torch.long).to(device)
        X_l2_train_t = torch.tensor(X_l2_train, dtype=torch.float32).to(device)

        # Pesos L2: SELL vs BUY — casi balanceado, pesos suaves
        counts_l2 = np.bincount(y_l2_train_filt.astype(int), minlength=2)
        w_l2 = 1.0 / (counts_l2 / len(y_l2_train_filt) + 1e-6)
        w_l2 = w_l2 / w_l2.sum()
        w_l2_t = torch.tensor(w_l2, dtype=torch.float32).to(device)

        model_l2 = model_class_l2(input_dim=len(feature_cols), output_dim=2).to(device)
        opt_l2   = torch.optim.Adam(model_l2.parameters(), lr=lr)
        crit_l2  = nn.CrossEntropyLoss(weight=w_l2_t)
        sched_l2 = torch.optim.lr_scheduler.OneCycleLR(
            opt_l2, max_lr=lr,
            steps_per_epoch=max(1, (len(X_l2_train) + batch_size - 1) // batch_size),  # ← ídem
            epochs=epochs, pct_start=0.3
        )
        loader_l2 = DataLoader(TensorDataset(X_l2_train_t, y_l2_train_t),
                               batch_size=batch_size, shuffle=True)

        model_l2.train()
        for epoch in range(epochs):
            for bx, by in loader_l2:
                opt_l2.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = crit_l2(model_l2(bx), by)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt_l2)
                nn.utils.clip_grad_norm_(model_l2.parameters(), 1.0)
                scaler_amp.step(opt_l2)
                scaler_amp.update()
                sched_l2.step()

        # =====================================================
        # INFERENCIA — Pipeline en cascada
        # =====================================================
        model_l1.eval()
        model_l2.eval()

        with torch.no_grad():
            # L1: probabilidad de que haya movimiento
            probs_l1 = torch.softmax(model_l1(X_test_t), dim=1).cpu().numpy()
            # L2: probabilidad de dirección
            probs_l2 = torch.softmax(model_l2(X_test_t), dim=1).cpu().numpy()

        # Decisión en cascada
        # threshold_move = 0.6    # L1: mínima confianza para considerar que hay movimiento
        # threshold_dir  = 0.55    # L2: mínima confianza para la dirección

        preds = np.ones(len(probs_l1), dtype=int)   # default: HOLD (1)

        for i in range(len(probs_l1)):
            if probs_l1[i, 1] > threshold_move:
                if threshold_dir is None:
                    preds[i] = 2 if probs_l2[i, 1] > probs_l2[i, 0] else 0
                else:
                    if probs_l2[i, 1] > threshold_dir:
                        preds[i] = 2
                    elif probs_l2[i, 0] > threshold_dir:
                        preds[i] = 0

        # --- RECOLECCIÓN ---
        step_res = pd.DataFrame({
            'datetime':   dt_test,
            'actual':     y_l1_test,               # ojo: guardamos el target original
            'pred':       preds,
            'prob_move':  probs_l1[:, 1],
            'prob_sell':  probs_l2[:, 0],
            'prob_buy':   probs_l2[:, 1],
        })
        # Reconstruimos actual en 0/1/2 para evaluate_model_classification
        # (y_l1_test es binario, necesitamos el original)
        original_test = df[target_col].values[lookback:][test_start:test_end]
        step_res['actual'] = original_test

        all_results.append(step_res)

        del y_l1_train_t, y_l2_train_t, X_l2_train_t
        torch.cuda.empty_cache()

    return pd.concat(all_results, ignore_index=True), all_results, model_l1, model_l2, sc

def filter_features(df, features, target):
    # 1. Preparar datos (importante usar los mismos que vería el modelo)
    X = df[features]
    y = df[target] # OJO: Lasso es para regresión, si es clasificación usa LogisticRegression(penalty='l1')

    # 2. ESCALAR ES OBLIGATORIO
    # LASSO penaliza el tamaño de los coeficientes; si una variable tiene números 
    # muy grandes, el modelo la penalizará por error.
    sc = RobustScaler()
    X_scaled = sc.fit_transform(X)

    X_sample, y_sample = resample(
        X_scaled, y,
        n_samples=20_000,
        stratify=y,
        random_state=42
    )

    sel = SelectFromModel(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        threshold="median"  # elimina la mitad menos importante
    )
    sel.fit(X_scaled, y)  # usa todos los datos, no necesita submuestra

    features_finales_test = np.array(features)[sel.get_support()]
    print(f"Features seleccionadas ({len(features_finales_test)}):")
    return features_finales_test

def tune_threshold(results_df, thresholds=np.arange(0.50, 0.95, 0.05)):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0, 0
    for t in thresholds:
        preds = np.ones(len(results_df), dtype=int)
        move_mask = results_df['prob_move'] > t
        preds[move_mask] = np.where(
            results_df.loc[move_mask, 'prob_buy'] > results_df.loc[move_mask, 'prob_sell'],
            2, 0
        )
        f1 = f1_score(results_df['actual'], preds, average='macro')
        print(f"threshold={t:.2f} → F1 Macro={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\n✅ Mejor threshold: {best_t:.2f} → F1={best_f1:.4f}")
    return best_t

def tune_threshold_wf(all_step_results, thresholds=np.arange(0.50, 0.95, 0.05)):
    """
    Optimiza threshold en los primeros 70% de folds,
    valida en el 30% restante — evita overfitting al threshold.
    """
    from sklearn.metrics import f1_score
    
    n_folds = len(all_step_results)
    split = int(n_folds * 0.7)
    
    df_train_folds = pd.concat(all_step_results[:split])
    df_val_folds   = pd.concat(all_step_results[split:])
    
    # Optimiza en train folds
    best_t = 0.85
    best_f1 = 0
    for t in thresholds:
        preds = np.ones(len(df_train_folds), dtype=int)
        mask = df_train_folds['prob_move'] > t
        preds[mask] = np.where(
            df_train_folds.loc[mask, 'prob_buy'] > df_train_folds.loc[mask, 'prob_sell'],
            2, 0
        )
        f1 = f1_score(df_train_folds['actual'], preds, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    # Valida en val folds
    preds_val = np.ones(len(df_val_folds), dtype=int)
    mask_val = df_val_folds['prob_move'] > best_t
    preds_val[mask_val] = np.where(
        df_val_folds.loc[mask_val, 'prob_buy'] > df_val_folds.loc[mask_val, 'prob_sell'],
        2, 0
    )
    f1_val = f1_score(df_val_folds['actual'], preds_val, average='macro')
    
    print(f"Threshold óptimo (train folds): {best_t:.2f} → F1={best_f1:.4f}")
    print(f"F1 en val folds con ese threshold: {f1_val:.4f}")
    print(f"Degradación: {best_f1 - f1_val:.4f} {'⚠️ posible overfit' if best_f1 - f1_val > 0.02 else '✅ estable'}")
    
    return best_t

def ensemble_soft_voting(results_lstm, results_gru, threshold_move=0.85):    
    df = results_lstm.copy()
    
    # Promedio de probabilidades entre LSTM y GRU
    df['prob_move_ens']  = (results_lstm['prob_move'] + results_gru['prob_move']) / 2
    df['prob_buy_ens']   = (results_lstm['prob_buy']  + results_gru['prob_buy'])  / 2
    df['prob_sell_ens']  = (results_lstm['prob_sell'] + results_gru['prob_sell']) / 2
    
    preds = np.ones(len(df), dtype=int)
    mask = df['prob_move_ens'] > threshold_move
    preds[mask] = np.where(
        df.loc[mask, 'prob_buy_ens'] > df.loc[mask, 'prob_sell_ens'],
        2, 0
    )
    
    df['pred'] = preds
    return df