from tradepy.load import import_dataset, clean, wavelet_denoising, resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features import generate_features
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
        weights = torch.tensor([1.5, 3.0, 1.5], dtype=torch.float32).to(device)
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
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
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
                            verbose=True, fallback_on_no_trades=False):
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
            # assign direction using argmax for rows where trade mask is True
            idx_trade = df.index[mask_trade]
            if len(idx_trade) > 0:
                buy_inds = df.loc[idx_trade, prob_buy_col] > df.loc[idx_trade, prob_sell_col]
                df.loc[idx_trade[buy_inds.values], 'pred_filtered'] = buy_label
                df.loc[idx_trade[~buy_inds.values], 'pred_filtered'] = sell_label
                if verbose:
                    n_buy2 = int(((df['pred_filtered'] == buy_label)).sum())
                    n_sell2 = int(((df['pred_filtered'] == sell_label)).sum())
                    n_hold2 = int(((df['pred_filtered'] == hold_label)).sum())
                    print(f"apply_confidence_filter (fallback argmax) -> buy:{n_buy2} sell:{n_sell2} hold:{n_hold2}")

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

            pnl = _pd.Series(0.0, index=_pd.RangeIndex(len(preds)))
            pnl.loc[(preds == 2)] = ret_array[(preds == 2)]
            pnl.loc[(preds == 0)] = -ret_array[(preds == 0)]

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