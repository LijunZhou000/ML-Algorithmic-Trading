import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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

# def train_walk_forward_2level(
#     df,
#     feature_cols,
#     target_col,
#     model_class_l1,
#     model_class_l2,
#     train_size=40000,
#     test_size=10000,
#     gap=2,
#     lookback=30,          # actualizado: usa params["lookback"]
#     epochs=15,
#     batch_size=256,
#     lr=5e-4,
#     threshold_move=0.5,
#     threshold_dir=0.5
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     all_results = []

#     df = df.copy()
#     df['_target_l1'] = (df[target_col] != 1).astype(int)
#     df['_target_l2'] = (df[target_col] == 2).astype(int)

#     X_all, y_l1_all = create_lstm_dataset(df, feature_cols, '_target_l1', lookback)
#     _,     y_l2_all = create_lstm_dataset(df, feature_cols, '_target_l2', lookback)
#     dt_all = df['datetime'].values[lookback:]

#     total_samples = len(X_all)

#     for start in tqdm(range(0, total_samples - train_size - test_size - gap, test_size), desc="WF 2-Level"):

#         train_end  = start + train_size
#         test_start = train_end + gap
#         test_end   = test_start + test_size

#         X_train = X_all[start:train_end]
#         X_test  = X_all[test_start:test_end]
#         y_l1_train = y_l1_all[start:train_end]
#         y_l1_test  = y_l1_all[test_start:test_end]
#         y_l2_train = y_l2_all[start:train_end]
#         dt_test    = dt_all[test_start:test_end]

#         sc = RobustScaler()
#         N_train, L, F = X_train.shape
#         X_train_scaled = sc.fit_transform(X_train.reshape(-1, F)).reshape(N_train, L, F)
#         N_test = X_test.shape[0]
#         X_test_scaled  = sc.transform(X_test.reshape(-1, F)).reshape(N_test, L, F)

#         X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
#         X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)

#         # =====================================================
#         # NIVEL 1
#         # =====================================================
#         y_l1_train_t = torch.tensor(y_l1_train, dtype=torch.long).to(device)

#         # Pesos de clase balanceados (más robusto que un multiplicador manual)
#         try:
#             classes_l1 = np.array([0, 1])
#             w_l1 = compute_class_weight(class_weight='balanced', classes=classes_l1, y=y_l1_train.astype(int))
#         except Exception:
#             counts_l1 = np.bincount(y_l1_train.astype(int), minlength=2)
#             w_l1 = 1.0 / (counts_l1 / len(y_l1_train) + 1e-6)
#         w_l1 = w_l1 / w_l1.sum()
#         w_l1_t = torch.tensor(w_l1.astype(np.float32), dtype=torch.float32).to(device)

#         model_l1 = model_class_l1(input_dim=len(feature_cols), output_dim=2, hidden_dim=256).to(device)
#         opt_l1   = torch.optim.Adam(model_l1.parameters(), lr=lr)
#         crit_l1  = nn.CrossEntropyLoss(weight=w_l1_t)
#         sched_l1 = torch.optim.lr_scheduler.OneCycleLR(
#             opt_l1, max_lr=lr,
#             steps_per_epoch=max(1, (len(X_train) + batch_size - 1) // batch_size),
#             epochs=epochs, pct_start=0.3
#         )

#         # scaler_amp independiente por modelo
#         scaler_amp_l1 = torch.amp.GradScaler(enabled=(device.type == "cuda"))

#         loader_l1 = DataLoader(TensorDataset(X_train_t, y_l1_train_t),
#                                batch_size=batch_size, shuffle=True)

#         model_l1.train()
#         for epoch in range(epochs):
#             for bx, by in loader_l1:
#                 opt_l1.zero_grad()
#                 with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
#                     loss = crit_l1(model_l1(bx), by)
#                 scaler_amp_l1.scale(loss).backward()
#                 scaler_amp_l1.unscale_(opt_l1)
#                 nn.utils.clip_grad_norm_(model_l1.parameters(), 1.0)
#                 scaler_amp_l1.step(opt_l1)
#                 scaler_amp_l1.update()
#                 sched_l1.step()

#         # =====================================================
#         # NIVEL 2
#         # =====================================================
#         mask_move = (y_l1_train == 1)
#         X_l2_train = X_train_scaled[mask_move]
#         y_l2_train_filt = y_l2_train[mask_move]

#         y_l2_train_t = torch.tensor(y_l2_train_filt, dtype=torch.long).to(device)
#         X_l2_train_t = torch.tensor(X_l2_train, dtype=torch.float32).to(device)

#         # Pesos balanceados para L2 (manejo de excepción si falta alguna clase)
#         try:
#             if len(y_l2_train_filt) == 0:
#                 raise ValueError("No hay muestras para L2")
#             classes_l2 = np.array([0, 1])
#             w_l2 = compute_class_weight(class_weight='balanced', classes=classes_l2, y=y_l2_train_filt.astype(int))
#         except Exception:
#             counts_l2 = np.bincount(y_l2_train_filt.astype(int), minlength=2)
#             w_l2 = 1.0 / (counts_l2 / max(1, len(y_l2_train_filt)) + 1e-6)
#         w_l2 = w_l2 / w_l2.sum()
#         w_l2_t = torch.tensor(w_l2.astype(np.float32), dtype=torch.float32).to(device)

#         model_l2 = model_class_l2(input_dim=len(feature_cols), output_dim=2, hidden_dim=256).to(device)
#         opt_l2   = torch.optim.Adam(model_l2.parameters(), lr=lr)
#         crit_l2  = nn.CrossEntropyLoss(weight=w_l2_t)
#         sched_l2 = torch.optim.lr_scheduler.OneCycleLR(
#             opt_l2, max_lr=lr,
#             steps_per_epoch=max(1, (len(X_l2_train) + batch_size - 1) // batch_size),
#             epochs=epochs, pct_start=0.3
#         )

#         # scaler_amp independiente para L2
#         scaler_amp_l2 = torch.amp.GradScaler(enabled=(device.type == "cuda"))

#         loader_l2 = DataLoader(TensorDataset(X_l2_train_t, y_l2_train_t),
#                                batch_size=batch_size, shuffle=True)

#         model_l2.train()
#         for epoch in range(epochs):
#             for bx, by in loader_l2:
#                 opt_l2.zero_grad()
#                 with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
#                     loss = crit_l2(model_l2(bx), by)
#                 scaler_amp_l2.scale(loss).backward()
#                 scaler_amp_l2.unscale_(opt_l2)
#                 nn.utils.clip_grad_norm_(model_l2.parameters(), 1.0)
#                 scaler_amp_l2.step(opt_l2)
#                 scaler_amp_l2.update()
#                 sched_l2.step()

#         # =====================================================
#         # INFERENCIA — vectorizada
#         # =====================================================
#         model_l1.eval()
#         model_l2.eval()

#         with torch.no_grad():
#             probs_l1 = torch.softmax(model_l1(X_test_t), dim=1).cpu().numpy()
#             probs_l2 = torch.softmax(model_l2(X_test_t), dim=1).cpu().numpy()

#         preds = np.ones(len(probs_l1), dtype=int)   # HOLD por defecto

#         move_mask = probs_l1[:, 1] > threshold_move

#         if threshold_dir is None:
#             # Sin umbral: ganador por probabilidad
#             dir_pred = np.where(probs_l2[:, 1] > probs_l2[:, 0], 2, 0)
#             preds[move_mask] = dir_pred[move_mask]
#         else:
#             buy_conf  = probs_l2[:, 1] > threshold_dir
#             sell_conf = probs_l2[:, 0] > threshold_dir
#             conflict  = buy_conf & sell_conf   # ambos altos → incertidumbre → HOLD

#             dir_pred = np.full(len(probs_l1), -1)        # -1 = sin decisión clara
#             dir_pred[buy_conf  & ~conflict] = 2
#             dir_pred[sell_conf & ~conflict] = 0

#             # Solo aplicar donde hay movimiento Y decisión clara
#             apply_mask = move_mask & (dir_pred != -1)
#             preds[apply_mask] = dir_pred[apply_mask]

#         original_test = df[target_col].values[lookback:][test_start:test_end]

#         step_res = pd.DataFrame({
#             'datetime':  dt_test,
#             'actual':    original_test,
#             'pred':      preds,
#             'prob_move': probs_l1[:, 1],
#             'prob_sell': probs_l2[:, 0],
#             'prob_buy':  probs_l2[:, 1],
#         })

#         all_results.append(step_res)

#         del y_l1_train_t, y_l2_train_t, X_l2_train_t
#         torch.cuda.empty_cache()

#     return pd.concat(all_results, ignore_index=True), all_results, model_l1, model_l2, sc

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

def grid_search_thresholds(
    results,
    move_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7],
    dir_thresholds=[0.5, 0.55, 0.6, None],
    metric='f1_macro',
):
    from itertools import product
    from sklearn.metrics import f1_score, precision_score

    prob_move = results['prob_move'].values
    prob_buy  = results['prob_buy'].values
    prob_sell = results['prob_sell'].values
    y_true    = results['actual'].astype(int).values

    def apply_thresholds(tm, td):
        preds = np.ones(len(results), dtype=int)
        move_mask = prob_move > tm
        if td is None:
            dir_pred = np.where(prob_buy > prob_sell, 2, 0)
            preds[move_mask] = dir_pred[move_mask]
        else:
            buy_conf  = prob_buy  > td
            sell_conf = prob_sell > td
            conflict  = buy_conf & sell_conf
            dir_pred  = np.full(len(results), -1)
            dir_pred[buy_conf  & ~conflict] = 2
            dir_pred[sell_conf & ~conflict] = 0
            apply_mask = move_mask & (dir_pred != -1)
            preds[apply_mask] = dir_pred[apply_mask]
        return preds

    def score(y_pred):
        if metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'f1_sell':
            return f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[0]
        elif metric == 'f1_buy':
            return f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[2]
        elif metric == 'precision_sell':
            return precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])[0]

    best_score, best_params = 0, {}
    grid = []

    for tm, td in product(move_thresholds, dir_thresholds):
        preds = apply_thresholds(tm, td)
        s = score(preds)
        grid.append({'threshold_move': round(float(tm), 2), 'threshold_dir': td, metric: round(s, 4)})
        if s > best_score:
            best_score = s
            best_params = {'threshold_move': tm, 'threshold_dir': td}

    df_grid = pd.DataFrame(grid).sort_values(metric, ascending=False)
    print(f"Métrica: {metric}")
    print(f"Mejor score: {best_score:.4f} con {best_params}")
    print(f"\nTop 10:\n{df_grid.head(10).to_string(index=False)}")

    return best_params, df_grid