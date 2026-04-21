from models import prepare_all, filter_high_correlation, calculate_optimal_params, filter_features, train_walk_forward_2level, tune_threshold_wf
from target import target_triple_barrier_realistic, apply_liquidity_filter
from load import save_trading_model_2level
import pandas as pd
from models_def import GoldLSTM_L1_Move, GoldLSTM_L2_Dir, GoldGRU_L1_Move, GoldGRU_L2_Dir

asset_list = ["bp1", "cl1", "es1", "gc1"]
# asset_list = ["bp1", "cl1", "es1", "gc1", "ibex1"]
MINUTES = 60
RETURN_HORIZON_MIN  = MINUTES * 2
TARGET_COLUMN_TRIPLE = 'target_class'
EXCLUDE_COLS_TRIPLE = [
    # Identificadores / temporales no cíclicos
    "datetime", "dtyyyymmdd", "trading_date", "ticker", "per", "openint",

    # OHLCV raw (el modelo no debería ver precios absolutos)
    "open", "high", "low", "close", "volume",

    # Targets
    "target_bin", f"target_logret_{MINUTES}", f"target_ret_{MINUTES}", f"target_ticks_{MINUTES}"
]
EXCLUDE_MORE = [
    # --- NO ESTACIONARIAS (Precios Absolutos) ---
    "open_day",           # El culpable nº 1 (Primer precio del día)
    "bb_std",             # Si es la desviación estándar en USD, varía con el precio
    "true_range",         # Es volatilidad en USD, no en % ni relativa
    
    # --- VOLUMEN ABSOLUTO (Crecen con el tiempo) ---
    "volume_cum",         # Volumen acumulado total
    "vpt",                # Volume Price Trend (usa precios y volumen raw)
    "obv",                # On-Balance Volume (es una suma acumulativa infinita)
    "ad",                 # Accumulation/Distribution (ídem que OBV)
    
    # --- REDUNDANTES / ALTA CORRELACIÓN ---
    # Si tienes 'hour_sin' y 'hour_cos', excluye las raw para no confundir a la red
    "hour", "minute", "dayofweek", "day", "month" 
]
def main():
    for asset in asset_list:
        df_final, spec = prepare_all(asset, MINUTES, RETURN_HORIZON_MIN, json_config_path="../Data/features_config.json")
        info_json = pd.read_json("../Data/futuros_specs_extend.json")
        
        feature_cols_triple = [c for c in df_final.columns if c not in EXCLUDE_COLS_TRIPLE and c not in EXCLUDE_MORE]
        
        df_final_triple_real = target_triple_barrier_realistic(df_final, info_json[asset[:2].upper()].to_dict(), return_horizon_min=RETURN_HORIZON_MIN, sampling_minutes=MINUTES, profit_factor=1.8, tp_atr_mult=1.4, sl_atr_mult=1.2)
        df_final_triple_real["target_class"] = df_final_triple_real[f"target_tb_{RETURN_HORIZON_MIN}m"]
        df_final_triple_liquid = apply_liquidity_filter(df_final_triple_real)
        
        # Úsalo solo con las features, no con el target
        features_to_remove = filter_high_correlation(df_final_triple_liquid[feature_cols_triple])
        print(f"Eliminando {len(features_to_remove)} features redundantes")
        
        # --- Ejemplo de uso para velas de 1 Hora (60 min) ---
        params = calculate_optimal_params(df_final_triple_liquid, sampling_minutes=60, horizon_min=120)
        print(f"Configuración sugerida:\n{params}")
        
        features_to_use = [f for f in feature_cols_triple if f not in features_to_remove]
        features_to_use = filter_features(df_final_triple_liquid, features_to_use, TARGET_COLUMN_TRIPLE)
        
        # Primera pasada: con todas las features filtradas por correlación
        # result_triple, model_triple, sc_triple, shap_triple = train_walk_forward(df_final_triple_liquid, features_to_use, TARGET_COLUMN_TRIPLE, GoldLSTM_Triple_Pro_v2, train_size=params["train_size"], test_size=params["test_size"], gap=params["gap"], lookback=48, epochs=15, batch_size=256, lr=5e-4)
        # result_triple_gru, model_triple_gru, sc_triple_gru, shap_triple_gru = train_walk_forward(df_final_triple_liquid, features_to_use, TARGET_COLUMN_TRIPLE, GoldGRU_Triple_Pro_v2, train_size=params["train_size"], test_size=params["test_size"], gap=params["gap"], lookback=48, epochs=15, batch_size=256, lr=5e-4)
        results, results_by_fold, model_l1, model_l2, sc = train_walk_forward_2level(
            df_final_triple_liquid,
            features_to_use,
            TARGET_COLUMN_TRIPLE,
            model_class_l1=GoldLSTM_L1_Move,
            model_class_l2=GoldLSTM_L2_Dir,
            train_size=params["train_size"],
            test_size=params["test_size"],
            gap=params["gap"],
            lookback=24,
            epochs=15,
            batch_size=256,
            lr=5e-4,
            threshold_move=0.85,
            threshold_dir=None
        )
        results_gru, results_gru_by_fold, model_l1_gru, model_l2_gru, sc_gru = train_walk_forward_2level(
            df_final_triple_liquid,
            features_to_use,
            TARGET_COLUMN_TRIPLE,
            model_class_l1=GoldGRU_L1_Move,   # ← GRU
            model_class_l2=GoldGRU_L2_Dir,    # ← GRU
            train_size=params["train_size"],
            test_size=params["test_size"],
            gap=params["gap"],
            lookback=24,
            epochs=15,
            batch_size=256,
            lr=5e-4,
            threshold_move=0.85,
            threshold_dir=None
        )
        # 2. Buscar mejor threshold para convertir probabilidades en señales binarias (BUY/SELL)
        best_threshold = tune_threshold_wf(results_by_fold)
        best_threshold_gru = tune_threshold_wf(results_gru_by_fold)
        
        # 3. Reentrenar usando el nuevo threshold
        results_best, results_by_fold_best, model_l1_best, model_l2_best, sc_best = train_walk_forward_2level(
            df_final_triple_liquid,
            features_to_use,
            TARGET_COLUMN_TRIPLE,
            model_class_l1=GoldLSTM_L1_Move,
            model_class_l2=GoldLSTM_L2_Dir,
            train_size=params["train_size"],
            test_size=params["test_size"],
            gap=params["gap"],
            lookback=24,
            epochs=15,
            batch_size=256,
            lr=5e-4,
            threshold_move=best_threshold,
            threshold_dir=None
        )
        results_gru_best, results_by_fold_gru_best, model_l1_gru_best, model_l2_gru_best, sc_gru_best = train_walk_forward_2level(
            df_final_triple_liquid,
            features_to_use,
            TARGET_COLUMN_TRIPLE,
            model_class_l1=GoldGRU_L1_Move,
            model_class_l2=GoldGRU_L2_Dir,
            train_size=params["train_size"],
            test_size=params["test_size"],
            gap=params["gap"],
            lookback=24,
            epochs=15,
            batch_size=256,
            lr=5e-4,
            threshold_move=best_threshold_gru,
            threshold_dir=None
        )

        
        model_params = {"input_dim": len(features_to_use), "output_dim": 2}

        save_trading_model_2level(
            model_l1_best, model_l2_best, sc_best,
            features_to_use,
            model_params,
            path=f"../Models/{asset}/trading_model_lstm_2level"
        )

        # Para el GRU cuando lo tengas:
        save_trading_model_2level(
            model_l1_gru_best, model_l2_gru_best, sc_gru_best,
            features_to_use,
            model_params,
            path=f"../Models/{asset}/trading_model_gru_2level"
        )

if __name__ == "__main__":
    main()