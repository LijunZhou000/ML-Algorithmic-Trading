from tradepy.data.loader import load_future
from tradepy.config.config import load_config, load_symbols, load_feature_config, load_exclude_config
from tradepy.data.cleaner import add_trading_date_by_gap
from tradepy.data.resampler import resample_ohlcv, daily_ohlcv_cummulative
from tradepy.features.generate import generate_features
from tradepy.features.quality_check import data_quality_report
from tradepy.supervised.load import prepare_all
from tradepy.paths import EXCLUDE_CONFIG, SYSTEM_CONFIG
from tradepy.supervised.target import target_triple_barrier_interday
from tradepy.supervised.pretrain import filter_features
from tradepy.supervised.models import GoldLSTM_L1_Move, GoldLSTM_L2_Dir, GoldGRU_L1_Move, GoldGRU_L2_Dir, BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicGRU_L1_Move, BasicGRU_L2_Dir, BasicLSTM_L3_Regression, BasicGRU_L3_Regression
from tradepy.supervised.train import train_walk_forward_3level, grid_search_thresholds, create_lstm_dataset
from tradepy.supervised.posttrain import evaluate_model_classification, tune_threshold_wf, save_trading_model_3level, get_predict
from tradepy.paths import MODELS_DIR

symbols = load_symbols()
cfg = load_config()
MINUTES             = cfg['sampling_minutes']        # 240
RETURN_HORIZON_MIN  = cfg['return_horizon_min']      # 2880

for ASSET in symbols:
    print(f"Procesando {ASSET}...")
    df_final, spec = prepare_all(ASSET, MINUTES, RETURN_HORIZON_MIN, json_config_path="../Data/features_config.json")
    exclude = load_exclude_config(MINUTES)
    df_final_triple_real = target_triple_barrier_interday(df_final, spec, return_horizon_days=int(RETURN_HORIZON_MIN/60/24), profit_factor=1.8, tp_atr_mult=1.4, sl_atr_mult=1.2)
    df_final_triple_real["target_class"] = df_final_triple_real[f"target_tb_{int(RETURN_HORIZON_MIN/60/24)}d"]
    df_final_triple_real["target_regression"] = df_final_triple_real[f"log_return_horizon_{RETURN_HORIZON_MIN}m"]
    feature_cols_triple = [c for c in df_final_triple_real.columns if c not in exclude['exclude_cols_triple']]
    df_final_triple_real = df_final_triple_real.dropna(subset=["target_class", "target_regression"])
    features_finales_test, params = filter_features(df_final_triple_real, feature_cols_triple, 'target_class', horizon_min=RETURN_HORIZON_MIN, sampling_minutes=MINUTES)
    result_3, result_by_fold_3, model_l1, model_l2, model_l3, sc, sc_l3 = train_walk_forward_3level(df_final_triple_real, features_finales_test, 'target_class', 'target_regression', BasicLSTM_L1_Move, BasicLSTM_L2_Dir, BasicLSTM_L3_Regression, 
                            train_size=params["train_size"],
                            test_size=params["test_size"],
                            gap=params["gap"],
                            lookback=params["lookback"],
                            epochs=15,
                            batch_size=256,
                            lr=5e-4)
    best_params, df_grid = grid_search_thresholds(result_3)
    res = get_predict(result_3, best_params)
    model_params = {"input_dim": len(features_finales_test), "output_dim": 2}
    save_trading_model_3level(
        model_l1, model_l2, model_l3,
        sc,
        sc_l3,
        features_finales_test,
        model_params,
        res,
        best_thresholds=best_params,
        path=f"{MODELS_DIR}/{ASSET.lower()}/trading_model_lstm_3level"
    )
    result_3_g, result_by_fold_3_g, model_l1_g, model_l2_g, model_l3_g, sc_g, sc_l3_g = train_walk_forward_3level(df_final_triple_real, features_finales_test, 'target_class', 'target_regression', BasicGRU_L1_Move, BasicGRU_L2_Dir, BasicGRU_L3_Regression, 
                            train_size=params["train_size"],
                            test_size=params["test_size"],
                            gap=params["gap"],
                            lookback=params["lookback"],
                            epochs=15,
                            batch_size=256,
                            lr=5e-4)
    best_params_g, df_grid_g = grid_search_thresholds(result_3_g)
    res_g = get_predict(result_3_g, best_params_g)
    model_params_g = {"input_dim": len(features_finales_test), "output_dim": 2}
    save_trading_model_3level(
        model_l1_g, model_l2_g, model_l3_g,
        sc_g,
        sc_l3_g,
        features_finales_test,
        model_params_g,
        res_g,
        best_thresholds=best_params_g,
        path=f"{MODELS_DIR}/{ASSET.lower()}/trading_model_gru_3level"
    )