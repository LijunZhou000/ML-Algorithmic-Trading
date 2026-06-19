import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import resample


def filter_high_correlation(df, threshold=0.95):
    corr_matrix = df.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def calculate_optimal_params(df, sampling_minutes=240, horizon_min=2880):
    if (24 * 60) % sampling_minutes != 0:
        raise ValueError(f"sampling_minutes={sampling_minutes} no divide exactamente el día")

    candles_per_day = (24 * 60) // sampling_minutes  # 6

    lookback   = candles_per_day * 5                          # 30

    # Con 29k filas totales, usar ~55% para train máximo
    max_train  = int(len(df) * 0.55)
    ideal_train = candles_per_day * 365 * 3                   # 3 años = 6570
    train_size  = min(ideal_train, max_train)
    train_size  = (train_size // 1000) * 1000                 # redondeo a miles sin float

    test_size  = int(candles_per_day * 30)                    # 1 mes = 180 velas
    test_size  = max(test_size, 100)

    gap        = max(int(np.ceil(horizon_min / sampling_minutes)), 2)  # 12

    return {
        "lookback":          lookback,
        "train_size":        train_size,
        "test_size":         test_size,
        "gap":               gap,
        "_candles_per_day":  candles_per_day,
        "_horizon_bars":     gap,
    }


def filter_features(df_final_triple_real, feature_cols_triple, TARGET_COLUMN_TRIPLE,
                    sampling_minutes=240, horizon_min=2880):
    target_leakage_cols = [f"return_horizon_{horizon_min}m", f"log_return_horizon_{horizon_min}m", f"target_tb_{int(horizon_min/60/24)}d", "target_class", "target_regression",
                           f"logreturn_tb_{int(horizon_min/60/24)}d"]
    features_limpias = [
        f for f in feature_cols_triple if f not in target_leakage_cols
    ]
    # 1. Correlación
    features_to_remove = filter_high_correlation(df_final_triple_real[features_limpias])
    print(f"Eliminando {len(features_to_remove)} features redundantes por correlación")

    features_to_use = [f for f in features_limpias if f not in features_to_remove]

    # 2. Params (ahora usa los valores reales, no hardcodeados)
    params = calculate_optimal_params(
        df_final_triple_real,
        sampling_minutes=sampling_minutes,
        horizon_min=horizon_min
    )
    print(f"Configuración sugerida:\n{params}")

    # 3. Preparar datos
    X = df_final_triple_real[features_to_use]
    y_class = df_final_triple_real["target_class"]
    y_reg = df_final_triple_real["target_regression"]

    # 4. Escalar
    sc = RobustScaler()
    X_scaled = sc.fit_transform(X)

    # 5. Submuestra estratificada para diagnóstico (no se usa en el fit del selector)
    # — se deja disponible pero el selector entrena sobre todos los datos
    n_samples = min(20_000, len(X_scaled))
    X_sample, y_sample = resample(
        X_scaled, y_class,
        n_samples=n_samples,
        stratify=y_class,
        random_state=42
    )

    # 6. Selección por importancia (RandomForest, umbral = mediana)
    sel = SelectFromModel(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        threshold="median"
    )
    sel.fit(X_scaled, y_class)  # todos los datos

    features_finales = np.array(features_to_use)[sel.get_support()]
    print(f"Features seleccionadas ({len(features_finales)}):")
    print(features_finales)
    
    sel_reg = SelectFromModel(
        RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        threshold="median"
    )
    sel_reg.fit(X_scaled, y_reg)  # todos los datos
    
    features_finales_reg = np.array(features_to_use)[sel_reg.get_support()]
    print(f"Features seleccionadas para regresión ({len(features_finales_reg)}):")
    print(features_finales_reg)
    
    features_finales = list(set(features_finales).union(set(features_finales_reg)))

    return features_finales, params