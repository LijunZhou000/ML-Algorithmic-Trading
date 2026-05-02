import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


def filter_high_correlation(df, threshold=0.95):
    corr_matrix = df.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def calculate_optimal_params(df, sampling_minutes=240, horizon_min=2880):
    """
    Calcula parámetros sugeridos para GoldLSTM_Triple_Pro.
    Mercado 24h (Oro/Forex). Velas de 4h por defecto.
    """
    # Validación
    if (24 * 60) % sampling_minutes != 0:
        raise ValueError(f"sampling_minutes={sampling_minutes} no divide exactamente el día (1440 min)")

    # 1. VELAS POR DÍA - mercado 24h forex/oro usa 365, no 252
    candles_per_day = (24 * 60) // sampling_minutes  # 240 → 6 velas/día

    # 2. LOOKBACK - 5 días para captar swing interday
    lookback = candles_per_day * 5  # 6 * 5 = 30 velas (puedes subir a 60 si quieres 10 días)

    # 3. TRAIN SIZE - 1.5 años con mercado 24h (365 días)
    train_size = int(round(candles_per_day * 365 * 1.5, -3))  # redondeado a miles, int

    # 4. TEST SIZE - re-entrenar cada ~2 meses
    test_size = int(round(candles_per_day * 60, -2))  # redondeado a centenas, int

    # 5. GAP - debe cubrir el horizonte completo para evitar leakage
    gap = int(np.ceil(horizon_min / sampling_minutes))  # 2880/240 = 12 velas
    gap = max(gap, 2)  # mínimo 2 por seguridad

    return {
        "lookback":   lookback,
        "train_size": train_size,
        "test_size":  test_size,
        "gap":        gap,
        # info útil para debugging
        "_candles_per_day": candles_per_day,
        "_horizon_bars":    gap,
    }


def filter_features(df_final_triple_real, feature_cols_triple, TARGET_COLUMN_TRIPLE,
                    sampling_minutes=240, horizon_min=2880):
    # 1. Correlación
    features_to_remove = filter_high_correlation(df_final_triple_real[feature_cols_triple])
    print(f"Eliminando {len(features_to_remove)} features redundantes por correlación")

    features_to_use = [f for f in feature_cols_triple if f not in features_to_remove]

    # 2. Params (ahora usa los valores reales, no hardcodeados)
    params = calculate_optimal_params(
        df_final_triple_real,
        sampling_minutes=sampling_minutes,
        horizon_min=horizon_min
    )
    print(f"Configuración sugerida:\n{params}")

    # 3. Preparar datos
    X = df_final_triple_real[features_to_use]
    y = df_final_triple_real[TARGET_COLUMN_TRIPLE]

    # 4. Escalar
    sc = RobustScaler()
    X_scaled = sc.fit_transform(X)

    # 5. Submuestra estratificada para diagnóstico (no se usa en el fit del selector)
    # — se deja disponible pero el selector entrena sobre todos los datos
    n_samples = min(20_000, len(X_scaled))
    X_sample, y_sample = resample(
        X_scaled, y,
        n_samples=n_samples,
        stratify=y,
        random_state=42
    )

    # 6. Selección por importancia (RandomForest, umbral = mediana)
    sel = SelectFromModel(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        threshold="median"
    )
    sel.fit(X_scaled, y)  # todos los datos

    features_finales = np.array(features_to_use)[sel.get_support()]
    print(f"Features seleccionadas ({len(features_finales)}):")
    print(features_finales)

    return features_finales, params