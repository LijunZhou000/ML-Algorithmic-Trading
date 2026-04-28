import json
import numpy as np
import pandas as pd
import inspect
from collections import defaultdict, deque

from .registry import FEATURE_REGISTRY


def generate_features(df, config_json=None, dropna_strategy="any"):
    # ─────────────────────────────────────────────────────────────
    # CONFIG
    # ─────────────────────────────────────────────────────────────
    cfg = {}
    if config_json:
        if isinstance(config_json, str):
            cfg = json.loads(config_json)
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            raise ValueError("config_json debe ser dict o JSON string")

    global_cfg = cfg.get("global", {})
    if "dropna_strategy" in global_cfg and (dropna_strategy == "any" or dropna_strategy is None):
        dropna_strategy = global_cfg.get("dropna_strategy")

    def p(name):
        return {**global_cfg, **cfg.get(name, {})}

    # ─────────────────────────────────────────────────────────────
    # DAG BUILD (dependencias)
    # ─────────────────────────────────────────────────────────────
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for name, meta in FEATURE_REGISTRY.items():
        indegree[name] = 0

    for name, meta in FEATURE_REGISTRY.items():
        for dep in meta["requires"]:
            # Solo conectar si la dependencia es otra feature
            if dep in FEATURE_REGISTRY:
                graph[dep].append(name)
                indegree[name] += 1

    # Topological sort (Kahn)
    queue = deque([n for n in FEATURE_REGISTRY if indegree[n] == 0])
    execution_order = []

    while queue:
        node = queue.popleft()
        execution_order.append(node)
        for neigh in graph[node]:
            indegree[neigh] -= 1
            if indegree[neigh] == 0:
                queue.append(neigh)

    if len(execution_order) != len(FEATURE_REGISTRY):
        raise RuntimeError("Ciclo detectado en dependencias de features")

    # ─────────────────────────────────────────────────────────────
    # EJECUCIÓN
    # ─────────────────────────────────────────────────────────────
    base = df.copy()
    buf = {}

    def _snapshot():
        nonlocal base
        if buf:
            base = pd.concat([base, pd.DataFrame(buf, index=base.index)], axis=1)
            buf.clear()

    def _call(name):
        meta = FEATURE_REGISTRY[name]
        func = meta["func"]

        params = p(name)

        # defaults especiales
        session_funcs = {"compute_atr", "obv", "ad", "rolling_slope"}
        if name in session_funcs and "session_key" not in params:
            params = {**params, "session_key": "trading_date"}

        # filtrar params según firma
        try:
            sig = inspect.signature(func)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            allowed = params if has_varkw else {k: v for k, v in params.items() if k in sig.parameters}
        except Exception:
            allowed = params

        result = func(base, **allowed) if allowed else func(base)

        # normalización salida
        out_items = {}
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                out_items[col] = result[col]
        elif isinstance(result, dict):
            out_items = result
        elif isinstance(result, pd.Series):
            out_items[result.name or "value"] = result
        else:
            try:
                arr = np.asarray(result)
                if arr.ndim == 1 and arr.shape[0] == len(base):
                    out_items["value"] = arr
            except Exception:
                return

        for col, series in out_items.items():
            if col in base.columns or col in buf:
                continue
            if not isinstance(series, pd.Series):
                try:
                    series = pd.Series(series, index=base.index)
                except Exception:
                    continue
            buf[col] = series.values

    # ─────────────────────────────────────────────────────────────
    # EJECUCIÓN POR CAPAS (snapshot automático)
    # ─────────────────────────────────────────────────────────────
    last_group = None

    for name in execution_order:
        group = FEATURE_REGISTRY[name].get("group", 1)

        if last_group is not None and group != last_group:
            _snapshot()

        _call(name)
        last_group = group

    _snapshot()

    # ─────────────────────────────────────────────────────────────
    # CLEAN
    # ─────────────────────────────────────────────────────────────
    base.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # DROPNA
    # ─────────────────────────────────────────────────────────────
    try:
        if dropna_strategy in (0, "0", None, False):
            pass
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == "all":
            base = base.dropna(how="all").reset_index(drop=True)
        elif isinstance(dropna_strategy, str) and dropna_strategy.lower() == "any":
            base = base.dropna(how="any").reset_index(drop=True)
        elif isinstance(dropna_strategy, float) and 0 < dropna_strategy < 1:
            thresh = int(np.ceil(base.shape[1] * float(dropna_strategy)))
            base = base.dropna(thresh=thresh).reset_index(drop=True)
        elif isinstance(dropna_strategy, int) and dropna_strategy >= 1:
            base = base.dropna(thresh=int(dropna_strategy)).reset_index(drop=True)
        else:
            val = float(dropna_strategy)
            if 0 < val < 1:
                thresh = int(np.ceil(base.shape[1] * val))
                base = base.dropna(thresh=thresh)
            elif val == 0:
                pass
            else:
                base = base.dropna(thresh=int(val)).reset_index(drop=True)
    except Exception:
        base = base.dropna(how="all").reset_index(drop=True)

    return base