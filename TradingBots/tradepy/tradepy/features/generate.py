from features.registry import FEATURE_REGISTRY

def generate_features(df, config_json=None, dropna_strategy='any'):
    groups = {}
    for name, meta in FEATURE_REGISTRY.items():
        groups.setdefault(meta['group'], []).append((name, meta['func']))
    
    for group_id in sorted(groups):
        for name, func in groups[group_id]:
            _call(func, name)
        _snapshot()