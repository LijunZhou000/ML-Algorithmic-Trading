FEATURE_REGISTRY = {}

def feature(group=1, requires=None):
    def decorator(func):
        FEATURE_REGISTRY[func.__name__] = {
            'func': func,
            'group': group,
            'requires': requires or []
        }
        return func
    return decorator