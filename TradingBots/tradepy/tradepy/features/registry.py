import inspect
import ast
import textwrap
from collections import defaultdict

FEATURE_REGISTRY = {}

def _extract_produces(func):
    produces = []
    try:
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        mod = ast.parse(src)

        # collect keys from any dict literal present (in return or assignments)
        for node in ast.walk(mod):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        produces.append(k.value)
                    elif isinstance(k, ast.Str):
                        produces.append(k.s)

        # collect keys from assignments like out['key'] = ...
        for node in ast.walk(mod):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in ('out', 'res', 'result'):
                        sl = target.slice
                        key = None
                        # different AST shapes across Python versions
                        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                            key = sl.value
                        else:
                            # ast.Index wrapper (older versions)
                            try:
                                if isinstance(sl, ast.Index):
                                    idx = sl.value
                                    if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
                                        key = idx.value
                                    elif isinstance(idx, ast.Str):
                                        key = idx.s
                            except Exception:
                                pass
                        if key:
                            produces.append(key)
            if isinstance(node, ast.AugAssign):
                target = node.target
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in ('out', 'res', 'result'):
                    sl = target.slice
                    key = None
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                        key = sl.value
                    else:
                        try:
                            if isinstance(sl, ast.Index):
                                idx = sl.value
                                if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
                                    key = idx.value
                                elif isinstance(idx, ast.Str):
                                    key = idx.s
                        except Exception:
                            pass
                    if key:
                        produces.append(key)
    except Exception:
        pass
    # unique preserve order
    seen = set()
    out = []
    for p in produces:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def feature(group=1, requires=None):
    def decorator(func):
        produces = _extract_produces(func)
        FEATURE_REGISTRY[func.__name__] = {
            'func': func,
            'group': group,
            'requires': requires or [],
            'produces': produces,
        }
        return func
    return decorator