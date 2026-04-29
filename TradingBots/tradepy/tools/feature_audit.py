"""
Static audit for tradepy features: parses decorators and function bodies to
report 'requires' vs 'produces' mismatches.

Run from repository root:
    python TradingBots/tradepy/tools/feature_audit.py
"""
import os
import ast
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(__file__)
FEATURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tradepy', 'features'))

BASELINE_COLUMNS = set([
    'open', 'high', 'low', 'close', 'volume', 'timestamp', 'date', 'time',
    'trading_date', 'symbol', 'exchange', 'bid', 'ask', 'size', 'adj_close'
])


def extract_requires_and_produces(path):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    try:
        mod = ast.parse(textwrap.dedent(src))
    except Exception:
        return []

    results = []

    for node in mod.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        func_name = node.name
        # decorators
        requires = []
        group = None
        for deco in node.decorator_list:
            # look for @feature(...)
            try:
                if isinstance(deco, ast.Call):
                    func = deco.func
                    deco_name = None
                    if isinstance(func, ast.Name):
                        deco_name = func.id
                    elif isinstance(func, ast.Attribute):
                        deco_name = func.attr
                    if deco_name == 'feature':
                        # keywords
                        for kw in deco.keywords:
                            if kw.arg == 'requires':
                                val = kw.value
                                if isinstance(val, (ast.List, ast.Tuple)):
                                    for elt in val.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            requires.append(elt.value)
                                        elif isinstance(elt, ast.Str):
                                            requires.append(elt.s)
                            if kw.arg == 'group':
                                v = kw.value
                                if isinstance(v, ast.Constant):
                                    group = v.value
                                elif isinstance(v, ast.Str):
                                    group = v.s
                        # positional args (less common) - try to infer second positional as requires
                        if not requires and len(deco.args) >= 2:
                            arg = deco.args[1]
                            if isinstance(arg, (ast.List, ast.Tuple)):
                                for elt in arg.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        requires.append(elt.value)
                                    elif isinstance(elt, ast.Str):
                                        requires.append(elt.s)
            except Exception:
                pass

        # produces: look for dict literals and out[...] assignments inside the function body
        produces = []
        for n in ast.walk(node):
            if isinstance(n, ast.Dict):
                for k in n.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        produces.append(k.value)
                    elif isinstance(k, ast.Str):
                        produces.append(k.s)
            if isinstance(n, ast.Assign):
                for target in n.targets:
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
            if isinstance(n, ast.AugAssign):
                target = n.target
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

        # dedupe preserves order
        seen = set()
        produces_u = []
        for p in produces:
            if p not in seen:
                seen.add(p)
                produces_u.append(p)

        results.append({'file': path, 'func': func_name, 'requires': requires, 'produces': produces_u, 'group': group})

    return results


if __name__ == '__main__':
    all_results = []
    for root, dirs, files in os.walk(FEATURES_DIR):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            res = extract_requires_and_produces(path)
            all_results.extend(res)

    produced_by = defaultdict(list)
    for r in all_results:
        for p in r['produces']:
            produced_by[p].append(r['func'])

    func_names = {r['func'] for r in all_results}

    missing = []
    unresolved_per_func = defaultdict(list)
    for r in all_results:
        for req in r['requires']:
            if req in produced_by:
                continue
            if req in BASELINE_COLUMNS:
                continue
            if req in func_names:
                # requires references a feature name rather than a column
                unresolved_per_func[r['func']].append((req, 'feature-name'))
            else:
                unresolved_per_func[r['func']].append((req, 'unknown'))

    # Print summary
    print(f"Scanned {len(all_results)} feature functions in {FEATURES_DIR}")
    print('\nSample features (func: requires -> produces):')
    for r in sorted(all_results, key=lambda x: x['func'])[:200]:
        print(f"{r['func']}: requires={r['requires']} produces={r['produces']} group={r['group']}")

    print('\nProduced columns (sample):')
    for col, prods in list(produced_by.items())[:200]:
        print(f"{col}: produced by {prods}")

    print('\nUnresolved requires by function:')
    for func, items in unresolved_per_func.items():
        if items:
            print(f"{func}:")
            for req, typ in items:
                print(f"  - {req} ({typ})")

    # Suggestions: if requires references feature name, propose replacing with that feature's produced columns
    print('\nSuggestions:')
    for func, items in unresolved_per_func.items():
        for req, typ in items:
            if typ == 'feature-name' and req in func_names:
                produced = []
                for r in all_results:
                    if r['func'] == req:
                        produced = r['produces']
                        break
                if produced:
                    print(f"- Feature '{func}' requires feature-name '{req}'. Consider replacing with produced columns: {produced}")
                else:
                    print(f"- Feature '{func}' requires feature-name '{req}' but no produced columns were statically detected in '{req}'.")

    print('\nDone.')
