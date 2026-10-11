"""Count all the custom code in each version: lines and tokens, without blanks, comments or docstrings.

Every line goes in one group, by what it is for:
  describe   the class line, fields and __init__, or a dtypes dict: what the data is
  boiler     __repr__, __eq__ and other dunder methods
  load/save  schemas, from_dict, to_dict, load, dump, and the checks inside them
  logic      the work itself
  imports    import lines
A method goes by its name. A top-level name can be given a group per file.
"""
import ast, io, tokenize
from pathlib import Path

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.ENDMARKER}
GROUPS = ['describe', 'boiler', 'load/save', 'logic', 'imports']
LOAD_SAVE_NAMES = {'from_dict', 'to_dict', 'load', 'dump', 'make'}

def docstring_lines(tree: ast.AST) -> set:
    lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines

def span(node: ast.AST) -> range:
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return range(start, node.end_lineno + 1)

def group_of_function(name: str) -> str:
    if name == '__init__':
        return 'describe'
    if name.startswith('__') and name.endswith('__'):
        return 'boiler'
    if name in LOAD_SAVE_NAMES:
        return 'load/save'
    return 'logic'

def assign_groups(tree: ast.Module, overrides: dict) -> dict:
    owner: dict = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            group = 'imports'
        elif isinstance(node, ast.ClassDef):
            group = overrides.get(node.name, 'describe')
            for ln in span(node):
                owner[ln] = group
            if node.name not in overrides:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        for ln in span(item):
                            owner[ln] = group_of_function(item.name)
            continue
        elif isinstance(node, ast.FunctionDef):
            group = overrides.get(node.name, group_of_function(node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, 'target', None)
            name = getattr(target, 'id', '') if target is not None else ''
            group = overrides.get(name, 'logic')
        else:
            group = 'logic'
        for ln in span(node):
            owner[ln] = group
    return owner

def measure(path: Path, overrides: dict) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    skip = docstring_lines(tree)
    owner = assign_groups(tree, overrides)
    out = {g: [0, 0] for g in GROUPS}
    seen: set = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        ln = tok.start[0]
        if tok.type in SKIP or ln in skip:
            continue
        rec = out[owner.get(ln, 'logic')]
        rec[1] += 1
        if ln not in seen:
            seen.add(ln)
            rec[0] += 1
    return out

VERSIONS = {
    'Ballot batches': [
        ('plain classes', 'impl_plain.py',       {}),
        ('marshmallow',   'impl_marshmallow.py', {'BatchSchema': 'load/save', 'schema': 'load/save'}),
        ('pydantic',      'impl_pydantic.py',    {'adapter': 'load/save'}),
        ('daffodil',      'impl_daffodil.py',    {'dtypes': 'describe'}),
    ],
    'Gradebook': [
        ('plain classes',   'gradebook/gb_classes_plain.py',    {}),
        ('pydantic',        'gradebook/gb_classes.py',          {}),
        ('daffodil table',  'gradebook/gb_daffodil.py',         {'dtypes': 'describe'}),
        ('daffodil indirect', 'gradebook/gb_daffodil_indirect.py', {'dtypes': 'describe', 'load_json': 'load/save', 'dump_json': 'load/save'}),
        ('daffodil + summary', 'gradebook/gb_daffodil_summary.py', {'dtypes': 'describe', 'summary_cols': 'describe'}),
    ],
}

if __name__ == '__main__':
    for app, versions in VERSIONS.items():
        print(f'\n{app}')
        print(f'  {"":16}' + ''.join(f'{g:>11}' for g in GROUPS) + f'{"total":>12}')
        for label, file, overrides in versions:
            m = measure(Path(file), overrides)
            cells = ''.join(f'{m[g][0]:>5} /{m[g][1]:>4}' for g in GROUPS)
            tot = (sum(v[0] for v in m.values()), sum(v[1] for v in m.values()))
            print(f'  {label:16}{cells}{tot[0]:>6} /{tot[1]:>5}')
    print('\nEach cell is lines / tokens.')
