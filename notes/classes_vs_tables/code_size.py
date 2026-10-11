"""Count the custom code in each version: lines and tokens, without blanks, comments or docstrings.

Each top-level class, function or assignment is counted on its own, and put in a group:
  records    the classes, schemas or dtypes that describe the data
  load/save  reading and writing the file
  logic      the work itself
"""
import ast, io, sys, tokenize
from pathlib import Path

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.ENDMARKER}

def docstring_lines(tree: ast.AST) -> set:
    lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines

def measure(path: Path, groups: dict) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    skip = docstring_lines(tree)
    owner = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = 'imports'
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else 'other'
        else:
            name = 'other'
        for ln in range(node.lineno, node.end_lineno + 1):
            owner[ln] = groups.get(name, 'logic' if name != 'imports' else 'imports')
    out: dict = {}
    seen = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        ln = tok.start[0]
        if tok.type in SKIP or ln in skip:
            continue
        g = owner.get(ln, 'other')
        rec = out.setdefault(g, [0, 0])
        rec[1] += 1
        if ln not in seen:
            seen.add(ln); rec[0] += 1
    return out

VERSIONS = {
    'Ballot batches': [
        ('marshmallow', 'impl_marshmallow.py', {'Batch': 'records', 'BatchSchema': 'records', 'schema': 'records', 'load': 'load/save', 'dump': 'load/save'}),
        ('pydantic',    'impl_pydantic.py',    {'Batch': 'records', 'adapter': 'records', 'load': 'load/save', 'dump': 'load/save'}),
        ('daffodil',    'impl_daffodil.py',    {'dtypes': 'records', 'load': 'load/save', 'dump': 'load/save'}),
    ],
    'Gradebook': [
        ('classes',     'gradebook/gb_classes.py',          {'Grade': 'records', 'load': 'load/save', 'dump': 'load/save'}),
        ('flat table',  'gradebook/gb_daffodil.py',         {'dtypes': 'records', 'load': 'load/save', 'dump': 'load/save'}),
        ('table + summary', 'gradebook/gb_daffodil_summary.py', {'dtypes': 'records', 'summary_cols': 'records', 'load': 'load/save', 'dump': 'load/save'}),
    ],
}

for app, versions in VERSIONS.items():
    print(f'\n{app}')
    print(f'  {"":16} {"records":>9} {"load/save":>10} {"logic":>9} {"imports":>8} {"total":>11}')
    for label, file, groups in versions:
        m = measure(Path(file), groups)
        cell = lambda g: f'{m.get(g, [0,0])[0]:3} / {m.get(g, [0,0])[1]:3}'
        tot = [sum(v[0] for v in m.values()), sum(v[1] for v in m.values())]
        print(f'  {label:16} {cell("records"):>9} {cell("load/save"):>10} {cell("logic"):>9} {cell("imports"):>8} {tot[0]:4} / {tot[1]:4}')
print('\nEach cell is lines / tokens.')
