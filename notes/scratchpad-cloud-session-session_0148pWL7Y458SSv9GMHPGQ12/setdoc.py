"""setdoc(path, {qualname: text}) replaces the first-string docstring of each function/class.
text is dedented body (no quotes). If the old docstring is followed by a second string,
that second string is left alone. If no docstring exists, one is inserted."""
import ast, textwrap, sys

def setdoc(path, docs):
    src = open(path).read()
    lines = src.split('\n')
    tree = ast.parse(src)
    targets = {}
    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + ch.name
                if q in docs:
                    targets[q] = ch
                walk(ch, q + '.')
    walk(tree, '')
    missing = set(docs) - set(targets)
    if missing:
        raise SystemExit(f"not found: {sorted(missing)}")
    edits = []
    for q, node in targets.items():
        body0 = node.body[0]
        indent = ' ' * (body0.col_offset)
        text = textwrap.dedent(docs[q]).strip('\n')
        assert '"""' not in text, q
        opener = 'r"""' if '\\' in text else '"""'
        new = [indent + opener] + [(indent + l if l.strip() else '') for l in text.split('\n')] + [indent + '"""']
        is_doc = (isinstance(body0, ast.Expr) and isinstance(body0.value, ast.Constant) and isinstance(body0.value.value, str))
        if is_doc:
            edits.append((body0.lineno - 1, body0.end_lineno, new))
        else:
            edits.append((body0.lineno - 1, body0.lineno - 1, new + ['']))
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = new
    open(path, 'w').write('\n'.join(lines))
