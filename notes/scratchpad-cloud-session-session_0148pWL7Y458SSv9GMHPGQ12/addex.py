import ast, sys, textwrap
def add_examples(path, cls_name, edits):
    """edits: {method: examples_block (unindented text starting with '>>>' lines)}"""
    src = open(path).read(); lines = src.split('\n'); tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    todo = []
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in edits:
            doc = n.body[0]
            assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant), n.name
            end = doc.end_lineno            # 1-based line of the closing quotes
            assert lines[end-1].strip() == '"""', (n.name, lines[end-1])
            assert 'Examples:' not in ast.get_docstring(n), n.name
            todo.append((end, n.name))
    assert len(todo) == len(edits), (set(edits) - {t[1] for t in todo})
    for end, name in sorted(todo, reverse=True):
        block = textwrap.indent(textwrap.dedent(edits[name]).strip('\n'), ' ' * 12)
        new = ['', '        Examples:'] + block.split('\n')
        # keep a blank line before Examples only if the docstring does not already end with one
        lines[end-1:end-1] = new
    open(path, 'w').write('\n'.join(lines))
    print('added Examples to:', ', '.join(n for _, n in sorted(todo)))
