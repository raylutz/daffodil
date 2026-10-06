import ast, textwrap
def edit_doc(path, cls_name, method, prose=None, examples=None):
    src = open(path).read(); lines = src.split('\n'); tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    n = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method)
    d = n.body[0]; first, last = d.lineno, d.end_lineno          # 1-based, first = opening quotes line
    assert lines[last-1].strip() == '"""', (method, lines[last-1])
    if examples:
        block = textwrap.indent(textwrap.dedent(examples).strip('\n'), ' ' * 12).split('\n')
        lines[last-1:last-1] = block
    if prose:
        args_i = next(i for i in range(first-1, last-1) if lines[i].strip() == 'Args:')
        para = textwrap.indent(textwrap.fill(prose, 92), ' ' * 8).split('\n')
        lines[args_i:args_i] = para + ['']
    open(path, 'w').write('\n'.join(lines))
    print('edited', method)
