import ast, json
src = open('src/daffodil/daf.py').read(); tree = ast.parse(src)
out = []
def visit(body, cls=None):
    for n in body:
        if isinstance(n, ast.ClassDef): visit(n.body, n.name)
        elif isinstance(n, ast.FunctionDef) and len(n.body) > 1:
            b = n.body[1]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, 'value', None), ast.Constant) and isinstance(b.value.value, str):
                out.append({'name': (cls + '.' if cls else '') + n.name, 'line': n.lineno, 'doc': ast.get_docstring(n) or '', 'second': b.value.value})
visit(tree.body)
json.dump(out, open('/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad/pairs.json', 'w'), indent=1)
print(len(out)); print([o['name'].split('.')[-1] for o in out])
