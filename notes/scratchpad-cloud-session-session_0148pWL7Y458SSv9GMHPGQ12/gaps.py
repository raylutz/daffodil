import ast, sys
p = sys.argv[1]
src = open(p).read(); t = ast.parse(src); lines = src.splitlines()
cls_of = {}
for c in ast.walk(t):
    if isinstance(c, ast.ClassDef):
        for n in c.body:
            if isinstance(n, ast.FunctionDef): cls_of[id(n)] = c.name
for node in ast.walk(t):
    if not isinstance(node, ast.FunctionDef): continue
    a = node.args
    params = a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else [])
    missing = [x.arg for x in params if x.annotation is None and x.arg not in ('self', 'cls')]
    noret = node.returns is None and node.name != '__init__'
    if not (missing or noret): continue
    rets = []
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and n.value is not None:
            rets.append(ast.unparse(n.value)[:60])
    decos = [ast.unparse(d) for d in node.decorator_list]
    print(f"{node.lineno:5} {cls_of.get(id(node), ''):10}.{node.name}({', '.join(x.arg for x in params)}) {decos or ''}")
    print(f"        missing: {missing}  return: {'MISSING' if noret else 'ok'}   returns: {sorted(set(rets))[:4] or '(none: returns None)'}")
