import ast, sys
src = open('src/daffodil/daf.py').read(); tree = ast.parse(src); lines = src.split('\n')
daf = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Daf')
want = sys.argv[1:]
for n in daf.body:
    if isinstance(n, ast.FunctionDef) and n.name in want:
        end = n.body[0].end_lineno if isinstance(n.body[0], ast.Expr) else n.lineno
        print(f'--- {n.name} (line {n.lineno})'); print('\n'.join(lines[n.lineno-1:end]))
        # first code line after docstring(s)
        body = [b for b in n.body[1:] if not (isinstance(b, ast.Expr) and isinstance(getattr(b,'value',None), ast.Constant))]
        if body: print('   ...CODE:', '\n'.join(lines[body[0].lineno-1:min(body[-1].end_lineno, body[0].lineno+7)]))
