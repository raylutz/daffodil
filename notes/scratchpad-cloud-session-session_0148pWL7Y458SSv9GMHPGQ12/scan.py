import ast,sys
src=open('src/daffodil/daf.py').read(); tree=ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name in ('Daf',):
        for f in node.body:
            if isinstance(f, ast.FunctionDef):
                doc=ast.get_docstring(f) or ''
                params=[a.arg for a in f.args.args+f.args.kwonlyargs if a.arg not in('self','cls')]
                if f.args.kwarg: params.append('**')
                bad=[]
                if not doc: bad.append('NO DOC')
                elif params and 'Args:' not in doc: bad.append('no Args')
                if doc and 'Examples:' not in doc and not f.name.startswith('_'): bad.append('no Examples')
                if bad and not f.name.startswith('_') : print(f.lineno, f.name, bad)
