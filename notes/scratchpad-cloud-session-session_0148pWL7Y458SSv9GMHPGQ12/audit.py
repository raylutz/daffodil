import ast, sys, re, glob, collections
files = ['src/daffodil/daf.py', 'src/daffodil/keyedlist.py'] + [f for f in sorted(glob.glob('src/daffodil/lib/*.py')) if not f.endswith(('daf_pdf.py','md_demo.py','__init__.py'))]
tot = collections.Counter(); rows = collections.defaultdict(list)
for f in files:
    tree = ast.parse(open(f).read())
    def visit(node, cls=None):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, ast.ClassDef): visit(n, n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name.startswith('_') : continue
                key = f'{f.split("/")[-1]}:{cls + "." if cls else ""}{n.name}'
                doc = ast.get_docstring(n)
                tot['public'] += 1
                if not doc: rows['no docstring'].append(key); continue
                body = n.body
                if len(body) > 1 and isinstance(body[1], ast.Expr) and isinstance(getattr(body[1],'value',None), ast.Constant) and isinstance(body[1].value.value, str):
                    rows['second string after docstring'].append(key)
                params = [a.arg for a in n.args.args + n.args.kwonlyargs if a.arg not in ('self','cls')]
                if n.args.vararg: params.append(n.args.vararg.arg)
                if n.args.kwarg: params.append(n.args.kwarg.arg)
                if params:
                    m = re.search(r'\n\s*Args:\n(.*?)(?:\n\s*\n\s*(?:Returns|Raises|Examples|Yields|Note)|\Z)', doc, re.S)
                    if not m: rows['params but no Args section'].append(key)
                    else:
                        named = set(re.findall(r'^\s*\*{0,2}(\w+)\s*:', m.group(1), re.M))
                        miss = [p for p in params if p not in named]
                        if miss: rows['Args missing some params'].append(f'{key} {miss}')
                unann = [a.arg for a in n.args.args + n.args.kwonlyargs if a.arg not in ('self','cls') and a.annotation is None]
                if unann: rows['unannotated params'].append(f'{key} {unann}')
                if n.returns is None: rows['no return annotation'].append(key)
                if 'Examples:' not in doc: rows['no Examples'].append(key)
                if re.search(r'\n\s*Example:', doc): rows['singular Example:'].append(key)
                if n.returns is not None and 'Returns:' not in doc and not (isinstance(n.returns, ast.Constant) and n.returns.value is None):
                    rows['no Returns section'].append(key)
    visit(tree)
print('public functions and methods checked:', tot['public'])
for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
    print(f'{len(v):4}  {k}')
import json; json.dump(rows, open('/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad/audit.json','w'), indent=1)
