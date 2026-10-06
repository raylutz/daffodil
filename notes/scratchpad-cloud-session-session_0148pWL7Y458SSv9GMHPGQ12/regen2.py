import doctest, importlib, ast, re
mods = {'daffodil.daf': 'src/daffodil/daf.py', 'daffodil.keyedlist': 'src/daffodil/keyedlist.py',
        'daffodil.lib.daf_md': 'src/daffodil/lib/daf_md.py', 'daffodil.lib.daf_pandas': 'src/daffodil/lib/daf_pandas.py'}
class Rec(doctest.DocTestRunner):
    def __init__(self): super().__init__(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE); self.fails = []; self.excs = []
    def report_start(self, *a): pass
    def report_success(self, *a): pass
    def report_failure(self, out, test, example, got): self.fails.append((test, example, got))
    def report_unexpected_exception(self, out, test, example, exc_info): self.excs.append((test, example, exc_info[1]))

def doc_ranges(path):
    tree = ast.parse(open(path).read()); out = {}
    def visit(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                q = prefix + n.name
                if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], 'value', None), ast.Constant) and isinstance(n.body[0].value.value, str):
                    out.setdefault(q, []).append((n.body[0].lineno - 1, n.body[0].end_lineno - 1))
                if isinstance(n, ast.ClassDef): visit(n.body, q + '.')
    visit(tree.body, '')
    return out

for modname, path in mods.items():
    mod = importlib.import_module(modname); runner = Rec(); finder = doctest.DocTestFinder(recurse=True)
    tests = [t for t in finder.find(mod, modname) if t.examples]
    for t in tests: runner.run(t, out=lambda s: None, clear_globs=True)
    ranges = doc_ranges(path); lines = open(path).read().split('\n'); edits = []
    failed_ids = {(id(t), id(e)): g for t, e, g in runner.fails}
    for t in tests:
        q = t.name[len(modname) + 1:]
        rs = ranges.get(q)
        if not rs: continue
        # the docstring that holds these examples: take the first range whose text contains the first example
        first = t.examples[0].source.split('\n')[0]
        start, end = next((a, b) for a, b in rs if any(l.strip() == '>>> ' + first for l in lines[a:b+1]))
        cur = start
        for ex in t.examples:
            first = ex.source.split('\n')[0]
            idx = next(i for i in range(cur, end + 1) if lines[i].strip() == '>>> ' + first)
            cur = idx + 1
            got = failed_ids.get((id(t), id(ex)))
            if got is not None:
                nsrc = ex.source.count('\n'); n = len(ex.want.splitlines()); indent = re.match(r'\s*', lines[idx]).group(0)
                new = [indent + g.rstrip() for g in got.splitlines() if g.strip()]
                edits.append((idx + nsrc, n, new))
    for s, n, new in sorted(edits, reverse=True): lines[s:s + n] = new
    open(path, 'w').write('\n'.join(lines))
    print(f'{path}: fixed {len(edits)} expected outputs; unexpected exceptions: {len(runner.excs)}')
    for t, ex, exc in runner.excs: print('   EXC', t.name.split('.')[-1], '|', ex.source.strip()[:70], '->', type(exc).__name__)
