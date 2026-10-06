import ast, io, sys, tokenize

def direct_returns(fn):
    out = []
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Return):
            out.append(ast.unparse(n.value) if n.value is not None else 'None')
        stack.extend(ast.iter_child_nodes(n))
    return out

def apply(path, edits, report=False):
    """edits: {lineno_of_def: (class_or_'', {param: type}, return_or_None)}"""
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    fns = {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    offs = [0]
    for l in lines: offs.append(offs[-1] + len(l))
    inserts = []   # (offset, text)
    for ln, (who, params, ret) in edits.items():
        fn = fns[ln]
        assert who == '' or fn.name == who, (ln, fn.name, who)
        if report:
            print(f"{ln:5} {fn.name:30} direct returns: {sorted(set(direct_returns(fn)))[:3]}  -> {ret}")
        end_hdr = max(ln, fn.body[0].lineno - 1)
        hdr = "".join(lines[ln-1:end_hdr])
        toks = list(tokenize.generate_tokens(io.StringIO(hdr).readline))
        depth = 0; seen_open = False; close_tok = None; prev = None
        for i, tk in enumerate(toks):
            if tk.type == tokenize.OP and tk.string == '(':
                depth += 1; seen_open = True
            elif tk.type == tokenize.OP and tk.string == ')':
                depth -= 1
                if depth == 0 and seen_open and close_tok is None:
                    close_tok = tk
            elif tk.type == tokenize.NAME and depth == 1 and tk.string in params:
                # find previous and next significant tokens
                j = i - 1
                while toks[j].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT): j -= 1
                k = i + 1
                while toks[k].type in (tokenize.NL, tokenize.COMMENT): k += 1
                if toks[j].string in ('(', ',', '*', '**') and toks[k].string in (',', ')', '='):
                    r, c = tk.end
                    base = offs[ln - 1 + r - 1] + c
                    inserts.append((base, f": {params[tk.string]}"))
                    params = {p: t for p, t in params.items() if p != tk.string}
        assert not params, (ln, fn.name, 'params not found', params)
        if ret is not None:
            assert close_tok is not None and fn.returns is None
            r, c = close_tok.end
            inserts.append((offs[ln - 1 + r - 1] + c, f" -> {ret}"))
    for off, text in sorted(inserts, reverse=True):
        src = src[:off] + text + src[off:]
    open(path, 'w').write(src)
