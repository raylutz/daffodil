"""
Move the Daf methods of one docs page into a module of their own, by way A of the split plan.
See notes/engineering_notebook/2026-10-25_1627_addon_hook_and_mixin_split_plan.md.
It was used once, on 2026-10-26, to split daf.py, one group per commit.
It writes a run-time import of each daf.py name into the functions that use it.
Those imports cost 0.2 us a call, so they were then replaced by _NAMES_FROM_DAF,
which daf.py fills at the end of its import. See the notebook entry above.
Run it on Python 3.10 or 3.11, where an f-string is one token:
    uv run --no-project --python 3.10 python notes/scripts/move_daf_group.py ...

usage: python move_daf_group.py <page> <module> <title> [--extra name,name] [--keep name,name] [--dry]

page:   the docs page in docsite/api/daf, without .md, as select.
module: the new module in src/daffodil, without .py, as daf_select.
title:  the words for the class-body header, as "selecting and indexing".
--extra: private methods on no page that move with the group.
--keep:  methods on the page that stay in daf.py.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAF_PY = ROOT / 'src/daffodil/daf.py'

# ---- the names daf.py imports, and where a moved module gets each one

STDLIB_MODULES = ['io', 'csv', 'copy', 're', 'json', 'time', 'functools', 'itertools', 'operator', 'typing']
LIB_MODULES = {
    'daf_utils': 'import daffodil.lib.daf_utils    as daf_utils',
    'md': 'import daffodil.lib.daf_md       as md',
    'daf_pandas': 'import daffodil.lib.daf_pandas   as daf_pandas',
    'daf_pdf': 'import daffodil.lib.daf_pdf      as daf_pdf',
    'daf_schema': 'import daffodil.lib.daf_schema   as daf_schema',
    'daf_ods': 'import daffodil.lib.daf_ods      as daf_ods',
    'daf_namespace': 'import daffodil.lib.daf_namespace as daf_namespace',
}
TYPING_NAMES = ['List', 'Dict', 'Any', 'Tuple', 'cast', 'Type', 'Callable', 'Generic', 'TypeVar', 'ClassVar', 'Literal', 'overload']
ABC_NAMES = ['Iterable', 'Collection', 'Sequence', 'Iterator', 'Hashable']
FROM_UTILS = ['_MISSING', 'NULL']                     # module-level, so a default value can use them.


def main() -> None:
    args = sys.argv[1:]
    dry = '--dry' in args
    args = [a for a in args if a != '--dry']
    extra, keep = [], []
    if '--extra' in args:
        i = args.index('--extra'); extra = args[i + 1].split(','); del args[i:i + 2]
    if '--keep' in args:
        i = args.index('--keep'); keep = args[i + 1].split(','); del args[i:i + 2]
    page, module, title = args

    src = DAF_PY.read_text()
    lines = src.split('\n')
    tree = ast.parse(src)
    daf_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Daf')

    # daf.py's own module-level names: defined there, not imported.
    daf_defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            daf_defined.add(node.name)
        elif isinstance(node, ast.Assign):
            daf_defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    t_aliases = set(re.findall(r'\bT_\w+', re.search(r'from daffodil\.lib\.daf_types import (.*?)# noqa', src, re.S).group(1)))

    # ---- the methods to move

    page_path = ROOT / f'docsite/api/daf/{page}.md'
    page_text = page_path.read_text()
    on_page = re.findall(r'^::: daffodil\.daf\.Daf\.(\w+)', page_text, re.M)
    nodes_of: dict = {}
    for n in daf_cls.body:
        if isinstance(n, ast.FunctionDef):
            nodes_of.setdefault(n.name, []).append(n)
    names, skipped = [], []
    for name in on_page + extra:
        nodes = nodes_of.get(name)
        decos = [ast.unparse(d) for n in nodes for d in n.decorator_list] if nodes else []
        if not nodes or name.startswith('__') or name in keep or any(d == 'property' or d.endswith('.setter') for d in decos):
            skipped.append(name)
            continue
        names.append(name)
    print('moving:', names)
    print('staying in daf.py:', skipped)

    # ---- the line spans, with any comment lines right above a def

    spans = []
    for name in names:
        nodes = nodes_of[name]
        start = min((n.decorator_list[0].lineno if n.decorator_list else n.lineno) for n in nodes)
        while lines[start - 2].strip().startswith('#') and not lines[start - 2].strip().startswith('#===='):
            start -= 1
        end = max(n.end_lineno for n in nodes)
        spans.append((start, end, name))
    spans.sort()
    for (s1, e1, n1), (s2, e2, n2) in zip(spans, spans[1:]):
        assert e1 < s2, (n1, n2)

    # ---- the lines inside multi-line strings that are not docstrings keep their text exactly

    docstring_lines = set()
    for name in names:
        for n in nodes_of[name]:
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                docstring_lines.update(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    frozen = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0] and tok.start[0] not in docstring_lines:
            frozen.update(range(tok.start[0] + 1, tok.end[0] + 1))
    assert sys.version_info < (3, 12), 'run on Python 3.10 or 3.11, where an f-string is one token'

    # ---- build each function's text

    def runtime_names(fn: ast.FunctionDef) -> set:
        skip = set()
        for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs + [fn.args.vararg, fn.args.kwarg]:
            if a is not None and a.annotation is not None:
                skip.update(id(s) for s in ast.walk(a.annotation))
        if fn.returns is not None:
            skip.update(id(s) for s in ast.walk(fn.returns))
        for s in ast.walk(fn):
            if isinstance(s, ast.AnnAssign):
                skip.update(id(x) for x in ast.walk(s.annotation))
        for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]:
            skip.update(id(x) for x in ast.walk(d))
        return {s.id for s in ast.walk(fn) if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Load) and id(s) not in skip}

    def default_names(fn: ast.FunctionDef) -> set:
        found = set()
        for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]:
            found.update(s.id for s in ast.walk(d) if isinstance(s, ast.Name))
        return found

    out_blocks, assigns, used_all = [], [], set()
    for start, end, name in spans:
        nodes = nodes_of[name]
        fname = name if name.startswith('_') else f'_{name}'
        kind = 'method'
        decos_all = {ast.unparse(d) for n in nodes for d in n.decorator_list}
        if 'staticmethod' in decos_all:
            kind = 'static'
        elif 'classmethod' in decos_all:
            kind = 'class'
        edits = {}                                    # line number -> new text
        work = {ln: lines[ln - 1] for ln in range(start, end + 1)}
        drop = set()
        for n in nodes:
            for d in n.decorator_list:
                if ast.unparse(d) in ('staticmethod', 'classmethod'):
                    drop.add(d.lineno)
            # the def name
            ln = n.lineno
            work[ln] = re.sub(rf'\bdef {name}\(', f'def {fname}(', work[ln], count=1)
            # the first parameter
            if kind != 'static' and n.args.args:
                a0 = n.args.args[0]
                if a0.annotation is None:
                    ann = {'self': ": 'Daf'", 'cls': ": 'type[Daf]'"}.get(a0.arg)
                    assert ann, (name, a0.arg)
                    text = work[a0.lineno]
                    col = a0.col_offset + len(a0.arg)
                    if a0.lineno == ln and fname != name:
                        col += 1                      # the leading underscore added to the name
                    assert text[col - len(a0.arg):col] == a0.arg, (name, text)
                    work[a0.lineno] = text[:col] + ann + text[col:]
        impl = nodes[-1]
        local = sorted((runtime_names(impl) & daf_defined) - {'NULL'})
        bad_defaults = (default_names(impl) & daf_defined) - {'NULL'}
        assert not bad_defaults, (name, bad_defaults)
        if local:
            body0 = impl.body[0]
            after = body0.end_lineno if (isinstance(body0, ast.Expr) and isinstance(body0.value, ast.Constant)) else impl.body[0].lineno - 1
            indent = ' ' * (impl.col_offset + 4)
            work[after] = work[after] + '\n' + f"{indent}from daffodil.daf import {', '.join(local)}"
        block = []
        for ln in range(start, end + 1):
            if ln in drop:
                continue
            text = work[ln]
            parts = text.split('\n')
            new_parts = []
            for p_i, part in enumerate(parts):
                if ln in frozen and p_i == 0:
                    new_parts.append(part)
                elif part.startswith('    '):
                    new_parts.append(part[4:])
                else:
                    assert part.strip() == '', (name, ln, part)
                    new_parts.append(part.strip())
            block.append('\n'.join(new_parts))
        out_blocks.append('\n'.join(block))
        for n in nodes:
            used_all |= {s.id for s in ast.walk(n) if isinstance(s, ast.Name)}
            # names inside a quoted annotation, as 'T_ls | Iterable | Daf'
            used_all |= {w for s in ast.walk(n) if isinstance(s, ast.Constant) and isinstance(s.value, str) and len(s.value) < 120
                         for w in re.findall(r'[A-Za-z_]\w*', s.value)}
            used_all |= {s.value.id for s in ast.walk(n) if isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name)}
        target = f'{module}.{fname}'
        assigns.append((name, {'method': target, 'static': f'staticmethod({target})', 'class': f'classmethod({target})'}[kind]))

    # ---- the module header

    header = [f'# {module}.py', '# copyright (c) 2026 Ray Lutz', '"""', f'Methods of Daf for {title}.', '',
              'Each function here takes the Daf as its first argument, or the class for a classmethod.',
              'It is assigned in the Daf class body under its Daf name, without the underscore.',
              'The docs show it under that name.', '',
              'A function that needs a name defined in daf.py, such as Daf, imports it when it runs.',
              'An import at the top would be circular, since daf.py imports this module.', '"""', '',
              'from __future__ import annotations', '']
    std = [m for m in STDLIB_MODULES if m in used_all]
    header += [f'import {m}' for m in std]
    if 'Path' in used_all:
        header.append('from pathlib import Path')
    if std or 'Path' in used_all:
        header.append('')
    ts = sorted(t for t in t_aliases if t in used_all)
    if ts:
        header.append(f"from daffodil.lib.daf_types import {', '.join(ts)}")
    libs = [v for k, v in LIB_MODULES.items() if k in used_all]
    header += libs
    fu = [n for n in FROM_UTILS if n in used_all]
    if fu:
        header.append(f"from daffodil.lib.daf_utils import {', '.join(fu)}")
    if 'KeyedList' in used_all:
        header.append('from daffodil.keyedlist import KeyedList')
    header.append('')
    tn = [n for n in TYPING_NAMES if n in used_all]
    an = [n for n in ABC_NAMES if n in used_all]
    header.append(f"from typing import {', '.join(tn + ['TYPE_CHECKING'])}")
    if an:
        header.append(f"from collections.abc import {', '.join(an)}")
    ann_only = sorted((used_all & {'T_klodaf', 'T_dodaf', 'DafIterator'}) | {'Daf'})
    header += ['', 'if TYPE_CHECKING:       # for the annotations only. A real import would be circular.',
               f"    from daffodil.daf import {', '.join(ann_only)}", '', '', '']
    body = '\n\n\n'.join(out_blocks)
    mod_path = ROOT / f'src/daffodil/{module}.py'
    if mod_path.exists():
        print('module exists, appending:', mod_path)
        new_mod = mod_path.read_text().rstrip('\n') + '\n\n\n' + body + '\n'
    else:
        new_mod = '\n'.join(header) + body + '\n'

    # ---- daf.py: the spans out, one block of assignments in at the first span

    width = max(len(n) for n, _ in assigns)
    block = [f'    #==== {title}: functions in {module}.py', '']
    block += [f'    {n.ljust(width)} = {v}' for n, v in assigns]
    new_lines = []
    first_start = spans[0][0]
    span_iter = {s: e for s, e, _ in spans}
    ln = 1
    while ln <= len(lines):
        if ln in span_iter:
            if ln == first_start:
                new_lines.extend(block + [''])
            end = span_iter[ln]
            ln = end + 1
            # drop the blank lines that separated the removed method from the next one
            while ln <= len(lines) and lines[ln - 1].strip() == '' and new_lines and new_lines[-1].strip() == '':
                ln += 1
            continue
        new_lines.append(lines[ln - 1])
        ln += 1
    new_daf = '\n'.join(new_lines)
    imp_anchor = 'import daffodil.daf_join          as daf_join\n'
    imp_line = f'import daffodil.{module}'.ljust(len('import daffodil.daf_join          ')) + f'as {module}\n'
    if imp_line not in new_daf and module != 'daf_join':
        new_daf = new_daf.replace(imp_anchor, imp_anchor + imp_line, 1)

    # ---- the docs: page entries and cross-references

    def entry(m: re.Match) -> str:
        name = m.group(1)
        if name not in names:
            return m.group(0)
        rest = m.group(2) or ''
        if 'options:' in rest:
            return f'::: daffodil.{module}._{name}{rest.rstrip()}\n      heading: "{name}"\n'
        return f'::: daffodil.{module}._{name}\n    options:\n      heading: "{name}"\n'
    new_page = re.sub(r'^::: daffodil\.daf\.Daf\.(\w+)\n((?:    .*\n)*)', entry, page_text, flags=re.M)
    xref_files = list((ROOT / 'docsite').rglob('*.md')) + list((ROOT / 'src/daffodil').rglob('*.py')) + [ROOT / 'README.md']
    if mod_path not in xref_files:
        xref_files.append(mod_path)                 # a new module is not on disk yet.
    xref_new = {}
    for f in xref_files:
        t = new_daf if f == DAF_PY else (new_mod if f == mod_path else (new_page if f == page_path else f.read_text()))
        t2 = re.sub(r'\]\[daffodil\.daf\.Daf\.(\w+)\]', lambda m: f']{"["}daffodil.{module}._{m.group(1)}]' if m.group(1) in names else m.group(0), t)
        if t2 != t:
            xref_new[f] = t2
    if dry:
        print(new_mod[:3000]); print('...'); print('\n'.join(block)); print(new_page); print('xrefs in', list(xref_new))
        return
    mod_path.write_text(new_mod)
    DAF_PY.write_text(xref_new.pop(DAF_PY, new_daf))
    page_path.write_text(xref_new.pop(page_path, new_page))
    if mod_path in xref_new:
        mod_path.write_text(xref_new.pop(mod_path))
    for f, t in xref_new.items():
        f.write_text(t)
    print('wrote', mod_path, 'and', len(xref_new), 'files of cross-references')


if __name__ == '__main__':
    main()
