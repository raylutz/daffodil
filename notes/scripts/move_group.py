"""
Move the methods of one API docs page of a class into a module of their own, by way A of the split plan.
See notes/engineering_notebook/2026-10-25_1627_addon_hook_and_mixin_split_plan.md.
It is the general form of move_daf_group.py, written for the split of sqldaf.py.

Each method becomes a function named with a leading underscore, which takes the table as self.
The class assigns it by name. A staticmethod or classmethod is wrapped there.
The names of the source module that a function uses when it runs are listed in the new module's
_NAMES_FROM_<SOURCE>, and the source module sets them at the end of its import.
The script writes that loop the first time, and adds each new module to it.

After a run, mypy may report an assignment in the class that differs from the Daf signature.
Add `# type: ignore[assignment]` on each line it names, as the def had `# type: ignore[override]`.

Run it on Python 3.10 or 3.11, where an f-string is one token:
    uv run --no-project --python 3.10 python notes/scripts/move_group.py ...

usage: python move_group.py <source> <class> <page> <module> <title> [--extra a,b] [--keep a,b] [--dry]

source: the module that holds the class, as sqldaf.
class:  the class, as SqlDaf.
page:   the docs page in docsite/api/<source>, without .md, as select.
module: the new module in src/daffodil, without .py, as sqldaf_select.
title:  the words for the class-body header, as "selecting".
--extra: private methods on no page that move with the group.
--keep:  methods on the page that stay in the source module.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = sys.argv[1:]
    dry = '--dry' in args
    args = [a for a in args if a != '--dry']
    extra, keep = [], []
    if '--extra' in args:
        i = args.index('--extra'); extra = args[i + 1].split(','); del args[i:i + 2]
    if '--keep' in args:
        i = args.index('--keep'); keep = args[i + 1].split(','); del args[i:i + 2]
    source, cls_name, page, module, title = args
    assert sys.version_info < (3, 12), 'run on Python 3.10 or 3.11, where an f-string is one token'

    src_path = ROOT / f'src/daffodil/{source}.py'
    src = src_path.read_text()
    lines = src.split('\n')
    tree = ast.parse(src)
    the_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    names_var = f'_NAMES_FROM_{source.upper()}'

    # ---- the source module's own names, and its imports

    defined = set()
    import_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset == 0 and node.lineno < the_cls.lineno:
            import_nodes.append(node)

    # ---- the methods to move

    page_path = ROOT / f'docsite/api/{source}/{page}.md'
    page_text = page_path.read_text()
    on_page = re.findall(rf'^::: daffodil\.{source}\.{cls_name}\.(\w+)', page_text, re.M)
    nodes_of: dict = {}
    for n in the_cls.body:
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
    print('staying:', skipped)

    # ---- the line spans, with any comment lines right above a def

    spans = []
    for name in names:
        nodes = nodes_of[name]
        start = min((n.decorator_list[0].lineno if n.decorator_list else n.lineno) for n in nodes)
        while lines[start - 2].strip().startswith('#') and not re.match(r'#\s*(=|-){3,}', lines[start - 2].strip()):
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

    # ---- the names each function uses

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

    def all_names(fn: ast.FunctionDef) -> set:
        found = {s.id for s in ast.walk(fn) if isinstance(s, ast.Name)}
        found |= {s.value.id for s in ast.walk(fn) if isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name)}
        # names inside a quoted annotation, as 'T_ls | Iterable | Daf'
        found |= {w for s in ast.walk(fn) if isinstance(s, ast.Constant) and isinstance(s.value, str) and len(s.value) < 120
                  for w in re.findall(r'[A-Za-z_]\w*', s.value)}
        return found

    # ---- build each function's text

    out_blocks, assigns, used_all, runtime_from_source = [], [], set(), set()
    for start, end, name in spans:
        nodes = nodes_of[name]
        fname = name if name.startswith('_') else f'_{name}'
        decos_all = {ast.unparse(d) for n in nodes for d in n.decorator_list}
        kind = 'static' if 'staticmethod' in decos_all else ('class' if 'classmethod' in decos_all else 'method')
        work = {ln: lines[ln - 1] for ln in range(start, end + 1)}
        drop = set()
        for n in nodes:
            for d in n.decorator_list:
                if ast.unparse(d) in ('staticmethod', 'classmethod'):
                    drop.add(d.lineno)
                work[d.lineno] = re.sub(r'\s*# type: ignore\[override\]', '', work[d.lineno])
            ln = n.lineno
            work[ln] = re.sub(rf'\bdef {name}\(', f'def {fname}(', work[ln], count=1)
            # A module function overrides nothing. The class assignment carries any ignore mypy needs.
            work[ln] = re.sub(r'\s*# type: ignore\[override\]', '', work[ln])
            if kind != 'static' and n.args.args:
                a0 = n.args.args[0]
                if a0.annotation is None:
                    ann = {'self': f": '{cls_name}'", 'cls': f": 'type[{cls_name}]'"}.get(a0.arg)
                    assert ann, (name, a0.arg)
                    text = work[a0.lineno]
                    col = a0.col_offset + len(a0.arg)
                    if a0.lineno == ln and fname != name:
                        col += 1                      # the leading underscore added to the name
                    assert text[col - len(a0.arg):col] == a0.arg, (name, text)
                    work[a0.lineno] = text[:col] + ann + text[col:]
        impl = nodes[-1]
        runtime_from_source |= runtime_names(impl) & defined
        bad = default_names(impl) & defined
        assert not bad, f'{name}: a default uses names of {source}.py: {bad}'
        block = []
        for ln in range(start, end + 1):
            if ln in drop:
                continue
            text = work[ln]
            if ln in frozen:
                block.append(text)
            elif text.startswith('    '):
                block.append(text[4:])
            else:
                assert text.strip() == '', (name, ln, text)
                block.append('')
        out_blocks.append('\n'.join(block))
        for n in nodes:
            used_all |= all_names(n)
        target = f'{module}.{fname}'
        assigns.append((name, {'method': target, 'static': f'staticmethod({target})', 'class': f'classmethod({target})'}[kind]))

    # ---- the module header: the source's imports that are used, then its own names for mypy

    import_lines = []
    for node in import_nodes:
        if isinstance(node, ast.Import):
            keep_aliases = [a for a in node.names if (a.asname or a.name.split('.')[0]) in used_all]
            if keep_aliases:
                import_lines.append(ast.unparse(ast.Import(names=keep_aliases)))
        else:
            keep_aliases = [a for a in node.names if (a.asname or a.name) in used_all]
            if keep_aliases:
                import_lines.append(ast.unparse(ast.ImportFrom(module=node.module, names=keep_aliases, level=node.level)))
    # TYPE_CHECKING and Tuple join the typing import, or start one.
    typing_i = next((i for i, l in enumerate(import_lines) if l.startswith('from typing import ')), None)
    if typing_i is None:
        import_lines.append('from typing import TYPE_CHECKING, Tuple')
    else:
        have_ls = [x.strip() for x in import_lines[typing_i][len('from typing import '):].split(',')]
        import_lines[typing_i] = 'from typing import ' + ', '.join(have_ls + [x for x in ('TYPE_CHECKING', 'Tuple') if x not in have_ls])
    ann_from_source = sorted((used_all & defined) | {cls_name}, key=lambda x: (x != cls_name, x))
    runtime_ls = sorted(runtime_from_source, key=lambda x: (x != cls_name, x))
    header = [f'# {module}.py', '# copyright (c) 2026 Ray Lutz', '"""', f'Methods of {cls_name} for {title}.', '',
              'Each function here takes the table as its first argument, or the class for a classmethod.',
              f'It is assigned in the {cls_name} class body under its name, without the underscore.',
              'The docs show it under that name.', '',
              f'Some functions use names defined in {source}.py, such as {cls_name}, when they run.',
              f'{source}.py sets those names here, as listed in {names_var}, once they exist.',
              f'An import at the top would be circular, since {source}.py imports this module.', '"""', '']
    # The future import only if the source has it. With it, a doctest that defines a @schemaclass
    # gets string annotations, so a SqlDaf example would read its dtypes as text.
    if 'from __future__ import annotations' in src:
        header += ['from __future__ import annotations', '']
    header += import_lines
    header += ['', f'if TYPE_CHECKING:       # for mypy. {source}.py sets the names below when the program runs.',
               f"    from daffodil.{source} import {', '.join(ann_from_source)}", '']
    if runtime_ls:
        header += [f'# The names of {source}.py that the functions here use when they run.',
                   f'# {source}.py sets each one here once it exists. An import at the top would be circular.',
                   f"{names_var}: Tuple[str, ...] = ({', '.join(repr(n) for n in runtime_ls)}{',' if len(runtime_ls) == 1 else ''})", '']
    header += ['', '']
    mod_path = ROOT / f'src/daffodil/{module}.py'
    assert not mod_path.exists(), f'{mod_path} exists. This script makes new modules only.'
    new_mod = '\n'.join(header) + '\n\n\n'.join(out_blocks) + '\n'

    # ---- the source: the spans out, one block of assignments in at the first span

    width = max(len(n) for n, _ in assigns)
    block_ls = [f'    #==== {title}: functions in {module}.py', '']
    block_ls += [f'    {n.ljust(width)} = {v}' for n, v in assigns]
    new_lines = []
    first_start = spans[0][0]
    span_end = {s: e for s, e, _ in spans}
    ln = 1
    while ln <= len(lines):
        if ln in span_end:
            if ln == first_start:
                new_lines.extend(block_ls + [''])
            ln = span_end[ln] + 1
            while ln <= len(lines) and lines[ln - 1].strip() == '' and new_lines and new_lines[-1].strip() == '':
                ln += 1
            continue
        new_lines.append(lines[ln - 1])
        ln += 1
    new_src = '\n'.join(new_lines)

    # The import of the new module, after the last import of a group module, or after the source's imports.
    imp_line = f'import daffodil.{module} as {module}\n'
    group_imps = list(re.finditer(rf'^import daffodil\.{source}_\w+ as {source}_\w+\n', new_src, re.M))
    if group_imps:
        at = group_imps[-1].end()
    else:
        last_import = max(n.end_lineno for n in import_nodes)
        at = len('\n'.join(new_src.split('\n')[:last_import])) + 1
    new_src = new_src[:at] + imp_line + new_src[at:]

    # The loop that sets each module's names, before the blocker if there is one.
    loop_head = f'for _group_module in ('
    if loop_head not in new_src:
        loop = (f"\n# The group modules hold most of the {cls_name} methods, by topic.\n"
                f"# Some of their functions use names defined here, such as {cls_name}, when they run.\n"
                f"# Each module lists those names in {names_var}, and they are set in it here, once they all exist.\n"
                f"# A module cannot import them at its top, since this module imports it first.\n"
                f"for _group_module in ({module},):\n"
                f"    for _name in getattr(_group_module, '{names_var}', ()):\n"
                f"        setattr(_group_module, _name, globals()[_name])\n\n")
        anchor = '# ---- block every Daf method' if '# ---- block every Daf method' in new_src else None
        if anchor:
            new_src = new_src.replace(anchor, loop.lstrip('\n') + '\n' + anchor, 1)
        else:
            new_src = new_src.rstrip('\n') + '\n\n' + loop
    else:
        new_src = re.sub(r'for _group_module in \(([^)]*)\):', lambda m: f"for _group_module in ({m.group(1).rstrip(',')}{', ' if m.group(1).rstrip(',') else ''}{module},):".replace(',,', ','), new_src, count=1)

    # ---- the docs: page entries and cross-references

    def entry(m: re.Match) -> str:
        name = m.group(1)
        if name not in names:
            return m.group(0)
        rest = m.group(2) or ''
        fname = name if name.startswith('_') else f'_{name}'
        if 'options:' in rest:
            return f'::: daffodil.{module}.{fname}{rest.rstrip()}\n      heading: "{name}"\n'
        return f'::: daffodil.{module}.{fname}\n    options:\n      heading: "{name}"\n'
    new_page = re.sub(rf'^::: daffodil\.{source}\.{cls_name}\.(\w+)\n((?:    .*\n)*)', entry, page_text, flags=re.M)
    xref_files = list((ROOT / 'docsite').rglob('*.md')) + list((ROOT / 'src/daffodil').rglob('*.py')) + [ROOT / 'README.md', mod_path]
    texts = {src_path: new_src, mod_path: new_mod, page_path: new_page}
    for f in xref_files:
        t = texts.get(f) if f in texts else f.read_text()
        t2 = re.sub(rf'\]\[daffodil\.{source}\.{cls_name}\.(\w+)\]',
                    lambda m: f']{"["}daffodil.{module}.{m.group(1) if m.group(1).startswith("_") else "_" + m.group(1)}]' if m.group(1) in names else m.group(0), t)
        if f in texts or t2 != t:
            texts[f] = t2
    if dry:
        print(new_mod[:2500]); print('...'); print('\n'.join(block_ls)); print([str(f) for f in texts])
        return
    for f, t in texts.items():
        f.write_text(t)
    print('wrote', mod_path, 'and', len(texts) - 1, 'other files')


if __name__ == '__main__':
    main()
