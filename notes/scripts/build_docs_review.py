# build_docs_review.py
#
# Build one Markdown file that holds all of daffodil's user documentation, for a review by
# another model, such as ChatGPT. It writes notes/docs_review/daffodil_docs_for_review.md.
#
# The file has a review prompt, then the README, then every page of the docs site in the order
# of the nav in mkdocs.yml. On an API page, each "::: name" directive is replaced by the
# signature and docstring of that name.
#
# The entries come from griffe, the parser that mkdocstrings uses. So a signature reads as on
# the site: a default is its source text, such as _MISSING, not the repr of an object. A
# "heading:" option names the entry, as on the site. A class directive without "members: false"
# also lists the public members that have a docstring, as the site does.
#
# Usage, from the repo root:   uv run python notes/scripts/build_docs_review.py

import os
import re
from typing import Any, Iterator, List, Tuple

import griffe
import yaml

T_ls = List[str]

OUT_PATH: str = 'notes/docs_review/daffodil_docs_for_review.md'

PROMPT: str = """# Daffodil documentation, for review

Daffodil is a small, fast, pure-Python library of 2-D tables. The main class is Daf: a list of rows,
each a plain Python list, with the column names in a dict. KeyedList is a row that can be read by name.
SqlDaf, new in version 0.8.0, keeps the rows in a SQLite table, for tables too large for memory.

This file holds all of the user documentation, in this order:

1. README.md, the front page on GitHub and PyPI.
2. Every page of the docs site, in the order of its menu: the Home page, the cheatsheet, the Guide
   pages, then the API reference for Daf, SqlDaf and KeyedList.

Every code example in the Guide pages and the docstrings, and every line of code in the cheatsheet,
runs as a test. In the API pages, a directive of the site is replaced by the signature and docstring
of the method it names.

## What to review

- Can a new user understand what Daffodil is, when to use it, and get started from the Guide alone?
- Is the mental model easy to assemble: what a selection shares, what makes a copy, what changes the original?
- Is anything missing, wrong, confusing, or explained twice in a way that could drift apart?
- Do the docstrings teach: what each method does, when to use it, what it returns, what can surprise a caller?
- Is the cheatsheet complete and explicit enough for an AI model writing code with Daffodil?
- Are the names consistent across pages: methods, parameters, and the terms for the same idea?
- Writing style the project follows: short sentences with one idea each, no em dashes, no semicolons,
  few parenthetical asides, plain words. Flag lines that break these, or that read as choppy.

Give findings as a numbered list, the most important first. For each: the page, the method or
heading, the problem, and a suggested rewrite. This review is of the documentation. Where you are
unsure how the code behaves, say so, and ask for the behavior to be checked and documented.

---
"""


def nav_pages(mkdocs_text: str) -> Iterator[Tuple[T_ls, str]]:
    """ The pages of the nav in mkdocs.yml, in order, each with the trail of its section titles. """

    start: int = mkdocs_text.index('\nnav:\n') + 1
    nav_text: str = mkdocs_text[start:]
    end_match: Any = re.search(r'\n[a-z_]+:', nav_text[len('nav:'):])
    nav_text = nav_text[:end_match.start() + len('nav:')] if end_match else nav_text
    nav_la: List[Any] = yaml.safe_load(nav_text)['nav']

    def walk(items_la: List[Any], trail_ls: T_ls) -> Iterator[Tuple[T_ls, str]]:
        item: Any
        for item in items_la:
            if isinstance(item, str):
                yield trail_ls, item
                continue
            title: str
            value: Any
            for title, value in item.items():
                if isinstance(value, str):
                    yield trail_ls + [title], value
                else:
                    yield from walk(value, trail_ls + [title])

    yield from walk(nav_la, [])


def signature(obj: Any) -> str:
    """ The signature of a function as source text, with its annotations and defaults. self and cls are left out. """

    parts_ls: T_ls = []
    param: Any
    for param in obj.parameters:
        if param.name in ('self', 'cls'):
            continue
        prefix: str = {'var positional': '*', 'var keyword': '**'}.get(param.kind.value, '')
        text: str = prefix + param.name
        if param.annotation is not None:
            text += f': {param.annotation}'
        if param.default is not None and not prefix:
            text += f' = {param.default}'
        parts_ls.append(text)
    returns: str = f' -> {obj.returns}' if obj.returns is not None else ''
    return f"({', '.join(parts_ls)}){returns}"


def api_entry(pkg: Any, dotted: str, heading: str = '', members: bool = True) -> str:
    """ The heading, signature and docstring of one documented name, and of a class's members. """

    obj: Any = pkg[dotted.split('.', 1)[1]]
    if obj.is_alias:
        obj = obj.final_target
    name: str = heading or obj.name
    sig: str = signature(obj) if obj.is_function else ''
    doc: str = obj.docstring.value if obj.docstring else ''
    if not doc and obj.is_attribute:
        doc = f'A constant: {obj.value}'
    entry: str = f"\n#### {name}{sig}\n\n{doc or '(no docstring)'}\n"
    if obj.is_class and members:
        member: Any
        for member in obj.members.values():
            if member.is_alias or not member.docstring or re.match(r'_[^_]', member.name):
                continue
            entry += api_entry(pkg, f'{dotted}.{member.name}', members=False)
    return entry


def render_page(pkg: Any, page_text: str) -> str:
    """ A page with each directive replaced by its entry. Its options give the heading and the members. """

    lines_ls: T_ls = page_text.splitlines()
    out_ls: T_ls = []
    i: int = 0
    while i < len(lines_ls):
        match: Any = re.match(r'::: ([\w.]+)\s*$', lines_ls[i])
        if match:
            options_ls: T_ls = []
            while i + 1 < len(lines_ls) and lines_ls[i + 1].startswith('    '):
                i += 1
                options_ls.append(lines_ls[i].strip())
            heading_match: Any = re.search(r'heading: "([^"]+)"', '\n'.join(options_ls))
            out_ls.append(api_entry(pkg, match.group(1),
                                    heading=heading_match.group(1) if heading_match else '',
                                    members='members: false' not in options_ls))
        else:
            out_ls.append(lines_ls[i])
        i += 1
    return '\n'.join(out_ls)


def demote_headings(text: str) -> str:
    """ Each Markdown heading one level lower, so a page sits under its own heading. Code blocks are left as they are. """

    out_ls: T_ls = []
    in_code: bool = False
    line: str
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            in_code = not in_code
        out_ls.append('#' + line if not in_code and re.match(r'#+ ', line) else line)
    return '\n'.join(out_ls)


def build(readme_text: str, mkdocs_text: str, docs_dir: str) -> str:
    """ The whole review file, from the README, the nav and the page files. """

    pkg: Any = griffe.load('daffodil', search_paths=['src'])
    parts_ls: T_ls = [PROMPT, '# README.md\n\n' + demote_headings(readme_text)]
    trail_ls: T_ls
    path: str
    for trail_ls, path in nav_pages(mkdocs_text):
        with open(os.path.join(docs_dir, path), encoding='utf-8') as page_f:
            page_text: str = page_f.read()
        title: str = ' / '.join(trail_ls) or path
        parts_ls.append(f'# Docs page: {title} ({path})\n\n' + demote_headings(render_page(pkg, page_text)))
    return '\n\n---\n\n'.join(parts_ls) + '\n'


def main() -> None:
    with open('README.md', encoding='utf-8') as readme_f:
        readme_text: str = readme_f.read()
    with open('mkdocs.yml', encoding='utf-8') as mkdocs_f:
        mkdocs_text: str = mkdocs_f.read()
    review_text: str = build(readme_text, mkdocs_text, 'docsite')
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as out_f:
        out_f.write(review_text)
    print(f'wrote {OUT_PATH}: {len(review_text):,} characters')


if __name__ == '__main__':
    main()
