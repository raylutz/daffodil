# mkdocs_hooks.py
#
# A hook that mkdocs.yml loads for the docs build.
#
# A doctest writes a blank line of output as <BLANKLINE>.
# mkdocstrings shows it as a blank line on the API pages.
# The Guide pages are plain Markdown, so this hook does the same for them.

import re

from typing import Any

BLANKLINE_RE: re.Pattern = re.compile(r'^<BLANKLINE>$', re.MULTILINE)


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """ Show each <BLANKLINE> line of a doctest as a blank line. """
    shown_s: str = BLANKLINE_RE.sub('', markdown)
    return shown_s
