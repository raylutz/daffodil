import re
import griffe

_RE = re.compile(r'^[ \t]*<BLANKLINE>[ \t]*$', re.MULTILINE)

class BlankLines(griffe.Extension):
    def on_instance(self, *, node=None, obj=None, agent=None, **kw):
        doc = getattr(obj, "docstring", None)
        if doc is not None and "<BLANKLINE>" in doc.value:
            doc.value = _RE.sub("", doc.value)
