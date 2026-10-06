import doctest, re
_p = doctest.DocTestParser._EXAMPLE_RE.pattern
_old = r'''(?P<want> (?:(?![ ]*$)    # Not a blank line
                     (?![ ]*>>>)  # Not a line starting with PS1
                     .+$\n?       # But any other line
                  )*)'''
assert _old in _p
_new = r'''(?P<want> (?:(?:(?![ ]*$)(?![ ]*>>>).+$\n?)
                  |(?:[ ]*\n(?=[ ]*(?:\||%%))))*)'''
doctest.DocTestParser._EXAMPLE_RE = re.compile(_p.replace(_old, _new), re.MULTILINE | re.VERBOSE)
