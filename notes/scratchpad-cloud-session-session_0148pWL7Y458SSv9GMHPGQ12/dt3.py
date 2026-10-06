import doctest, re, io, contextlib
# D: patch the parser so expected output may contain blank lines when the next
# nonblank line starts with '|' or '%%'.
src_re = doctest.DocTestParser._EXAMPLE_RE.pattern
old_want = r'''(?P<want> (?:(?![ ]*$)    # Not a blank line
                     (?![ ]*>>>)  # Not a line starting with PS1
                     .+$\n?       # But any other line
                  )*)'''
assert old_want in src_re
new_want = r'''(?P<want> (?:(?:(?![ ]*$)(?![ ]*>>>).+$\n?)
                  |(?:[ ]*\n(?=[ ]*(?:\||%%))))*)'''
pat = re.compile(src_re.replace(old_want, new_want), re.MULTILINE | re.VERBOSE)
doctest.DocTestParser._EXAMPLE_RE = pat
def run(label, src, want):
    t = doctest.DocTestParser().get_doctest(src + want, {}, label, None, 0)
    r = doctest.DocTestRunner(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): r.run(t)
    print(label, (r.failures, r.tries))
src = ">>> print('\\n| A  | B  |\\n| -: | -: |\\n|  1 |  2 |\\n\\n%% shape')\n"
run("D real output, patched parser", src, "\n| A  | B  |\n| -: | -: |\n|  1 |  2 |\n\n%% shape\n")
run("D2 misaligned real? (wrong spacing)", src, "\n| A | B |\n| -: | -: |\n| 1 | 2 |\n\n%% shape\n")
