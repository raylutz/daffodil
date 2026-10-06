import doctest, re
real = "\n| A  | B  |\n| -: | -: |\n|  1 |  2 |\n\n%% shape\n"
def run(label, src, want, flags=0):
    t = doctest.DocTestParser().get_doctest(src + want, {}, label, None, 0)
    r = doctest.DocTestRunner(optionflags=flags, verbose=False)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.run(t)
    s = r.summarize(verbose=False) if False else (r.failures, r.tries)
    print(label, "failures/tries =", s)
src = ">>> print(%r)\n" % real[:-1] if False else ">>> print('\\n| A  | B  |\\n| -: | -: |\\n|  1 |  2 |\\n\\n%% shape')\n"
# A: real output, blank lines kept
run("A real output w/ blank lines", src, "\n| A  | B  |\n| -: | -: |\n|  1 |  2 |\n\n%% shape\n")
# B: BLANKLINE markers
run("B <BLANKLINE>", src, "<BLANKLINE>\n| A  | B  |\n| -: | -: |\n|  1 |  2 |\n<BLANKLINE>\n%% shape\n")
# C: today's style, misaligned, NORMALIZE_WHITESPACE
run("C misaligned, no blanks, NORMALIZE_WHITESPACE", src, "| A | B |\n| -: | -: |\n| 1 | 2 |\n%% shape\n", doctest.NORMALIZE_WHITESPACE)
run("C2 same without NORMALIZE_WHITESPACE", src, "| A | B |\n| -: | -: |\n| 1 | 2 |\n%% shape\n")
