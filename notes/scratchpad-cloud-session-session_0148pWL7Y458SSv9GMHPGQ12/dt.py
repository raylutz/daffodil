import doctest, io, re, sys
from daffodil.daf import Daf

REAL = repr(Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c']).apply_dtypes(dtypes={'a': int, 'b': float, 'c': str}))
print("what repr() gives, with its blank lines made visible:")
print(REAL.replace("\n", "\\n\n"))

def run(label, body, flags=0, parser=None):
    text = ">>> d = Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c'])\n>>> d.apply_dtypes(dtypes={'a': int, 'b': float, 'c': str})\n" + body
    test = (parser or doctest.DocTestParser()).get_doctest(text, {'Daf': Daf}, 't', None, 0)
    out = io.StringIO()
    runner = doctest.DocTestRunner(optionflags=flags, verbose=False)
    runner.run(test, out=out.write)
    r = runner.summarize(verbose=False)
    print(f"{label:68} -> {'PASS' if r.failed == 0 else 'FAIL'}")
    return out.getvalue()

NW = doctest.NORMALIZE_WHITESPACE
real_body = "\n| a  |  b  | c  |\n| -: | --: | -: |\n|  1 | 2.5 |  x |\n\n%% daf rows=1; cols=3; keyfield=''; name=''\n"
print("\nA. The real output, with its blank lines, written into the docstring:")
msg = run("   exact comparison", real_body)
print("   doctest says:", msg.split('Expected:')[1].split('Got:')[0].strip().replace('\n', ' / ')[:60], "...  (it stopped reading at the first blank line)")
run("   with NORMALIZE_WHITESPACE", real_body, NW)
print("\nB. Blank lines written as <BLANKLINE>:")
bl = "<BLANKLINE>\n| a  |  b  | c  |\n| -: | --: | -: |\n|  1 | 2.5 |  x |\n<BLANKLINE>\n%% daf rows=1; cols=3; keyfield=''; name=''\n<BLANKLINE>\n"
run("   exact comparison", bl)
print("\nC. The style used today: no blank lines, and a table with the old, misaligned spacing:")
old = "| a |  b  | c |\n| -: | --: | -: |\n| 1 | 2.5 | x |\n%% daf rows=1; cols=3; keyfield=''; name=''\n"
run("   exact comparison", old)
run("   with NORMALIZE_WHITESPACE (what pytest.ini sets)", old, NW)
print("\nD. A parser that keeps reading across a blank line when the next line starts with | or %%:")
class TolerantParser(doctest.DocTestParser):
    _EXAMPLE_RE = re.compile(r'''
        (?P<source> (?:^(?P<indent> [ ]*) >>>    .*)
                    (?:\n           [ ]* \.\.\. .*)*)
        \n?
        (?P<want> (?: (?![ ]*$) (?![ ]*>>>) .+$\n?
                    | (?:[ ]*\n)+ (?=[ ]*(?:\||%%)) )* )
        ''', re.MULTILINE | re.VERBOSE)
tp = TolerantParser()
run("   real output, exact comparison", real_body, 0, tp)
run("   misaligned table, exact comparison (the old bug would be caught)", "\n" + old.replace("%%", "\n%%"), 0, tp)
