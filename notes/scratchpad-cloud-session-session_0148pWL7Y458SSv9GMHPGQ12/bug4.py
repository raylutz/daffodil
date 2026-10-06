from daffodil.daf import Daf
def run(label, fn):
    try: print(f"{label}: returns {fn()!r}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e}")
a2 = Daf(cols=['k1', 'k2', 'v'], lol=[[1, 2, 3]], keyfield=('k1', 'k2'))
b2 = Daf(cols=['k1', 'k2', 'w'], lol=[[1, 2, 4]], keyfield=('k1', 'k2'))
a1 = Daf(cols=['k', 'v'], lol=[[1, 'a']], keyfield='k', name='A')
b1 = Daf(cols=['k', 'w'], lol=[[1, 'b']], keyfield='k', name='B')
run("both composite            ", lambda: a2.join(b2))
run("self str, other composite ", lambda: a1.join(b2))
run("self composite, other str ", lambda: a2.join(b1))
run("normal single-key join    ", lambda: a1.join(b1).lol)
