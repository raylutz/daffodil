from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'a',5],[2,'b',6]],cols=['id','v','n'],keyfield='id')
def show(label, d):
    try: k=d.keys()
    except Exception as e: k=type(e).__name__
    print(f"{label:34} cols={d.columns()} keyfield={d.keyfield!r} keys()={k}")
d=mk(); d.drop_cols(['id']); show("drop_cols(['id']) in place", d)
show("select_cols(['v','n'])", mk().select_cols(['v','n']))
d=mk(); d.rename_cols({'id':'ident'}); show("rename_cols({'id':'ident'})", d)
d=mk(); d.set_cols(['x','y','z']); show("set_cols(['x','y','z'])", d)
show("Daf(cols=[p,q], keyfield='a')", Daf(lol=[[1,2]],cols=['p','q'],keyfield='a'))
d=mk(); d.set_keyfield('nope'); show("set_keyfield('nope')", d)
