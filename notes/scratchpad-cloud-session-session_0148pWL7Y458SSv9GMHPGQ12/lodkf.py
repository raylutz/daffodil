from daffodil.daf import Daf
lod=[{'id':1,'v':'a'},{'id':2,'v':'b'}]
def t(label, f):
    try:
        d=f(); print(f"{label:46} cols={d.columns()} keyfield={d.keyfield!r} keys={d.keys()}")
    except Exception as e:
        print(f"{label:46} {type(e).__name__}: {str(e)[:80]}")
t("from_lod(lod, keyfield='id')", lambda: Daf.from_lod(lod, keyfield='id'))
t("from_lod(lod, keyfield='zz')", lambda: Daf.from_lod(lod, keyfield='zz'))
t("from_lod([], keyfield='id')", lambda: Daf.from_lod([], keyfield='id'))
t("Daf(keyfield='id').extend(lod)", lambda: Daf(keyfield='id').extend(lod))
t("Daf(keyfield='id').append(dict) x2", lambda: Daf(keyfield='id').append(lod[0]).append(lod[1]))
t("from_lod_to_cols(lod, keyfield='id')", lambda: Daf.from_lod_to_cols(lod, keyfield='id'))
