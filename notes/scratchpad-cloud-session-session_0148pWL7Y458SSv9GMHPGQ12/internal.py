from daffodil.daf import Daf
lod=[{'x':1,'y':2,'z':3},{'x':4,'y':5,'z':6}]
def t(label,f):
    try: d=f(); print(f"{label:48} cols={d.columns()} lol={d.lol}")
    except Exception as e: print(f"{label:48} {type(e).__name__}: {str(e)[:100]}")
t("from_lod_to_cols(lod, dtypes={'x':int,'y':int})", lambda: Daf.from_lod_to_cols(lod, dtypes={'x':int,'y':int}))
dod={'r1':{'a':1,'b':2},'r2':{'a':3,'b':4}}
t("from_dod(dod, keyfield='id')", lambda: Daf.from_dod(dod, keyfield='id'))
t("from_dod(dod, keyfield='id', dtypes={'a':int,'b':int})", lambda: Daf.from_dod(dod, keyfield='id', dtypes={'a':int,'b':int}))
t("from_dod(dod, keyfield='id', dtypes incl. id)", lambda: Daf.from_dod(dod, keyfield='id', dtypes={'id':str,'a':int,'b':int}))
