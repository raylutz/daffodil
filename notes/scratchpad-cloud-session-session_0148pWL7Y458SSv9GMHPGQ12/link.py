from daffodil.daf import Daf
def show(label, d):
    print(f"{label:38} cols={d.columns()} dtypes={ {k:v.__name__ for k,v in (d.dtypes or {}).items()} } keyfield={d.keyfield!r}")
show("cols only", Daf(lol=[[1,2]], cols=['a','b']))
show("dtypes only (names come from dtypes)", Daf(lol=[[1,2]], dtypes={'a':int,'b':int}))
show("cols + same dtypes", Daf(lol=[[1,2]], cols=['a','b'], dtypes={'a':int,'b':int}))
show("cols + different dtypes", Daf(lol=[[1,2]], cols=['p','q'], dtypes={'a':int,'b':int}))
show("cols + keyfield not a column", Daf(lol=[[1,2]], cols=['p','q'], keyfield='a'))
show("hd + cols differ", Daf(lol=[[1,2]], hd={'x':0,'y':1}, cols=['p','q']))
d=Daf(lol=[[1,2]], cols=['a','b'], dtypes={'a':int,'b':int}, keyfield='a', disp_cols=['a'])
show("clone_empty(cols=[p,q]) today", d.clone_empty(cols=['p','q']))
show("set_cols([p,q]) today", d.copy('editable').set_cols(['p','q']))
