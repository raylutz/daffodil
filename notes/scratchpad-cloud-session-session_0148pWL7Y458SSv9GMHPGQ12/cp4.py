from daffodil.daf import Daf
def mk(): return Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'],keyfield='id')
def outer(d):
    c=d.copy(); c.lol=list(d.lol); return c
for name,f in [("drop_cols",lambda c:c.drop_cols(['v'])),("rename_col",lambda c:c.rename_cols({'v':'w'}) if hasattr(c,'rename_cols') else c.set_cols(['id','w'])),("append",lambda c:c.append([3,'c'])),("insert_icol",lambda c:c.insert_icol(1,[0,0],'w')),("set_keyfield",lambda c:c.set_keyfield(''))]:
    d=mk(); c=outer(d)
    try: f(c)
    except Exception as e: print(name,"error",type(e).__name__,e); continue
    print(f"outer-only {name}: d cols {d.columns()} rows {d.lol} keyfield {d.keyfield!r} | c cols {c.columns()}")
