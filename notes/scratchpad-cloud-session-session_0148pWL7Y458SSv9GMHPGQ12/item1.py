from daffodil.daf import Daf
def base(): return Daf(lol=[[1,2,3],[4,5,6],[7,8,9]], cols=['a','b','c'])
def show(label, fn):
    d=base()
    try:
        fn(d); print(f"{label}: {[[type(v).__name__ if isinstance(v,Daf) else v for v in r] for r in d.lol]}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e}")
small=Daf(lol=[[10,20],[30,40]], cols=['x','y'])
def f1(d): d[0:2,0:2]=small
def f2(d): d[0:2,0:2]=base()
def f3(d):
    v=Daf(lol=[[10,20],[30,40]], cols=['x','y']); v.retmode='val'; d[0:2,0:2]=v[:, :]
show("2x2 Daf into 2x2 region ", f1)
show("3x3 Daf into 2x2 region ", f2)
show("retmode='val' source    ", f3)
