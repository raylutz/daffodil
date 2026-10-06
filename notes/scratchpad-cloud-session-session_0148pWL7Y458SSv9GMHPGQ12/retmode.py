from daffodil.daf import Daf

def target():
    return Daf(lol=[[1,2,3],[4,5,6],[7,8,9]], cols=['a','b','c'])

def source():
    v = Daf(lol=[[10,20],[30,40]], cols=['x','y'])
    v.retmode = 'val'
    return v

for label, expr in [("v[:, :]", lambda v: v[:, :]),
                    ("v[0:2, 0:2]", lambda v: v[0:2, 0:2]),
                    ("v[0:2]", lambda v: v[0:2]),
                    ("v[0, 0:2]", lambda v: v[0, 0:2])]:
    got = expr(source())
    d = target()
    d[0:2, 0:2] = got
    cells = [[type(c).__name__ if isinstance(c, Daf) else c for c in r] for r in d.lol]
    print(f"{label:12} returns {type(got).__name__:5} -> target {cells}")
