from daffodil.daf import Daf
def grid(n): return Daf(lol=[[0]*n for _ in range(n)], cols=[f"c{i}" for i in range(n)])
def show(label, d):
    print(label); [print("   ", r) for r in d.lol]
d = grid(4); d[0:4, 0:4] = Daf(lol=[[1,2],[3,4]], cols=['x','y']); show("2x2 Daf into 4x4 region:", d)
d = grid(4); d[0:2, 0:2] = Daf(lol=[[i*10+j for j in range(4)] for i in range(4)], cols=list('wxyz')); show("4x4 Daf into 2x2 region:", d)
d = grid(4); d[0:4, 1] = [7, 8]; show("2 values into a 4-row column:", d)
d = grid(4); d[0:2, 1] = [7, 8, 9, 9]; show("4 values into a 2-row column:", d)
d = grid(4); d[0:2, 0:4] = [5, 6]; show("2 values into a 2x4 block (same row repeated):", d)
