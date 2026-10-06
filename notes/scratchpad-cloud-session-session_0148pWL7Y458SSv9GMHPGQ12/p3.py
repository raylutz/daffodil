from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,2,3],[4,5,6],[7,8,9]], cols=['a','b','c'])
src = Daf(lol=[[10,20,30],[40,50,60]], cols=['a','b','c'])
d=mk(); d.set_irows_icols(None, [1], 0); print('none', d.lol)
d=mk(); d.set_irows_icols(0, None, src); print('3627', type(d.lol[0]), d.lol[0] is src)
d=mk(); d.set_irows_icols([0,1], None, src); print('3661', [type(r) for r in d.lol], d.lol[2])
v=src[0]; print('src[0]', type(v), v)
d=mk(); d.set_irows_icols([0,1], 1, Daf(lol=[[100],[200]], cols=['x'])); print('3690', d.lol)
d=mk(); d.set_irows_icols([0,1,2], 1, [100]); print('3685', d.lol)
d=mk(); d.set_irows_icols([0,1], [0,1], [100]); print('3711', d.lol)
d=mk(); 
try:
  d.set_irows_icols([0,1], [0,1], Daf(lol=[[100,200]], cols=['x','y'])); print('3715', d.lol)
except Exception as e: print('3715 err', type(e), e)
d=mk(); d[0:2, 0:2] = Daf(lol=[[100,200],[300,400]], cols=['a','b']); print('setitem', d.lol)
