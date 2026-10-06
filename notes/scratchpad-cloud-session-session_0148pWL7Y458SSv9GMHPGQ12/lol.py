from daffodil.daf import Daf
d = Daf(lol=[[1,2,3],[4,5,6],[7,8,9]], cols=['a','b','c'])
small = Daf(lol=[[10,20],[30,40]], cols=['x','y'])
d[0:2, 0:2] = small.lol
print(d.lol)
