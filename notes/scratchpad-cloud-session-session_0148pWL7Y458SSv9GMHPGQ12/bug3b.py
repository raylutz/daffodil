from daffodil.daf import Daf
shared = ['x']
a = Daf(cols=['k', 'v'], lol=[[1, 'a1']], keyfield='k', name='A')
b = Daf(cols=['k', 'v'], lol=[[1, 'b1']], keyfield='k', name='B')
a.join(b, shared_fields=shared)
print("list after 1st join:", shared)
c = Daf(cols=['id', 'k', 'v'], lol=[[1, 'c_k', 'c1']], keyfield='id', name='C')
d = Daf(cols=['id', 'k', 'v'], lol=[[1, 'd_k', 'd1']], keyfield='id', name='D')
r = c.join(d, shared_fields=shared)
print("2nd join, reused list :", r.columns(), r.lol)
r = c.join(d, shared_fields=['x'])
print("2nd join, fresh list  :", r.columns(), r.lol)
