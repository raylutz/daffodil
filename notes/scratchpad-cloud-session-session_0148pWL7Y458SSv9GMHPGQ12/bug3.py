from daffodil.daf import Daf
shared = ['a']
a = Daf(cols=['k', 'a', 'x'], lol=[[1, 2, 3]], keyfield='k', name='A')
b = Daf(cols=['k', 'a', 'y'], lol=[[1, 4, 5]], keyfield='k', name='B')
r = a.join(b, shared_fields=shared)
print("join result columns :", r.columns())
print("caller's list after :", shared)
c = Daf(cols=['id', 'a', 'z'], lol=[[1, 6, 7]], keyfield='id', name='C')
d = Daf(cols=['id', 'a', 'w'], lol=[[1, 8, 9]], keyfield='id', name='D')
r2 = c.join(d, shared_fields=shared)
print("reused list, 2nd join:", r2.columns())
print("caller's list after :", shared)
t = Daf.derive_join_translator_daf('k', 'k', ['k', 'a'], ['k', 'a'], shared_fields=('a',))
print("tuple input works   :", t.num_rows(), "rows")
