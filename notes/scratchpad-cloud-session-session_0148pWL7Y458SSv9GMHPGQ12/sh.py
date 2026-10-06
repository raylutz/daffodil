from daffodil.daf import Daf
d = Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id', dtypes={'id':int,'v':str}); d.keys()
c = d.copy()
print('c is d           :', c is d)
for n in ['lol','hd','dtypes','_kd','attrs']:
    print(f'c.{n} is d.{n:6}:', getattr(c,n) is getattr(d,n))
print('rows shared      :', all(a is b for a,b in zip(c.lol,d.lol)))
c.append([4,'d']); print('after c.append  : d.lol =', d.lol, '| d has key 4:', 4 in d.keys())
