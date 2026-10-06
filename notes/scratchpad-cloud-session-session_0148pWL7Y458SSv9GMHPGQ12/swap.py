from daffodil.daf import Daf
def run(f):
    try: return f().lol
    except Exception as e: return f'{type(e).__name__}'
d = Daf(lol=[[1,'5','a'],[2,5,'b'],[3,'5','c']], cols=['id','n','v'], keyfield='id')
print('same, plain equality     :', run(lambda: d.select_where(lambda r: r['v']=='b')) == run(lambda: d.select_by_dict({'v':'b'})))
print('two equalities (and)     :', run(lambda: d.select_where(lambda r: r['n']=='5' and r['v']=='c')), run(lambda: d.select_by_dict({'n':'5','v':'c'})))
print('TRAP text vs number      : where int(r["n"])==5 ->', run(lambda: d.select_where(lambda r: int(r['n'])==5)))
print('                           by_dict({"n": 5})   ->', run(lambda: d.select_by_dict({'n': 5})))
print('not equal                : where !=  ->', run(lambda: d.select_where(lambda r: r['v']!='b')), '| inverse ->', run(lambda: d.select_by_dict({'v':'b'}, inverse=True)))
print('TRAP inverse of two      : where (n!="5" and v!="c") ->', run(lambda: d.select_where(lambda r: r['n']!='5' and r['v']!='c')), '| inverse of both ->', run(lambda: d.select_by_dict({'n':'5','v':'c'}, inverse=True)))
print('unknown column           : where ->', run(lambda: d.select_where(lambda r: r['zz']=='b')), '| by_dict ->', run(lambda: d.select_by_dict({'zz':'b'})))
print('shared rows              :', d.select_by_dict({'v':'b'}).lol[0] is d.lol[1])
