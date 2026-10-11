import make_data, fmt, impl_marshmallow as M, impl_pydantic as P, impl_daffodil as D
from daffodil.lib.daf_utils import test_strbool
recs = make_data.make_records(1000)
def norm(r):  # CSV gives text; compare as typed values
    return {**r, 'num_ballots': int(r['num_ballots']), 'is_duplicate': test_strbool(r['is_duplicate']),
            'flagged': test_strbool(r['flagged'])}
res = {}
for name, mod in [('marshmallow', M), ('pydantic', P), ('daffodil', D)]:
    obj = mod.load(fmt.encode(name, recs)); t = mod.totals(obj); mod.flag(obj)
    out = [norm(r) for r in fmt.decode(name, mod.dump(obj))]
    res[name] = (t, out)
    print(f'{name:12}', len(out), sum(t.values()), sum(r['flagged'] for r in out))
print('same totals:', res['marshmallow'][0] == res['pydantic'][0] == res['daffodil'][0])
print('same output:', res['marshmallow'][1] == res['pydantic'][1] == res['daffodil'][1])
