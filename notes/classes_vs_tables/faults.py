"""Inject one fault at a time and see where each implementation reports it."""
import json, fmt, make_data, impl_marshmallow as M, impl_pydantic as P, impl_daffodil as D
from daffodil.daf import Daf

def bad_data(change, name):
    recs = make_data.make_records(20); change(recs[5]); return fmt.encode(name, recs)

data_faults = {
    'number as text "abc"':  lambda r: r.update(num_ballots='abc'),
    'bool as text "no"':     lambda r: r.update(is_duplicate='no'),
    'field missing':         lambda r: r.pop('precinct'),
    'extra field':           lambda r: r.update(precint='P99'),
    'null number':           lambda r: r.update(num_ballots=None),
}

def typo_obj(batches):
    return sum(b.num_balots for b in batches)
def typo_daf(daf):
    return sum(daf.col('num_balots'))
typo = {'marshmallow': typo_obj, 'pydantic': typo_obj, 'daffodil': typo_daf}

def run(name, mod, text, extra=None):
    try:
        obj = mod.load(text)
    except Exception as e:
        return f'at load: {type(e).__name__}: {str(e)[:70]!r}'
    try:
        if extra: extra(obj)
        t = mod.totals(obj); mod.flag(obj); out = mod.dump(obj)
    except Exception as e:
        return f'later: {type(e).__name__}: {str(e)[:70]!r}'
    rec = fmt.decode(name, out)[5]
    return f'no error; totals {sum(t.values())}; record 5 out: {rec}'

mods = [('marshmallow', M), ('pydantic', P), ('daffodil', D)]
for fname, change in data_faults.items():
    print(f'== {fname}')
    for name, mod in mods:
        print(f'   {name:12} {run(name, mod, bad_data(change, name))}')
print('== typo in code: num_balots')
for name, mod in mods:
    print(f'   {name:12} {run(name, mod, bad_data(lambda r: None, name), typo[name])}')
