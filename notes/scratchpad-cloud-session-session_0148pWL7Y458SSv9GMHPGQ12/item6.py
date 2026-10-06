from daffodil.daf import Daf
for use_pyon in (True, False):
    d = Daf(lol=[[{'x': 1, 'ok': True}, [1, 'a'], True]], cols=['d', 'l', 'b'],
            dtypes={'d': dict, 'l': list, 'b': bool})
    d.flatten(use_pyon=use_pyon)
    print(f"use_pyon={use_pyon}: {d.lol}")
d = Daf(lol=[[{'x': 1, 'ok': True}, [1, 'a'], True]], cols=['d', 'l', 'b'],
        dtypes={'d': dict, 'l': list, 'b': bool})
d.flatten()
print(f"default:        {d.lol}")
