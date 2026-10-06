from daffodil.daf import Daf
class Sub(Daf): pass
d=Sub(lol=[[1]],cols=['a'],name='orig')
for lv in ['shallow','sortable','editable','deep']:
    c=d.copy(lv)
    print(f"{lv:9} new object: {c is not d}  same class: {type(c) is Sub}  name: {c.name!r}")
c=d.copy(); c.name='other'
print("rename copy -> original name:", d.name, "| copy name:", c.name)
