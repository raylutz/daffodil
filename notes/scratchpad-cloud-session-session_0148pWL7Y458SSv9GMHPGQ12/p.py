from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList
d = Daf(lol=[[1,2],[3,4]], cols=['a','b'])
print(format(Daf(lol=[['x']], cols=['a']), '>5'), format(Daf(lol=[[3.14159]], cols=['a']), '.2f'))
print('in empty', 'x' in Daf(keyfield='a'))
try: d._is_keyfield_valid(1.5)
except Exception as e: print('kf', type(e), e)
d2 = Daf(lol=[['1','2']]); print(d2.hd); d2.apply_dtypes(dtypes={'a':int,'b':int}); print(d2.hd, d2.lol)
d3 = Daf(lol=[['1','2']], cols=['a','b']); d3.apply_dtypes(dtypes={'a':int}, silent_error=True); print(d3.dtypes, d3.lol)
d4 = Daf(lol=[['1','2']], cols=['a','b']); 
try:
  d4.dtypes = int; d4.apply_dtypes(); print('single', d4.lol, d4.dtypes)
except Exception as e: print('single err', type(e), e)
d5 = Daf(lol=[[{'x':1}, True]], cols=['a','b'], dtypes={'a':dict,'b':bool}); d5.flatten(use_pyon=False); print('flat', d5.lol)
print(d.to_attrib_dict())
x = Daf.from_csv_buff("a,b\n1,2\n\n\n"); print(x.lol)
print(d.to_donpa())
d._itermode='bogus'
try: iter(d)
except Exception as e: print(type(e), e)
print(list(Daf(lol=[[1,2]],cols=['a','b']).iter_list()))
