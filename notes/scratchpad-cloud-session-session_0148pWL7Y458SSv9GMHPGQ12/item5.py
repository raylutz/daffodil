from daffodil.daf import Daf
# list values
row1 = {'l': [1, 2]}
acc = Daf.count_values_da(row1, {}, ['l'])
acc = Daf.count_values_da({'l': [3]}, acc, ['l'])
print("list: acc  =", acc)
print("list: row1 =", row1)
# dict values
row1 = {'d': {'x': 1}}
acc = Daf.count_values_da(row1, {}, ['d'])
acc = Daf.count_values_da({'d': {'x': 2, 'y': 1}}, acc, ['d'])
print("dict: acc  =", acc)
print("dict: row1 =", row1)
# normal scalar use, for reference
daf = Daf(cols=['g'], lol=[['M'], ['F'], ['M']])
print("scalar via reduce:", daf.reduce(Daf.count_values_da, cols=['g'], initial_da={}))
