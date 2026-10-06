from daffodil.daf import Daf
store = {
    'c1': Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]]),
    'c2': Daf(cols=['x', 'y'], lol=[[4, 8], [6, 12]]),
}
manifest = Daf(cols=['name'], lol=[['c1'], ['c2']])
result = manifest.manifest_reduce(Daf.sum_da, load_func=lambda spec: store[spec['name']])
print("result:", result)
print("expected: {'x': 14, 'y': 26}")
