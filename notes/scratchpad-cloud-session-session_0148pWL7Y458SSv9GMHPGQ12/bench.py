import timeit
from daffodil.daf import Daf
d = Daf(lol=[[i]*10 for i in range(1000)], cols=[f"c{i}" for i in range(10)])
print("setitem col (1000 rows):", round(min(timeit.repeat(lambda: d.set_irows_icols(range(1000), 3, list(range(1000))), number=200, repeat=5))/200*1e6,1), "us")
print("setitem row block      :", round(min(timeit.repeat(lambda: d.set_irows_icols([0,1], [0,1], [5,6]), number=20000, repeat=5))/20000*1e6,2), "us")
print("Daf(cols=10)           :", round(min(timeit.repeat(lambda: Daf(cols=[f"c{i}" for i in range(10)]), number=20000, repeat=5))/20000*1e6,2), "us")
