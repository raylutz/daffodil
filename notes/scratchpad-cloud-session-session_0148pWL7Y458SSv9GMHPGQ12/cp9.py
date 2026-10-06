import copy, time
from daffodil.daf import Daf
big=Daf(lol=[[i,i,i] for i in range(200000)],cols=['a','b','c'],keyfield='a')
def t(f,n=200):
    s=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-s)/n*1e6
print("copy.copy(self):           %.1f us" % t(lambda: copy.copy(big)))
print("Daf(cols=, lol=, ...):     %.1f us" % t(lambda: Daf(cols=big.columns(), lol=big.lol, keyfield='a', dtypes=dict(big.dtypes) if big.dtypes else {}, name='x')))
print("clone_empty(lol=big.lol):  %.1f us" % t(lambda: big.clone_empty(lol=big.lol)))
class Sub(Daf):
    def __init__(self,*a,extra='x',**k): super().__init__(*a,**k); self.extra=extra
s=Sub(lol=[[1]],cols=['a'],extra='keep')
print("copy.copy keeps extra:", copy.copy(s).extra, "| clone_empty class:", type(s.clone_empty()).__name__)
