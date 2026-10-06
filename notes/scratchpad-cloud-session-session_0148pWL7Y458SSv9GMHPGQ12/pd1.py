import pandas as pd, numpy as np, time
df=pd.DataFrame({'id':[2,1],'v':['b','a']}); df.attrs['k']=[1]
print("pandas", pd.__version__, " copy_on_write:", pd.options.mode.copy_on_write)
for name,c in [("copy()",df.copy()),("copy(deep=False)",df.copy(deep=False))]:
    print(f"{name:17} new object:{c is not df}  same index obj:{c.index is df.index}  shares data:{np.shares_memory(c['id'].values, df['id'].values)}  attrs same obj:{c.attrs is df.attrs}")
c=df.copy(deep=False); c.loc[0,'v']='X'; print("deep=False, set cell on copy -> original:", df.loc[0,'v'])
df=pd.DataFrame({'id':[2,1],'v':['b','a']})
c=df.copy(deep=False); c['w']=[0,0]; print("deep=False, add column on copy -> original cols:", list(df.columns))
c=df.copy(deep=False); c.loc[2]=[3,'c']; print("deep=False, add row on copy -> original rows:", len(df))
df.name='orig' if False else None
e=df.iloc[:0]; print("df.iloc[:0] columns kept:", list(e.columns), " len", len(e), "| df.head(0) dtypes:", dict(df.head(0).dtypes.astype(str)))
class Sub(pd.DataFrame):
    @property
    def _constructor(self): return Sub
s=Sub({'a':[1]}); print("subclass kept by copy():", type(s.copy()).__name__, "| iloc[:0]:", type(s.iloc[:0]).__name__)
big=pd.DataFrame(np.random.rand(200000,50))
for lab,f in [("copy()",lambda:big.copy()),("copy(deep=False)",lambda:big.copy(deep=False))]:
    t=time.perf_counter(); f(); print(lab, round(time.perf_counter()-t,5),'s (200,000 x 50)')
