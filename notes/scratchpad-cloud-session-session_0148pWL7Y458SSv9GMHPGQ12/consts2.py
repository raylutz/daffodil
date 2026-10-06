import pandas as pd, re, inspect
print("pandas", pd.__version__)
df=pd.DataFrame({'a':[1,2],'b':[3,4]})
print("merge how= is a string:", df.merge(df, how='inner', on='a').shape, "| axis= is an int:", df.sum(axis=1).tolist())
print("str.contains flags= uses the stdlib enum:", type(re.IGNORECASE).__mro__[:3], int(re.I|re.M))
print("pd.read_csv quoting= uses csv module ints:", __import__('csv').QUOTE_ALL, type(__import__('csv').QUOTE_ALL).__name__)
import pandas._libs.lib as lib
print("no_default sentinel:", lib.no_default, type(lib.no_default).__name__)
print("enum classes in pandas.api.types:", [n for n in dir(pd.api.types) if n[0].isupper()][:5])
