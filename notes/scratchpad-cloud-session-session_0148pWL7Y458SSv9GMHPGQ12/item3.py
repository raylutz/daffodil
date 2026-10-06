from daffodil.daf import Daf
for cols in (['a', 'a', 'b'], ['a_2', 'a', 'a']):
    try:
        d = Daf(cols=cols, lol=[[1, 2, 3], [4, 5, 6]])
        print(f"{cols}: columns={d.columns()} shape={d.shape()} to_lod={d.to_lod()[0]}")
    except Exception as e:
        print(f"{cols}: {type(e).__name__}: {e}")
