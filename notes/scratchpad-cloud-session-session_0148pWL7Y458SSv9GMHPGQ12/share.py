from daffodil.daf import Daf
def run(label, fn):
    try: print(f"{label}: {fn()!r}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e}")
def adds_key_in_loop():
    d = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4], [5, 6]], itermode='keyedlist')
    for row in d:
        row['total'] = row['a'] + row['b']
    return d.lol
def adds_key_via_iloc():
    d = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4], [5, 6]])
    d.iloc(0, rtype='klist')['total'] = 3
    return d.iloc(1, rtype='klist').keys()
def deletes_key_in_loop():
    d = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]], itermode='keyedlist')
    for row in d:
        del row['b']
    return d.lol
run("add a key to each row in a loop ", adds_key_in_loop)
run("add a key to an iloc row, then another iloc row", adds_key_via_iloc)
run("delete a key from each row      ", deletes_key_in_loop)
