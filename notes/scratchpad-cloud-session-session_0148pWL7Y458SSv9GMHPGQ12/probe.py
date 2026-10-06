from daffodil.keyedlist import KeyedList, KeyedIndex
def run(f):
    try: return repr(f())
    except Exception as e: return f'{type(e).__name__}: {e}'
ki = KeyedIndex(['a','b','c'])
ctor = {
 'KeyedIndex+list':        lambda: KeyedList(ki, [1,2,3]).values(),
 'KeyedIndex+list wrong n':lambda: KeyedList(ki, [1,2]),
 'KeyedIndex+None':        lambda: KeyedList(ki),
 'KeyedIndex+tuple':       lambda: KeyedList(ki, (1,2,3)),
 'dict':                   lambda: KeyedList({'a':1}).values(),
 'dict+list':              lambda: KeyedList({'a':1}, [5]).values(),
 'list+list':              lambda: KeyedList(['a','b'], [1,2]).values(),
 'list+None':              lambda: KeyedList(['a','b'], default=0).values(),
 'KeyedList':              lambda: KeyedList(KeyedList({'a':1})).values(),
 'None':                   lambda: KeyedList().values(),
 'int':                    lambda: KeyedList(5),
}
kl = KeyedList(ki, [10,20,30])
get = {
 'one key':          lambda: kl['b'],
 'list of keys':     lambda: kl[['a','c']],
 'list with missing':lambda: kl[['a','zz']],
 'missing key':      lambda: kl['zz'],
 'tuple key':        lambda: kl[('a',)],
 'dict key':         lambda: kl[{'a':1}],
 'set key':          lambda: kl[{1}],
 'int key':          lambda: kl[1],
 'None key':         lambda: kl[None],
 'empty list':       lambda: kl[[]],
}
for k,f in {**ctor, **get}.items(): print(f'{k:24} {run(f)}')
# shared index is still copied before a key is added
a = KeyedList(ki, [1,2,3]); a['new'] = 4; print('hd shared copy:', list(ki), list(a))
