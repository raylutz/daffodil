from daffodil.daf import Daf, KeysDisabledError

def optB(d, keyfield=''):
    """default means the Daf's own keyfield. The Daf is not changed. No keyfield at all raises."""
    kf = keyfield or d.keyfield
    if not kf:
        raise KeysDisabledError("remove_dups: give a keyfield, or set one on the Daf.")
    if isinstance(kf, (str, int)):
        kd = Daf._build_kd(d.hd[kf], d.lol)
    else:
        kd = Daf._build_kd([d.hd[c] for c in kf], d.lol)
    unique_irows = list(kd.values())
    unique = d.select_irows(unique_irows); unique.keyfield = kf
    dups = d.select_irows(unique_irows, invert=True); dups.keyfield = kf
    return unique, dups

def optC(d, keyfield=''):
    """only the default changes: use the Daf's own keyfield. Everything else as before."""
    if not keyfield and d.keyfield:
        keyfield = d.keyfield
    return d.remove_dups(keyfield)

def mk(kf=''):
    return Daf(lol=[[1,'a'],[2,'b'],[1,'c'],[3,'d'],[2,'e']], cols=['id','v'], keyfield=kf)

def show(label, f, d, *a):
    try:
        u, p = f(d, *a)
        print(f'   {label:9} unique {u.lol} dups {p.lol} | unique.keyfield {u.keyfield!r} | Daf.keyfield now {d.keyfield!r}, same obj {u is d}')
    except Exception as e:
        print(f'   {label:9} EXC {type(e).__name__}: {e} | Daf.keyfield now {d.keyfield!r}')

orig = lambda d, kf='': d.remove_dups(kf)
for title, kf, arg in [('keyed Daf (id), remove_dups()', 'id', ''),
                       ('Daf with no keyfield, remove_dups()', '', ''),
                       ("Daf with no keyfield, remove_dups('id')", '', 'id'),
                       ("keyed on v, remove_dups('id')", 'v', 'id')]:
    print('==', title)
    for name, f in [('orig', orig), ('B', optB), ('C', optC)]:
        show(name, f, mk(kf), arg)
print("== no repeats, remove_dups('id')")
for name, f in [('orig', orig), ('B', optB)]:
    d = Daf(lol=[[1,'a'],[2,'b']], cols=['id','v']); show(name, f, d, 'id')
print("== composite keyfield")
d = Daf(lol=[[1,'a',0],[1,'a',1],[1,'b',2]], cols=['p','q','r'])
for name, f in [('orig', orig), ('B', optB)]:
    d = Daf(lol=[[1,'a',0],[1,'a',1],[1,'b',2]], cols=['p','q','r']); show(name, f, d, ('p','q'))
