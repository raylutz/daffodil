import doctest, importlib, pkgutil, io, contextlib, sys, collections
import daffodil
APPLY = len(sys.argv) > 1 and sys.argv[1] == 'apply'
mods=[daffodil]
for m in pkgutil.walk_packages(daffodil.__path__, 'daffodil.'):
    if m.name.endswith(('daf_pdf','md_demo')): continue
    mods.append(importlib.import_module(m.name))
class R(doctest.DocTestRunner):
    def __init__(s,*a,**k): super().__init__(*a,**k); s.rec=[]
    def report_success(s,out,test,ex,got): s.rec.append((test,ex,got))
    def report_failure(s,out,test,ex,got): s.rec.append((test,ex,got))
    def report_unexpected_exception(s,out,test,ex,ei): s.rec.append((test,ex,None))
def is_table(text):
    return any(l.lstrip().startswith(('|','%%')) for l in text.splitlines())
edits=collections.defaultdict(list); seen=set()
for m in mods:
    for t in doctest.DocTestFinder().find(m):
        key=(t.filename,t.lineno,t.name)
        if key in seen: continue
        seen.add(key)
        r=R(optionflags=doctest.NORMALIZE_WHITESPACE)
        with contextlib.redirect_stdout(io.StringIO()): r.run(t)
        for test,ex,got in r.rec:
            if got is None or not is_table(got): continue
            est=(t.lineno or 0)+ex.lineno+ex.source.count(chr(10))
            edits[t.filename].append((est,ex,got,t.name))
changed=0
for fn,es in edits.items():
    lines=open(fn).read().split('\n')
    t_has=True
    used=set(); resolved=[]
    for est,ex,got,name in es:
        n_old=len(ex.want.splitlines()); ns=ex.source.count('\n')
        first='>>> '+ex.source.split('\n')[0].strip()
        wl=[w.strip() for w in ex.want.splitlines()]
        cands=[i+ns for i,l in enumerate(lines) if l.strip()==first and [x.strip() for x in lines[i+ns:i+ns+n_old]]==wl and (i+ns) not in used]
        if not cands: print('NOT FOUND',name,first); continue
        start=min(cands,key=lambda c:abs(c-est)) if t_has else cands[0]
        used.add(start); resolved.append((start,ex,got,name))
    for start,ex,got,name in sorted(resolved,key=lambda e:-e[0]):
        n_old=len(ex.want.splitlines())
        ind=' '*ex.indent
        new=[(ind+('<BLANKLINE>' if l=='' else l)) if True else l for l in got[:-1].split('\n')]
        # keep trailing whitespace out
        new=[l.rstrip() for l in new]
        old=lines[start:start+n_old]
        if [o.strip() for o in old]!=[o.strip() for o in new]:
            changed+=1
            if not APPLY and changed<=2:
                print('---',fn.split('/')[-1],name,'line',start+1); print('\n'.join(old)); print('=>'); print('\n'.join(new))
        lines[start:start+n_old]=new
    if APPLY: open(fn,'w').write('\n'.join(lines))
print('table examples',sum(len(v) for v in edits.values()),'changed',changed)
