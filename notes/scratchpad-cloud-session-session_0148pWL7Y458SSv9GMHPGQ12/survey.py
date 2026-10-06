import doctest, importlib, pkgutil, sys, io, contextlib
import daffodil
mods=[]
for m in pkgutil.walk_packages(daffodil.__path__, 'daffodil.'):
    if m.name.endswith(('daf_pdf','md_demo')): continue
    mods.append(importlib.import_module(m.name))
mods.append(daffodil)
class R(doctest.DocTestRunner):
    def __init__(s,*a,**k): super().__init__(*a,**k); s.rec=[]
    def report_success(s,out,test,ex,got): s.rec.append((test,ex,got,True))
    def report_failure(s,out,test,ex,got): s.rec.append((test,ex,got,False))
    def report_unexpected_exception(s,out,test,ex,ei): s.rec.append((test,ex,None,False))
tot=tab=0; seen=set()
for m in mods:
    for t in doctest.DocTestFinder().find(m):
        if t.filename in seen and False: pass
        r=R(optionflags=doctest.NORMALIZE_WHITESPACE)
        with contextlib.redirect_stdout(io.StringIO()): r.run(t)
        for test,ex,got,ok in r.rec:
            tot+=1
            if got and any(l.lstrip().startswith(('|','%%')) for l in got.splitlines()):
                tab+=1
                if got.startswith('\n') or '\n\n' in got: pass
print(tot, tab)
