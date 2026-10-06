import ast,re,subprocess
p='src/daffodil/lib/daf_utils.py'
t=ast.parse(open(p).read())
src_all=subprocess.run("cat src/daffodil/*.py src/daffodil/lib/*.py | grep -v 'daf_utils.py'",shell=True,capture_output=True,text=True).stdout
tests=subprocess.run("cat tests/*.py",shell=True,capture_output=True,text=True).stdout
own=open(p).read()
for n in t.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
        d=ast.get_docstring(n)
        nm=n.name
        u=len(re.findall(r'\b'+nm+r'\b',src_all)); ow=len(re.findall(r'\b'+nm+r'\b',own))-1; te=len(re.findall(r'\b'+nm+r'\b',tests))
        print(f"{n.lineno:5} {nm:34} doc={'Y' if d else '-'} len={len(d or ''):4} src={u:3} own={ow:2} tests={te:3}")
