import os, tempfile
from daffodil.daf import Daf
def show(label, f):
    try: d = f(); print(f'{label:34} ok  cols={d.columns()} lol={d.lol}')
    except Exception as e: print(f'{label:34} {type(e).__name__}: {str(e)[:70]}')
show("from_csv_buff('')",               lambda: Daf.from_csv_buff(''))
show("from_csv_buff(b'')",              lambda: Daf.from_csv_buff(b''))
show("from_csv_buff('\\n')",            lambda: Daf.from_csv_buff('\n'))
show("from_csv_buff('  ')",             lambda: Daf.from_csv_buff('  '))
show("from_csv_buff('id,v')  header only", lambda: Daf.from_csv_buff('id,v'))
show("from_csv_buff('id,v\\n') header only", lambda: Daf.from_csv_buff('id,v\n'))
show("from_csv_buff('', noheader=True)", lambda: Daf.from_csv_buff('', noheader=True))
show("from_csv_buff(iter([]))",         lambda: Daf.from_csv_buff(iter([])))
p = os.path.join(tempfile.mkdtemp(), 'empty.csv'); open(p, 'w').close()
show("from_csv(empty file)",            lambda: Daf.from_csv(p))
show("from_md('')",                     lambda: Daf.from_md(''))
