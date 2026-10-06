from daffodil.lib import daf_sql as s
names = ['data__ab', 'rev__2024', 'my_col', 'My Col', '"Quoted"', 'select', 'a-b']
for q in (True, False):
    print(f"quoting_ok={q}")
    for n in names:
        e1 = s.sql_escape_str(n, quoting_ok=q)
        e2 = s.sql_escape_str(e1, quoting_ok=q)
        back = s.sql_unesc_str(e1)
        print(f"  {n!r:12} -> {e1!r:16} again -> {e2!r:16} {'same' if e1 == e2 else 'CHANGED'}   unescape -> {back!r}")
