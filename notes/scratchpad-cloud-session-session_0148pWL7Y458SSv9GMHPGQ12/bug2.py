import re
from daffodil.lib import daf_sql as s

def encode_char(ch):
    n = ord(ch)
    if n <= 0xFF:
        return f"__{n:02X}"
    if n <= 0xFFFF:
        return f"__u{n:04X}"
    return f"__U{n:08X}"

_DECODE_RE = re.compile(r'__U([0-9A-Fa-f]{8})|__u([0-9A-Fa-f]{4})|__([0-9A-Fa-f]{2})')

def decode(name):
    return _DECODE_RE.sub(lambda m: chr(int(m.group(1) or m.group(2) or m.group(3), 16)), name)

def escape_noquote(name):
    new = re.sub(r'[^0-9A-Za-z_]', lambda m: encode_char(m.group()), name)
    if re.search(r'^\d', new) or new.lower() in s._RESERVED_SQL_WORDS_SET:
        new = encode_char(new[0]) + new[1:]
    return new

names = ['€x', 'café', 'emoji😀', 'My Col', 'a-b', '9lives', 'select', 'plain']
print(f"{'name':10} {'original':22} {'back':10} | {'prototype':26} {'back'}")
for n in names:
    o = s.sql_escape_str(n, quoting_ok=False)
    ob = s.sql_unesc_str(o)
    p = escape_noquote(n)
    pb = decode(p)
    print(f"{n!r:10} {o!r:22} {ob!r:10} | {p!r:26} {pb!r} {'ok' if pb == n else 'FAIL'}")
import sqlite3
con = sqlite3.connect(':memory:')
cols = [escape_noquote(n) for n in names]
con.execute(f"create table t ({', '.join(cols)})")
print("sqlite accepts all prototype names as bare identifiers:", [r[1] for r in con.execute('pragma table_info(t)')] == cols)
