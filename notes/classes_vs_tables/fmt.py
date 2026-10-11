"""Each implementation reads and writes its own format."""
import csv, io, json

def to_csv(recs: list[dict]) -> str:
    names: list = []
    for r in recs:
        names += [k for k in r if k not in names]
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=names, restval='', lineterminator='\n')
    w.writeheader(); w.writerows(recs)
    return f.getvalue()

def encode(name: str, recs: list[dict]) -> str:
    return to_csv(recs) if name == 'daffodil' else json.dumps(recs)

def decode(name: str, text: str) -> list[dict]:
    if name == 'daffodil':
        return list(csv.DictReader(io.StringIO(text)))
    return json.loads(text)
