"""daffodil: the records are rows of one table, stored as CSV."""
from daffodil.daf import Daf

dtypes = {'batch_id': str, 'precinct': str, 'ballot_type': str,
          'num_ballots': int, 'is_duplicate': bool, 'flagged': bool}

def load(text: str) -> Daf:
    return Daf.from_csv_buff(text, keyfield='batch_id', dtypes=dtypes)

def totals(daf: Daf) -> dict:
    out: dict = {}
    for p, n, dup in zip(daf.col('precinct'), daf.col('num_ballots'), daf.col('is_duplicate')):
        if not dup:
            out[p] = out.get(p, 0) + n
    return out

def flag(daf: Daf) -> None:
    daf.insert_col('flagged', [n > 500 for n in daf.col('num_ballots')])

def dump(daf: Daf) -> str:
    return daf.to_csv_buff()
