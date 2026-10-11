"""The gradebook as one row per student. The grades are a dict in one column: an indirect column.

The dict maps each subject to a list of [score, weight] pairs. Daffodil saves a dict cell as its
Python text and reads it back with ast.literal_eval. `dump_json()` and `load_json()` keep compact
JSON in the cell instead, which is smaller and faster to read.
"""
import json
from daffodil.daf import Daf

dtypes = {'student': str, 'grades': dict}

def build(rows: list[dict]) -> Daf:
    per: dict = {}
    for r in rows:
        per.setdefault(r['student'], {}).setdefault(r['subject'], []).append([r['score'], r['weight']])
    return Daf(lol=[[student, grades] for student, grades in per.items()], cols=['student', 'grades'], keyfield='student')

def load(text: str) -> Daf:
    return Daf.from_csv_buff(text, keyfield='student', dtypes=dtypes)

def dump(book: Daf) -> str:
    return book.to_csv_buff()

def load_json(text: str) -> Daf:
    book = Daf.from_csv_buff(text, keyfield='student', dtypes={'student': str, 'grades': str})
    book.apply_to_col('grades', json.loads)
    book.dtypes = dtypes
    return book

def dump_json(book: Daf) -> str:
    out = book.copy('editable')
    out.apply_to_col('grades', lambda g: json.dumps(g, separators=(',', ':')))
    return out.to_csv_buff()

def report_grade(book: Daf, student: str, subject: str, score: int, weight: float) -> None:
    book.retmode = 'val'
    book[student, 'grades'].setdefault(subject, []).append([score, weight])

def _average(pairs: list) -> float:
    return sum(s * w for s, w in pairs) / sum(w for _, w in pairs)

def student_averages(book: Daf) -> dict:
    out = {}
    for student, grades in zip(book.col('student'), book.col('grades')):
        avgs = [_average(pairs) for pairs in grades.values()]
        out[student] = sum(avgs) / len(avgs)
    return out

def subject_average(book: Daf, subject: str) -> float:
    avgs = [_average(g[subject]) for g in book.col('grades') if subject in g]
    return sum(avgs) / len(avgs)
