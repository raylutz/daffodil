"""The gradebook as one table, one row per grade, saved as CSV."""
from daffodil.daf import Daf

dtypes = {'student': str, 'subject': str, 'score': int, 'weight': float}

def build(rows: list[dict]) -> Daf:
    return Daf.from_lod(rows, dtypes=dtypes)

def load(text: str) -> Daf:
    return Daf.from_csv_buff(text, dtypes=dtypes)

def dump(book: Daf) -> str:
    return book.to_csv_buff()

def report_grade(book: Daf, student: str, subject: str, score: int, weight: float) -> None:
    book.append({'student': student, 'subject': subject, 'score': score, 'weight': weight})

def subject_averages(book: Daf) -> dict:
    """Weighted average for each (student, subject)."""
    sums: dict = {}
    cols = [book.col(name) for name in ('student', 'subject', 'score', 'weight')]
    for student, subject, score, weight in zip(*cols):
        s = sums.setdefault((student, subject), [0.0, 0.0])
        s[0] += score * weight
        s[1] += weight
    return {key: total / weight for key, (total, weight) in sums.items()}

def student_averages(book: Daf) -> dict:
    per: dict = {}
    for (student, _), avg in subject_averages(book).items():
        per.setdefault(student, []).append(avg)
    return {student: sum(avgs) / len(avgs) for student, avgs in per.items()}

def subject_average(book: Daf, subject: str) -> float:
    avgs = list(subject_averages(book.select_by_dict({'subject': subject})).values())
    return sum(avgs) / len(avgs)
