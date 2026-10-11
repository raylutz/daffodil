"""The gradebook as two tables: the grades, and a summary with one row per student.

The summary has two columns for each subject: the weighted total and the total weight.
It is built once, at load, and kept up to date when a grade is added. The questions
then read the summary, with no grouping.
"""
from daffodil.daf import Daf
from gb_data import SUBJECTS

dtypes = {'student': str, 'subject': str, 'score': int, 'weight': float}
summary_cols = ['student'] + [f'{s}_{part}' for s in SUBJECTS for part in ('total', 'weight')]

def build_summary(grades: Daf) -> Daf:
    rows: dict = {}
    pos = {s: 1 + 2 * i for i, s in enumerate(SUBJECTS)}
    width = len(summary_cols)
    for student, subject, score, weight in zip(grades.col('student'), grades.col('subject'),
                                               grades.col('score'), grades.col('weight')):
        row = rows.get(student)
        if row is None:
            row = rows[student] = [student] + [0.0] * (width - 1)
        i = pos[subject]
        row[i]     += score * weight
        row[i + 1] += weight
    return Daf(lol=list(rows.values()), cols=summary_cols, keyfield='student')

def load(text: str) -> tuple[Daf, Daf]:
    grades = Daf.from_csv_buff(text, dtypes=dtypes)
    return grades, build_summary(grades)

def dump(book: tuple[Daf, Daf]) -> str:
    return book[0].to_csv_buff()

def report_grade(book: tuple[Daf, Daf], student: str, subject: str, score: int, weight: float) -> None:
    grades, summary = book
    grades.append({'student': student, 'subject': subject, 'score': score, 'weight': weight})
    summary.retmode = 'val'
    summary[student, f'{subject}_total']  += score * weight
    summary[student, f'{subject}_weight'] += weight

def student_averages(book: tuple[Daf, Daf]) -> dict:
    out = {}
    for row in book[1].lol:
        avgs = [row[i] / row[i + 1] for i in range(1, len(row), 2) if row[i + 1]]
        out[row[0]] = sum(avgs) / len(avgs)
    return out

def subject_average(book: tuple[Daf, Daf], subject: str) -> float:
    summary = book[1]
    avgs = [t / w for t, w in zip(summary.col(f'{subject}_total'), summary.col(f'{subject}_weight')) if w]
    return sum(avgs) / len(avgs)
