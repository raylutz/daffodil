"""Make the same grades each time, as flat rows: student, subject, score, weight."""
import random

SUBJECTS = ['math', 'gym', 'history', 'art', 'science']

def make_grades(num_students: int, per_subject: int = 4) -> list[dict]:
    rng = random.Random(2)
    rows = []
    for i in range(num_students):
        for subject in SUBJECTS:
            for _ in range(per_subject):
                rows.append({'student': f's{i:05d}', 'subject': subject,
                             'score': rng.randrange(40, 101), 'weight': rng.choice([0.1, 0.2, 0.4])})
    return rows
