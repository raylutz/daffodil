"""The gradebook as plain classes. Everything is written by hand, including the JSON form."""
import json

class Grade:
    def __init__(self, score: int, weight: float) -> None:
        self.score  = score
        self.weight = weight

    def __repr__(self) -> str:
        return f'Grade(score={self.score!r}, weight={self.weight!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grade):
            return NotImplemented
        return (self.score, self.weight) == (other.score, other.weight)

    @classmethod
    def from_dict(cls, d: dict) -> 'Grade':
        if not isinstance(d.get('score'), int) or isinstance(d.get('score'), bool):
            raise ValueError(f"score is not an int: {d.get('score')!r}")
        if not isinstance(d.get('weight'), (int, float)) or isinstance(d.get('weight'), bool):
            raise ValueError(f"weight is not a number: {d.get('weight')!r}")
        return cls(d['score'], float(d['weight']))

    def to_dict(self) -> dict:
        return {'score': self.score, 'weight': self.weight}

class Subject:
    def __init__(self, grades: list[Grade] | None = None) -> None:
        self.grades = grades if grades is not None else []

    def __repr__(self) -> str:
        return f'Subject(grades={self.grades!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subject):
            return NotImplemented
        return self.grades == other.grades

    @classmethod
    def from_dict(cls, d: dict) -> 'Subject':
        return cls([Grade.from_dict(g) for g in d.get('grades', [])])

    def to_dict(self) -> dict:
        return {'grades': [g.to_dict() for g in self.grades]}

    def report_grade(self, score: int, weight: float) -> None:
        self.grades.append(Grade(score, weight))

    def average_grade(self) -> float:
        total = total_weight = 0.0
        for grade in self.grades:
            total        += grade.score * grade.weight
            total_weight += grade.weight
        return total / total_weight

class Student:
    def __init__(self, subjects: dict[str, Subject] | None = None) -> None:
        self.subjects = subjects if subjects is not None else {}

    def __repr__(self) -> str:
        return f'Student(subjects={self.subjects!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Student):
            return NotImplemented
        return self.subjects == other.subjects

    @classmethod
    def from_dict(cls, d: dict) -> 'Student':
        return cls({name: Subject.from_dict(s) for name, s in d.get('subjects', {}).items()})

    def to_dict(self) -> dict:
        return {'subjects': {name: s.to_dict() for name, s in self.subjects.items()}}

    def get_subject(self, name: str) -> Subject:
        if name not in self.subjects:
            self.subjects[name] = Subject()
        return self.subjects[name]

    def average_grade(self) -> float:
        averages = [subject.average_grade() for subject in self.subjects.values()]
        return sum(averages) / len(averages)

class Gradebook:
    def __init__(self, students: dict[str, Student] | None = None) -> None:
        self.students = students if students is not None else {}

    def __repr__(self) -> str:
        return f'Gradebook(students={self.students!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gradebook):
            return NotImplemented
        return self.students == other.students

    @classmethod
    def from_dict(cls, d: dict) -> 'Gradebook':
        return cls({name: Student.from_dict(s) for name, s in d.get('students', {}).items()})

    def to_dict(self) -> dict:
        return {'students': {name: s.to_dict() for name, s in self.students.items()}}

    def get_student(self, name: str) -> Student:
        if name not in self.students:
            self.students[name] = Student()
        return self.students[name]

    def subject_average(self, subject: str) -> float:
        averages = [student.subjects[subject].average_grade()
                    for student in self.students.values() if subject in student.subjects]
        return sum(averages) / len(averages)

def build(rows: list[dict]) -> Gradebook:
    book = Gradebook()
    for r in rows:
        book.get_student(r['student']).get_subject(r['subject']).report_grade(r['score'], r['weight'])
    return book

def load(text: str) -> Gradebook:
    return Gradebook.from_dict(json.loads(text))

def dump(book: Gradebook) -> str:
    return json.dumps(book.to_dict())

def student_averages(book: Gradebook) -> dict:
    return {name: student.average_grade() for name, student in book.students.items()}
