"""The gradebook as nested classes, saved as JSON with pydantic."""
from pydantic import BaseModel

class Grade(BaseModel):
    score:  int
    weight: float

class Subject(BaseModel):
    grades: list[Grade] = []

    def report_grade(self, score: int, weight: float) -> None:
        self.grades.append(Grade(score=score, weight=weight))

    def average_grade(self) -> float:
        total = total_weight = 0.0
        for grade in self.grades:
            total        += grade.score * grade.weight
            total_weight += grade.weight
        return total / total_weight

class Student(BaseModel):
    subjects: dict[str, Subject] = {}

    def get_subject(self, name: str) -> Subject:
        if name not in self.subjects:
            self.subjects[name] = Subject()
        return self.subjects[name]

    def average_grade(self) -> float:
        averages = [subject.average_grade() for subject in self.subjects.values()]
        return sum(averages) / len(averages)

class Gradebook(BaseModel):
    students: dict[str, Student] = {}

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
    return Gradebook.model_validate_json(text)

def dump(book: Gradebook) -> str:
    return book.model_dump_json()

def student_averages(book: Gradebook) -> dict:
    return {name: student.average_grade() for name, student in book.students.items()}
