import gb_data, gb_classes as C, gb_daffodil as D
rows = gb_data.make_grades(50)
book_c = C.load(C.dump(C.build(rows)))          # round trip through JSON
book_d = D.load(D.dump(D.build(rows)))          # round trip through CSV
C_avg, D_avg = C.student_averages(book_c), D.student_averages(book_d)
print('students:', len(C_avg), len(D_avg))
print('same student averages:', all(abs(C_avg[k] - D_avg[k]) < 1e-9 for k in C_avg) and C_avg.keys() == D_avg.keys())
print('math average:', round(book_c.subject_average('math'), 6), round(D.subject_average(book_d, 'math'), 6))
book_c.get_student('s00000').get_subject('math').report_grade(100, 0.4)
D.report_grade(book_d, 's00000', 'math', 100, 0.4)
print('after a new grade:', round(C.student_averages(book_c)['s00000'], 6), round(D.student_averages(book_d)['s00000'], 6))
print(D.dump(book_d)[:120])
print(C.dump(book_c)[:160])
