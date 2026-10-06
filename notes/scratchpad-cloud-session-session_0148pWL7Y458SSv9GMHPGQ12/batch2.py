import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from addex import add_examples
add_examples('src/daffodil/daf.py', 'Daf', {
'from_excel_buff': """
    >>> import io, xlsxwriter
    >>> buff = io.BytesIO()
    >>> workbook = xlsxwriter.Workbook(buff, {'in_memory': True})
    >>> sheet = workbook.add_worksheet()
    >>> for irow, row in enumerate([['id', 'v'], [1, 'a'], [2, 'b']]):
    ...     for icol, value in enumerate(row):
    ...         _ = sheet.write(irow, icol, value)
    >>> workbook.close()
    >>> d = Daf.from_excel_buff(buff.getvalue(), keyfield='id')
    >>> d.columns(), d.lol
    (['id', 'v'], [['1', 'a'], ['2', 'b']])
    >>> Daf.from_excel_buff(buff.getvalue(), dtypes={'id': int, 'v': str}).lol
    [[1, 'a'], [2, 'b']]
""",
'from_csv_file': """
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> with open(path, 'w') as f:
    ...     n = f.write('id,v\\n1,a\\n')
    >>> Daf.from_csv_file(path).lol
    [['1', 'a']]
""",
'buff_to_file': """
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> Daf.buff_to_file('id,v\\n1,a\\n', path) == path
    True
    >>> open(path).read()
    'id,v\\n1,a\\n'
""",
'from_directory': """
    >>> import os, tempfile
    >>> folder = tempfile.mkdtemp()
    >>> os.mkdir(os.path.join(folder, 'sub'))
    >>> for name in ('b.txt', 'a.csv', os.path.join('sub', 'c.csv')):
    ...     with open(os.path.join(folder, name), 'w') as f:
    ...         n = f.write('hello')
    >>> d = Daf.from_directory(folder)
    >>> sorted(d.col('basename')), d.col('size')
    (['a.csv', 'b.txt', 'c.csv'], [5, 5, 5])
    >>> Daf.from_directory(folder, recursive=False).col('basename')
    ['b.txt', 'a.csv']
    >>> sorted(Daf.from_directory(folder, file_pat=r'\\.csv$').col('basename'))
    ['a.csv', 'c.csv']
    >>> sorted(Daf.from_directory(folder, include_dirs=True).col('basename'))
    ['a.csv', 'b.txt', 'c.csv', 'sub']
""",
'from_googlesheet': """
    >>> Daf.from_googlesheet('some-id', service_account_file='key.json')
    Traceback (most recent call last):
        ...
    NotImplementedError: from_googlesheet() is not implemented yet.
""",
'to_googlesheet': """
    >>> Daf(cols=['x'], lol=[[1]]).to_googlesheet('some-id', service_account_file='key.json')
    Traceback (most recent call last):
        ...
    NotImplementedError: to_googlesheet() is not implemented yet.
""",
})
