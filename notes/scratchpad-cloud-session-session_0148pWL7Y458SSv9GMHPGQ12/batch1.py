import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from addex import add_examples
add_examples('src/daffodil/daf.py', 'Daf', {
'iter_dict': """
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> list(d.iter_dict())
    [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
    >>> for row in d.iter_dict():
    ...     row['y'] = 0
    >>> d.lol
    [[1, 2], [3, 4]]
""",
'iter_klist': """
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> [row['x'] for row in d.iter_klist()]
    [1, 3]
    >>> for row in d.iter_klist():
    ...     row['y'] = 0
    >>> d.lol
    [[1, 0], [3, 0]]
""",
'iter_list': """
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> [row[0] for row in d.iter_list()]
    [1, 3]
    >>> for row in d.iter_list():
    ...     row[1] = 9
    >>> d.lol
    [[1, 9], [3, 9]]
""",
'num_rows': """
    >>> Daf(cols=['x'], lol=[[1], [2], [3]]).num_rows()
    3
    >>> Daf().num_rows()
    0
""",
'len': """
    >>> d = Daf(cols=['x'], lol=[[1], [2], [3]])
    >>> d.len(), len(d)
    (3, 3)
""",
})
