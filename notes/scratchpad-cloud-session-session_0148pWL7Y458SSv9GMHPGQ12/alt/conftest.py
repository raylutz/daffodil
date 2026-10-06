from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser
from sybil.parsers.doctest import DocTestParser
pytest_collect_file = Sybil(parsers=[DocTestParser(), PythonCodeBlockParser()], patterns=['*.md']).pytest()
