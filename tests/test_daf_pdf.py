# test_daf_pdf.py
#
# Minimal smoke tests for daffodil/lib/daf_pdf.py. The PDF import API is experimental and
# not yet settled, so these only check that the module imports and is wired onto Daf.

from daffodil.daf import Daf
from daffodil.lib import daf_pdf


def test_daf_pdf_module_exposes_from_pdf():
    assert isinstance(daf_pdf.__dict__['_from_pdf'], classmethod)
    assert isinstance(daf_pdf.__dict__['_from_pdf_new'], classmethod)


def test_daf_from_pdf_is_wired_as_classmethod():
    assert callable(Daf.from_pdf)
    assert Daf.from_pdf.__self__ is Daf
    assert Daf.from_pdf.__func__ is daf_pdf._from_pdf.__func__
