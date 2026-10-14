# test_daf_join_other_only_rows.py
# Rows found only in the other Daf of a right or outer join. Found 2026-10-14 while matching SqlDaf.join() to Daf.join().

import pytest

from daffodil.daf import Daf


def make_pair(other_key: str) -> tuple:
    a_daf: Daf = Daf(lol=[['k3', 'A3', 1], ['k1', 'A1', 2]], cols=['id', 'name', 'v'], keyfield='id')
    b_daf: Daf = Daf(lol=[['k3', 'B3', 7], ['k4', 'B4', 8]], cols=[other_key, 'name', 'w'], keyfield=other_key)
    return a_daf, b_daf


def test_outer_join_keeps_the_key_of_an_other_only_row_when_the_key_names_match():
    a_daf, b_daf = make_pair('id')
    assert a_daf.join(b_daf, how='outer').lol[-1] == ['k4', '', '', 'B4', 8]


@pytest.mark.xfail(strict=True, reason="BUG: src/daffodil/daf.py:13053 join_records() looks up the key by the first Daf's "
                                       "column name, so an other-only row gets NULL when the key columns are named differently.")
def test_outer_join_keeps_the_key_of_an_other_only_row_when_the_key_names_differ():
    a_daf, b_daf = make_pair('ballot')
    assert a_daf.join(b_daf, how='outer').lol[-1][0] == 'k4'


@pytest.mark.xfail(strict=True, reason="BUG: src/daffodil/daf.py:13064 join_records() takes a shared field only from the first Daf, "
                                       "so an other-only row of a right or outer join loses it.")
def test_outer_join_keeps_a_shared_field_of_an_other_only_row():
    a_daf, b_daf = make_pair('id')
    assert a_daf.join(b_daf, how='outer', shared_fields=['name']).lol[-1] == ['k4', 'B4', '', 8]
