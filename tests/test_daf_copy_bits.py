# test_daf_copy_bits.py
# Tests for the CopyBits flags of copy(), the class setting copy_level_default, the name of a copy,
# and clone_empty(), which is copy() with the layout bits and no rows.

import pytest

from daffodil.daf import Daf
from daffodil.lib.daf_types import CopyBits


def make_daf() -> Daf:
    daf = Daf(
        lol=[[1, 'a'], [2, 'b'], [3, 'c']],
        cols=['id', 'v'],
        keyfield='id',
        dtypes={'id': int, 'v': str},
        name='orig',
        disp_cols=['id'],
    )
    daf.attrs['x'] = [1]
    daf.keys()      # build the key index.
    return daf


# =====================================================================
# the bits
# =====================================================================

def test_none_shares_everything_but_is_a_new_object():
    daf = make_daf()
    copied = daf.copy(CopyBits.NONE)
    assert copied is not daf
    assert copied.lol is daf.lol and copied.hd is daf.hd and copied.dtypes is daf.dtypes
    assert copied.attrs is daf.attrs and copied.disp_cols is daf.disp_cols
    assert copied._kd is daf._kd


def test_each_bit_gives_only_its_own_part():
    daf = make_daf()
    c = daf.copy(CopyBits.HD)
    assert c.hd is not daf.hd and c.lol is daf.lol and c.dtypes is daf.dtypes and c.attrs is daf.attrs
    c = daf.copy(CopyBits.DTYPES)
    assert c.dtypes is not daf.dtypes and c.hd is daf.hd and c.lol is daf.lol
    c = daf.copy(CopyBits.ATTRS)
    assert c.attrs is not daf.attrs and c.attrs == daf.attrs
    assert c.disp_cols is not daf.disp_cols and c.disp_cols == daf.disp_cols
    assert c.lol is daf.lol


def test_outer_gives_own_row_list_and_shares_rows():
    daf = make_daf()
    c = daf.copy(CopyBits.OUTER)
    assert c.lol is not daf.lol and c.lol[0] is daf.lol[0]
    assert c.hd is daf.hd


def test_outer_implies_a_fresh_key_index():
    daf = make_daf()
    c = daf.copy(CopyBits.OUTER)
    assert c._kd == {} and c._kd is not daf._kd
    c.append([4, 'd'])
    assert daf.num_rows() == 3
    assert daf._kd == {1: 0, 2: 1, 3: 2}
    assert c.keys() == [1, 2, 3, 4]


def test_rows_implies_outer_and_key_index():
    daf = make_daf()
    c = daf.copy(CopyBits.ROWS)
    assert c.lol is not daf.lol and c.lol[0] is not daf.lol[0]
    assert c._kd == {}


def test_sum_of_bits_and_plain_int_are_accepted():
    daf = make_daf()
    c = daf.copy(CopyBits.OUTER | CopyBits.HD)
    assert c.lol is not daf.lol and c.hd is not daf.hd and c.dtypes is daf.dtypes
    c = daf.copy(int(CopyBits.HD))
    assert c.hd is not daf.hd and c.lol is daf.lol


def test_deep_bit_is_a_deep_copy():
    daf = Daf(lol=[['abc', [1, 2]]], cols=['a', 'b'])
    c = daf.copy(CopyBits.DEEP)
    assert c.lol[0][1] is not daf.lol[0][1]


def test_presets_are_sums_of_bits():
    daf = make_daf()
    for name, bits in [('shallow', CopyBits.ATTRS),
                       ('sortable', CopyBits.ATTRS | CopyBits.OUTER | CopyBits.HD | CopyBits.DTYPES | CopyBits.KD),
                       ('editable', CopyBits.ATTRS | CopyBits.OUTER | CopyBits.HD | CopyBits.DTYPES | CopyBits.KD | CopyBits.ROWS)]:
        a, b = daf.copy(name), daf.copy(bits)
        assert (a.lol is daf.lol, a.hd is daf.hd, a.dtypes is daf.dtypes, a.attrs is daf.attrs) == \
               (b.lol is daf.lol, b.hd is daf.hd, b.dtypes is daf.dtypes, b.attrs is daf.attrs)
        assert (a.lol[0] is daf.lol[0]) == (b.lol[0] is daf.lol[0])


# =====================================================================
# the key index and the keyfield
# =====================================================================

def test_list_keyfield_is_copied_with_the_key_index():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield=['id', 'v'])
    c = daf.copy(CopyBits.KD)
    assert c.keyfield == daf.keyfield and c.keyfield is not daf.keyfield
    assert daf.copy(CopyBits.NONE).keyfield is daf.keyfield


def test_adopted_key_index_without_keyfield_is_copied_not_cleared():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], kd={1: 0, 2: 1})
    c = daf.copy(CopyBits.KD)
    assert c._kd == {1: 0, 2: 1} and c._kd is not daf._kd


# =====================================================================
# the class default
# =====================================================================

def test_class_default_is_sortable():
    assert Daf.copy_level_default == 'sortable'


def test_subclass_can_change_the_default():
    class SafeDaf(Daf):
        copy_level_default = 'editable'

    safe = SafeDaf(lol=[[1, 'a']], cols=['id', 'v'])
    plain = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    assert safe.copy().lol[0] is not safe.lol[0]
    assert plain.copy().lol[0] is plain.lol[0]
    assert safe.copy('shallow').lol is safe.lol


def test_default_can_be_a_sum_of_bits():
    class ViewDaf(Daf):
        copy_level_default = CopyBits.ATTRS

    daf = ViewDaf(lol=[[1, 'a']], cols=['id', 'v'])
    assert daf.copy().lol is daf.lol


# =====================================================================
# the shell and the name
# =====================================================================

@pytest.mark.parametrize('level', ['shallow', 'sortable', 'editable', 'deep', CopyBits.NONE, CopyBits.ROWS])
def test_copy_is_a_new_object_of_the_same_class_without_a_name(level):
    class Sub(Daf):
        pass

    daf = Sub(lol=[[1, 'a']], cols=['id', 'v'], name='orig')
    c = daf.copy(level)
    assert c is not daf and type(c) is Sub
    assert c.name == '' and daf.name == 'orig'


@pytest.mark.parametrize('level', ['shallow', 'sortable', 'editable', 'deep'])
def test_name_argument_names_the_copy(level):
    daf = make_daf()
    assert daf.copy(level, name='work').name == 'work'
    assert daf.name == 'orig'


def test_settings_are_taken_over():
    daf = make_daf()
    daf.md_max_rows = 3
    daf.itermode = Daf.ITERMODE_KEYEDLIST
    c = daf.copy()
    assert c.md_max_rows == 3 and c.itermode == Daf.ITERMODE_KEYEDLIST


# =====================================================================
# clone_empty
# =====================================================================

def test_clone_empty_keeps_the_class_layout_and_settings():
    class Sub(Daf):
        pass

    daf = Sub(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id', dtypes={'id': int, 'v': str}, name='orig', disp_cols=['id'])
    daf.attrs['x'] = [1]
    daf.md_max_rows = 3
    c = daf.clone_empty()
    assert type(c) is Sub
    assert c.columns() == ['id', 'v'] and c.keyfield == 'id' and c.lol == []
    assert c.dtypes == daf.dtypes and c.dtypes is not daf.dtypes
    assert c.attrs == daf.attrs and c.attrs is not daf.attrs
    assert c.disp_cols == ['id'] and c.md_max_rows == 3
    assert c.hd is not daf.hd and c._kd == {}
    assert c.name == ''


def test_clone_empty_name_and_adopted_rows():
    daf = make_daf()
    rows = [[9, 'z']]
    c = daf.clone_empty(lol=rows, name='new')
    assert c.name == 'new' and c.lol is rows
    assert daf.lol != rows


def test_clone_empty_with_cols_drops_schema_and_display_columns():
    daf = make_daf()
    daf.schema = object()       # type: ignore[assignment]
    c = daf.clone_empty(lol=[[1, 2]], cols=['p', 'q'])
    assert c.columns() == ['p', 'q']
    assert c.schema is None and c.disp_cols == []
    assert daf.columns() == ['id', 'v'] and daf.disp_cols == ['id']


def test_clone_empty_with_cols_makes_names_unique():
    c = make_daf().clone_empty(lol=[[1, 2]], cols=['a', 'a'])
    assert c.columns() == ['a', 'a_1']


def test_clone_empty_keeps_the_schema_without_cols():
    daf = make_daf()
    marker = object()
    daf.schema = marker         # type: ignore[assignment]
    assert daf.clone_empty().schema is marker
