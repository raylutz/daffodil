# test_daf_schema.py
#
# Tests for schemaclass.py (the @schemaclass decorator and SchemaBase) and daf_schema.py
# (_apply_schema/_attach_schema/_default_record, the glue between Daf and either a
# @schemaclass-decorated class or a "schema_daf" -- a Daf instance with Name/dtype/Default
# columns describing another Daf's structure).
#
# This batch surfaced 2 real issues, now fixed:
#   - daf_schema.py contained a complete, verbatim duplicate of schemaclass.py's entire
#     SchemaBase class and schemaclass() decorator (an apparent accidental copy-paste),
#     confirmed unused anywhere in the codebase and removed.
#   - _apply_schema()'s schema_daf branch used `if`/`elif` between setting cols and setting
#     dtypes, but these are independent conditions that should both apply when a schema_daf
#     has both 'Name' and 'dtype' columns (the normal case) -- the elif silently skipped
#     setting dtypes whenever cols was also being set. Fixed to two independent `if` blocks.

import pytest

from daffodil.lib.schemaclass import schemaclass, SchemaBase
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList


# =====================================================================
# schemaclass.py: @schemaclass decorator / SchemaBase
# =====================================================================

@schemaclass
class PersonSchema(SchemaBase):
    name: str = ''
    age: int = 0
    tags: list = []


def test_default_record_basic():
    assert PersonSchema.default_record() == {'name': '', 'age': 0, 'tags': []}


def test_default_record_with_override():
    rec = PersonSchema.default_record(name='Alice')
    assert rec == {'name': 'Alice', 'age': 0, 'tags': []}


def test_default_record_mutable_defaults_are_isolated():
    rec1 = PersonSchema.default_record()
    rec1['tags'].append('x')
    rec2 = PersonSchema.default_record()
    assert rec1['tags'] == ['x']
    assert rec2['tags'] == []


def test_get_dtypes_dict_no_origins():
    assert PersonSchema.get_dtypes_dict() == {'name': str, 'age': int, 'tags': list}


def test_get_dtypes_dict_use_origins():
    assert PersonSchema.get_dtypes_dict(use_origins=True) == {'name': str, 'age': int, 'tags': list}


def test_get_columns():
    assert PersonSchema.get_columns() == ['name', 'age', 'tags']


def test_get_pandas_dtypes_from_schema():
    result = PersonSchema.get_pandas_dtypes_from_schema(PersonSchema)
    assert result == {'name': 'string', 'age': 'Int64', 'tags': 'object'}


def test_get_pandas_dtypes_from_schema_float_and_unmapped():
    @schemaclass
    class MixedSchema(SchemaBase):
        score: float = 0.0
        flag: bool = False  # bool isn't explicitly mapped -> falls to the "object" else branch

    result = MixedSchema.get_pandas_dtypes_from_schema(MixedSchema)
    assert result == {'score': 'Float64', 'flag': 'object'}


def test_record_from_dict():
    rec = PersonSchema.record_from({'name': 'Bob', 'age': '30'})
    assert rec == {'name': 'Bob', 'age': 30, 'tags': []}


def test_record_from_keyedlist():
    kl = KeyedList(['name', 'age'], ['Carol', '25'])
    rec = PersonSchema.record_from(kl)
    assert rec == {'name': 'Carol', 'age': 25, 'tags': []}


def test_validate_keys_debug_invalid_key_raises():
    with pytest.raises(AssertionError):
        PersonSchema.validate_keys_debug({'bogus': 1})


def test_validate_keys_debug_valid_keys_ok():
    PersonSchema.validate_keys_debug({'name': 'x', 'age': 1})  # should not raise


def test_cannot_instantiate_schemaclass():
    with pytest.raises(TypeError):
        PersonSchema()


def test_schemaclass_requires_default_value():
    with pytest.raises(TypeError):
        @schemaclass
        class BadSchema(SchemaBase):
            x: int  # no default


def test_schemaclass_requires_at_least_one_field():
    with pytest.raises(TypeError):
        @schemaclass
        class EmptySchema(SchemaBase):
            pass


def test_schemaclass_inheritance_merges_annotations():
    @schemaclass
    class ExtendedSchema(PersonSchema):
        extra: str = 'x'

    assert ExtendedSchema.get_columns() == ['name', 'age', 'tags', 'extra']
    assert ExtendedSchema.default_record() == {'name': '', 'age': 0, 'tags': [], 'extra': 'x'}


def test_is_schemaclass_marker_set():
    assert getattr(PersonSchema, '__is_schemaclass__', False) is True


# =====================================================================
# daf_schema.py: _apply_schema / _attach_schema / _default_record
# =====================================================================

def test_daf_init_with_schemaclass_sets_cols_and_dtypes():
    daf = Daf(schema=PersonSchema)
    assert list(daf.hd.keys()) == ['name', 'age', 'tags']
    assert daf.dtypes == {'name': str, 'age': int, 'tags': list}


def test_daf_default_record_via_schemaclass():
    daf = Daf(schema=PersonSchema)
    assert daf.default_record() == {'name': '', 'age': 0, 'tags': []}


def test_daf_default_record_no_schema_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(AttributeError):
        daf.default_record()


def test_attach_schema_sets_keyfield_from_dunder():
    @schemaclass
    class KeyedSchema(SchemaBase):
        id: int = 0
        name: str = ''
    KeyedSchema.__keyfield__ = 'id'

    daf = Daf(schema=KeyedSchema)
    assert daf.keyfield == 'id'


def test_attach_schema_non_schemaclass_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(TypeError):
        daf.attach_schema(int)


def test_apply_schema_noop_when_no_schema():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.apply_schema()
    assert result is daf


# --- schema_daf support ---

def _make_schema_daf():
    return Daf(
        lol=[['id', 'int', 0], ['name', 'str', '']],
        cols=['Name', 'dtype', 'Default'],
        attrs={'keyfield': 'id'},
    )


def test_apply_schema_daf_sets_cols_dtypes_and_keyfield():
    # this is the bug we found: dtypes was previously silently skipped (elif instead of if)
    # whenever cols was also being set, which is the normal case.
    schema_daf = _make_schema_daf()
    daf = Daf(schema=schema_daf)
    assert list(daf.hd.keys()) == ['id', 'name']
    assert daf.dtypes == {'id': int, 'name': str}
    assert daf.keyfield == 'id'


def test_apply_schema_daf_default_record():
    schema_daf = _make_schema_daf()
    daf = Daf(schema=schema_daf)
    assert daf.default_record() == {'id': 0, 'name': ''}


def test_apply_schema_daf_all_dtype_aliases():
    schema_daf = Daf(
        lol=[['a', 'str', ''], ['b', 'int', 0], ['c', 'float', 0.0],
             ['d', 'bool', False], ['e', 'list', []], ['f', 'dict', {}]],
        cols=['Name', 'dtype', 'Default'],
    )
    daf = Daf(schema=schema_daf)
    assert daf.dtypes == {'a': str, 'b': int, 'c': float, 'd': bool, 'e': list, 'f': dict}


def test_apply_schema_daf_unsupported_dtype_raises():
    bad_schema_daf = Daf(lol=[['x', 'bogus_type', '']], cols=['Name', 'dtype', 'Default'])
    with pytest.raises(RuntimeError):
        Daf(schema=bad_schema_daf)


def test_default_record_schema_daf_missing_name_col_raises():
    bad_schema_daf = Daf(lol=[['str', '']], cols=['dtype', 'Default'])
    daf = Daf()
    daf.schema = bad_schema_daf
    with pytest.raises(RuntimeError):
        daf.default_record()


def test_default_record_schema_daf_no_default_col_uses_empty_string():
    schema_daf = Daf(lol=[['id'], ['name']], cols=['Name'])
    daf = Daf()
    daf.schema = schema_daf
    assert daf.default_record() == {'id': '', 'name': ''}


def test_default_record_unsupported_schema_type_raises():
    daf = Daf()
    daf.schema = 42
    with pytest.raises(TypeError):
        daf.default_record()
