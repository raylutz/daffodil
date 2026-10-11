# test_daf_check_schema.py
#
# check_schema() raises ValueError for cells that do not fit the dtypes, for empty
# cells in required columns, and for missing required columns. Required columns come
# from the Required column of an attached schema Daf.

import pytest
from daffodil.daf import Daf
from daffodil.lib.schemaclass import schemaclass


SCHEMA_MD = """
| Name         | dtype | Default | Required |
|:-------------|:------|:--------|:---------|
| batch_id     | str   |         | yes      |
| precinct     | str   |         | true     |
| ballot_type  | str   |         |          |
| num_ballots  | int   | 0       | 1        |
| share        | float |         |          |
| is_duplicate | bool  | 0       |          |
| tags         | list  |         |          |
| info         | dict  |         | no       |
"""

HEADER = 'batch_id,precinct,ballot_type,num_ballots,share,is_duplicate,tags,info\n'


def _load(body: str) -> Daf:
    daf = Daf.from_csv_buff(HEADER + body, keyfield='batch_id')
    daf.apply_schema(Daf.from_md(SCHEMA_MD))
    return daf.apply_dtypes(dtypes=daf.dtypes)


def test_good_table_passes_and_returns_self() -> None:
    daf = _load("b1,P1,mail,5,0.5,0,\"['a']\",\"{'k': 1}\"\nb2,P2,,7,,1,,\n")
    assert daf.check_schema() is daf


def test_each_kind_of_problem() -> None:
    daf = _load("b1,,mail,abc,x,maybe,\"['a']\",\n"
                "b2,P2,mail,,0.5,0,,\n")
    with pytest.raises(ValueError) as exc_info:
        daf.check_schema()
    assert str(exc_info.value) == (
        "check_schema(): 5 cells do not fit the schema:\n"
        "  row 'b1', column 'precinct': required, but empty\n"
        "  row 'b1', column 'num_ballots': 'abc' is not int\n"
        "  row 'b2', column 'num_ballots': required, but empty\n"
        "  row 'b1', column 'share': 'x' is not float\n"
        "  row 'b1', column 'is_duplicate': 'maybe' is not bool")


def test_float_column_may_hold_int_and_bool_column_holds_0_or_1() -> None:
    daf = Daf(lol=[['b1', 'P1', 'm', 3, 2, True, [], {}]], cols=HEADER.strip().split(','), keyfield='batch_id')
    daf.apply_schema(Daf.from_md(SCHEMA_MD))
    assert daf.check_schema() is daf
    daf[0, 'is_duplicate'] = 2
    with pytest.raises(ValueError, match="row 'b1', column 'is_duplicate': 2 is not bool"):
        daf.check_schema()


def test_text_in_list_or_dict_column() -> None:
    daf = _load("b1,P1,mail,5,,0,not a list,{broken\n")
    with pytest.raises(ValueError, match="2 cells") as exc_info:
        daf.check_schema()
    assert "column 'tags': 'not a list' is not list" in str(exc_info.value)
    assert "column 'info': '{broken' is not dict" in str(exc_info.value)


def test_missing_required_column() -> None:
    daf = Daf.from_csv_buff('batch_id,ballot_type,num_ballots\nb1,mail,5\n', keyfield='batch_id')
    daf.apply_schema(Daf.from_md(SCHEMA_MD))
    with pytest.raises(ValueError, match="column 'precinct': required, but missing"):
        daf.check_schema()


def test_max_report_and_total() -> None:
    daf = _load(''.join(f"b{i},,mail,5,,0,,\n" for i in range(25)))
    with pytest.raises(ValueError) as exc_info:
        daf.check_schema(max_report=3)
    msg = str(exc_info.value)
    assert msg.startswith("check_schema(): 25 cells do not fit the schema. The first 3:\n")
    assert msg.count('\n  ') == 3


def test_no_keyfield_reports_the_row_position() -> None:
    daf = Daf.from_csv_buff('id,n\na,1\nb,x\n')
    daf.apply_schema(Daf.from_md('| Name | dtype |\n|:--|:--|\n| id | str |\n| n | int |\n'))
    daf.apply_dtypes(dtypes=daf.dtypes)
    with pytest.raises(ValueError, match="row 1, column 'n': 'x' is not int"):
        daf.check_schema()


def test_short_row_counts_as_null() -> None:
    daf = Daf(lol=[['b1', 'P1', 'm', 5], ['b2']], cols=['batch_id', 'precinct', 'ballot_type', 'num_ballots'], keyfield='batch_id')
    daf.apply_schema(Daf.from_md(SCHEMA_MD))
    with pytest.raises(ValueError, match="2 cells") as exc_info:
        daf.check_schema()
    assert "row 'b2', column 'precinct': required, but empty" in str(exc_info.value)


def test_schemaclass_checks_dtypes_only() -> None:
    @schemaclass
    class Batch:
        batch_id: str = ''
        num_ballots: int = 0
    daf = Daf(lol=[['', 'abc']], schema=Batch)
    with pytest.raises(ValueError, match="row 0, column 'num_ballots': 'abc' is not int"):
        daf.check_schema()


def test_no_schema_no_dtypes_passes() -> None:
    daf = Daf(lol=[['a', 'x']], cols=['k', 'v'])
    assert daf.check_schema() is daf


def test_load_is_unchanged_without_the_call() -> None:
    daf = _load("b1,,mail,abc,,0,,\n")
    assert daf.lol[0][1] == '' and daf.lol[0][3] == 'abc'


def test_composite_key_reports_the_row_position() -> None:
    daf = Daf(lol=[['a', 1, 'x']], cols=['g', 'h', 'n'], keyfield=['g', 'h'])
    daf.apply_schema(Daf.from_md('| Name | dtype |\n|:--|:--|\n| g | str |\n| h | int |\n| n | int |\n'))
    with pytest.raises(ValueError, match="row 0, column 'n': 'x' is not int"):
        daf.check_schema()
