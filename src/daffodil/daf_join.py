# daf_join.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for joins and annotation, which combine two tables.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

from daffodil.lib.daf_types import T_cs, T_da, T_ds, T_la, T_ls, T_ma
import daffodil.lib.daf_utils    as daf_utils
from daffodil.lib.daf_utils import NULL

from typing import List, Any, Tuple, cast, TYPE_CHECKING

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf, KeysDisabledError, T_klodaf, logs

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf', 'KeysDisabledError', 'logs')


def _update_with_history(
        self:       'Daf',                      # the table as it is, such as one read from its CSV file
        new_daf:    'Daf',                      # the rerun of some of its records, typed or not
        by:         str = '',                   # the column that names a record. The keyfield if empty.
        key_order:  'T_la | Daf | None' = None, # the order of the keys in the result
        ) -> Tuple['Daf', 'Daf']:               # updated, history
    """
    Merge a rerun of some records into this table, and return the records it replaced.

    Use it when some records of a saved table are made again, and the table must hold the new
    version while the old one is kept. This Daf is the table as it is, usually read from its CSV
    file with no `apply_dtypes()`, so every cell is text. `new_daf` holds the rerun records, with
    the same columns, in any order.

    A record is all the rows with one value in the `by` column, as `groupby()` gives them. Where
    `by` is the keyfield, each record is one row. A rerun record replaces the whole prior record,
    even when it has fewer rows.

    The rerun is compared as text. A copy of it is converted with `apply_dtypes()` to `str` with
    `from_str=False`, so 3 becomes `'3'` and a dict becomes its text, as a CSV file holds them.
    So a rerun record that equals the prior one cell by cell is no change.

    The keys are taken in `key_order`, or in this table's order, with new keys last. For each key:

    - With no rerun record, or one equal to the prior record, the prior record is kept.
    - Otherwise the rerun record goes into the result as text. The prior record goes to history.
      A new key, with no prior record, gets a history row with only its key.

    So history is empty exactly when nothing was taken from the rerun, and the table need not be
    written again. A rerun with no rows changes nothing.

    Neither input is changed. Both results are new Dafs with rows of their own. `updated` has this
    Daf's columns and keyfield. `history` has the same columns and no keyfield.

    Args:
        new_daf: The rerun records. Its columns must be the same set as this Daf's.
        by: The column that names a record. The keyfield of this Daf if empty.
        key_order: The keys in the order wanted, as a list, or a Daf whose keys are used. It must hold every key of both tables. This Daf's order if None.

    Returns:
        A tuple of two Dafs. The first is the updated table. The second is the history, the prior
            rows of each changed record.

    Raises:
        ValueError: No `by` is given and this Daf has no keyfield. The columns of the two tables
            differ. A key of either table is not in `key_order`.

    Examples:
        >>> from daffodil.daf import Daf
        >>> prior_daf = Daf(cols=['bid', 'opt', 'v'],
        ...                 lol=[['a', '1', '0'], ['a', '2', '1'], ['b', '1', '1'], ['c', '1', '0']])
        >>> prior_daf
        <BLANKLINE>
        | bid | opt | v  |
        | --: | --: | -: |
        |   a |   1 |  0 |
        |   a |   2 |  1 |
        |   b |   1 |  1 |
        |   c |   1 |  0 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>

        The rerun changes a, repeats c as it was, and adds d:

        >>> new_daf = Daf(cols=['bid', 'opt', 'v'], lol=[['a', 1, 1], ['a', 2, 0], ['c', 1, 0], ['d', 1, 1]])
        >>> updated_daf, history_daf = prior_daf.update_with_history(new_daf, by='bid')
        >>> updated_daf
        <BLANKLINE>
        | bid | opt | v  |
        | --: | --: | -: |
        |   a |   1 |  1 |
        |   a |   2 |  0 |
        |   b |   1 |  1 |
        |   c |   1 |  0 |
        |   d |   1 |  1 |
        <BLANKLINE>
        %% daf rows=5; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> history_daf
        <BLANKLINE>
        | bid | opt | v  |
        | --: | --: | -: |
        |   a |   1 |  0 |
        |   a |   2 |  1 |
        |   d |     |    |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """

    keyfield_a: Any = self.keyfield
    by = by or (keyfield_a if isinstance(keyfield_a, str) else '')
    if not by:
        raise ValueError("update_with_history() needs a by column, or a keyfield on the table.")

    cols_ls: T_ls = self.columns()
    updated_daf: Daf = Daf(cols=cols_ls, keyfield=keyfield_a)
    history_daf: Daf = Daf(cols=cols_ls)

    #---- A rerun with no rows changes nothing

    if not new_daf:
        updated_daf.append(self)
        return updated_daf, history_daf

    if set(new_daf.columns()) != set(cols_ls):
        raise ValueError(f"update_with_history(): the rerun's columns differ from the table's. "
                         f"Only in the rerun: {sorted(set(new_daf.columns()) - set(cols_ls))}. "
                         f"Only in the table: {sorted(set(cols_ls) - set(new_daf.columns()))}.")

    #---- The rerun as text, in this table's columns and keyfield

    # select_cols() makes new rows, so the conversion leaves new_daf as it is.
    new_text_daf: Daf = new_daf.select_cols(cols_ls)
    new_text_daf.apply_dtypes(dtypes={col: str for col in cols_ls}, from_str=False)
    new_text_daf.set_keyfield(keyfield_a)

    #---- The records of each table, by key

    prior_records_klodaf: T_klodaf = self.groupby(by)
    new_records_klodaf: T_klodaf   = new_text_daf.groupby(by)

    #---- The order of the walk

    # Without a key order, this table's order holds, and keys new to it follow.
    # A given key order must hold every key of both tables.
    key: Any
    walk_keys_la: T_la
    if key_order is None:
        walk_keys_la = list(prior_records_klodaf) + [key for key in new_records_klodaf if key not in prior_records_klodaf]
    else:
        walk_keys_la = list(key_order.keys() if isinstance(key_order, Daf) else key_order)
        walk_keys_set: set = set(walk_keys_la)
        stray_la: T_la = ([key for key in prior_records_klodaf if key not in walk_keys_set]
                          + [key for key in new_records_klodaf if key not in walk_keys_set])
        if stray_la:
            raise ValueError(f"update_with_history(): keys not in key_order: {stray_la[:10]}")

    #---- Walk the keys in order

    # Each record is a selection of its table, which copies no rows.
    # append() copies them into the result, and adds nothing for None.
    prior_rec_daf: 'Daf | None'
    new_rec_daf: 'Daf | None'
    for key in walk_keys_la:
        prior_rec_daf = prior_records_klodaf.get(key)
        new_rec_daf   = new_records_klodaf.get(key)

        if not new_rec_daf or prior_rec_daf == new_rec_daf:
            updated_daf.append(prior_rec_daf)
            continue

        updated_daf.append(new_rec_daf)
        history_daf.append(prior_rec_daf or {by: key})

    return updated_daf, history_daf


def _annotate_daf(self: 'Daf', other_daf: 'Daf', my_to_other_dict: T_ds) -> 'Daf':
    """
    Copy fields from another Daf into this one, row by row, matching on the key.

    Both Daf instances need a keyfield. For each row here, the row with the same
    key in `other_daf` is found, and each field of this row named in
    `my_to_other_dict` gets the value of the other field. A key that is missing
    in `other_daf` raises `KeyError`. A field that is not a column here is added as
    a new column at the right.

    Args:
        other_daf: The Daf to copy from.
        my_to_other_dict: Maps the field to set here to the field to read there.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: A keyfield is not set, or a key is not found in `other_daf`.

    Examples:
        >>> a_daf = Daf(lol=[[1, 'x'], [2, 'y']], cols=['id', 'v'], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  x |
        |  2 |  y |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> o_daf = Daf(lol=[[1, 'P'], [2, 'Q']], cols=['id', 'w'], keyfield='id')
        >>> o_daf
        <BLANKLINE>
        | id | w  |
        | -: | -: |
        |  1 |  P |
        |  2 |  Q |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.annotate_daf(o_daf, {'v': 'w'})
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  P |
        |  2 |  Q |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """

    my_keyfield = self.keyfield
    if not my_keyfield:
        raise KeyError("annotate_daf: self must have keyfield defined")

    self._rebuild_kd_if_invalidated()

    other_keyfield = other_daf.keyfield
    if not other_keyfield:
        raise KeyError("annotate_daf: other daf array must have keyfield defined")

    other_daf._rebuild_kd_if_invalidated()

    # a field that is not a column is added first, filled with NULL, so the names match the data.
    for my_field in my_to_other_dict:
        if my_field not in self.hd:
            self.assign_col(my_field)

    for my_rec_klist in self.iter_klist():

        rowkey = self._get_keyval(my_rec_klist)

        other_rec = other_daf.select_record(rowkey)

        for my_field, other_field in my_to_other_dict.items():
            my_rec_klist[my_field] = other_rec[other_field]

    # keys still valid

    return self


def _derive_join_translator(
    self: 'Daf',
    other_daf: 'Daf',                               # The other Daf instance to join with
    shared_fields: T_ls | None = None,           # Columns shared between tables that do not require renaming
                                                    # keyfields do not need to be added here.
    omit_other_cols: T_ls | None = None,           # cols to omit from other (use instead of shared fields)
    tag_other: bool = False,                        # if True, and col not in shared_fields, add suffix tag to other_daf cols

    ) -> 'Daf':  # Translator Daf
    """
    Work out how the columns of two Daf instances are named in a join.

    The translator is a Daf with a row for each column of the result. Its
    columns are `resolved_colname`, `source_name`, `source_colname` and
    `is_keyfield`. A column name that both Daf instances have is given a suffix
    with the name of its source, such as `name_daf1`, unless it is a shared
    field. The names of the instances are `daf1` and `daf2` if they have none.
    The keyfields are always shared.

    `join()` calls this. Call it yourself to see the names, or to edit the
    translator and give it back as `custom_translator_daf`.

    Args:
        other_daf: The Daf to join with.
        shared_fields: Columns that both have and that appear only once.
        omit_other_cols: Columns of the other Daf to leave out.
        tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.

    Returns:
        The translator Daf. Its keyfield is `resolved_colname`.

    Examples:
        >>> a_daf = Daf(lol=[[1, 'x']], cols=['id', 'name'], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id | name |
        | -: | ---: |
        |  1 |    x |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> b_daf = Daf(lol=[[1, 'y']], cols=['id', 'name'], keyfield='id')
        >>> b_daf
        <BLANKLINE>
        | id | name |
        | -: | ---: |
        |  1 |    y |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.derive_join_translator(b_daf).col('resolved_colname')
        ['id', 'name_daf1', 'name_daf2']
    """
    assert isinstance(self.keyfield, str)
    assert isinstance(other_daf.keyfield, str)

    translator_daf = Daf.derive_join_translator_daf(
        self_keyfield       = self.keyfield,
        other_keyfield      = other_daf.keyfield,
        self_cols           = self.hd.keys(),
        other_cols          = other_daf.hd.keys(),
        self_name           = self.name,
        other_name          = other_daf.name,
        shared_fields       = shared_fields,
        omit_other_cols     = omit_other_cols,
        tag_other           = tag_other,
        )
    return translator_daf


def _derive_join_translator_daf(
    cls: 'type[Daf]',
    self_keyfield:      str,                        # pass self.keyfield (daf)          or esc_my_index_col (sql)
    other_keyfield:     str,                        # pass other_daf.keyfield (daf)     or esc_other_index_col (sql)
    self_cols:          T_cs,                       # pass self.columns() (daf)         or esc_sql_cols (sql)
    other_cols:         T_cs,                       # pass other_daf.columns() (daf)    or esc_sql_other_cols (sql)
    self_name:          str = '',                   # pass self.name (daf)              or esc_sql_table_name (sql)
    other_name:         str = '',                   # pass other_daf.name (daf)         or esc_sql_other_table_name (sql)
    shared_fields:      T_cs | None = None,         # Columns shared between tables that do not require renaming (esc if sql)
                                                    # use omit_other_cols instead of shared_fields due to better name.
    omit_other_cols:    T_cs  | None = None,        # cols to omit from other (esc if sql)
    tag_other:          bool = False,               # if True, and col not in shared_fields, add suffix tag to other_cols

    ) -> 'Daf':  # Translator Daf
    """
    Work out a join translator from column names, with no Daf instances.

    This is the form that does not need two Daf instances, so SQL joins can use
    it. See `derive_join_translator()` for what the translator holds.

    Args:
        self_keyfield: The keyfield of the first table.
        other_keyfield: The keyfield of the other table.
        self_cols: The column names of the first table.
        other_cols: The column names of the other table.
        self_name: The name of the first table.
        other_name: The name of the other table.
        shared_fields: Columns that both have and that appear only once. The list is not changed.
        omit_other_cols: Columns of the other table to leave out.
        tag_other: If True, every column of the other table, except shared ones, gets a suffix.

    Returns:
        The translator Daf. Its keyfield is `resolved_colname`.

    Examples:
        >>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
        >>> tr.columns()
        ['resolved_colname', 'source_name', 'source_colname', 'is_keyfield']
        >>> tr
        <BLANKLINE>
        | resolved_colname | source_name | source_colname | is_keyfield |
        | ---------------: | ----------: | -------------: | ----------: |
        |               id |           L |             id |        True |
        |                v |           L |              v |       False |
        |                w |           R |              w |       False |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='resolved_colname'; name=''
        <BLANKLINE>
        >>> tagged = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R', tag_other=True)
        >>> tagged.col('resolved_colname')
        ['id', 'v', 'w_R']
        >>> omitted = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w', 'x'], 'L', 'R', omit_other_cols=['x'])
        >>> omitted.col('resolved_colname')
        ['id', 'v', 'w']
    """

    shared_fields   = list(shared_fields or [])     # a copy, so the caller's list is not changed.
    omit_other_cols = omit_other_cols or []

    if self_keyfield and self_keyfield not in shared_fields:
        shared_fields.append(self_keyfield)

    if other_keyfield and other_keyfield not in shared_fields:
        shared_fields.append(other_keyfield)

    # to_dn_if_list()'s T_ca return type is broader (str|int|tuple keys, untyped dict) than
    # T_cs -- but these are always column-name collections here, so str keys only.
    shared_fields   = cast(T_cs, daf_utils.to_dn_if_list(shared_fields))
    self_cols       = cast(T_cs, daf_utils.to_dn_if_list(self_cols))
    other_cols      = cast(T_cs, daf_utils.to_dn_if_list(other_cols))

    # Get column names and names from both Dafs  (list of KeyViews of Any)
    colnames_locs: List[T_cs] = [self_cols, other_cols]

    # Assign suffixes for source tables based on table names or default
    source_suffixes = [
        f"_{self_name}"         if self_name else "_daf1",
        f"_{other_name}"        if other_name else "_daf2",
        ]

    source_names = [
        self_name if self_name else "daf1",
        other_name if other_name else "daf2",
        ]

    # Identify columns that are common but not shared fields
    common_colnames = [col for col in colnames_locs[0]
                            if col in colnames_locs[1] and col not in shared_fields and col not in omit_other_cols]

    # set.intersection(*[set(colnames_lists[0]) for cols in colnames_lists[1]]) - shared_fields

    # Define the Daf structure first
    translator_daf = Daf(cols=["resolved_colname", "source_name", "source_colname", "is_keyfield"])

    # Populate the Daf by appending rows directly
    for idx, (colnames_cs, source_suffix) in enumerate(zip(colnames_locs, source_suffixes)):
        source_name = source_names[idx]
        tag_with_suffix = idx and tag_other
        for col in colnames_cs:
            if idx and (col in shared_fields or col in omit_other_cols):
                continue

            if (tag_with_suffix or
                    col and col in common_colnames):
                resolved_colname = f"{col}{source_suffix}"
            else:
                resolved_colname = str(col)

            translator_daf.append([
                resolved_colname,   # resolved_colname
                #idx,               # source_index
                source_name,        # source_name
                col,                # source_colname
                bool(col == self_keyfield or col == other_keyfield),   # is_keyfield
            ])

    # Set the keyfield for the resulting Daf
    translator_daf.set_keyfield("resolved_colname")
    return translator_daf


def _join(
    self: 'Daf',
    other_daf: 'Daf',                                   # The other Daf instance to join with.
    how: str = 'inner',                                 # Type of join - 'inner', 'left', 'right', 'outer'. Default is 'inner'.
    shared_fields: T_ls | None = None,                  # List of fields to ignore in conflict resolution (shared fields)
    tag_other: bool = False,                            # if not a shared field or keyfield, suffix all other_daf fields with _{name}
    custom_translator_daf: 'Daf' | None = None,         # provide a custom translater to provide all naming details.
    diagnose: bool = False,                             # Enable diagnostic logging.
    name: str='',                                       # name for the joined instance.
    fill: Any = NULL,                                   # value for cells that have no match.
) -> 'Daf':
    """
    Join two Daf instances on their keyfields, as in SQL.

    Both need a keyfield, and it must be a single column. A composite keyfield, a
    tuple of columns, raises `KeyError`. A row of this Daf is joined with the row of
    `other_daf` that has the same key.

    The join is one to one. It is not a many to many join, as in SQL. Each row of this
    Daf finds at most one row of the other, through its key index. If a key repeats in
    the other Daf, every match gets its last row. If a key repeats in this Daf, each of
    those rows is in the result. Check the keys first if they may repeat. See `keys()`.

    The types of join are:

        inner    only the keys that are in both.
        left     all keys of this Daf.
        right    all keys of the other Daf.
        outer    all keys of both.

    Columns that only one Daf has are in the result. A column that both have
    is given the name of its source as a suffix, such as `name_daf1`, unless it is
    in `shared_fields`. The names are `daf1` and `daf2` if the instances have no
    names. For other names use `custom_translator_daf`, which you can start from
    `derive_join_translator()`.

    When a key has no match, its cells from the other side are NULL, the empty string. Pass
    `fill=None` to get `None` there, as earlier versions did.
    The keyfield of the result is the keyfield of this Daf. The result is a new Daf.

    A row that only the other Daf has, in a right or outer join, comes after the rows
    of this Daf. It takes its key, and the columns in `shared_fields`, from the other
    Daf, since this Daf has no row for it. This holds when the keyfields have different
    names too. The key column of the result has the name of this Daf's keyfield.

    Args:
        other_daf: The Daf to join with.
        how: `inner`, `left`, `right` or `outer`.
        shared_fields: Columns that both have and that appear only once.
        tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.
        custom_translator_daf: A translator that sets all the names.
        diagnose: If True, print progress messages.
        name: The name of the result.
        fill: The value for cells that have no match.

    Returns:
        The joined Daf.

    Raises:
        ValueError: `how` is not one of the four.
        KeysDisabledError: A Daf has no keyfield.
        KeyError: A keyfield is a tuple.

    Examples:
        >>> a_daf = Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id | name  |
        | -: | ----: |
        |  1 | Alice |
        |  2 |   Bob |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> b_daf = Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id')
        >>> b_daf
        <BLANKLINE>
        | id | salary |
        | -: | -----: |
        |  1 |     50 |
        |  3 |     70 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.join(b_daf)
        <BLANKLINE>
        | id | name  | salary |
        | -: | ----: | -----: |
        |  1 | Alice |     50 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.join(b_daf, how='left')
        <BLANKLINE>
        | id | name  | salary |
        | -: | ----: | -----: |
        |  1 | Alice |     50 |
        |  2 |   Bob |        |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.join(b_daf, how='left', fill=None)
        <BLANKLINE>
        | id | name  | salary |
        | -: | ----: | -----: |
        |  1 | Alice |     50 |
        |  2 |   Bob |   None |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"Unsupported join type: {how}")

    if not self.keyfield or not other_daf.keyfield:
        raise KeysDisabledError("Both Daf instances must have a keyfield defined for a join operation.")

    if not isinstance(self.keyfield, str) or not isinstance(other_daf.keyfield, str):
        raise KeyError("join not supported for complex keys (i.e. tuples)")

    self._rebuild_kd_if_invalidated()
    other_daf._rebuild_kd_if_invalidated()

    if diagnose:
        logs.stsloc(f"Initiating join:\nself:\n{self}\nother_daf:\n{other_daf}", 3)

    # Derive or use custom translator Daf
    if custom_translator_daf:
        translator_daf = custom_translator_daf
    else:
        translator_daf = self.derive_join_translator(other_daf, shared_fields=shared_fields, tag_other=tag_other)

    if diagnose:
        logs.stsloc(f"Translator Daf:\n{translator_daf}", 3)

    join_names_ls = [self.name or 'daf1', other_daf.name or 'daf2']

    # we allow the translator to contain records for more than two dafs that may be joined in a chain opeation.
    eff_translator_daf = translator_daf.select_where(lambda row: bool(row.get('source_name') in join_names_ls))

    resolved_colnames = eff_translator_daf.col("resolved_colname")

    # Prepare the resulting Daf
    result_daf = Daf(cols=resolved_colnames, name=name)
    keyfield = self.keyfield
    if keyfield and result_daf._is_keyfield_valid(keyfield):    # an empty table has no key column to give.
        result_daf.keyfield = keyfield                          # okay to set now with lazy kd generation.

    # Helper function to fetch a record by key, with silent error
    def fetch_record(daf: 'Daf', mykey: str | int) -> T_da:
        return daf.select_record(mykey, silent_error=True)

    # Track matched keys for outer joins
    matched_keys = set()

    # Perform the join
    for row_da in self:
        key = row_da[keyfield]
        if not isinstance(key, (str, int)):
            raise KeyError("join not supported for complex keys (i.e. tuples)")

        other_row_da = fetch_record(other_daf, key)

        if other_row_da:
            matched_keys.add(key)
            combined_record = Daf.join_records(
                [row_da, other_row_da],
                translator_daf,
                join_names_ls,
                fill,
            )
            result_daf.append(combined_record)
        elif how in ("left", "outer"):
            combined_record = Daf.join_records(
                [row_da, None],
                translator_daf,
                join_names_ls,
                fill,
            )
            result_daf.append(combined_record)

    if how in ("right", "outer"):
        # A row that only the other Daf has. join_records() takes the key and the shared
        # fields from this Daf, which has no such row. So those cells come from the other row.
        # A shared field is a column of this Daf that the other Daf has too, and that the
        # translator does not take from the other Daf.
        other_mapped_set: set = {row['source_colname'] for row in eff_translator_daf
                                 if row['source_name'] == join_names_ls[1]}
        fallback_lot: List[Tuple[str, str]] = []
        for translator_row in eff_translator_daf:
            if translator_row['source_name'] != join_names_ls[0]:
                continue
            if translator_row.get('is_keyfield', False):
                fallback_lot.append((translator_row['resolved_colname'], other_daf.keyfield))
            elif translator_row['source_colname'] in other_daf.hd and translator_row['source_colname'] not in other_mapped_set:
                fallback_lot.append((translator_row['resolved_colname'], translator_row['source_colname']))

        unmatched_keys = [
            key for key in other_daf._kd if key not in matched_keys
        ]
        for key in unmatched_keys:
            other_row_da = fetch_record(other_daf, key)
            combined_record = Daf.join_records(
                [None, other_row_da],
                translator_daf,
                join_names_ls,
                fill,
            )
            resolved_col: str
            other_col: str
            for resolved_col, other_col in fallback_lot:
                combined_record[resolved_col] = other_row_da[other_col]
            result_daf.append(combined_record)

    if diagnose:
        logs.stsloc(f"Resulting Daf:\n{result_daf}", 3)

    return result_daf


def _join_records(
    records: List[T_ma | None],     # List of dictionaries representing the source records
                                    # only two records are supported here.
                                    # sometimes either one can be None if a corresponding record is not available.

    translator_daf: 'Daf',          # Translator Daf mapping resolved columns to source

    join_names_ls: T_ls | None = None,  # names of the two Daf arrays supplying the records.
                                    #  required only if there are more than two source_names specified in the translator.
                                    # this is used when a single translator is used for chained joins.

    fill: Any = NULL,               # value for a column that the record does not have.

    ) -> T_da:

    """
    Combine one record from each table into one record, using a translator.

    This is a static method, and the step that `join()` repeats. A record may be
    None, which gives `fill` for the columns of that side. A column that a record
    does not have also gets `fill`.

    Args:
        records: The two records, in the order of the source names.
        translator_daf: The translator, as from `derive_join_translator()`.
        join_names_ls: The two source names. Needed only if the translator names more than two sources.
        fill: The value for a column that has no record, or that the record lacks. NULL by default.

    Returns:
        The combined record, as a dict.

    Raises:
        ValueError: The translator names more than two sources and `join_names_ls` is not given.

    Examples:
        >>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
        >>> Daf.join_records([{'id': 1, 'v': 'a'}, {'id': 1, 'w': 'b'}], tr)
        {'id': 1, 'v': 'a', 'w': 'b'}
        >>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr)
        {'id': 1, 'v': 'a', 'w': ''}
        >>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr, fill=0)
        {'id': 1, 'v': 'a', 'w': 0}
    """
    combined_record = {}

    if not join_names_ls:
        join_names_ls = translator_daf.col('source_name', unique=True)
        if len(join_names_ls) > 2 or not join_names_ls:
            raise ValueError ("join_names_ls must be specified as translator_daf has more than two source_names")

    # Iterate through the translator Daf
    for translator_row in translator_daf:
        source_name     = translator_row["source_name"]
        if join_names_ls and source_name not in join_names_ls:
            continue
        source_index    = join_names_ls.index(source_name)
        resolved_col    = translator_row["resolved_colname"]
        source_colname  = translator_row["source_colname"]
        is_keyfield     = translator_row.get("is_keyfield", False)

        # Handle keyfield explicitly
        if is_keyfield:
            # Take the keyfield value from the first non-None record
            for record in records:
                if record and source_colname in record:
                    combined_record[resolved_col] = record[source_colname]
                    break
            else:
                combined_record[resolved_col] = fill
            continue

        # Handle all other columns
        rec_da = records[source_index]

        if rec_da is not None:
            rec_da = cast(T_da, rec_da)
            combined_record[resolved_col] = rec_da.get(source_colname, fill)
        else:
            combined_record[resolved_col] = fill

    return combined_record
