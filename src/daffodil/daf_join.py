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

from daffodil.keyedlist import KeyedList

from typing import List, Any, Callable, Dict, Tuple, cast, TYPE_CHECKING

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
        Create a small Daf of order lines, several lines for some orders. The values are text, as from a CSV file:

        >>> from daffodil.daf import Daf
        >>> prior_daf = Daf(cols=['order_id', 'line', 'qty'],
        ...                 lol=[['o1', '1', '5'], ['o1', '2', '2'], ['o2', '1', '8'], ['o3', '1', '4']])
        >>> prior_daf
        <BLANKLINE>
        | order_id | line | qty |
        | -------: | ---: | --: |
        |       o1 |    1 |   5 |
        |       o1 |    2 |   2 |
        |       o2 |    1 |   8 |
        |       o3 |    1 |   4 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>

        The new table changes order o1, repeats o3 as it was, and adds o4:

        >>> new_daf = Daf(cols=['order_id', 'line', 'qty'], lol=[['o1', 1, 6], ['o1', 2, 2], ['o3', 1, 4], ['o4', 1, 3]])

        Update it, and keep the replaced rows as history:

        >>> updated_daf, history_daf = prior_daf.update_with_history(new_daf, by='order_id')
        >>> updated_daf
        <BLANKLINE>
        | order_id | line | qty |
        | -------: | ---: | --: |
        |       o1 |    1 |   6 |
        |       o1 |    2 |   2 |
        |       o2 |    1 |   8 |
        |       o3 |    1 |   4 |
        |       o4 |    1 |   3 |
        <BLANKLINE>
        %% daf rows=5; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> history_daf
        <BLANKLINE>
        | order_id | line | qty |
        | -------: | ---: | --: |
        |       o1 |    1 |   5 |
        |       o1 |    2 |   2 |
        |       o4 |      |     |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        With `key_order`, the keys come in that order. It must hold every key of both tables:

        >>> prior_daf.update_with_history(new_daf, by='order_id', key_order=['o4', 'o3', 'o2', 'o1'])[0]
        <BLANKLINE>
        | order_id | line | qty |
        | -------: | ---: | --: |
        |       o4 |    1 |   3 |
        |       o3 |    1 |   4 |
        |       o2 |    1 |   8 |
        |       o1 |    1 |   6 |
        |       o1 |    2 |   2 |
        <BLANKLINE>
        %% daf rows=5; cols=3; keyfield=''; name=''
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
        Create a Daf of people whose city is not known yet, and a Daf that has their cities:

        >>> a_daf = Daf(cols=['id', 'city'], lol=[[1, 'unknown'], [2, 'unknown']], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id |  city   |
        | -: | ------: |
        |  1 | unknown |
        |  2 | unknown |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> o_daf = Daf(cols=['id', 'home_city'], lol=[[1, 'Denver'], [2, 'Austin']], keyfield='id')
        >>> o_daf
        <BLANKLINE>
        | id | home_city |
        | -: | --------: |
        |  1 |    Denver |
        |  2 |    Austin |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        Fill in each city from the other Daf, matched by key:

        >>> a_daf.annotate_daf(o_daf, {'city': 'home_city'})
        <BLANKLINE>
        | id |  city  |
        | -: | -----: |
        |  1 | Denver |
        |  2 | Austin |
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
        Create two small Daf tables of people, which both have a name column:

        >>> a_daf = Daf(cols=['id', 'name'], lol=[[1, 'Ana Lopez']], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id |   name    |
        | -: | --------: |
        |  1 | Ana Lopez |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> b_daf = Daf(cols=['id', 'name'], lol=[[1, 'Ana L. Lopez']], keyfield='id')
        >>> b_daf
        <BLANKLINE>
        | id |     name     |
        | -: | -----------: |
        |  1 | Ana L. Lopez |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        See the names the join would give the columns. Each name column gets its table's suffix:

        >>> a_daf.derive_join_translator(b_daf).col('resolved_colname')
        ['id', 'name_daf1', 'name_daf2']

        The tables used below. Both have `item`:

        >>> stock = Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3]])
        >>> price = Daf(cols=['id', 'item', 'price'], keyfield='id', lol=[['a1', 'pen', 1.5]])

        With `shared_fields`, the shared column comes from this Daf only:

        >>> stock.derive_join_translator(price, shared_fields=['item'])
        <BLANKLINE>
        | resolved_colname | source_name | source_colname | is_keyfield |
        | ---------------: | ----------: | -------------: | ----------: |
        |               id |        daf1 |             id |        True |
        |             item |        daf1 |           item |       False |
        |              qty |        daf1 |            qty |       False |
        |            price |        daf2 |          price |       False |
        <BLANKLINE>
        %% daf rows=4; cols=4; keyfield='resolved_colname'; name=''
        <BLANKLINE>

        `omit_other_cols` leaves columns of the other Daf out:

        >>> stock.derive_join_translator(price, omit_other_cols=['item']).col('resolved_colname')
        ['id', 'item', 'qty', 'price']

        `tag_other=True` gives every column of the other Daf a suffix:

        >>> stock.derive_join_translator(price, tag_other=True).col('resolved_colname')
        ['id', 'item_daf1', 'qty', 'item_daf2', 'price_daf2']
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
        Make the translator for a join of people and pay, from their keyfields, columns and names:

        >>> translator = Daf.derive_join_translator_daf('id', 'id', ['id', 'name'], ['id', 'salary'], 'people', 'pay')
        >>> translator.columns()
        ['resolved_colname', 'source_name', 'source_colname', 'is_keyfield']
        >>> translator
        <BLANKLINE>
        | resolved_colname | source_name | source_colname | is_keyfield |
        | ---------------: | ----------: | -------------: | ----------: |
        |               id |      people |             id |        True |
        |             name |      people |           name |       False |
        |           salary |         pay |         salary |       False |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='resolved_colname'; name=''
        <BLANKLINE>

        Give every column of the other table a suffix, with `tag_other=True`:

        >>> tagged = Daf.derive_join_translator_daf('id', 'id', ['id', 'name'], ['id', 'salary'], 'people', 'pay', tag_other=True)
        >>> tagged.col('resolved_colname')
        ['id', 'name', 'salary_pay']

        Leave a column of the other table out, with `omit_other_cols`:

        >>> omitted = Daf.derive_join_translator_daf('id', 'id', ['id', 'name'], ['id', 'salary', 'bonus'], 'people', 'pay', omit_other_cols=['bonus'])
        >>> omitted.col('resolved_colname')
        ['id', 'name', 'salary']
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
        Create two small Daf tables, people and their salaries, keyed by id:

        >>> a_daf = Daf(cols=['id', 'name'], lol=[[1, 'Ana Lopez'], [2, 'Ben Okafor']], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id |    name    |
        | -: | ---------: |
        |  1 |  Ana Lopez |
        |  2 | Ben Okafor |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> b_daf = Daf(cols=['id', 'salary'], lol=[[1, 72000], [3, 58000]], keyfield='id')
        >>> b_daf
        <BLANKLINE>
        | id | salary |
        | -: | -----: |
        |  1 |  72000 |
        |  3 |  58000 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        Join them. By default only the keys in both are kept:

        >>> a_daf.join(b_daf)
        <BLANKLINE>
        | id |   name    | salary |
        | -: | --------: | -----: |
        |  1 | Ana Lopez |  72000 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Keep every person, with NULL where there is no salary, or with another `fill` value:

        >>> a_daf.join(b_daf, how='left')
        <BLANKLINE>
        | id |    name    | salary |
        | -: | ---------: | -----: |
        |  1 |  Ana Lopez |  72000 |
        |  2 | Ben Okafor |        |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.join(b_daf, how='left', fill=None)
        <BLANKLINE>
        | id |    name    | salary |
        | -: | ---------: | -----: |
        |  1 |  Ana Lopez |  72000 |
        |  2 | Ben Okafor |   None |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        The tables used below. Both have `item`:

        >>> stock = Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]])
        >>> price = Daf(cols=['id', 'item', 'price'], keyfield='id', lol=[['a1', 'pen', 1.5], ['b2', 'ink', 4.0], ['d4', 'cap', 0.5]])

        A column that both have gets a suffix for each source:

        >>> stock.join(price).columns()
        ['id', 'item_daf1', 'qty', 'item_daf2', 'price']

        `shared_fields` names columns that hold the same values in both, so they appear once:

        >>> stock.join(price, shared_fields=['item'])
        <BLANKLINE>
        | id | item | qty | price |
        | -: | ---: | --: | ----: |
        | a1 |  pen |   3 |   1.5 |
        | b2 |  ink |  10 |   4.0 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name=''
        <BLANKLINE>

        `tag_other=True` gives every column of the other Daf a suffix, even one only it has:

        >>> stock.join(price, tag_other=True).columns()
        ['id', 'item_daf1', 'qty', 'item_daf2', 'price_daf2']

        Name the result:

        >>> stock.join(price, how='left', shared_fields=['item'], name='stock_prices').name
        'stock_prices'

        For other names, edit the translator from `derive_join_translator()` and pass it in. Here the
        price column is renamed:

        >>> translator = stock.derive_join_translator(price, shared_fields=['item'])
        >>> translator[3, 'resolved_colname'] = 'unit_price'
        >>> stock.join(price, custom_translator_daf=translator).columns()
        ['id', 'item', 'qty', 'unit_price']
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


def _join_with_residue(
        self: 'Daf',
        other_daf:              'Daf',                  # the other Daf, joined by key.
        shared_fields:          T_ls | None = None,     # columns both have that appear once, as in join().
        tag_other:              bool = False,           # suffix the other's columns, as in join().
        custom_translator_daf:  'Daf | None' = None,    # the naming of the columns, as in join().
        diagnose:               bool = False,           # log the steps, as in join().
        name:                   str = '',               # the name of the joined Daf.
        ) -> Tuple['Daf', 'Daf', 'Daf']:
    """
    Join two tables by key, and keep the rows of each that have no partner.

    It gives three Daf objects:

    - the inner join: the rows whose key both tables have, with the columns of both, as
      `join()` gives them.
    - the residue of this Daf: its rows whose key the other lacks, in its own columns.
    - the residue of the other Daf: its rows whose key this Daf lacks, in its own columns.

    Together they hold every row of both tables. No row is lost, and no column is added to say
    how a row matched. An outer join keeps every row too, but pads each unmatched row with the
    other table's columns, so only blank cells show it had no partner.

    Use it where an unmatched row is a finding to report, not a row to drop. Neither table is
    changed. Each residue is a selection, which shares its rows with its table, as
    `select_records()` gives. Both tables need a keyfield.

    Args:
        other_daf: The Daf to join with, by its keyfield.
        shared_fields: Columns that both have and that appear only once in the join.
        tag_other: If True, every column of the other, except shared ones, gets a suffix.
        custom_translator_daf: A translator that sets the names of the joined columns. See `join()`.
        diagnose: If True, log the steps of the join.
        name: The name of the joined Daf.

    Returns:
        The joined Daf, the residue of this Daf, and the residue of the other.

    Raises:
        KeysDisabledError: A table has no keyfield.

    Examples:
        Create two small Daf tables, people and their salaries. Each has a key the other lacks:

        >>> people_daf = Daf(cols=['id', 'name', 'city'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver'], ['p2', 'Ben Okafor', 'Austin'], ['p3', 'Chen Wu', 'Boston']])
        >>> people_daf
        <BLANKLINE>
        | id |    name    |  city  |
        | -: | ---------: | -----: |
        | p1 |  Ana Lopez | Denver |
        | p2 | Ben Okafor | Austin |
        | p3 |    Chen Wu | Boston |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> pay_daf = Daf(cols=['id', 'salary', 'city'], keyfield='id', lol=[
        ...     ['p2', 64000, 'Austin'], ['p3', 58000, 'Boston'], ['p9', 81000, 'Denver']])
        >>> pay_daf
        <BLANKLINE>
        | id | salary |  city  |
        | -: | -----: | -----: |
        | p2 |  64000 | Austin |
        | p3 |  58000 | Boston |
        | p9 |  81000 | Denver |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Join them, and keep the rows of each that have no partner:

        >>> joined_daf, people_res_daf, pay_res_daf = people_daf.join_with_residue(pay_daf, shared_fields=['city'])
        >>> joined_daf
        <BLANKLINE>
        | id |    name    |  city  | salary |
        | -: | ---------: | -----: | -----: |
        | p2 | Ben Okafor | Austin |  64000 |
        | p3 |    Chen Wu | Boston |  58000 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name=''
        <BLANKLINE>
        >>> people_res_daf.to_lod()
        [{'id': 'p1', 'name': 'Ana Lopez', 'city': 'Denver'}]
        >>> pay_res_daf.to_lod()
        [{'id': 'p9', 'salary': 81000, 'city': 'Denver'}]

        The options of the join act as in `join()`:

        >>> people_daf.join_with_residue(pay_daf, tag_other=True, name='people_pay')[0].columns()
        ['id', 'name', 'city_daf1', 'salary_daf2', 'city_daf2']
    """
    joined_daf: 'Daf' = self.join(other_daf, how='inner', shared_fields=shared_fields, tag_other=tag_other,
                                  custom_translator_daf=custom_translator_daf, diagnose=diagnose, name=name)

    # Each residue is the rows whose key the other table lacks.
    self_res_daf: 'Daf' = self.select_records(other_daf, inverse=True, silent_error=True)
    other_res_daf: 'Daf' = other_daf.select_records(self, inverse=True, silent_error=True)
    return joined_daf, self_res_daf, other_res_daf

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
        Make the translator for a join of people and pay:

        >>> translator = Daf.derive_join_translator_daf('id', 'id', ['id', 'name'], ['id', 'salary'], 'people', 'pay')

        Join one row of each into one row:

        >>> Daf.join_records([{'id': 1, 'name': 'Ana Lopez'}, {'id': 1, 'salary': 72000}], translator)
        {'id': 1, 'name': 'Ana Lopez', 'salary': 72000}

        With no row from the other table, its cells are NULL, or the `fill` value:

        >>> Daf.join_records([{'id': 1, 'name': 'Ana Lopez'}, None], translator)
        {'id': 1, 'name': 'Ana Lopez', 'salary': ''}
        >>> Daf.join_records([{'id': 1, 'name': 'Ana Lopez'}, None], translator, fill=0)
        {'id': 1, 'name': 'Ana Lopez', 'salary': 0}
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


def _joint_apply(
        required_klodaf:    'T_klodaf',                 # the tables a key must be in, by name, each with a keyfield.
        func:               Callable[..., Any],         # func(key, rows_daf) gives a dict of output name to rows.
        optional_klodaf:    'T_klodaf | None' = None,   # tables that add their row where they have the key.
        ) -> 'Tuple[T_klodaf, T_klodaf]':               # outputs, residues
    """
    Combine several tables row by row, by key, with a function, and keep the rows that do not match.

    Use it when several tables describe the same items, each keyed by the same id, and each row
    of the result is made from the rows of all of them. `join()` combines two tables by the names
    of their columns. This combines any number, and the function decides what each result row holds,
    such as a flag where two tables disagree.

    The tables are of two kinds:

    - A required table must have the key. A key is combined only if every required table has it.
    - An optional table adds its row where it has the key, and None where it does not.
      It never keeps a key from being combined.

    Each table is matched by its own keyfield. The keys come in the order of the first required
    table. A key that repeats in a table finds its last row, as a lookup by key does.

    The function is called once for each combined key, as `func(key, rows_daf)`. `rows_daf` is a
    small Daf with one row for each table that has the key, keyed by the table's name in a first
    column, `table`. Its other columns are those of all the tables, in the order first seen. A column
    that several tables have is one column, so the tables can be compared on it, as
    `rows_daf.col('city')`. A cell of a column that a table lacks is NULL, so leave NULL out when
    comparing a column that only some tables have. An optional table that
    lacks the key has no row, so `'notes' in rows_daf` tells whether it has one. No table may have a
    column named `table`.

    `rows_daf` is one Daf, refilled for each key, since that is several times faster than a new one.
    So it is valid only during the call. To keep it, keep `rows_daf.copy('editable')`.

    The function returns a dict of output name to a row, a list of rows, or None, or it returns None
    for no output. A row is a dict. Each output has the columns its rows use, in the order first
    seen, as `from_lod()` gives them. A row without a column has NULL there. An output has no
    keyfield. Set one with `set_keyfield()`.

    The residues are made here, not by the function. Each required table has one: its rows whose key
    another required table lacks. A residue is a selection of its own table, in its own columns, as
    `select_records()` gives, so its rows are shared with the table. An optional table has no residue.
    With two required tables and no optional ones, the result is the same as `join_with_residue()`,
    with the joined rows made by the function.

    Neither table is changed. A SqlDaf among the tables makes the whole call run as
    [`SqlDaf.joint_apply()`][daffodil.sqldaf_apply._joint_apply], with SqlDaf results.

    Args:
        required_klodaf: The tables a key must be in, by name.
        func: Called as `func(key, rows_daf)` for each combined key. Returns a dict of output name to rows.
        optional_klodaf: Tables that add their row where they have the key, by name.

    Returns:
        The outputs, by name, in the order first written, and the residues, by the name of each required table.

    Raises:
        ValueError: No required table is given, a table has no keyfield, or a table has a column named `table`.
        TypeError: The function returns something other than a dict or None, or a row that is not a dict.

    Examples:
        Create three small tables about the same people, keyed by id. Each lacks someone:

        >>> people = Daf(cols=['id', 'name', 'city'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver'], ['p2', 'Ben Okafor', 'Austin'], ['p3', 'Chen Wu', 'Boston']])
        >>> people
        <BLANKLINE>
        | id |    name    |  city  |
        | -: | ---------: | -----: |
        | p1 |  Ana Lopez | Denver |
        | p2 | Ben Okafor | Austin |
        | p3 |    Chen Wu | Boston |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> visits = Daf(cols=['id', 'city', 'visits'], keyfield='id', lol=[
        ...     ['p1', 'Denver', 4], ['p2', 'Dallas', 7], ['p9', 'Reno', 2]])
        >>> visits
        <BLANKLINE>
        | id |  city  | visits |
        | -: | -----: | -----: |
        | p1 | Denver |      4 |
        | p2 | Dallas |      7 |
        | p9 |   Reno |      2 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> notes = Daf(cols=['id', 'note'], keyfield='id', lol=[['p2', 'moved']])
        >>> notes
        <BLANKLINE>
        | id | note  |
        | -: | ----: |
        | p2 | moved |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        Combine people and visits, both required. Flag the people whose two cities disagree:

        >>> def combine(key, rows_daf):
        ...     person, visit = rows_daf.select_record('people'), rows_daf.select_record('visits')
        ...     return {'joined': {'id': key, 'name': person['name'], 'visits': visit['visits'],
        ...                        'city_differs': int(len(set(rows_daf.col('city'))) > 1)}}
        >>> outputs_klodaf, residues_klodaf = Daf.joint_apply(KeyedList({'people': people, 'visits': visits}), combine)
        >>> outputs_klodaf['joined']
        <BLANKLINE>
        | id |    name    | visits | city_differs |
        | -: | ---------: | -----: | -----------: |
        | p1 |  Ana Lopez |      4 |            0 |
        | p2 | Ben Okafor |      7 |            1 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield=''; name=''
        <BLANKLINE>

        The function gets a small Daf for each key. Here is the one for p2, with notes as optional:

        >>> def show(key, rows_daf):
        ...     if key == 'p2':
        ...         print(rows_daf)
        >>> outputs_klodaf, residues_klodaf = Daf.joint_apply(
        ...     KeyedList({'people': people, 'visits': visits}), show, optional_klodaf=KeyedList({'notes': notes}))
        | table  | id |    name    |  city  | visits | note  |
        | -----: | -: | ---------: | -----: | -----: | ----: |
        | people | p2 | Ben Okafor | Austin |        |       |
        | visits | p2 |            | Dallas |      7 |       |
        |  notes | p2 |            |        |        | moved |
        <BLANKLINE>
        %% daf rows=3; cols=6; keyfield='table'; name=''
        <BLANKLINE>

        Each required table keeps, as its residue, the rows whose key the other lacks:

        >>> residues_klodaf['people']
        <BLANKLINE>
        | id |  name   |  city  |
        | -: | ------: | -----: |
        | p3 | Chen Wu | Boston |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> residues_klodaf['visits']
        <BLANKLINE>
        | id | city | visits |
        | -: | ---: | -----: |
        | p9 | Reno |      2 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Add notes as an optional table. Its row is None where it lacks the key, and it has no residue:

        >>> def with_note(key, rows_daf):
        ...     note = rows_daf.select_record('notes')['note'] if 'notes' in rows_daf else ''
        ...     return {'joined': {'id': key, 'note': note}}
        >>> outputs_klodaf, residues_klodaf = Daf.joint_apply(
        ...     KeyedList({'people': people, 'visits': visits}), with_note, optional_klodaf=KeyedList({'notes': notes}))
        >>> outputs_klodaf['joined']
        <BLANKLINE>
        | id | note  |
        | -: | ----: |
        | p1 |       |
        | p2 | moved |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> list(residues_klodaf.keys())
        ['people', 'visits']

        Write to several outputs, a list of rows to one, or nothing at all for a key:

        >>> def split(key, rows_daf):
        ...     cities_ls = rows_daf.col('city')
        ...     if len(set(cities_ls)) == 1:
        ...         return None
        ...     return {'moved': {'id': key}, 'cities': [{'id': key, 'city': city} for city in cities_ls]}
        >>> outputs_klodaf, residues_klodaf = Daf.joint_apply(KeyedList({'people': people, 'visits': visits}), split)
        >>> list(outputs_klodaf.keys())
        ['moved', 'cities']
        >>> outputs_klodaf['cities']
        <BLANKLINE>
        | id |  city  |
        | -: | -----: |
        | p2 | Austin |
        | p2 | Dallas |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    tables_klodaf: 'T_klodaf' = _joint_tables(required_klodaf, optional_klodaf)
    table: Any
    for table in tables_klodaf.values():
        if type(table).joint_apply is not Daf.joint_apply:              # a SqlDaf runs the whole call.
            return type(table).joint_apply(required_klodaf, func, optional_klodaf)

    # ---- The key index and the columns of each table
    kd_dodi: Dict[str, Dict[Any, int]] = {}
    cols_dols: Dict[str, List[Any]] = {}
    name: str
    daf: 'Daf'
    for name, daf in tables_klodaf.items():
        daf._rebuild_kd_if_invalidated()
        kd_dodi[name] = daf._kd
        cols_dols[name] = list(daf.hd)
    required_ls: List[str] = list(required_klodaf.keys())
    first_name: str = required_ls[0]
    others_ls: List[str] = required_ls[1:]

    # ---- The combined keys, in the first required table's order, and the function for each
    rows_by_output: Dict[str, List[Any]] = {}
    combined_keys_ls: List[Any] = []
    rows_maker: _RowsDafMaker = _RowsDafMaker(cols_dols)
    key: Any
    for key in kd_dodi[first_name]:
        if not all(key in kd_dodi[other] for other in others_ls):
            continue
        combined_keys_ls.append(key)
        rows_la: List[Any] = []
        for name, daf in tables_klodaf.items():
            irow: int | None = kd_dodi[name].get(key)
            rows_la.append(None if irow is None else daf.lol[irow])
        _collect_outputs(func(key, rows_maker.make(rows_la)), rows_by_output)

    # ---- The outputs, and a residue for each required table
    outputs_klodaf: 'T_klodaf' = KeyedList({out_name: Daf.from_lod(rows_ls) for out_name, rows_ls in rows_by_output.items()})
    residues_klodaf: 'T_klodaf' = KeyedList({name: required_klodaf[name].select_records(combined_keys_ls, inverse=True, silent_error=True)
                                             for name in required_ls})
    return outputs_klodaf, residues_klodaf


def _joint_tables(required_klodaf: Any, optional_klodaf: Any) -> Any:
    """
    The required and optional tables of `joint_apply()` together, by name, after the checks.

    There must be a required table, and each table needs a keyfield.
    """
    if not required_klodaf:
        raise ValueError("joint_apply(): give at least one required table.")
    tables_da: Dict[str, Any] = dict(required_klodaf.items())
    tables_da.update(dict((optional_klodaf or {}).items()))
    name: str
    for name, table in tables_da.items():
        if not table.keyfield:
            raise ValueError(f"joint_apply(): the table {name!r} has no keyfield. Each table is matched by its keyfield.")
    return KeyedList(tables_da)


class _RowsDafMaker:
    """
    Makes the small Daf that `joint_apply()` gives its function for each key. Internal use.

    It has a first column, `table`, the keyfield, then the columns of all the tables, in the order
    first seen. Each table's cells go to their columns. A column the table lacks is NULL.
    """

    def __init__(self, cols_dols: Dict[str, List[Any]]) -> None:
        self.names_ls: List[str] = list(cols_dols)
        self.cols_ls: List[Any] = ['table']
        name: str
        col: Any
        for name, cols_ls in cols_dols.items():
            if 'table' in cols_ls:
                raise ValueError(f"joint_apply(): the table {name!r} has a column named 'table', which holds the table names.")
            for col in cols_ls:
                if col not in self.cols_ls:
                    self.cols_ls.append(col)
        positions_di: Dict[Any, int] = {col: icol for icol, col in enumerate(self.cols_ls)}
        self.positions_lol: List[List[int]] = [[positions_di[col] for col in cols_dols[name]] for name in self.names_ls]
        self.rows_daf: 'Daf' = Daf(cols=self.cols_ls, keyfield='table')       # one Daf, refilled for each key.

    def make(self, rows_la: List[Any]) -> 'Daf':
        """
        The small Daf for one key, from each table's row as a list of its cells, or None.

        It is the same Daf for every key, refilled with set_lol(), which rebuilds the key index.
        """
        width: int = len(self.cols_ls)
        lol: List[List[Any]] = []
        itable: int
        vals: Any
        for itable, vals in enumerate(rows_la):
            if vals is None:
                continue
            row_la: List[Any] = [NULL] * width
            row_la[0] = self.names_ls[itable]
            pos: int
            val: Any
            for pos, val in zip(self.positions_lol[itable], vals):
                row_la[pos] = val
            lol.append(row_la)
        self.rows_daf.set_lol(lol)
        return self.rows_daf


def _collect_outputs(result: Any, rows_by_output: Dict[str, List[Any]]) -> None:
    """
    Add what the function of `joint_apply()` returned for one key to the rows of each output.

    The result is None, or a dict of output name to a row, a list of rows, or None.
    """
    if result is None:
        return
    if not isinstance(result, dict):
        raise TypeError(f"joint_apply(): the function returned a {type(result).__name__}. "
                        "It must return a dict of output name to rows, or None.")
    out_name: str
    rows: Any
    for out_name, rows in result.items():
        if rows is None:
            continue
        row_da: Any
        for row_da in (rows if isinstance(rows, list) else [rows]):
            if not isinstance(row_da, (dict, KeyedList)):
                raise TypeError(f"joint_apply(): the row for {out_name!r} is a {type(row_da).__name__}, not a dict.")
            rows_by_output.setdefault(out_name, []).append(row_da if isinstance(row_da, dict) else row_da.to_dict())

