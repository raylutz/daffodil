# daf_join.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf that combine two tables.

Each function here takes the Daf as its first argument and is assigned in the Daf class body,
as the methods in daffodil.lib.daf_schema are. The docs show it under its Daf name.
"""

from typing import Any, Tuple

from daffodil.lib.daf_types import T_la, T_ls

from typing import TYPE_CHECKING
if TYPE_CHECKING:       # for the annotations only. A real import would be circular.
    from daffodil.daf import Daf, T_klodaf


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
    from daffodil.daf import Daf

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
