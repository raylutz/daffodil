from __future__ import annotations
# daf.py
"""

# Daffodil -- Python Dataframes

The Daffodil class provides a lightweight, simple and fast alternative to provide
2-d data arrays with mixed types.

"""

r"""
    MIT License

    Copyright (c) 2026 Ray Lutz

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""


r"""
See README file: https://github.com/raylutz/daffodil/blob/main/README.md
See CHANGELOG file: https://github.com/raylutz/daffodil/blob/main/CHANGELOG.md
See ROADMAP file: https://github.com/raylutz/daffodil/blob/main/ROADMAP.md
"""

r"""
    To update the package:
        1. run all tests and demos to verify validity of the version.
            pytest will run all tests.
            demos
                python tests/daf_demo.py
                python tests/daf_benchmarks.py
        2. Update CHANGELOG.md
        3. Increment version number in pyproject.toml
        4. Create GitHub release.
        5. Remove prior release
                rm -r dist; rm -r build; rm -r *.egg-info
        6. Upgrade tools
                python.exe -m pip install --upgrade pip
                pip install --upgrade build setuptools wheel twine
                pip install --upgrade build
        7. build the release
                python -m build
        8. Check that documentation md files exist in release (only in linux)
                tar -tzf dist/daffodil-*.tar.gz | grep -E 'CHANGELOG\.md|ROADMAP\.md|LICENSE'
        8. check distribution        
                twine check dist/*
        9. Upload it
                twine upload dist/*
            You will need pypi security token    
                
     
    venv311\Scripts\activate     
"""


#VERSION  = 'v0.5.9'  <-- update in pyproject.toml !!
#VERSDATE = '2025-09-22'

# import os
# import sys
import io
import csv
import copy
import re
import json
import time
import functools
import itertools
import operator
#import numpy as np         # moved to uses of numpy
#from typing_extensions import deprecated       # can't get this to import correctly
from pathlib import Path

# no longer need the following due to using pytest
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from daffodil.lib.daf_types import T_ls, T_lola, T_di, T_loda, T_da, T_li, T_dtype_dict, \
                            T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_ds, T_lb, T_rli, \
                            T_ta, T_ts, T_lor, T_kva, T_donpa, T_npa, T_lsi, T_cs, T_ca, T_ma, T_rma, T_copybits  # noqa: F401

import daffodil.lib.daf_utils    as daf_utils
import daffodil.lib.daf_md       as md
import daffodil.lib.daf_pandas   as daf_pandas
import daffodil.lib.daf_pdf      as daf_pdf
import daffodil.lib.daf_schema   as daf_schema
import daffodil.lib.daf_ods      as daf_ods
import daffodil.lib.daf_namespace as daf_namespace
import daffodil.daf_join          as daf_join
import daffodil.daf_structure     as daf_structure
import daffodil.daf_convert       as daf_convert
import daffodil.daf_modify        as daf_modify
import daffodil.daf_select        as daf_select
import daffodil.daf_apply         as daf_apply
import daffodil.daf_dtypes        as daf_dtypes

from daffodil.keyedlist import KeyedList

import typing
from typing import List, Dict, Any, Tuple, cast, Type, Callable, Generic, TypeVar, ClassVar, Literal, overload  # noqa: F401
from collections.abc import Iterable, Collection, Sequence, Iterator, Hashable    # noqa: F401


#T_Daf = Type['Daf']
# No T_daf alias -- mypy does not treat a plain string-valued module attribute as an implicit
# forward-reference type alias (confirmed directly; needs either a real class object, only
# available after Daf is defined below, or a `TypeAlias`-annotated assignment, which needs
# Python 3.10+). Using the 'Daf' forward-reference string
# literally at each use site instead avoids the whole issue.

T_dodaf = Dict[str, 'Daf']
T_klodaf = KeyedList[Any, 'Daf']   # a KeyedList of Daf, one Daf for each group, as groupby() gives.

logs = daf_utils                # alias

NULL = ''                       # instead of var == '' use var is NULL

# define a sentinel object to express a missing item where None is a valid value.
from daffodil.lib.daf_utils import _MISSING

# Values that the single-cell fast path of __setitem__ stores as they are.
# Any other value, such as a dict, a list or a Daf, takes the general path.
_PLAIN_VALUE_TYPES = (int, float, str, bool)

def _uncountable_message(funcname: str, col: str, val: Any) -> str:
    """ The message for a value that cannot be counted. Internal, for count_values_da() and count_values_kl(). """
    return (f"{funcname}(): column {col!r} holds a {type(val).__name__}, which cannot be counted. "
            f"Use the cols parameter to select the columns to count, and leave this one out.")


def _short_list(items: T_la, limit: int = 5) -> str:
    """ A list as text, cut to its first few items. Internal, for error messages. """
    if len(items) <= limit:
        return repr(items)
    return repr(items[:limit])[:-1] + f", ... {len(items) - limit} more]"


def _keys_mismatch(keys: Iterable[Any], cols: T_la) -> str:
    """ How keys differ from cols: the missing columns, the keys that are not columns, or the first one out of order. Internal. """
    keys_la = list(keys)
    keyset, colset = set(keys_la), set(cols)
    missing = [col for col in cols if col not in keyset]
    extra = [key for key in keys_la if key not in colset]
    parts = []
    if missing:
        parts.append(f"it lacks the columns {_short_list(missing)}")
    if extra:
        parts.append(f"it has keys that are not columns, {_short_list(extra)}")
    if not parts:
        for pos, (key, col) in enumerate(zip(keys_la, cols)):
            if key != col:
                parts.append(f"its keys are in another order: position {pos} holds {key!r}, where the column is {col!r}")
                break
    return '; '.join(parts) if parts else "its keys are the columns, in order"


class DaffodilError(Exception):
    """Base exception for Daffodil."""

class KeysDisabledError(DaffodilError, LookupError):
    """Row-key lookups are unavailable because there is no keyfield and no key index (kd), or a lookup by column name is unavailable because the Daf has no column names."""

class ColumnNotFoundError(DaffodilError, KeyError, RuntimeError):
    """A column name is not in the Daf. It is a `KeyError`, and also a `RuntimeError`, which older code may catch."""

class Daf:
    """
    A table of data, stored as a list of rows.

    Daf is a small, fast, pure Python table. Use it to read, reshape and write
    2-D data without the weight of pandas. It is not meant for heavy numeric work.

    The rows are a list of lists named `lol`. A row holds only values. The column
    names are kept once, in a header dict named `hd` that maps each name to its
    position. That makes rows cheap to build, append and copy.

    A Daf may have a keyfield. This is a column, or a tuple of columns, whose
    values name the rows. The key index is built the first time a key is needed.
    Methods that add, remove or reorder rows keep it up to date, or mark it to be rebuilt.
    Changing a key value inside a row does not. See `__setitem__()`.

    A missing value is `NULL`, which is the empty string. It prints as nothing.

    A cell can hold any Python object, such as a number, text, a list, a dict, a set or
    an object of your own. Nothing limits what a cell can hold. The `dtypes` are not a
    schema that checks or forces the cells. They say how to convert columns, as when
    `apply_dtypes()` turns the text of a CSV file into numbers. A Daf built with `dtypes`
    keeps its cells as they are.

    Rows are returned as dicts or as [KeyedList][daffodil.keyedlist.KeyedList]
    objects. See [retmode][daffodil.daf.Daf.retmode] and
    [itermode][daffodil.daf.Daf.itermode].

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.columns()
        ['id', 'v']
        >>> daf.select_record(2)
        {'id': 2, 'v': 'b'}
        >>> daf.shape()
        (2, 2)
        >>> e_daf = Daf(lol=[[[1, 2], {'k': 1}]], cols=['a', 'b'], dtypes={'a': int, 'b': str})
        >>> e_daf
        <BLANKLINE>
        |   a    |    b     |
        | -----: | -------: |
        | [1, 2] | {'k': 1} |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> e_daf.iloc(0)
        {'a': [1, 2], 'b': {'k': 1}}
    """

    RETMODE_OBJ  = 'obj'
    RETMODE_VAL  = 'val'

    ITERMODE_DICT = 'dict'
    ITERMODE_KEYEDLIST = 'keyedlist'

    # The parts that copy() can give a copy of its own. Add them with |. A part without its bit is shared.
    COPY_NONE   = 0     # nothing: the copy shares every part.
    COPY_ATTRS  = 1     # the attrs dict, deep copied, and the disp_cols list.
    COPY_OUTER  = 2     # the list that holds the rows. The rows themselves are still shared.
    COPY_HD     = 4     # the header dict that maps a column name to its position.
    COPY_DTYPES = 8     # the dtypes dict.
    COPY_KD     = 16    # the key index, which is rebuilt on first use, and a keyfield that is a list.
    COPY_ROWS   = 32    # a new list for every row. Text and numbers in the cells are not copied.

    # The names that copy() accepts, as sums of the bits. The name 'deep' is not here, because it
    # is a deep copy and cannot be combined with any bit.
    _COPY_PRESETS: ClassVar[Dict[str, int]] = {
        'shallow':  COPY_ATTRS,
        'sortable': COPY_ATTRS | COPY_OUTER | COPY_HD | COPY_DTYPES | COPY_KD,
        'editable': COPY_ATTRS | COPY_OUTER | COPY_HD | COPY_DTYPES | COPY_KD | COPY_ROWS,
    }

    copy_level_default: ClassVar[str | int] = 'sortable'     # what copy() does when no level is given. See copy().

    # The rows are in memory. An add-on registered for Daf alone is refused on a class without them.
    _rows_in_memory: ClassVar[bool] = True


    def __init__(self,
            lol:        T_lola | None       = None,     # used to initialize the data array.
            hd:         T_di | None         = None,     # used to initialize the hd array. If used, then cols not needed.
            kd:         Dict[str | int, int] | None = None,  # used to initialize the kd array if no keyfield is set.
                                                        # keys can be int too (an int-valued single-column keyfield), not just str.
            cols:       T_cs | None         = None,     # Optional column names to use.
            dtypes:     T_dtype_dict | None = None,     # Optional dtype_dict describing the desired type of each column.
                                                        #   also used to define column names if provided and cols not provided.
            schema:     type | Daf | None   = None,     # Optional schema class used to define columns and defaults.
                                                        # can also be a daffodil table specifying the schema.
            keyfield:   str | int | T_ta | T_la      = '',  # A field of the columns to be used as a key.
                                                            # can be set even if columns not set yet.
                                                            # can be tuple or list of colnames, and then they are used as tuple keys.
            name:       str                 = '',       # An optional name of the Daf array.
            use_copy:   bool                = False,    # If True, make a deep copy of the lol data.
            disp_cols:  T_cs | None         = None,     # Optional list of strings to use for display, if initialized.

            retmode:    str                 = 'obj',    # default retmode
            itermode:   str                 = 'dict',   # default itermode, either 'dict' or 'keyedlist'
            attrs:      T_da | None         = None,     # arbitrary additional attributes.
        ):
        """
        Create a Daf from rows, column names and options.

        Every argument is optional, so `Daf()` makes an empty table. The usual call
        gives the rows and the column names.

        The rows are not copied. The `lol` list you pass in becomes the data of the
        Daf, so changing one changes the other. The same holds for `hd`, `kd` and
        `attrs`. Pass `use_copy=True` to deep copy `lol` first.

        The `dtypes` can only name columns. If `dtypes` has a name that is not a column, the Daf
        keeps a new dict without it, and your dict is not changed. A dict that is already right is adopted.

        If `cols` is given, it sets the column names. If it is not given, the keys of
        `dtypes` set them. Otherwise `hd` is used. A name that is empty or repeated
        is made unique, so `['a', 'a']` becomes `['a', 'a_1']`.

        With no names at all, the Daf has no columns and `lol` is only a list of rows.
        Call `set_cols()` to name them later.

        Args:
            lol: Rows, as a list of lists. Adopted, not copied.
            hd: Header dict that maps column name to position.
            kd: Key index to adopt when no keyfield is set.
            cols: Column names. These win over `hd` and `dtypes`.
            dtypes: Type for each column, used when converting, as from the text of a CSV file. It does not limit what a cell can hold. Entries for names that are not columns are left out.
            schema: A `@schemaclass` or a schema Daf that supplies columns and defaults.
            keyfield: Column, or tuple or list of columns, whose values identify rows. Each must be a column. See `set_keyfield()`.
            name: Free text name of this Daf.
            use_copy: If True, deep copy `lol` instead of adopting it.
            disp_cols: Column names to show when the Daf is printed.
            retmode: Whether a one cell result is returned as a Daf or a bare value.
            itermode: Whether iteration yields dicts or KeyedList objects.
            attrs: Free form dict of extra information. Adopted, not copied.

        A `keyfield` must be made of columns, or the Daf raises `KeyError`. A Daf with no column
        names yet can still be given one, for the columns that come later, as in
        `Daf(keyfield='id')` followed by `append()` of dicts.

        Raises:
            TypeError: `disp_cols` is not a list, a tuple or None.
            KeyError: The `keyfield` is not a column, or one of its columns is not.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf.num_rows()
            2
            >>> Daf(lol=[[1, 2]], cols=['a', 'a']).columns()
            ['a', 'a_1']
        """

        self.name           = name              # str - arbitrary name for this daffodil array
        self.schema         = schema            # Optional schema class of schema daf used to define columns and defaults.
        self.dtypes         = dtypes            # col types used for conversion from csv str values.

        if keyfield is None:                    # not specified
            # if schema is not None:
            #     keyfield = getattr(schema, "__keyfield__", "")
            # else:
            keyfield = NULL
        self.keyfield       = keyfield          # str - field of array to assign to the kd dict.

        if lol is None:
            lol = []

        # if dtypes is None:
        #     dtypes = self._get_dtypes_from_schema(schema)
            
        if cols is None:
            cols = []
        if attrs and isinstance(attrs, dict):
            self.attrs = attrs
        else:
            self.attrs = {}

        if hd and isinstance(hd, dict):
            self.hd         = hd                # header dict, use to index columns
        else:
            self.hd         = {}

        if use_copy:
            self.lol        = copy.deepcopy(lol)
        else:
            self.lol        = lol

        """
        _kd semantics:

        - If keyfield != '':
            _kd is a managed index derived from lol and keyfield.
            It may be invalidated (set to {}) and rebuilt lazily.

        - If keyfield == '':
            _kd is unmanaged (external/adopted).
            It is not maintained and may become stale after mutation.
        """
        if kd and isinstance(kd, dict):
            self._kd         = kd
        else:
            self._kd         = {}                # indicates kd not built.

        if disp_cols is None:
            self.disp_cols  = []
        elif isinstance(disp_cols, (list, tuple)):
            self.disp_cols  = list(disp_cols)
        else:
            raise TypeError("disp_cols must be a list/tuple or None")

        self.md_max_rows    = 10    # default number of rows when used with __repr__ and __str__
        self.md_max_cols    = 10    # default number of cols when used with __repr__ and __str__

        self._retmode       = retmode       # retmode can be either RETMODE_OBJ or RETMODE_VAL
        self._itermode      = itermode      # itermode can be either ITERMODE_DICT or ITERMODE_KEYEDLIST

        # Initialize iterator variables
        self._iter_index = 0

        # Initialize metadata storage.
        # These attributes can be set for any instance. They are not automatically propagated
        #   when copying or slicing. The metadata must be explicitly propagated:
        #       new_daf.attrs = daf.attrs.copy()
        # This can be used for providing information about a dataframe, such as
        #   the 'first_data_colidx' to keep track of the metadata columns vs. data columns.

        if not cols:
            if dtypes and isinstance(dtypes, dict):
                self.hd = type(self)._build_hd(dtypes.keys())
        else:
            if isinstance(cols, str):
                cols = [cols]
            # cols will be sanitized only if necessary: for repeated names, or for a blank name.
            self._cols_to_hd(cols)
            if len(cols) != len(self.hd) or NULL in self.hd:
                cols = daf_utils._sanitize_cols(cols=cols)
                self._cols_to_hd(cols)

        # if self.hd and dtypes:
            # effective_dtypes = {col: dtypes.get(col, str) for col in self.hd}

            # # setting dtypes may be better done manually if required.
            # if self.num_cols():

                # self.lol = daf_utils.apply_dtypes_to_hdlol((self.hd, self.lol), effective_dtypes, from_str=False)[1]

        self._align_with_columns()          # the dtypes can only name columns that exist.

        if self.keyfield and self.hd and not self._is_keyfield_valid():
            raise KeyError(f"Daf(): the keyfield {self.keyfield!r} is not a column. The columns are {list(self.hd)}.")

        # rebuild kd if possible, only if keyfield is defined.
        # Now using lazy kd building. Leave as {} if not manually defined.
        # self._rebuild_kd()
        self._invalidate_kd()    # use lazy kd rebuilding

        self.apply_schema()
        

    #===========================
    # basic attributes and methods

    @property
    def retmode(self) -> str:
        """
        What a selection like `daf[row, col]` returns: a Daf, or plain values.

        'obj' is the default. Every selection returns a new Daf, so you can keep calling Daf
        methods on the result.

        'val' returns plain Python values when the selection is one cell, one row or one
        column. You get a single value, or a list. A selection of several rows and several
        columns is still a Daf.

        A one-row result is the row itself, not a copy. Changing it changes this Daf.

        In 'val' mode, one row or one cell is read straight from the rows, with no Daf
        built on the way. This is when the row is an int position or a str key, and the
        column, if given, is an int position or a str name. On a 10-row table it took
        about 0.6 µs by position and 3.3 µs by key, against 12 to 29 µs through a Daf.

        Use 'val' when you read values for your own code. Use 'obj' when you want to keep
        working with Daf methods.

        The mode belongs to the Daf you set it on. A Daf made by a selection starts again at
        'obj'. A copy keeps the mode.

        Set it with `daf.retmode = 'val'`, or with the retmode argument when you create the
        Daf. Any other value raises ValueError.

        Returns:
            str: 'obj' or 'val'.

        Examples:
            >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> type(daf[0, 'x']).__name__
            'Daf'
            >>> daf.retmode = 'val'
            >>> daf[0, 'x']
            1
            >>> daf[1, :]
            [3, 4]
            >>> daf[:, 'y']
            [2, 4]
        """
        return self._retmode

    @retmode.setter
    def retmode(self, new_retmode: str) -> None:
        """
        Set the return mode.

        Args:
            new_retmode: New return mode to apply.
                RETMODE_OBJ: return a daffodil object (default)
                RETMODE_VAL: return a value from that cell in the array or a list of values.
        """
        if new_retmode in [self.RETMODE_OBJ, self.RETMODE_VAL]:
            self._retmode = new_retmode
        else:
            raise ValueError("Invalid retmode")

    @property
    def itermode(self) -> str:
        """
        What each row is when you loop over a Daf: a dict, or a KeyedList.

        Your code reads a row the same way in both modes, by column name, as in `row['qty']`.
        The difference is what the row is.

        'dict' is the default. Each row is a new dict of column name to value. The values are
        copied into it, so changing it does not change the Daf.

        'keyedlist' gives each row as a [KeyedList][daffodil.keyedlist.KeyedList]. It points
        into the table instead of copying it. See there for how it works, what changes the
        Daf, and when it is faster.

        The mode is used by `for row in daf`, and by methods that loop over the rows, such as
        `reduce()`. The methods `iter_dict()`, `iter_klist()` and `iter_list()` ignore it and
        always give their own kind of row. A Daf made by a selection starts again at 'dict'.

        Set it with `daf.itermode = 'keyedlist'`, or with the itermode argument when you
        create the Daf. Any other value raises ValueError.

        Returns:
            str: 'dict' or 'keyedlist'.

        Examples:
            >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> for row in daf:
            ...     row['y'] = 0
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.itermode = 'keyedlist'
            >>> for row in daf:
            ...     row['y'] = 0
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  0 |
            |  3 |  0 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
        """
        return self._itermode

    @itermode.setter
    def itermode(self, new_itermode: str) -> None:
        """
        Set the iterator mode.

        Args:
            new_itermode: New iterator mode ('dict' or 'keyedlist').
        """

        if new_itermode in [self.ITERMODE_DICT, self.ITERMODE_KEYEDLIST]:
            self._itermode = new_itermode
        else:
            raise ValueError("Invalid itermode")

    def __iter__(self) -> Iterator[T_da]:
        """
        Loop over the rows, as in `for row in daf`.

        Each row is a dict or a KeyedList, depending on itermode. For plain lists, use
        `iter_list()`.

        The annotation gives a dict, as in the default itermode. A type checker cannot see
        the itermode. Code that loops over KeyedLists can use `iter_klist()`, which is
        typed for them.

        In the default itermode, each row is a new dict. Assigning a new value to it does not
        change the Daf. To change values in a loop, use `iter_klist()`, whose rows write
        through, or `apply_in_place()`. A list or dict held in a cell is the same object in
        the dict and the Daf, so changing it in place does show.

        Each loop gets a new iterator, which goes through the rows once. It goes through
        the list of rows it started with: rows appended during the loop are seen, but a
        new list put in place, as by a sort, is not.

        Returns:
            Iterator: The rows, as dicts or KeyedList objects.
        """
        return cast(Iterator[T_da], self._default_iterator())     # typed for the default itermode, as above.


    def _default_iterator(self) -> Iterator[T_ma]:
        if self._itermode == self.ITERMODE_DICT:
            return self.iter_dict()
        elif self._itermode == self.ITERMODE_KEYEDLIST:
            return self.iter_klist()
        else:
            raise ValueError(f"Invalid iteration mode: {self._itermode}")


    def iter_dict(self) -> Iterator[Dict[str,Any]]:
        """
        Loop over the rows as dicts, whatever the itermode is.

        Each dict is a copy, so changing it does not change the Daf.

        Returns:
            Iterator[Dict[str, Any]]: Each row as a dict of column name to value.

        Examples:
            >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> list(daf.iter_dict())
            [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
            >>> for row in daf.iter_dict():
            ...     row['y'] = 0
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
        """
        self._check_iter_cols()
        keys = self.hd.keys()
        return (dict(zip(keys, row_la)) for row_la in self.lol)


    def iter_klist(self) -> Iterator[KeyedList]:
        """
        Loop over the rows as [KeyedList][daffodil.keyedlist.KeyedList] objects, whatever the
        itermode is.

        See [KeyedList][daffodil.keyedlist.KeyedList] for how these rows work.

        Returns:
            Iterator[KeyedList]: Each row as a KeyedList.

        Examples:
            >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> [row['x'] for row in daf.iter_klist()]
            [1, 3]
            >>> for row in daf.iter_klist():
            ...     row['y'] = 0
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  0 |
            |  3 |  0 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
        """
        self._check_iter_cols()
        return self._klist_rows()

    def _klist_rows(self) -> Iterator[KeyedList]:
        """ The rows as KeyedList objects, for `iter_klist()`. Internal. """
        # Each KeyedList is made as KeyedList(hd, row_la) would make it, without its checks.
        # A row that is not as long as the columns gets the normal constructor, and its error.
        hd = self.hd
        num_cols = len(hd)
        new_klist = KeyedList.__new__
        for row_la in self.lol:
            if len(row_la) != num_cols:
                yield KeyedList(hd, row_la)
                continue
            row_kl = new_klist(KeyedList)
            row_kl.hd = hd
            row_kl._values = row_la
            row_kl._hd_shared = True
            yield row_kl

    def _check_iter_cols(self) -> None:
        """ Raise if the rows cannot be named, for `iter_dict()` and `iter_klist()`. Internal. """
        if self.lol and not self.hd:
            raise KeysDisabledError(
                "Iterating the rows as dicts or KeyedList objects needs column names. This Daf has none. "
                "Call set_cols() to name them, or use iter_list().")


    def iter_list(self) -> Iterator[list]:
        """
        Loop over the rows as the plain lists that the Daf stores.

        Values are read by position, not by column name. These are the Daf's own row lists,
        not copies, so assigning to one changes the Daf.

        Returns:
            Iterator[list]: Each row as a list.

        Examples:
            >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  2 |
            |  3 |  4 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> [row[0] for row in daf.iter_list()]
            [1, 3]
            >>> for row in daf.iter_list():
            ...     row[1] = 9
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  9 |
            |  3 |  9 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
        """
        return iter(self.lol)


    # def __next__(self) -> Dict[str, int] | KeyedList:
        # if self._iter_index < len(self.lol):
            # if self._itermode == self.ITERMODE_DICT:
                # row_dict = dict(zip(self.hd.keys(), self.lol[self._iter_index]))
                # self._iter_index += 1
                # return row_dict
            # elif self._itermode == self.ITERMODE_KEYEDLIST:
                # row_klist = KeyedList(self.hd, self.lol[self._iter_index])
                # self._iter_index += 1
                # return row_klist
            # else:
                # raise NotImplementedError
        # else:
            # self._iter_index = 0
            # raise StopIteration

    def __bool__(self) -> bool:
        """
        Say whether the Daf holds data.

        `bool(d)` and `if d:` are true when the Daf has at least one row with a value.
        A Daf that has column names but no rows is false. Daffodil uses this as its
        test for an empty table.

        Returns:
            True if there is data.

        Examples:
            >>> bool(Daf(cols=['a']))
            False
            >>> bool(Daf(lol=[[1]], cols=['a']))
            True
        """

        return bool(self.num_cols())


    def __format__(self, format_spec: str) -> str:
        """
        Format a Daf that holds one cell.

        With no format spec this is the same as `str()`. With a spec, the Daf must
        have exactly one cell. A number is formatted with the spec. Anything else
        is converted with `str()`.

        Args:
            format_spec: A format spec such as `.2f`.

        Returns:
            The formatted text.

        Raises:
            ValueError: A spec is given and the Daf does not have exactly one cell.

        Examples:
            >>> format(Daf(lol=[[3.14159]], cols=['a']), '.2f')
            '3.14'
        """
        # Assuming the current object is a single cell when called in formatting
        # If a format_spec is provided, use it; otherwise, use __str__
        if format_spec:
            value = self.to_value()
            if isinstance(value, (int, float)):
                return format(value, format_spec)
            return str(value)
        return self.__str__()


    def __eq__(self, other: object) -> bool:
        """
        Compare two Daf instances.

        Two Daf instances are equal when their rows, their column names and their
        keyfield are equal. The order of the columns matters. The name, dtypes,
        attrs and modes are not compared. Anything that is not a Daf is not equal.

        Args:
            other: The object to compare with.

        Returns:
            True if equal.

        Examples:
            >>> a_daf = Daf(lol=[[1]], cols=['x'])
            >>> a_daf
            <BLANKLINE>
            | x  |
            | -: |
            |  1 |
            <BLANKLINE>
            %% daf rows=1; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> a_daf == Daf(lol=[[1]], cols=['x'])
            True
            >>> a_daf == Daf(lol=[[1]], cols=['x'], keyfield='x')
            False
        """
        if not isinstance(other, Daf):
            return False

        return (self.lol == other.lol and self.columns() == other.columns() and self.keyfield == other.keyfield)


    def __str__(self) -> str:
        """
        Show the Daf as a Markdown table.

        The table shows at most `md_max_rows` rows and `md_max_cols` columns, 10 by
        default. Larger tables show the first five and the last five, with `...`
        between them. A summary line with the size and keyfield follows the table.
        Use `md_daf_table_snippet()` for control over the output.

        Returns:
            The Markdown text.
        """
        return self.md_daf_table_snippet()


    def __repr__(self) -> str:
        """
        Show the Daf as a Markdown table, for the Python prompt.

        This is the same text as `str()`, with a newline in front, so the table
        starts at the left margin when it is echoed at the prompt.

        Returns:
            The Markdown text.
        """
        return "\n"+self.md_daf_table_snippet()


    def __contains__(self, key: Any) -> bool:
        """
        Test whether a key is in the keyfield column.

        Use it as `key in my_daf`. For a composite keyfield, the key is a tuple.
        An empty Daf contains nothing.

        Args:
            key: The key to look for.

        Returns:
            True if a row has this key.

        Raises:
            KeyError: The Daf has rows but no keyfield, and no key index was adopted.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> 2 in daf, 5 in daf
            (True, False)
        """
        if not self:
            return False

        self._rebuild_kd_if_invalidated()

        if not self._kd:
            raise KeyError("The dictionary 'kd' is not initialized.")

        return key in self._kd


    #===========================
    # size and shape

    # Please note that daffodil supports pre-allocated arrays with None values to speed appending.


    def num_cols(self) -> int:
        """
        Count the columns by looking at the rows.

        This does not use the column names. It returns the length of the longest of
        the first 10 rows, and 0 if there are no rows. For a table with equal row
        lengths that is the real width. Use `len(d.columns())` to count the names, and
        `is_rectangular()` to check the rows.

        Returns:
            The number of columns.

        Examples:
            >>> Daf(lol=[[1, 2, 3]], cols=['a', 'b']).num_cols()
            3
            >>> Daf(cols=['a', 'b']).num_cols()
            0
        """
        # unit tested

        if not self.lol:
            return 0
        _, result = daf_utils.min_max_cols_lol(self.lol, limit=10)
        return result


    def __len__(self) -> int:
        """
        Return the number of rows, so `len(d)` works.

        Returns:
            The number of rows.
        """
        return self.num_rows()


    def num_rows(self) -> int:
        """
        Return the number of rows.

        Returns:
            The number of rows.

        Examples:
            >>> Daf(cols=['x'], lol=[[1], [2], [3]]).num_rows()
            3
            >>> Daf().num_rows()
            0
        """
 
        if not self.lol:
            return 0

        return len(self.lol)


    def len(self) -> int:
        """
        Return the number of rows.

        This does the same as `num_rows()` and `len(d)`.

        Returns:
            The number of rows.

        Examples:
            >>> daf = Daf(cols=['x'], lol=[[1], [2], [3]])
            >>> daf
            <BLANKLINE>
            | x  |
            | -: |
            |  1 |
            |  2 |
            |  3 |
            <BLANKLINE>
            %% daf rows=3; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.len(), len(daf)
            (3, 3)
        """
        return self.num_rows()


    #==== columns and shape: functions in daf_structure.py

    is_rectangular    = daf_structure._is_rectangular
    force_rectangular = daf_structure._force_rectangular
    calc_cols         = daf_structure._calc_cols
    rename_cols       = daf_structure._rename_cols
    set_cols          = daf_structure._set_cols

    def shape(self) -> Tuple[int, int]:
        """
        Return the number of rows and columns as a tuple.

        This is a method, not a property, because the column count is worked out from
        the rows. It is the same as `(num_rows(), num_cols())`. A Daf with column
        names but no rows has shape `(0, 0)`.

        Returns:
            A tuple of the number of rows and the number of columns.

        Examples:
            >>> Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b']).shape()
            (3, 2)
        """
        # test exists in test_daf.py

        if not len(self):
            return (0, 0)

        return (self.num_rows(), self.num_cols())


    #===========================
    # copying convenience function to mimic pandas syntax.

    def copy(
            self,
            level:       str | T_copybits | None = None,  # a name, a sum of COPY_ bits, or None for copy_level_default
            name:        str  = '',                            # the name of the copy
            deep:        bool = False,                         # deprecated: use level='deep'
            for_sorting: bool = False,                         # deprecated: use level='sortable'
            inplace:     bool = False,                         # if True, copy into this Daf and return it
            ) -> 'Daf':
        """
        Make a copy of the Daf. You choose what the copy shares with the original.

        Unless `inplace=True`, the copy is a new object of the same class. What it holds is shared
        with the original unless you ask for its own. Sharing is what makes daffodil
        fast, because nothing is copied that does not have to be. Pick the cheapest
        `level` that is safe for what you do next.

        The level is one of four names, or a sum of the `COPY_` constants of `Daf`.
        Without a level, the class setting `copy_level_default` is used. It is `sortable`.
        For copies with their own row lists by default, set it on a subclass, as in
        `class SafeDaf(Daf): copy_level_default = 'editable'`. The table gives the cost for
        200,000 rows of 50 columns.

        | Level      | Parts copied                                          | Cost            |
        |------------|-------------------------------------------------------|-----------------|
        | `shallow`  | `attrs` and `disp_cols`                               | 0.00002 s       |
        | `sortable` | `shallow`, the row list, `hd`, `dtypes`, the key index | 0.002 s, 2 MB   |
        | `editable` | `sortable`, plus a new list for every row             | 0.5 s, 93 MB    |
        | `deep`     | everything that is a container                        | 4 s, 107 MB     |

        To choose your own mix, add these constants of `Daf` with `|`. A part without its bit is
        shared. A copy is always a new object, so there is no bit for that. The two bits `COPY_ROWS`
        and `COPY_OUTER` imply others: `COPY_ROWS` adds `COPY_OUTER`, and `COPY_OUTER` adds
        `COPY_KD`. They are added for you, because a row list that is copied while the key index
        is shared would leave the index wrong for one of the two. The name `deep` cannot be
        combined with any bit.

        | Constant      | Value | Parts copied                                              |
        |---------------|------:|-----------------------------------------------------------|
        | `COPY_NONE`   |     0 | nothing. Every part is shared.                            |
        | `COPY_ATTRS`  |     1 | the `attrs` dict, deep copied, and the `disp_cols` list   |
        | `COPY_OUTER`  |     2 | the list that holds the rows. The rows are still shared   |
        | `COPY_HD`     |     4 | the header dict                                           |
        | `COPY_DTYPES` |     8 | the dtypes dict                                           |
        | `COPY_KD`     |    16 | the key index, and a `keyfield` that is a list            |
        | `COPY_ROWS`   |    32 | a new list for every row                                  |

        With `shallow`, the copy shares the row list, the rows, `hd` and `dtypes` with the
        original. Reading is safe. Sorting with `sort_by_colname()` is safe too, because it
        builds a new row list. Appending a row, sorting the row list in place or dropping
        a column is seen by the original, and can leave it with a key index that is wrong.

        With `sortable`, the copy has its own row list, so you can append, extend, insert,
        remove, sort or reverse rows, rename columns, change the keyfield or dtypes, and
        use `drop_cols()`. The key index is cleared on the copy and is rebuilt on first use.
        The rows are still shared. Changing values is seen by the original, whichever way you do it.
        That makes a selection a live view for editing. Adding a column is not seen by the original.
        `insert_col()`, `insert_icol()`, `insert_idx_col()` and `assign_col()` with a new name
        copy shared rows first.

        With `editable`, each row is a new list. You can also add columns and change cells.
        The cells themselves are shared. That is safe for text and numbers, and not for a
        list or dict held in a cell.

        With `deep`, nothing that can be changed is shared. Text and numbers are not copied,
        because they cannot change. A list or dict held in a cell is copied.

        The copy has no name unless you give one with `name`, so that two tables do not
        claim to be the same one. The other settings, such as `md_max_rows`, the
        `schema` and the retmode and itermode, are taken over. A `schema` is shared.

        With `inplace=True`, the Daf copies into itself and returns itself, instead of making a
        new one. Use it on a selection, to make it a table of its own: its rows then no longer
        share anything with the Daf they came from. The default level is then `deep`. `editable`
        may be asked for, to copy only the rows. A lower level would leave the rows shared, so it
        raises ValueError. The name stays, unless `name` gives a new one.

        The arguments `deep` and `for_sorting` are deprecated. Use `level='deep'` and
        `level='sortable'` instead. They still work: `deep=True` gives the level `deep`, and
        `for_sorting=True` gives `sortable`. A higher level wins if you give both. A call like
        `copy(True)` still means `deep`. `copy(False)` uses the class setting.

        Args:
            level: A name, a sum of the `COPY_` constants, or None for the class setting `copy_level_default`.
            name: The name of the copy. It is empty if not given.
            deep: Deprecated. Use `level='deep'`. True gives the level `deep`.
            for_sorting: Deprecated. Use `level='sortable'`. True gives the level `sortable`.
            inplace: If True, copy into this Daf and return it. The level must copy the rows.

        Returns:
            The new Daf, or this one if `inplace`.

        Raises:
            ValueError: `level` is a name that is not one of the four, or `inplace` with a level that does not copy the rows.

        Examples:
            >>> daf = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'], name='orig')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  2 |  b |
            |  1 |  a |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name='orig'
            <BLANKLINE>
            >>> c = daf.copy(name='work')
            >>> c.name, daf.name
            ('work', 'orig')
            >>> _ = c.append([3, 'c'])
            >>> daf.num_rows()
            2
            >>> s = daf.copy('shallow')
            >>> _ = s.append([3, 'c'])
            >>> daf.num_rows()
            3
            >>> daf = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'])
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  2 |  b |
            |  1 |  a |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> e = daf.copy('editable')
            >>> e[0, 'v'] = 'X'
            >>> daf.iloc(0)
            {'id': 2, 'v': 'b'}
            >>> m = daf.copy(Daf.COPY_OUTER | Daf.COPY_HD)
            >>> m[0, 'v'] = 'Y'
            >>> daf.iloc(0)
            {'id': 2, 'v': 'Y'}
            >>> sel = daf.select_where(lambda row: row['id'] == 1)
            >>> sel.copy(inplace=True) is sel
            True
            >>> daf[1, 'v'] = 'Z'
            >>> sel.lol
            [[1, 'a']]
        """

        if inplace:
            if isinstance(level, bool):
                deep, level = deep or level, None
            if level is None and not deep and not for_sorting:
                level = 'deep'
            if not (deep or level == 'deep'):
                check_bits: int = self._COPY_PRESETS.get(level, -1) if isinstance(level, str) else int(level or 0)
                if check_bits != -1 and not check_bits & self.COPY_ROWS:
                    raise ValueError(f"copy(inplace=True): level {level!r} does not copy the rows. Use 'editable' or 'deep'.")
            new_daf: 'Daf' = self.copy(level=level, name=name or self.name, deep=deep, for_sorting=for_sorting)
            self.__dict__.update(new_daf.__dict__)
            return self

        if isinstance(level, bool):     # an old call, copy(True), meant deep.
            deep, level = deep or level, None

        if level is None:
            level = self.copy_level_default

        if deep or level == 'deep':
            new_daf = copy.deepcopy(self)       # fully independent copy
            new_daf.name = name
            return new_daf

        if isinstance(level, str):
            if level not in self._COPY_PRESETS:
                raise ValueError(f"copy: level must be one of {[*self._COPY_PRESETS, 'deep']}, not {level!r}")
            bits = self._COPY_PRESETS[level]
        else:
            bits = int(level)

        if for_sorting:
            bits |= self._COPY_PRESETS['sortable']

        if bits & self.COPY_ROWS:               # a copied row list must not share a key index.
            bits |= self.COPY_OUTER
        if bits & self.COPY_OUTER:
            bits |= self.COPY_KD

        # A shallow copy, as copy.copy() makes, without its 1.7 us of finding out how to copy.
        # A subclass's own __copy__ is not called.
        new_daf = type(self).__new__(type(self))    # a new object, which shares everything so far.
        new_daf.__dict__.update(self.__dict__)
        new_daf.name = name

        if bits & self.COPY_ATTRS:
            new_daf.attrs = {} if type(self.attrs) is dict and not self.attrs else copy.deepcopy(self.attrs)   # deepcopy takes 1 us even for {}.
            if isinstance(self.disp_cols, list):
                new_daf.disp_cols = list(self.disp_cols)

        if bits & self.COPY_ROWS:
            new_daf.lol = [list(row) for row in self.lol]
        elif bits & self.COPY_OUTER:
            new_daf.lol = list(self.lol)

        if bits & self.COPY_HD:
            new_daf.hd = dict(self.hd) if type(self.hd) is dict else copy.deepcopy(self.hd)

        if bits & self.COPY_DTYPES and isinstance(self.dtypes, dict):
            new_daf.dtypes = dict(self.dtypes)

        if bits & self.COPY_KD:
            if isinstance(self.keyfield, list):
                new_daf.keyfield = list(self.keyfield)
            if self.keyfield:
                new_daf._kd = {}                # a managed index is rebuilt on first use.
            elif self._kd:
                new_daf._kd = dict(self._kd)    # an adopted index cannot be rebuilt, so it is copied.

        return new_daf

    #===========================
    # column names
    @staticmethod
    def _build_hd(keys: T_cs) -> T_di:
        """
        Build header dictionary from keys (internal).

        Args:
            keys: Iterable of column names.

        Returns:
            Dict[str, int]: Mapping of column name to index.
        """
        # it is necessary to rebuild hd whenever it the array or cols is changed.
        # this is equivalent to:
        #   {col: idx for idx, col in enumerate(keys)}
        # but this is substantially faster
        # see https://github.com/raylutz/daffodil/issues/6
        # self.hd = {col: idx for idx, col in enumerate(dtypes.keys())}

        # usage here: self.hd = type(self)._build_hd(keys)

        return dict(zip(keys, range(len(keys))))


    def columns(self) -> T_ls:
        """
        Return the column names.

        The list is a new copy, so changing it does not change the Daf. Use
        `set_cols()` or `rename_cols()` to change the names.

        Returns:
            The column names, in order.

        Examples:
            >>> Daf(cols=['a', 'b']).columns()
            ['a', 'b']
        """
        # test exists in test_daf.py
        return list(self.hd.keys())


    def _cols_to_hd(self, cols: T_cs) -> None:
        """
        Rebuild internal header dictionary from column list.

        Args:
            cols: List of column names.
        """
        # see https://github.com/raylutz/daffodil/issues/6
        self.hd = type(self)._build_hd(cols)
        #self.hd = {col:idx for idx, col in enumerate(cols)}


    # @staticmethod
    # def normalize(da: T_da, defined_cols: T_ls):
        # """ if given a dict, fill out any columns that exist to match 'defined_cols'
        # """

        # if not self:
            # return

        # # from utilities import daf_utils

        # for key in defined_cols:

            # record_da = daf_utils.set_cols_da(da, defined_cols)

            # self.update_record_irow(irow, record_da)

        # return self


    def _remap_keyfield(self, old_to_new: Dict[Any, str]) -> None:
        """
        Give the keyfield the new names of its columns, in place. Clear it if a key column has no new name.

        A single name stays a single name, and a tuple or a list stays what it is. The key index
        is cleared, so that it is rebuilt on first use. Internal use.

        Args:
            old_to_new: Maps each old column name to its new name.
        """
        keyfield = self.keyfield

        if keyfield:
            if isinstance(keyfield, (str, int)):
                self.keyfield = old_to_new.get(keyfield, '')
            elif all(key_col in old_to_new for key_col in keyfield):
                self.keyfield = type(keyfield)(old_to_new[key_col] for key_col in keyfield)
            else:
                self.keyfield = ''

        self._invalidate_kd()


    #===========================
    # keyfield

    @overload
    def keys(self, *, silent_error: bool=True, astype: Literal['list']='list') -> T_la: ...
    @overload
    def keys(self, *, silent_error: bool=True, astype: Literal['view']) -> T_kva: ...
    @overload
    def keys(self, *, silent_error: bool=True, astype: str='list') -> T_la | T_kva: ...

    def keys(self, 
            *,
            silent_error: bool=True, 
            astype: str = 'list',           # 'list' | 'view'
            ) -> T_la | T_kva:

        """
        Return the row keys.

        The keys come from the key index. There is one for each distinct value of the
        keyfield, in the order of the first row that has it. With a composite keyfield
        each key is a tuple. The key index is built here if it is needed.

        A repeated key appears once. The index then points to the last row that has it,
        so a lookup of that key finds the last row. So `len(keys())` is less than
        `len(daf)` when keys repeat.

        Without a keyfield the answer is an empty list, unless `silent_error` is False.
        Then it raises an error. An index passed in as `kd` is not used by this method.

        With `astype='view'` you get a view of the key index, not a copy. It keeps the
        old keys after the Daf changes, so use it right away.

        Args:
            silent_error: If False, raise an error when there is no keyfield.
            astype: `list` for a new list, or `view` for a view of the index.

        Returns:
            The keys.

        Raises:
            KeysDisabledError: There is no keyfield and no key index, and `silent_error` is False.
            ValueError: `astype` is neither `list` nor `view`.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf.keys()
            [1, 2]
            >>> dup = Daf(cols=['id', 'q'], lol=[['a', 1], ['b', 2], ['a', 3]], keyfield='id')
            >>> dup
            <BLANKLINE>
            | id | q  |
            | -: | -: |
            |  a |  1 |
            |  b |  2 |
            |  a |  3 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> dup.keys(), dup.select_record('a')
            (['a', 'b'], {'id': 'a', 'q': 3})
        """

        # test exists in test_daf.py

        if not self.keyfield and not self._kd:
            if silent_error:
                return [] if astype == 'list' else {}.keys()  # empty KeysView
            else:
                raise self._no_keys_error('keys')

        self._rebuild_kd_if_invalidated()

        if astype == 'list':
            return list(self._kd.keys())

        elif astype == 'view':
            return self._kd.keys()   # safe: kd is now valid

        else:
            raise ValueError("astype must be 'list' or 'view'")


    def set_keyfield(
            self, 
            keyfield: str | T_ta | T_la='', 
            *,
            silent_error: bool=False,
            force_kd_rebuild: bool=False,
            ) -> 'Daf':
        """
        Choose the column, or columns, that identify the rows.

        Give a column name, or a tuple or list of names for a composite key. A
        composite key is a tuple of those values. An empty keyfield turns key lookups
        off. The key index is built later, when it is first needed.

        The Daf must have column names first. With none, nothing happens.

        A name that is not a column raises `KeyError`. Pass `silent_error=True` to store it
        anyway, as older versions did. Then key lookups find nothing. A key that is not
        unique is not checked. A lookup finds the last row that has it.

        A Daf that has column names and no rows can have a keyfield. It applies to the rows that are
        added later.

        Args:
            keyfield: Column name, or a tuple or list of names. Empty to turn off.
            silent_error: If True, store a name that is not a column, and do not raise.
            force_kd_rebuild: If True, build the key index now.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: The keyfield is not a column, and `silent_error` is not True.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.set_keyfield('id').keys()
            [1, 2]
            >>> daf.set_keyfield(['id', 'v']).keys()
            [(1, 'a'), (2, 'b')]
            >>> empty_daf = Daf(cols=['id', 'v']).set_keyfield('id')
            >>> empty_daf
            <BLANKLINE>
            <BLANKLINE>
            %% daf rows=0; cols=0; keyfield='id'; name=''
            <BLANKLINE>
            >>> empty_daf.keyfield
            'id'
            >>> empty_daf.append({'id': 7, 'v': 'x'}).keys()
            [7]
        """
        """ set the indexing keyfield to a new column
            if keyfield == '', then reset the keyfield.
            if keyfield not in columns,
                then KeyError, if not silent_error
                else set it anyway.
            Otherwise, set keyfield
            keyfield can be a tuple or list of colnames.
            If provided, then the row keys are constructed as tuples of those fields.

            set_keyfield sets the keyfield attribute but does not force rebuild of
                the _kd dictionary unless 'force_kd_rebuild' is True.

            if there are no columns declared at all, do nothing -- there is no schema yet
                to validate a keyfield against.

            Note: this checks self.hd directly, not `if not self:` -- __bool__/num_cols()
                answer "does this Daf have any ROWS" (relied on throughout calling code as
                an emptiness check), which is a different question from "are there COLUMN
                definitions to set a keyfield against." A Daf constructed as Daf(cols=[...])
                with nothing appended yet has real columns and zero rows -- `if not self:`
                read that as empty and silently skipped setting the keyfield at all, with no
                error to signal it; every append() afterward silently kept behaving as if no
                keyfield were set, since self.keyfield never actually changed from ''.
        """
        if not self.hd:
            return self

        if not keyfield:
            self.keyfield = ''
            self._kd = {}
            return self

        if not silent_error and not self._is_keyfield_valid(keyfield):
            raise KeyError(f"set_keyfield(): the keyfield {keyfield!r} is not a column. The columns are {list(self.hd)}.")
        self.keyfield = keyfield
        self._invalidate_kd()    # use lazy kd rebuilding

        if force_kd_rebuild:
            self._rebuild_kd()

        return self


    def _invalidate_kd(self) -> None:
        """
        Mark the key dictionary as invalid if keyfield is set.

        Notes:
            Enables lazy rebuilding of `_kd` after data modifications
                if keyfield != ''.
        """
        """ Mark kd as invalid for future lookups.

            Any time row deletions or additions are performed,
            kd must be rebuilt before using it. Instead of constantly
            maintaining it with every append, for example. 

            Instead, we mark the kd as invalid and then prior to
            using it if keyfield is set, then rebuild it at that time.

            This lazy kd rebuilding is an important performance and 
            usability improvement. There is no need to worry about
            setting the keyfield or disabling it during appends to
            a new Daf array for example. Can now just set the keyfield
            and not worry that time will be wasted during appends in
            a loop.

        """
        if self.keyfield:
            self._kd = {}


    def _key_icols(self) -> T_li:
        """
        The positions of the keyfield columns, or an empty list with no keyfield.

        A composite keyfield gives one position for each of its columns. A keyfield
        name that is not a column is left out. Internal use.
        """
        keyfield = self.keyfield
        if not keyfield:
            return []
        hd = self.hd
        if isinstance(keyfield, (str, int)):
            icol = hd.get(cast(str, keyfield))
            return [] if icol is None else [icol]
        return [hd[col] for col in keyfield if col in hd]


    def _invalidate_kd_if_key(self, icols: Iterable[int]) -> None:
        """
        Invalidate the kd with `_invalidate_kd()` if any of these columns is a keyfield column.

        Every method that writes into a column calls this, or `_invalidate_kd()`, so that
        a changed key is not looked up in a stale index. Internal use.
        """
        if self.keyfield:
            key_icols = self._key_icols()
            if any(icol in key_icols for icol in icols):
                self._invalidate_kd()


    def _rebuild_kd_if_invalidated(self) -> 'Daf':
        """
        Rebuild key dictionary if it has been invalidated if keyfield is valid.

        Notes:
            Internal use.
            Only rebuilds when keyfield is set and `_kd` is empty.
        """
        if self and not self._kd:
            self._rebuild_kd()
        return self


    def _rebuild_kd(self) -> None:
        """
        Rebuild key dictionary from current data if keyfield is valid.

        Notes:
            Internal.
            Required after deletions when keyfield is active.
            Checks the full validity of keyfield before applying.
        """

        if self._is_keyfield_valid():
            if isinstance(self.keyfield, (str, int)):
                # self.hd is typed Dict[str, int] (the common case), but per this class's own
                # support for numeric column names, an int keyfield is a valid key into it too.
                col_idx = self.hd[cast(str, self.keyfield)]
                self._kd = type(self)._build_kd(col_idx, self.lol)
            else:
                col_idx_list = [self.hd[cast(str, key_tup)] for key_tup in self.keyfield]
                self._kd = type(self)._build_kd(col_idx_list, self.lol)


    @staticmethod
    def _build_kd(col_idx: int | T_li, lol: T_lola) -> Dict[str | int, int]:
        """
        Build key dictionary from column index and data.

        Args:
            col_idx: Column index or indices used as key.
            lol: Data array.

        Returns:
            Dict: Mapping of key values to row indices.

        Note:
            Internal.
        """

        # build key dictionary from col_idx col of lol
        # _build_hd()'s own declared return type is Dict[str, int] (correct for its primary use
        # building a column-NAME header dict), but it's reused here to build a dict from the
        # actual VALUES in a keyfield column, which -- unlike column names -- can legitimately be
        # int too (an int-valued single-column keyfield); cast reflects that broader real return.
        if isinstance(col_idx, int):
            key_col = daf_utils.select_col_of_lol_by_col_idx(lol, col_idx)

            # see https://github.com/raylutz/daffodil/issues/6
            kd = cast(Dict[str | int, int], Daf._build_hd(key_col))
            #kd = {key: index for index, key in enumerate(key_col)}
        else:
            col_idx_list = col_idx
            kd = cast(Dict[str | int, int], Daf._build_hd(Daf(lol=lol)[:, col_idx_list].to_lota()))
        return kd


    def _get_keyval(self, data_item: T_ma) -> Any:
        """
        Extract key value from a data item.

        Returns:
            Key value corresponding to current keyfield.

        Note:
            Internal
        """
        if isinstance(self.keyfield, (str, int)):
            keyval = data_item[self.keyfield]   # type: ignore[index]  # an int keyfield is a key, not a position
        elif isinstance(self.keyfield, (tuple, list)):
            keyval = tuple((data_item[key_tup] for key_tup in self.keyfield))
        return keyval


    def _is_keyfield_valid(self, keyfield: str | int | T_ta | T_la='') -> bool:
        """
        Validate keyfield against available columns.

        Args:
            keyfield: Optional keyfield override.

        Returns:
            bool: True if keyfield is valid.

        Note:
            Internal use.
        """

        """
            evaluate self.keyfield or passed keyfield instead,
            to confirm that the key is composed of valid colnames,
            either as a single str or int, or as a tuple of str or int.
        """

        keyfield_to_test = keyfield or self.keyfield


        if isinstance(keyfield_to_test, (str, int)):
            return bool(keyfield_to_test in self.hd)
        elif isinstance(keyfield_to_test, (tuple, list)):
            return all(key_tup in self.hd for key_tup in keyfield_to_test)

        raise RuntimeError(f"keyfield '{keyfield_to_test}' invalid")



    def get_existing_keys(self, keylist: T_ls) -> T_ls:
        """
        Keep the keys that are in the Daf.

        Use it to find out which of a list of keys have a row. The order of the list
        is kept. The result is empty if the Daf has no keyfield.

        Args:
            keylist: The keys to check.

        Returns:
            The keys from the list that have a row.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf.get_existing_keys([1, 5, 2])
            [1, 2]
        """

        # unit tested

        self. _rebuild_kd_if_invalidated()

        return [key for key in keylist if key in self._kd]

    #===========================

    apply_schema = daf_schema._apply_schema
    default_record = daf_schema._default_record
    attach_schema = daf_schema._attach_schema
    check_schema = daf_schema._check_schema


    #===========================
    # dtypes

    #==== types and conversion: functions in daf_dtypes.py

    set_dtypes            = daf_dtypes._set_dtypes
    apply_dtypes          = daf_dtypes._apply_dtypes
    _apply_dtypes_to_cols = daf_dtypes._apply_dtypes_to_cols
    flatten               = daf_dtypes._flatten
    strip                 = daf_dtypes._strip

    # Unflatten has moved to .apply_dtypes()

    # def unflatten_cols(self, cols: T_ls):
        # """
            # given a daf and list of cols,
            # convert cols named to either list or dict if col exists and it appears to be
                # stringified using f"{}" functionality.

        # """

        # if not self:
            # return

        # # from utilities import daf_utils

        # self.hd, self.lol = daf_utils.unflatten_hdlol_by_cols((self.hd, self.lol), cols)

        # return self


    # def unflatten_by_dtypes(self):
        # # deprecated. Use unflatten()

        # if not self or not self.dtypes:
            # return self

        # unflatten_cols = self.calc_cols(include_types = [list, dict])

        # if not unflatten_cols:
            # return self

        # self.unflatten_cols(unflatten_cols)

        # return self


    # def flatten_cols(self, cols: T_ls):
        # # given a daf, convert given list of columns to json.

        # if not self:
            # return self

        # # from utilities import daf_utils

        # for irow, da in enumerate(self):
            # record_da = copy.deepcopy(da)
            # for col in cols:
                # if col in da:
                    # record_da[col] = daf_utils.json_encode(record_da[col])
            # self.update_record_irow(irow, record_da)

        # return self


    # def flatten(self):

        # if not self or not self.dtypes or not self.hd:
            # return self

        # flatten_cols = self.calc_cols(include_types = [list, dict])

        # if not flatten_cols:
            # return self

        # self._flatten_cols(cols=flatten_cols)

        # return self


    #==== apply, reduce and group: functions in daf_apply.py

    _safe_tofloat                         = daf_apply._safe_tofloat
    _basic_append                         = daf_apply._basic_append
    apply_formulas                        = daf_apply._apply_formulas
    _parse_formulas                       = daf_apply._parse_formulas
    cols_to_dol                           = daf_apply._cols_to_dol
    insert_dif_row                        = daf_apply._insert_dif_row
    insert_dif_rows                       = daf_apply._insert_dif_rows
    apply                                 = daf_apply._apply
    update_row                            = staticmethod(daf_apply._update_row)
    apply_in_place                        = daf_apply._apply_in_place
    manifest_apply                        = daf_apply._manifest_apply
    manifest_reduce                       = daf_apply._manifest_reduce
    manifest_process                      = daf_apply._manifest_process
    _cols_scope                           = daf_apply._cols_scope
    _reduce_scope                         = daf_apply._reduce_scope
    _new_group_dafs                       = daf_apply._new_group_dafs
    groupby                               = daf_apply._groupby
    groupby_cols                          = daf_apply._groupby_cols
    group_where                           = daf_apply._group_where
    groupby_repeated_cols                 = daf_apply._groupby_repeated_cols
    groupby_cols_reduce                   = daf_apply._groupby_cols_reduce
    _groupby_cols_sum                     = daf_apply._groupby_cols_sum
    groupby_reduce                        = daf_apply._groupby_reduce
    multi_groupby                         = daf_apply._multi_groupby
    reduce_dodaf_to_daf                   = staticmethod(daf_apply._reduce_dodaf_to_daf)
    multi_groupby_reduce                  = daf_apply._multi_groupby_reduce
    apply_colwise                         = daf_apply._apply_colwise
    daf_sum                               = daf_apply._daf_sum
    reduce                                = daf_apply._reduce
    _sum_by_col                           = daf_apply._sum_by_col
    sum_da                                = staticmethod(daf_apply._sum_da)
    _sum_sparse_keys                      = staticmethod(daf_apply._sum_sparse_keys)
    _sum_sparse_cols                      = staticmethod(daf_apply._sum_sparse_cols)
    _sparse_sum_plan                      = staticmethod(daf_apply._sparse_sum_plan)
    _sum_sparse                           = staticmethod(daf_apply._sum_sparse)
    _is_plain_sum                         = staticmethod(daf_apply._is_plain_sum)
    sum_kl                                = staticmethod(daf_apply._sum_kl)
    groupsum_daf                          = daf_apply._groupsum_daf
    multi_groupsum                        = daf_apply._multi_groupsum
    set_col2_from_col1_using_regex_select = daf_apply._set_col2_from_col1_using_regex_select
    apply_replace_regex                   = daf_apply._apply_replace_regex
    alter_daf_per_setting                 = daf_apply._alter_daf_per_setting
    alter_daf_per_alter_specs_daf         = daf_apply._alter_daf_per_alter_specs_daf
    apply_to_col                          = daf_apply._apply_to_col
    diff_da                               = staticmethod(daf_apply._diff_da)
    count_values_da                       = staticmethod(daf_apply._count_values_da)
    count_values_kl                       = staticmethod(daf_apply._count_values_kl)
    sum_dodis                             = staticmethod(daf_apply._sum_dodis)
    sum                                   = daf_apply._sum
    sum_np                                = daf_apply._sum_np

    #===========================
    # schema support: apply_schema, default_record and attach_schema are assigned above, near the keyfield methods.


    #===========================
    # initializers

    def _align_with_columns(self, columns_changed: bool=False) -> None:
        """
        Make the attributes that are tied to the columns agree with the columns, in place.

        The `dtypes`, the `keyfield`, the `schema` and `disp_cols` all describe columns, so
        they must agree with `hd`. This is the one place that decides what stays. It is called
        by the constructor and by the methods that make a Daf from another one. Internal use.

        The `dtypes` keep only the entries for columns that exist, and a dict
        that is already right is left as it is. This is all that is done when the columns
        are the ones that the caller gave, because the caller's other values then stand.

        With `columns_changed`, the columns are not the ones that the values were made for.
        Then the `keyfield` is cleared if any of its columns is missing. The `schema` and the
        `disp_cols` are dropped, because they cannot be cut to fit.

        Args:
            columns_changed: The columns are new, and the other attributes were inherited.
        """
        hd = self.hd

        if isinstance(self.dtypes, dict) and not self.dtypes.keys() <= hd.keys():
            self.dtypes = {col: typ for col, typ in self.dtypes.items() if col in hd}

        if columns_changed:
            if self.keyfield:
                key_cols = [self.keyfield] if isinstance(self.keyfield, (str, int)) else list(self.keyfield)
                if any(key_col not in hd for key_col in key_cols):
                    self.keyfield = ''
            self.schema    = None
            self.disp_cols = []


    def clone_empty(self, lol: T_lola | None = None, cols: T_ls | None = None, name:str='') -> 'Daf':
        """
        Make a new Daf with the same layout and no rows.

        This is [`copy()`][daffodil.daf.Daf.copy] with the layout bits and then no rows. The
        new Daf is an object of the same class. It has the same column names, keyfield,
        dtypes, `attrs`, `schema` and display settings. It has its own `hd`, `dtypes`,
        `attrs` and key index. The name is not carried over. Give `name` to set it.

        Give `lol` to fill the new Daf with rows. They are adopted, not copied.

        If you give `cols` that differ from the columns of the original, the columns are
        new. The `dtypes` keep the entries for columns that are still there. The keyfield
        is cleared if any of its columns is gone. The `schema` and the `disp_cols` describe
        the old columns, so they are not carried over. If `cols` are the same as the
        columns of the original, everything is carried over.

        Args:
            lol: Rows for the new Daf. If None, it has no rows.
            cols: Column names to use instead of the existing ones.
            name: The name of the new Daf.

        Returns:
            The new Daf.

        Examples:
            >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> c = daf.clone_empty()
            >>> c.columns(), c.keyfield, c.num_rows()
            (['id', 'v'], 'id', 0)
        """
        if self is None:
            return Daf()

        # as 'sortable', but without COPY_OUTER: the row list is replaced next, so copying it
        # would cost time in proportion to the rows, for nothing.
        new_daf = self.copy(self.COPY_ATTRS | self.COPY_HD | self.COPY_DTYPES | self.COPY_KD, name=name)

        new_daf.lol = lol if lol is not None else []        # adopted, as in Daf().

        if cols:
            if isinstance(cols, str):
                cols = [cols]
            if list(cols) != list(new_daf.hd):
                new_daf._cols_to_hd(cols)
                if len(cols) != len(new_daf.hd) or NULL in new_daf.hd:
                    new_daf._cols_to_hd(daf_utils._sanitize_cols(cols=cols))
                new_daf._align_with_columns(columns_changed=True)

        return new_daf


    def set_lol(self, new_lol: T_lola) -> 'Daf':
        """
        Replace the rows with a new list of lists.

        The list is adopted, not copied. The column names, the keyfield and the other
        settings stay. The key index is rebuilt when it is next needed.

        Args:
            new_lol: The new rows.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf.set_lol([[5, 'q'], [6, 'r']]).keys()
            [5, 6]
        """
        """ set the lol with the value passed, leaving other settings,
            and recalculating kd if required (i.e. if keyfield is defined).
        """

        self.lol = new_lol
        self._invalidate_kd() # use lazy kd building.

        return self



    #===========================
    # convert from / to other data or files.

    #==== import and export: functions in daf_convert.py

    from_lod         = classmethod(daf_convert._from_lod)
    from_lot         = classmethod(daf_convert._from_lot)
    to_daf           = daf_convert._to_daf
    to_lod           = daf_convert._to_lod
    from_dod         = classmethod(daf_convert._from_dod)
    to_dod           = daf_convert._to_dod
    from_cols_dol    = classmethod(daf_convert._from_cols_dol)
    to_cols_dol      = daf_convert._to_cols_dol
    to_attrib_dict   = daf_convert._to_attrib_dict
    from_lod_to_cols = classmethod(daf_convert._from_lod_to_cols)
    from_excel_buff  = classmethod(daf_convert._from_excel_buff)
    from_ods         = classmethod(daf_convert._from_ods)
    from_csv         = classmethod(daf_convert._from_csv)
    from_csv_buff    = classmethod(daf_convert._from_csv_buff)
    from_csv_file    = classmethod(daf_convert._from_csv_file)
    to_csv_file      = daf_convert._to_csv_file
    to_csv_buff      = daf_convert._to_csv_buff
    buff_to_file     = staticmethod(daf_convert._buff_to_file)
    from_directory   = classmethod(daf_convert._from_directory)
    from_numpy       = classmethod(daf_convert._from_numpy)
    to_numpy         = daf_convert._to_numpy
    to_donpa         = daf_convert._to_donpa
    from_googlesheet = classmethod(daf_convert._from_googlesheet)
    to_googlesheet   = daf_convert._to_googlesheet
    to_json          = daf_convert._to_json
    from_json        = classmethod(daf_convert._from_json)
    to_value         = daf_convert._to_value
    to_list          = daf_convert._to_list
    to_lota          = daf_convert._to_lota
    to_dict          = daf_convert._to_dict
    to_klist         = daf_convert._to_klist

    #==== Excel

    #==== CSV

    #==== Directory capture

    #==== md

    from_md = md._from_md
    dodaf_to_md = md.dodaf_to_md
    dodaf_from_md = md._dodaf_from_md

    #==== PDF to Daf

    from_pdf = daf_pdf._from_pdf
    from_pdf = from_pdf                             # fool linter.

    #==== Pandas
    #@classmethod
    from_pandas_df = daf_pandas._from_pandas_df
    from_pandas_df = from_pandas_df                 # fool linter.

    to_pandas_df = daf_pandas._to_pandas_df
    to_pandas_df = to_pandas_df                     # fool linter.


    #==== Numpy


    #==== Googlesheets

    #===========================
    # JSON compatible representation.

    # there are some limitations to JSON encoding, as it does not correctly allow
    # for non-str keys of dictionaries. The kd can be corrected if the keyfield is
    # defined and there is a dtypes that defines the type of that column, so it can
    # be corrected if it is not a str. Also, the dtypes dict cannot contain values
    # that are types.

    # Define a mapping of string representations to Python types
    TYPE_MAP = {
        'int':      int,
        'float':    float,
        'str':      str,
        'bool':     bool,
        'list':     list,
        'dict':     dict,
        # Add more types as needed, if apply_dtypes() can convert to them.
    }

    #===========================
    # convert to other format

    # #@deprecated("use Daf instead")
    # def to_hllola(self) -> T_hllola:
        # """ Create hllola from daf
            # test exists in test_daf.py

            # DEPRECATED
        # """
        # return (list(self.hd.keys()), self.lol)

    #===========================
    # append

    #==== modifying: functions in daf_modify.py

    append                      = daf_modify._append
    _append_row_la              = daf_modify._append_row_la
    concat                      = daf_modify._concat
    extend                      = daf_modify._extend
    _extend_lol                 = daf_modify._extend_lol
    record_append               = daf_modify._record_append
    _col_names                  = daf_modify._col_names
    _check_first_fast_row       = daf_modify._check_first_fast_row
    _raise_fast_length          = daf_modify._raise_fast_length
    _err_name                   = daf_modify._err_name
    remove_key                  = daf_modify._remove_key
    remove_keylist              = daf_modify._remove_keylist
    delete_records              = daf_modify._delete_records
    remove_dups                 = daf_modify._remove_dups
    drop_cols                   = daf_modify._drop_cols
    assign_record               = daf_modify._assign_record
    assign_record_irow          = daf_modify._assign_record_irow
    update_by_keylist           = daf_modify._update_by_keylist
    update_record_irow          = daf_modify._update_record_irow
    update_record_krow          = daf_modify._update_record_krow
    assign_icol                 = daf_modify._assign_icol
    _own_rows                   = daf_modify._own_rows
    _new_colname                = daf_modify._new_colname
    insert_icol                 = daf_modify._insert_icol
    insert_irow                 = daf_modify._insert_irow
    assign_col                  = daf_modify._assign_col
    insert_col                  = daf_modify._insert_col
    insert_idx_col              = daf_modify._insert_idx_col
    set_col_irows               = daf_modify._set_col_irows
    set_icol                    = daf_modify._set_icol
    set_icol_irows              = daf_modify._set_icol_irows
    find_replace                = daf_modify._find_replace
    replace_in_columns          = daf_modify._replace_in_columns
    split_daf_into_ranges       = daf_modify._split_daf_into_ranges
    split_daf_into_chunks_lodaf = daf_modify._split_daf_into_chunks_lodaf
    sort_by_colname             = daf_modify._sort_by_colname
    sort_by_colnames            = daf_modify._sort_by_colnames
    transpose                   = daf_modify._transpose
    wide_to_narrow              = daf_modify._wide_to_narrow
    narrow_to_wide              = daf_modify._narrow_to_wide

    #=========================
    # remove records per keyfield; drop cols

    #===========================
    # indexing

    def __getitem__(self,
            slice_spec:   (slice | int | str | T_li | T_ls | range | T_lor
                           | Tuple[slice | int | str | T_li | T_ls | range | T_lor | Tuple[str, str],
                                   slice | int | str | T_li | T_ls | range | T_lor | Tuple[str, str]]),
            ) -> Any:
        """
        Select rows, columns or cells, as in `my_daf[rows, cols]`.

        The selector is `[rows]` or `[rows, cols]`. With one selector, all columns are
        returned. Use `:` for all. A selector is one of these.

            integer       A position. Negative counts from the end.
            slice         `2:5`, `:3`, `::2`, as for a Python list.
            list          A list of positions, in the order given, or a list of ranges.
            range         A range of positions.
            string        A key of the keyfield for rows, or a column name for columns.
            list of str   Several keys, or several column names, in the order given.
            tuple         An inclusive range of keys, or of column names. Use None to
                          start at the first or to end at the last, as in `(None, 'r3')`.

        A tuple of two items is read as `[rows, cols]` when it stands alone. To give a
        range of row keys, add the column selector, as in `my_daf[('r1', 'r3'), :]`.

        Integers are always positions. A keyfield or column names that are integers
        cannot be used in brackets. Use `select_krows()` and `select_kcols()` instead.

        The result is a new Daf. Its rows are shared with this Daf when you select
        rows, so changing a cell in the result changes it here too. Selecting columns
        makes new row lists, so replacing a cell in the result does not change this Daf.
        A list or dict held in a cell is still the same object in both. Use `copy('deep')`
        when those must be separate too. The keyfield and dtypes carry
        over if their columns are still there. See [retmode][daffodil.daf.Daf.retmode]
        for getting a bare value or list when the result is one cell, row or column.

        A column slice works as it does for a Python list.

        Args:
            slice_spec: A row selector, or a tuple of a row selector and a column selector.

        Returns:
            A new Daf, or a value or list if `retmode` is `val`.

        Raises:
            IndexError: A row or column position is out of range.
            KeyError: A key or column name is not found.
            KeysDisabledError: Rows are selected by key and there is no keyfield.
            TypeError: A selector is None, or has a type that is not accepted.

        Examples:
            >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  1 |  a | 10 |
            |  2 |  b | 20 |
            |  3 |  c | 30 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[1]
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  2 |  b | 20 |
            <BLANKLINE>
            %% daf rows=1; cols=3; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[1:]
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  2 |  b | 20 |
            |  3 |  c | 30 |
            <BLANKLINE>
            %% daf rows=2; cols=3; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[:, 'v']
            <BLANKLINE>
            | v  |
            | -: |
            |  a |
            |  b |
            |  c |
            <BLANKLINE>
            %% daf rows=3; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> daf[[2, 0], ['n', 'id']]
            <BLANKLINE>
            | n  | id |
            | -: | -: |
            | 30 |  3 |
            | 10 |  1 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[1, 'n'].to_value()
            20
            >>> daf[(1, 2), :]
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  1 |  a | 10 |
            |  2 |  b | 20 |
            <BLANKLINE>
            %% daf rows=2; cols=3; keyfield='id'; name=''
            <BLANKLINE>
        """
        if self._retmode == self.RETMODE_VAL and self.lol:
            # One row or one cell in val mode: read it from lol and build no Daf.
            # Other selectors, and rows of no columns, take the general path below.
            val = self._getitem_val_fast(slice_spec)
            if val is not _MISSING:
                return val

        irows, icols = self._parse_selectors(slice_spec)

        if icols is None:
            ret_daf = self.select_irows(irows=irows)
        else:
            ret_daf = self.select_irows(irows=irows).select_icols(icols=icols)

        return ret_daf._adjust_return_val(self.retmode)


    def _getitem_val_fast(self, slice_spec: Any) -> Any:
        """
        Read one row or one cell for `__getitem__` in val mode, without building a Daf.

        The row is an int position or a str key. The column, if given, is an int
        position or a str name. The result is what the general path returns: the row
        itself, not a copy, or the bare value of a row of one column, or the cell.
        Errors are the same too, since the lookups are the same.

        Args:
            slice_spec: The selector given to `__getitem__`.

        Returns:
            The row or the value, or `_MISSING` when the general path must be taken.
        """
        spec_type = type(slice_spec)

        if spec_type is int or spec_type is str:
            row_la = self.lol[slice_spec if spec_type is int else self._key_to_irow(slice_spec)]
            num_cols = len(row_la)
            if num_cols > 1:
                return row_la
            if num_cols == 1:
                return row_la[0]
            return _MISSING

        if spec_type is tuple and len(slice_spec) == 2:
            row_spec, col_spec = slice_spec
            row_type = type(row_spec)
            col_type = type(col_spec)
            if (row_type is int or row_type is str) and (col_type is int or col_type is str and self.hd):
                # Look up the key, then the column, then the row, as the general path does.
                irow = row_spec if row_type is int else self._key_to_irow(row_spec)
                icol = col_spec if col_type is int else self.hd[col_spec]
                return self.lol[irow][icol]

        return _MISSING


    def _key_to_irow(self, key: str) -> int:
        """
        Find the row position of one key, for the single-cell fast paths.

        A key that is in the key index is read from it directly. Otherwise the lookup
        goes through `krows_to_irows()`, which rebuilds an index that was cleared, and
        raises the same errors as the general path.

        Args:
            key: The row key.

        Returns:
            The row position.
        """
        irow = self._kd.get(key)
        if irow is None:
            irow = cast(T_li, self.krows_to_irows(krows=key))[0]
        return irow


    def __setitem__(self,
            slice_spec:   (slice | int | str | range | T_li | T_ls | T_lb | T_lor
                           | Tuple[slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any],
                                   slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any]]),
            value: Any,
            ) -> 'Daf':
        """
        Assign values to a selection, as in `my_daf[rows, cols] = value`.

        The selector is the same as for `[]`. The Daf is changed in place.

        A single value fills every cell of the selection. A list fills the selection
        in order. A dict given for a row sets that row from the dict. The cells whose
        columns are not in the dict become NULL. See `set_irows_icols()` for what
        happens when the source and the selection differ in size.

        A `str` or `bytes` is one value, not a list of characters, so
        `my_daf[:, 'v'] = 'xyz'` sets every row of `v` to `xyz`. A list assigned to
        several whole rows is copied, so each row has its own list.

        A write does not update the key index (kd). Writing into a keyfield cell, or
        writing a whole row, which includes the key, leaves the kd as it was: a lookup
        still finds the row by its old key, and not by the new one. So don't change keys
        this way. If you must, call `set_keyfield()` with the same keyfield afterwards,
        which invalidates the kd, so it is rebuilt when it is next needed. Writing other
        columns is safe.

        A selection that shares the row has a key index of its own. Resetting one Daf does
        not reset the other. Call `set_keyfield()` on each Daf that will look up keys.

        A single name, as in `my_daf['x'] = value`, selects the row with that key. It is
        not a column, as it would be in pandas. Use `my_daf[:, 'x']` for a column, or
        `assign_col()` to add one. A name that is not a row key raises an error that
        says this.

        Args:
            slice_spec: A row selector, or a tuple of a row selector and a column selector.
            value: The value, list, dict or Daf to assign.

        Raises:
            KeyError: A row key or a column name is not found.
            KeysDisabledError: A row is selected by key, and the Daf has no keyfield.

        Examples:
            >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> daf
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  1 |  a | 10 |
            |  2 |  b | 20 |
            |  3 |  c | 30 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[1, 'v'] = 'z'
            >>> daf[:, 'n'] = [1, 2, 3]
            >>> daf
            <BLANKLINE>
            | id | v  | n  |
            | -: | -: | -: |
            |  1 |  a |  1 |
            |  2 |  z |  2 |
            |  3 |  c |  3 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield='id'; name=''
            <BLANKLINE>
            >>> daf[0] = {'v': 'q'}
            >>> daf.iloc(0)
            {'id': '', 'v': 'q', 'n': ''}

            A key changed after the index is built is found only after `set_keyfield()`:

            >>> k_daf = Daf(cols=['id', 'q'], lol=[['a', 1], ['b', 2]], keyfield='id')
            >>> k_daf
            <BLANKLINE>
            | id | q  |
            | -: | -: |
            |  a |  1 |
            |  b |  2 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name=''
            <BLANKLINE>
            >>> k_daf.select_record('a')
            {'id': 'a', 'q': 1}
            >>> k_daf[0, 'id'] = 'z'
            >>> k_daf.select_record('z')
            {}
            >>> _ = k_daf.set_keyfield(k_daf.keyfield)
            >>> k_daf.select_record('z')
            {'id': 'z', 'q': 1}
        """
        # One cell and a plain value: write it into lol directly.
        # The general path below does the same for this case, after parsing the selectors.
        if type(slice_spec) is tuple and len(slice_spec) == 2 and type(value) in _PLAIN_VALUE_TYPES:
            # Any, since mypy does not narrow on the `type(x) is int` checks below.
            row_spec, col_spec = cast(Tuple[Any, Any], slice_spec)
            row_type = type(row_spec)
            col_type = type(col_spec)
            if (row_type is int or row_type is str) and (col_type is int or col_type is str and self.hd):
                # Look up the key, then the column, then the row, as the general path does,
                # so that a bad selector raises the same error.
                irow = row_spec if row_type is int else self._key_to_irow(row_spec)
                icol = col_spec if col_type is int else self.hd[col_spec]
                self.lol[irow][icol] = value
                return self

        try:
            irows, icols = self._parse_selectors(slice_spec)
        except (KeysDisabledError, KeyError) as exc_info:
            if type(slice_spec) is not str:
                raise
            # daf['name'] = ... selects a row by key. Someone used to pandas often means a column.
            hint = (f"daf[name] selects a row. For a column use daf[:, {slice_spec!r}], "
                    f"or assign_col({slice_spec!r}, values) to add one.")
            if isinstance(exc_info, KeysDisabledError):
                raise KeysDisabledError(f"{exc_info} {hint}") from exc_info
            raise KeyError(f"{slice_spec!r} is not a row key. {hint}") from exc_info

        return self.set_irows_icols(irows=irows, icols=icols, value=value)

    def _parse_selectors(
            self,
            slice_spec: (
                slice
                | int
                | str
                | range
                | T_li
                | T_ls
                | T_lb
                | T_lor
                | Tuple[
                    slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any],
                    slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any],
                ]
            ),
        ) -> Tuple[
            int | slice | range | T_li,
            int | slice | range | T_li | None,
        ]:
        """
        Normalize and validate row/column selectors.

        Args:
            slice_spec: Indexing selector passed to __getitem__ / __setitem__.

        Returns:
            (irows, icols):
                irows: selector for rows (int, slice, range, or list[int])
                icols: selector for cols (same types) or None (means all columns)

        Raises:
            TypeError: on invalid selector usage (including None as selector)
        """

        # --- unpack ---
        if isinstance(slice_spec, tuple) and len(slice_spec) == 2:
            row_spec, col_spec = slice_spec
            col_provided = True
        else:
            row_spec = slice_spec
            col_spec = None
            col_provided = False

        # --- reject None selectors ---
        if row_spec is None:
            raise TypeError("None is not a valid row selector")

        if col_provided and col_spec is None:
            raise TypeError("None is not a valid column selector")

        # --- ROWS ---
        if row_spec == slice(None):
            irows: int | slice | range | T_li = list(range(len(self)))

        elif (
            isinstance(row_spec, str)
            or daf_utils.is_list_of_type(row_spec, str)
            or isinstance(row_spec, tuple)   # <-- allow all tuples
            or (isinstance(row_spec, list) and not row_spec)
            # or daf_utils.is_tuple_of_type_len(row_spec, str, 2)

        ):
            irows = self.krows_to_irows(krows=row_spec)

        elif (
            isinstance(row_spec, (int, slice, range))
            or daf_utils.is_list_of_type(row_spec, (int, range))
        ):
            # is_list_of_type() already confirmed (at runtime) row_spec is one of the declared
            # irows alternatives -- it's just not a TypeGuard, so mypy can't narrow on it itself.
            irows = cast('int | slice | range | T_li', row_spec)

        else:
            raise TypeError(f"Invalid row selector: {row_spec}")

        # --- COLS ---
        if not col_provided or col_spec == slice(None):
            icols: int | slice | range | T_li | None = None

        elif (
            isinstance(col_spec, str)
            or daf_utils.is_list_of_type(col_spec, str)
            or isinstance(col_spec, tuple)   # <-- allow all tuples
            or (isinstance(col_spec, list) and not col_spec)
            # or daf_utils.is_tuple_of_type_len(col_spec, str, 2)
        ):
            icols = self.kcols_to_icols(kcols=col_spec)

        elif (
            isinstance(col_spec, (int, slice, range))
            or daf_utils.is_list_of_type(col_spec, (int, range))
        ):
            icols = cast('int | slice | range | T_li', col_spec)

        else:
            raise TypeError(f"Invalid column selector: {col_spec}")

        return irows, icols


    def _adjust_return_val(self, retmode: str = '') -> Any:
        """
        Adjust return value based on retmode.

        Args:
            retmode: Return mode override.

        Returns:
            Any: Adjusted result.
        """
        """
            There is currently defined two ways to return data from __getitem__ which
            is controlled by the _retmode property setting.

            RETMODE_OBJ: return a full daf object.
            RETMODE_VAL: return just the value, when possible.

            It is helpful to just get a single column as a list, for example,
            instead of returning the entire array.

            This implementation is after the fact, and results in additional
            processing, but it is also feasible with this design to avoid
            creating the intervening daf structure and improve efficiency.
        """
        if not retmode:
            retmode = self.retmode

        if retmode == self.RETMODE_OBJ:
            # do nothing in this case.
            return self

        num_rows, num_cols = self.shape()

        if num_rows == 1 and num_cols == 1:
            # single value, just return it.
            return self.lol[0][0]

        elif num_rows == 1 and num_cols > 1:
            # single row, return as list.
            return self.lol[0]

        elif num_rows > 1 and num_cols == 1:
            # single column result as a list.
            return self.icol(0)

        return self


    #==== selecting and indexing: functions in daf_select.py

    set_irows_icols                = daf_select._set_irows_icols
    krows_to_irows                 = daf_select._krows_to_irows
    kcols_to_icols                 = daf_select._kcols_to_icols
    gkeys_to_idxs                  = staticmethod(daf_select._gkeys_to_idxs)
    select_krows                   = daf_select._select_krows
    select_kcols                   = daf_select._select_kcols
    select_irows                   = daf_select._select_irows
    select_icols                   = daf_select._select_icols
    select_record                  = daf_select._select_record
    _basic_get_record              = daf_select._basic_get_record
    select_records_daf             = daf_select._select_records_daf
    irow_la                        = daf_select._irow_la
    irow                           = daf_select._irow
    iloc                           = daf_select._iloc
    _rows_matching_any             = daf_select._rows_matching_any
    select_by_dict                 = daf_select._select_by_dict
    select_first_row_by_dict       = daf_select._select_first_row_by_dict
    select_where                   = daf_select._select_where
    select_where_idxs              = daf_select._select_where_idxs
    split_where                    = daf_select._split_where
    col                            = daf_select._col
    col_to_la                      = daf_select._col_to_la
    icol                           = daf_select._icol
    icol_to_la                     = daf_select._icol_to_la
    select_cols                    = daf_select._select_cols
    _keyfield_not_a_column_message = daf_select._keyfield_not_a_column_message

    #=============================================
    # the following methods might be absorbed into the above.
    #

    # def select_irows(self, irows_li: T_li) -> 'Daf':
        # """ Select multiple records from daf using row indexes and create new daf.

        # """

        # selected_daf = self.clone_empty()

        # for row_idx in irows_li:
            # record = self._basic_get_record(row_idx)

            # selected_daf.append(record)

        # return selected_daf


    # def select_by_dict_to_lod(self, selector_da: T_da, expectmax: int=-1, inverse: bool=False) -> T_loda:
        # """ Select rows in daf which match the fields specified in d, returning lod
            # test exists in test_daf.py

            # DEPRECATE, use select_by_dict().to_lod()
        # """

        # result_lod = self.select_by_dict(selector_da=selector_da, expectmax=expectmax, inverse=inverse).to_lod()

        # return result_lod


    # def from_selected_cols(self, cols: T_ls | None = None, exclude_cols: T_ls | None = None) -> 'Daf':
        # """ given a list of colnames, create a new daf of those cols.
            # creates as new daf

            # use my_daf[:, colnames_ls]

        # """

        # if not cols:
            # cols = []
        # if not exclude_cols:
            # exclude_cols = []

        # desired_cols = self.calc_cols(include_cols=cols, exclude_cols=exclude_cols)

        # selected_idxs = [self.hd[col] for col in desired_cols if col in self.hd]

        # new_lol = []

        # for irow, la in enumerate(self.lol):
            # la = [la[idx] for idx in range(len(la)) if idx in selected_idxs]
            # new_lol.append(la)

        # old_cols = list(self.hd.keys())
        # new_cols = [old_cols[idx] for idx in range(len(old_cols)) if idx in selected_idxs]

        # new_dtypes = {col: typ for col, typ in self.dtypes.items() if col in new_cols}

        # return Daf(lol=new_lol, cols=new_cols, dtypes=new_dtypes)


    #=========================
    #   modify records

    def _no_keys_error(self, who: str) -> 'KeysDisabledError':
        """
        Make the error for a key lookup on a Daf that has no keyfield and no key index.

        A Daf with a key index (kd) and no keyfield can look up keys, so both must be missing. Internal use.
        """
        return KeysDisabledError(f"{who}(): key lookups are disabled, as the keyfield is not set and there is no key index (kd).")


    #=========================
    # find/replace

    #=========================
    # split and grouping

    #=========================
    #   sort

    #=========================
    #   apply formulas

    #===============================
    # annotate and join

    def annotate_daf(self, other_daf: 'Daf', my_to_other_dict: T_ds) -> 'Daf':
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


    #===============================
    # apply and reduce

    # def reduce(
            # self,
            # func: Callable[[T_da, T_da], T_da | T_la],
            # by: str='row',
            # cols: T_la | None = None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> T_da | T_la:
        # """
        # Apply a function to each 'row', 'col', or 'table' and accumulate to a single T_da
        # Note: to apply a function to a portion of the table, first select the columns or rows desired
                # using a selection process.

        # Args:
            # func (Callable): The function to apply to each 'row', 'col', or 'table'.
            # It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
                # if by == 'table', function should create a new Daf instance.
            # **kwargs: Additional parameters to pass to the function.

        # Returns:
            # either a dict (by='rows' or 'table') or list (by='cols')
        # """
        # if by == 'table':
            # reduction_da = func(self, cols, **kwargs)
            # return reduction_da

        # if by == 'row':
            # reduction_da = {}
            # for row_da in self:
                # reduction_da = func(row_da, reduction_da, cols, **kwargs)
            # return reduction_da

        # elif by == 'col':
            # reduction_la = []
            # num_cols = self.num_cols()
            # for icol in range(num_cols):
                # col_la = self.icol(icol)
                # reduction_la = func(col_la, reduction_la, **kwargs)
            # return reduction_la

        # else:
            # raise NotImplementedError
        # return [] # for mypy only.


    #===================================
    # apply / reduce convenience methods

    # daf_sum2 / daf_sum3 (and the sum_da2 / sum_da3 functions they called) were investigatory
    # variants used to find out why daf_sum() (sum_da) was slow. The key finding: comparing
    # `value == ''` instead of `value is NULL` made the loop take ~10x longer -- Python implements
    # '' as a singleton (like None), so the very fast `is` comparison works and was adopted into
    # sum_da and other NULL comparisons throughout daf.py. Commented out rather than deleted, as
    # a record of that investigation/finding for future reference.
    #
    # def daf_sum2(
    #         self,
    #         by: str = 'row',
    #         cols: T_la | None = None,
    #         **kwargs: Any,
    #         ) -> T_da:
    #     # this one to investigate why daf_sum is so slow!
    #
    #     return self.reduce(func=Daf.sum_da2, by=by, cols=cols, **kwargs)
    #
    #
    # def daf_sum3(
    #         self,
    #         by: str = 'row',
    #         cols: T_la | None = None,
    #         **kwargs: Any,
    #         ) -> T_da:
    #     # this one to investigate why daf_sum is so slow!
    #
    #     return self.reduce(func=Daf.sum_da3, by=by, cols=cols, **kwargs)


    # def reduce2(
            # self,
            # func: Callable[[T_da, T_da], T_da | T_la],
            # by: str='row',
            # cols: T_la | None = None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> T_da | T_la:
        # """
        # Apply a function to each 'row', 'col', or 'table' and accumulate to a single T_da
        # Note: to apply a function to a portion of the table, first select the columns or rows desired
                # using a selection process.

        # Args:
            # func (Callable): The function to apply to each 'row', 'col', or 'table'.
            # It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
                # if by == 'table', function should create a new Daf instance.
            # **kwargs: Additional parameters to pass to the function.

        # Returns:
            # either a dict (by='rows' or 'table') or list (by='cols')
        # """
        # if by == 'table':
            # reduction_da = func(self, cols, **kwargs)
            # return reduction_da

        # if by == 'row':

            # if not self:
                # return {}

            # allcols = self.hd.keys()

            # if cols is None:
                # cols = allcols

            # reduction_da = dict.fromkeys(cols, 0)

            # for row in self:
                # for col in cols:
                    # reduction_da[col] += row[col]
            # return reduction_da


            # # for row_da in self:
                # # reduction_da = func(row_da, reduction_da, cols, **kwargs)
            # # return reduction_da

        # elif by == 'col':
            # reduction_la = []
            # num_cols = self.num_cols()
            # for icol in range(num_cols):
                # col_la = self.icol(icol)
                # reduction_la = func(col_la, reduction_la, **kwargs)
            # return reduction_la

        # else:
            # raise NotImplementedError
        # return [] # for mypy only.


    #===================================
    # Sparse sums
    #
    # A plain sum of sparse rows picks its loop once, before the first row.
    # The loop itself tests nothing.

    _SPARSE_SAMPLE_ROWS: ClassVar[int] = 100         # the rows read to judge how many keys a row has.

    # sum_da2 / sum_da3 -- investigatory variants used to find out why daf_sum() (sum_da) was
    # slow. The key finding: comparing `value == ''` instead of `value is NULL` made the loop
    # take ~10x longer -- Python implements '' as a singleton (like None), so the very fast
    # `is` comparison works and was adopted into sum_da and other NULL comparisons throughout
    # daf.py. Commented out rather than deleted, as a record of that investigation/finding.

#     @staticmethod
#     def sum_da2(row_da: T_da,                   # the current row from the daf array.
#                 accum_da: T_da,                 # an accumulated result. Must be initialized for all columns in cols.
#                 cols: Iterable,                 # defines the active columns. Can be a list, keys(), range, or slice
#                 astype: Type | None = None,    # a type like int, float, str to cast the value if it is not that type. Optional.
#                 diagnose:bool=False
#                 ) -> T_da:     # result_da
#         """ sum values in row and accum dicts per colunms provided.
#             will safely skip data that can't be summed.
#         """

#         diagnose = diagnose
#         # nan_indicator = ''

#         # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
#             # if col not in cols:                 # this check is not needed in the version below.
#                 # continue                        # 251 vs 207.

#         for col in cols:

#             value = row_da[col]
#             #if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
#             # if isinstance(value, str):        # this makes the loop take 42% longer (294)
#             # if isinstance(value, str) and value == '':  # same (294)
#             # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
#             if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
#             # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
#                 continue

#             # the try/except below is the most time efficient way to handle this while still
#             # allowing for astype and nan values. (212 ms for 1000x1000 array)
#             # Please note that the cols value is determined
#             # prior to entering the function and must contain an iterable, even if all columns
#             # are specified.

#             # writing this loop the other way around, by going through all columns and skipping those not
#             # mentioned in cols is also very inefficient.

#             # 213 for the version below, which seems like it should be fastest.
#             # but it is slightly less advantageous because initial assignment is inside the try/except block.

#             # try:
#                 # if astype:
#                     # value = row_da[col]
#                     # if astype==int and isinstance(value, (str, float, bool)):
#                         # value = int(float(value))
#                     # elif astype==float and isinstance(value, (str, int, bool)):
#                         # value = float(value)
#                     # elif astype==str and isinstance(value, (float, int, bool)):
#                         # value = str(value)
#                     # accum_da[col] += value
#                 # else:
#                     # accum_da[col] += row_da[col]
#             # except Exception:
#                 # continue

#             # this one measured at 230
#             # value = row_da[col]
#             # try:
#                 # if astype:
#                     # if astype==int and isinstance(value, (str, float, bool)):
#                         # value = int(float(value))
#                     # elif astype==float and isinstance(value, (str, int, bool)):
#                         # value = float(value)
#                     # elif astype==str and isinstance(value, (float, int, bool)):
#                         # value = str(value)

#                 # accum_da[col] += value

#             # except Exception:
#                 # continue

#             # this one measured at 209 with all cols and no astype.
#             if astype:
#                 try:
#                     if astype is int and isinstance(value, (str, float, bool)):
#                         value = int(float(value))
#                     elif astype is float and isinstance(value, (str, int, bool)):
#                         value = float(value)
#                     elif astype is str and isinstance(value, (float, int, bool)):
#                         value = str(value)

#                     accum_da[col] += value

#                 except Exception:
#                     continue

#             else:
#                 try:
#                     accum_da[col] += value

#                 except Exception:
#                         continue



#         return accum_da


#     @staticmethod
#     def sum_da3(row_da: T_da,                   # the current row from the daf array.
#                 accum_da: T_da,                 # an accumulated result. Must be initialized for all columns in cols.
#                 cols: Iterable,                 # defines the active columns. Can be a list, keys(), range, or slice
#                 astype: Type | None = None,    # a type like int, float, str to cast the value if it is not that type. Optional.
#                 diagnose:bool=False
#                 ) -> T_da:     # result_da
#         """ sum values in row and accum dicts per colunms provided.
#             will safely skip data that can't be summed.
#         """

#         diagnose = diagnose
#         #nan_indicator = ''

#         # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
#             # if col not in cols:                 # this check is not needed in the version below.
#                 # continue                        # 251 vs 207.

#         for col in cols:

#             value = row_da[col]
#             # if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
#             # if isinstance(value, str):        # this makes the loop take 42% longer (294)
#             # if isinstance(value, str) and value == '':  # same (294)
#             # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
#             # if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
#             # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
#             #     continue

#             if value is NULL:                   # about 10% faster than above.
#                 continue

#             # the try/except below is the most time efficient way to handle this while still
#             # allowing for astype and nan values. (212 ms for 1000x1000 array)
#             # Please note that the cols value is determined
#             # prior to entering the function and must contain an iterable, even if all columns
#             # are specified.

#             # writing this loop the other way around, by going through all columns and skipping those not
#             # mentioned in cols is also very inefficient.

#             # 213 for the version below, which seems like it should be fastest.
#             # but it is slightly less advantageous because initial assignment is inside the try/except block.

#             # try:
#                 # if astype:
#                     # value = row_da[col]
#                     # if astype==int and isinstance(value, (str, float, bool)):
#                         # value = int(float(value))
#                     # elif astype==float and isinstance(value, (str, int, bool)):
#                         # value = float(value)
#                     # elif astype==str and isinstance(value, (float, int, bool)):
#                         # value = str(value)
#                     # accum_da[col] += value
#                 # else:
#                     # accum_da[col] += row_da[col]
#             # except Exception:
#                 # continue

#             # this one measured at 230
#             # value = row_da[col]
#             # try:
#                 # if astype:
#                     # if astype==int and isinstance(value, (str, float, bool)):
#                         # value = int(float(value))
#                     # elif astype==float and isinstance(value, (str, int, bool)):
#                         # value = float(value)
#                     # elif astype==str and isinstance(value, (float, int, bool)):
#                         # value = str(value)

#                 # accum_da[col] += value

#             # except Exception:
#                 # continue

#             # this one measured at 209 with all cols and no astype.
#             if astype:
#                 # value = row_da[col]
#                 try:
#                     if astype is int and isinstance(value, (str, float, bool)):
#                         value = int(float(value))
#                     elif astype is float and isinstance(value, (str, int, bool)):
#                         value = float(value)
#                     elif astype is str and isinstance(value, (float, int, bool)):
#                         value = str(value)

#                     accum_da[col] += value

#                 except Exception:
#                     continue

#             else:
#                 try:
#                     accum_da[col] += value

#                 except Exception:
#                         continue



#         return accum_da


    def daf_valuecount(
            self,
            by:     str         = 'row',
            cols:   T_cs | None = None,
            ) -> T_ma:
        """
        Count how often each value occurs, in each column, using `reduce()` and `count_values_kl()`.

        Args:
            by: How the function is applied. See `reduce()`.
            cols: The columns to count. All if None.

        Returns:
            A dict that maps each column to a dict of value and count. A column that is
            not in `cols` is empty.

        Examples:
            >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> daf
            <BLANKLINE>
            | g  | x  | y  |
            | -: | -: | -: |
            |  a |  1 | 10 |
            |  b |  2 | 20 |
            |  a |  3 | 30 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.daf_valuecount(cols=['g'])['g']
            {'a': 2, 'b': 1}
        """

        # by='row' (the default, and the only mode exercised) always reduces to a single record.
        return cast(T_ma, self.reduce(func=type(self).count_values_kl, by=by, cols=cols))


    # for example:
    #   my_daf.apply_to_col(col='colname', func=lambda x: re.sub(r'^\D+', '', x))

    #====================================
    # reduction atomic functions

    # requirements for reduction functions:
    #   1. reduction will produce a single dictionary of results, for each daf chunk.
    #   2. each atomic function will be staticmethod which accepts a single row dictionary, this_da
    #       and contributes to an accum_da. The accum_da is mutated by each row call.
    #   3. the reduction atomic function must be able to deal with combining results
    #       in a daf where each record is the result of processing one chunk.
    #   4. each atomic function will also accept a cols parameter which identifies which
    #       columns are to be included in the reduction, if it is not None or []
    #       Otherwise, all columns will be processed. This columns parameter can be
    #       initialized explicitly or using my_daf.calc_cols(include_cols, exclude_cols, include_dtypes, excluded_dtypes)
    #   5. Even if columns are reduced, the result of the function will include all columns
    #       and non-specified columns will be initialized to '' empty string. This complies
    #       with design goal of always producing a result that will be useful in a report.
    #   6. Therefore, the reduction result may be appended to the daf if desired.



    # @staticmethod
    # def sum_da(row_da: T_da, accum_da: T_da, cols: T_la | None = None, astype:Type=int, diagnose:bool=False) -> T_da:     # result_da
        # """ sum values in row and accum dicts per colunms provided.
            # will safely skip data that can't be summed.
        # """
        # diagnose = diagnose
        # nan_indicator = ''

        # if cols is None:
            # cols_list = []
        # elif not isinstance(cols, list):
            # cols_list = [cols]
        # else:
            # cols_list = cols

        # if len(cols_list) > 10:
            # cols_list_or_dict = dict.fromkeys(cols_list)
        # else:
            # cols_list_or_dict = cols_list

        # for key, value in row_da.items():
            # if value == nan_indicator:
                # continue

            # if cols_list_or_dict and key not in cols_list_or_dict:
                # # accum_da[key] = ''
                # continue

            # if astype==int and isinstance(value, (str, float, bool)):
                # value = int(float(value))
            # elif astype==float and isinstance(value, (str, int, bool)):
                # value = float(value)
            # elif astype==str and isinstance(value, (float, int, bool)):
                # value = str(value)

            # try:
                # if key in accum_da:
                    # accum_da[key] += value
                # else:
                    # accum_da[key] = value
            # except Exception:
                # pass
        # return accum_da


    # def classify_by_logic_spec(self, groupcol, input_da: T_da) -> T_da: # groupname

        # for logic_spec_da in self:




    #===============================================
    # functions not following apply or reduce pattern

    def valuecounts_for_colname(
            self,
            colname:    str,
            sort:       bool=False,
            reverse:    bool=True,
            omit_nulls: bool=False,
            ) -> Dict[Any, int]:
        """
        Count how often each value occurs in one column.

        With `sort=True` the dict is ordered from the most common value to the least.
        Use `reverse=False` for the other way. A column that does not exist gives an
        empty dict. An empty cell is counted as the empty string, unless
        `omit_nulls` is True.

        Args:
            colname: The column to count.
            sort: If True, order by count.
            reverse: With `sort`, True puts the most common first.
            omit_nulls: If True, leave out the count of empty cells.

        Returns:
            A dict that maps each value to its count.

        Examples:
            >>> daf = Daf(lol=[['a'], ['b'], ['a'], ['']], cols=['g'])
            >>> daf
            <BLANKLINE>
            | g  |
            | -: |
            |  a |
            |  b |
            |  a |
            |    |
            <BLANKLINE>
            %% daf rows=4; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.valuecounts_for_colname('g', sort=True, omit_nulls=True)
            {'a': 2, 'b': 1}
        """

        valuecounts_di: T_di = {}

        if colname not in self.hd:
            return {}

        icol = self.hd[colname]

        for irow in range(len(self.lol)):
            val = self.lol[irow][icol]
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if omit_nulls:
            daf_utils.safe_del_key(valuecounts_di, '')

        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_di


    def valuecounts_for_colnames_ls(
            self,
            colnames_ls:    T_cs | None = None,
            sort:           bool=False,
            reverse:        bool=True,
            omit_nulls:     bool=False,
            ) -> Dict[str, Dict[Any, int]]:
        """
        Count how often each value occurs, in each of several columns.

        Args:
            colnames_ls: The columns to count. All if None.
            sort: If True, order each count by size.
            reverse: With `sort`, True puts the most common first.
            omit_nulls: If True, leave out the count of empty cells.

        Returns:
            A dict that maps each column to a dict of value and count.

        Examples:
            >>> daf = Daf(lol=[['a', 'x'], ['b', 'x']], cols=['g', 'h'])
            >>> daf
            <BLANKLINE>
            | g  | h  |
            | -: | -: |
            |  a |  x |
            |  b |  x |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.valuecounts_for_colnames_ls()
            {'g': {'a': 1, 'b': 1}, 'h': {'x': 2}}
        """

        if not colnames_ls:
            colnames_ls = self.columns()

        colnames_ls = cast(T_ls, colnames_ls)

        valuecounts_dodi: T_dodi = {}

        for colname in colnames_ls:
            valuecounts_dodi[colname] = \
                self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)

        return valuecounts_dodi


    def valuecounts_for_colname_selectedby_colname(
            self,
            colname: str,
            selectedby_colname: str,
            selectedby_colvalue: str,
            sort: bool = False,
            reverse: bool = True,
            ) -> Dict[Any, int]:
        """
        Count the values of a column, in the rows where another column has a value.

        Args:
            colname: The column to count.
            selectedby_colname: The column to test.
            selectedby_colvalue: Only rows where that column equals this are counted.
            sort: If True, order the counts by size.
            reverse: With `sort`, True puts the most common first.

        Returns:
            A dict that maps each value to its count. It is empty if a column does not exist.

        Examples:
            >>> daf = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
            >>> daf
            <BLANKLINE>
            | g  | h  |
            | -: | -: |
            |  a |  x |
            |  b |  x |
            |  a |  y |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.valuecounts_for_colname_selectedby_colname('g', 'h', 'x')
            {'a': 1, 'b': 1}
        """


        valuecounts_di: T_di = {}

        if colname not in self.hd or selectedby_colname not in self.hd:
            return {}

        icol = self.hd[colname]
        selectedby_colidx = self.hd[selectedby_colname]

        for irow in range(len(self.lol)):
            val = self.lol[irow][selectedby_colidx]
            if val != selectedby_colvalue:
                continue
            val = self.lol[irow][icol]
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_di


    def valuecounts_for_colnames_ls_selectedby_colname(
            self,
            colnames_ls: T_cs | None = None,
            selectedby_colname: str = '',
            selectedby_colvalue: str = '',
            sort: bool = False,
            reverse: bool = True,
            ) -> Dict[str, Dict[Any, int]]:

        """
        Count the values of several columns, in the rows where another column has a value.

        Args:
            colnames_ls: The columns to count. All if None.
            selectedby_colname: The column to test.
            selectedby_colvalue: Only rows where that column equals this are counted.
            sort: If True, order the counts by size.
            reverse: With `sort`, True puts the most common first.

        Returns:
            A dict that maps each column to a dict of value and count.

        Examples:
            >>> daf = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
            >>> daf
            <BLANKLINE>
            | g  | k  | n  |
            | -: | -: | -: |
            |  a |  x |  1 |
            |  a |  y |  2 |
            |  b |  x |  3 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['k'], 'g', 'a')
            {'k': {'x': 1, 'y': 1}}
            >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['g', 'k'], 'g', 'a')
            {'g': {'a': 2}, 'k': {'x': 1, 'y': 1}}
        """


        if not colnames_ls:
            colnames_ls = self.columns()

        colnames_ls = cast(T_ls, colnames_ls)

        valuecounts_dodi: T_dodi = {}

        for colname in colnames_ls:
            valuecounts_dodi[colname] = \
                self.valuecounts_for_colname_selectedby_colname(
                        colname,
                        selectedby_colname,
                        selectedby_colvalue,
                        sort=sort,
                        reverse=reverse,
                        )

        return valuecounts_dodi


    def valuecounts_for_colname1_groupedby_colname2(
            self,
            colname1: str,
            groupedby_colname2: str,
            sort: bool = False,
            reverse: bool = True,
            ) -> Dict[Any, Dict[Any, int]]:
        """
        Count the values of one column, for each value of another.

        The data is read once. Use it to see whether two columns relate one to one: each
        group should then hold a single value.

        Args:
            colname1: The column whose values are counted.
            groupedby_colname2: The column whose values form the groups.
            sort: If True, order each count by size.
            reverse: With `sort`, True puts the most common first.

        Returns:
            A dict that maps each value of the second column to a dict of value and count.

        Examples:
            >>> daf = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
            >>> daf
            <BLANKLINE>
            | g  | h  |
            | -: | -: |
            |  a |  x |
            |  b |  x |
            |  a |  y |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.valuecounts_for_colname1_groupedby_colname2('g', 'h')
            {'x': {'a': 1, 'b': 1}, 'y': {'a': 1}}
        """


        valuecounts_dodi: T_dodi = {}

        if colname1 not in self.hd or groupedby_colname2 not in self.hd:
            return {}

        icol1 = self.hd[colname1]
        groupedby_col2idx = self.hd[groupedby_colname2]

        for row in self.lol:
            groupval = row[groupedby_col2idx]
            val = row[icol1]
            if groupval not in valuecounts_dodi:
                valuecounts_dodi[groupval] = {}
            valuecounts_di: T_di = valuecounts_dodi[groupval]
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if sort:
            # for group, valuecounts_di in valuecounts_dodi.items():
                # valuecounts_dodi[group] = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

            valuecounts_dodi = {group: dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))
                                    for group, valuecounts_di in valuecounts_dodi.items()}


        return valuecounts_dodi



    def gen_stats_daf(self, col_def_lot: T_lota) -> T_doda:

        """
        Work out statistics for columns, given a profile for each.

        `col_def_lot` has a tuple for each column of interest. The tuple is the column
        name, a type, a format, and a profile. The profile is one of `index`,
        `attrib`, `file_paths`, `scalar` or `localidx`, and chooses what is measured.
        An index looks for repeats, an attribute counts the values, and a scalar gives
        the minimum, maximum, mean and standard deviation. The type and format are not
        used.

        Args:
            col_def_lot: A list of tuples of column name, type, format and profile.

        Returns:
            A dict that maps each column name to its statistics, as a dict.

        Raises:
            NotImplementedError: A profile is not one of the five.

        Examples:
            >>> daf = Daf(lol=[[1], [3]], cols=['n'])
            >>> daf
            <BLANKLINE>
            | n  |
            | -: |
            |  1 |
            |  3 |
            <BLANKLINE>
            %% daf rows=2; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.gen_stats_daf([('n', int, '', 'scalar')])['n']['mean']
            2
        """

        info_dod = {}

        # from utilities import daf_utils

        for col_def_ta in col_def_lot:
            col_name, col_dtype, col_format, col_profile = col_def_ta

            col_data_la = self.col(col_name)                                       # perflint-reviewed (loop-invariant-statement)

            info_dod[col_name] = daf_utils.list_stats(col_data_la, profile=col_profile)     # perflint-reviewed (loop-invariant-statement)

        return info_dod


    def derive_join_translator(
        self,
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


    @classmethod
    def derive_join_translator_daf(
        cls,
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


    update_with_history = daf_join._update_with_history

    def join(
        self,
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


    @staticmethod
    def join_records(
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

    #===============================
    # wide to narrow and narrow to wide conversion

    #===============================
    # reporting

    def md_daf_table_snippet(
            self,
            ) -> str:
        """
        Make a short Markdown table of the Daf, with a summary line.

        This is what `str()` shows. It keeps at most `md_max_rows` rows and
        `md_max_cols` columns, 10 by default. Longer text is shortened to 80
        characters.

        If the Daf has no rows, it is empty, and it shows no header. The summary line says
        `rows=0; cols=0`, because the number of columns is counted from the rows.
        `columns()` still gives the names.

        Returns:
            The Markdown text.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> print(daf.md_daf_table_snippet(), end='')
            | x  | y  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''

            An empty Daf has no rows and no columns, so only the summary line shows. This is so even if it has column names:

            >>> Daf(cols=['x', 'y'])
            <BLANKLINE>
            <BLANKLINE>
            %% daf rows=0; cols=0; keyfield=''; name=''
            <BLANKLINE>
        """

        return self.to_md(
                max_rows        = self.md_max_rows,
                max_cols        = self.md_max_cols,
                shorten_text    = True,
                max_text_len    = 80,
                smart_fmt       = False,
                include_summary = True,
                )

    # the following alias is defind at the bottom of this file.
    # Daf.md_daf_table = Daf.to_md

    def to_md(
            self,
            max_rows:       int     = 0,         # limit number of rows by keeping leading and trailing rows and omitting middle rows.
            max_cols:       int     = 0,         # limit number of cols by keeping leading and trailing cols and omitting middle cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = False,     # include a one-line summary after the table, describing shape, keyfield, name
            disp_cols:      T_cs | None = None,    # use these column names instead of those defined in daf.
            header:         T_cs | None = None,    # use this header instead.
            ) -> str:
                
        """
        Make a Markdown table of the Daf.

        Use it to put a table into a report or a log, as in `daf[0:50].to_md()`. An f-string of a
        Daf gives `str()`, which keeps only 10 rows and 10 columns and adds a summary line. `to_md()`
        writes every row and column, and leaves the summary line out, unless asked. So it gives the
        rows selected, as most code wants. A SqlDaf gives the same text, reading only those rows.

        Without limits the whole table is written. With `max_rows` or `max_cols`, the
        first and last are kept, and the middle is replaced by `...`. Text longer
        than `max_text_len` is shortened by cutting out its middle. `just` has one
        character for each column: `<` left, `^` center, `>` right. The default is
        right. With no column names, `A`, `B` and so on are used. With `include_summary`,
        the Markdown can be read back with `from_md()`, as text.

        A ragged Daf, with rows of different lengths, is shown with its short rows padded
        with blank cells. The Daf itself is not changed. With `include_summary`, the summary
        line ends with `ragged rows detected`. `from_md()` reads such a table back as
        rectangular.

        Use `max_rows` and `max_cols` together, or neither. With only `max_cols`, a row of
        `...` is added under the header by mistake.

        Args:
            max_rows: The most rows to show. 0 for all.
            max_cols: The most columns to show. 0 for all.
            just: The justification of each column.
            shorten_text: If True, shorten text that is longer than `max_text_len`.
            max_text_len: The longest text to show in full.
            smart_fmt: If True, show numbers with fewer decimal places.
            include_summary: If True, add a line with the size, keyfield and name.
            disp_cols: Column names to show instead of the real ones.
            header: A header to use instead.

        Returns:
            The Markdown text.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
            >>> print(daf.to_md())
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
        """

        daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)

        header_exists = bool(self.hd)

        if header_exists and daf_lol:
            # daf_to_lol_summary embeds colnames as the first row; pull it back out and
            # pass it explicitly via `header` rather than relying on the data's first row
            # (daffodil never treats the first row of a lol as an implicit header).
            embedded_header = daf_lol[0]
            data_lol = daf_lol[1:]
        elif daf_lol:
            # Daffodil arrays may have no header at all (a bare lol). A Markdown table needs a
            # header row and a separator row, and Daf.from_md() requires them, so write
            # spreadsheet-style names (A, B, C, ...) in the header. The names are for the
            # text only. The Daf is not given names.
            embedded_header = daf_utils._generate_spreadsheet_column_names_list(len(daf_lol[0]))
            data_lol = daf_lol
            header_exists = True
        else:
            embedded_header = None
            data_lol = daf_lol

        final_header = header if header is not None else embedded_header

        mdstr = md.md_lol_table(
            data_lol,
            header              = final_header,
            just                = just or ('>' * len(data_lol[0])) if data_lol else just,
            omit_header         = not header_exists,
            shorten_text        = shorten_text,
            max_text_len        = max_text_len,
            smart_fmt           = smart_fmt,

            )
        if include_summary:
            parts = []
            parts.append(f"rows={self.num_rows()}")
            parts.append(f"cols={self.num_cols()}")
            parts.append(f"keyfield='{self.keyfield or ''}'")
            parts.append(f"name='{self.name or ''}'")

            schema = self.attrs.get('schema')
            if schema:
                parts.append(f"schema='{schema}'")
            if not self.is_rectangular():
                parts.append("ragged rows detected")

            mdstr += "\n%% daf " + "; ".join(parts) + "\n"

        # if include_summary:
        #     # (compatible with tuple keys.)
        #     mdstr += f"\n\\[{self.num_rows():,} rows x {self.num_cols():,} cols; keyfield='{self.keyfield}'; {len(self._kd):,} keys ] ({self.name or type(self).__name__})\n"
        return mdstr


    def to_md_cols(
            self,
            max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
            max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = False,     # include a one-line summary after the table.
            disp_cols:      T_ls | None = None, # use these column names instead of those defined in daf.
            ) -> str:
        """
        Make a Markdown table in which each row of the Daf is a column.

        There is no header row and no separator row. The first column holds the column names of the Daf.
        Use it for a Daf with few rows and many columns. Some Markdown renderers, such as Python-Markdown,
        show this text as plain text, because they need a header and a separator row to make a table.

        Args:
            max_rows: The most rows to show. 0 for all.
            max_cols: The most columns to show. 0 for all.
            just: The justification of each column.
            shorten_text: If True, shorten text that is longer than `max_text_len`.
            max_text_len: The longest text to show in full.
            smart_fmt: If True, show numbers with fewer decimal places.
            include_summary: Not used.
            disp_cols: Column names to show instead of the real ones.

        Returns:
            The Markdown text.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> print(daf.to_md_cols(), end='')
            | x  | 1  | 2  |
            | y  | a  | b  |
        """

        daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)

        #header_exists = bool(self.hd)
        #breakpoint() #temp

        mdstr = md.md_cols_lol_table(
                cols_lol        = daf_lol,
                header          = None,
                just            = just,
                omit_header     = True,
                shorten_text    = shorten_text,
                max_text_len    = max_text_len,
                smart_fmt       = smart_fmt,
                )
        # if include_summary:
            # mdstr += f"\n\[{len(self.lol)} rows x {len(self.hd)} cols; keyfield={self.keyfield}; {len(self._kd)} keys ] ({type(self).__name__})\n"
        return mdstr

    def daf_to_lol_summary(
            self, 
            max_rows: int=10, 
            max_cols: int=10, 
            disp_cols: T_cs | None =None,
            ) -> T_lola:

        # from utilities import daf_utils

        # first build a basic summary by adding colnames, if they exist.
        """
        Make a list of lists for display, with the column names first.

        If there are more rows or columns than the limits, the first and last are
        kept and the middle is replaced by `...`. The limit is the number of data rows
        kept, and an odd limit keeps the extra row at the start. A limit of 0 means no
        limit, so with only `max_cols` all the rows are kept. The rows are not copied.

        Args:
            max_rows: The most rows to keep. 0 for no limit.
            max_cols: The most columns to keep. 0 for no limit.
            disp_cols: Column names to use instead of the real ones.

        Returns:
            The rows, with a header row first if there are column names.

        Examples:
            >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> daf
            <BLANKLINE>
            | x  | y  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.daf_to_lol_summary()
            [['x', 'y'], [1, 'a'], [2, 'b']]
            >>> big = Daf(lol=[[i, i] for i in range(20)], cols=['x', 'y'])
            >>> big
            <BLANKLINE>
            |  x  |  y  |
            | --: | --: |
            |   0 |   0 |
            |   1 |   1 |
            |   2 |   2 |
            |   3 |   3 |
            |   4 |   4 |
            | ... | ... |
            |  15 |  15 |
            |  16 |  16 |
            |  17 |  17 |
            |  18 |  18 |
            |  19 |  19 |
            <BLANKLINE>
            %% daf rows=20; cols=2; keyfield=''; name=''
            <BLANKLINE>
            >>> big.daf_to_lol_summary(max_rows=4)
            [['x', 'y'], [0, 0], [1, 1], ['...', '...'], [18, 18], [19, 19]]
        """

        if disp_cols:
            if isinstance(disp_cols, list):
                colnames_ls = disp_cols
            else:
                colnames_ls = list(disp_cols)
        else:
            colnames_ls = list(self.hd.keys())

        colnames_ls = cast(list, colnames_ls)    

        result_lol = self.lol
        if colnames_ls:
            result_lol = [colnames_ls] + result_lol

        # no limits, return summary. A ragged Daf is shown with its short rows padded, as copies.
        if not max_rows and not max_cols:
            return daf_utils.pad_ragged_lol(result_lol)

        num_rows    = self.num_rows()
        num_cols    = self.num_cols()

        if not max_rows or num_rows <= max_rows:
            # Get all the rows, but potentially limit columns
            result_lol = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)

        else:
            # Get the first and last portion of rows. An odd limit gives the extra row to the first part.
            # The last part is cut from the front, because a slice of [-0:] would be every row.

            first_lol   = self.lol[:(max_rows + 1)//2]
            last_lol    = self.lol[num_rows - max_rows//2:]
            divider_lol = [['...'] * num_cols]

            result_lol  = [colnames_ls] + first_lol + divider_lol + last_lol
            result_lol  = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)

        return result_lol

    @staticmethod
    def dict_to_md(da: T_da, cols: T_ls | None = None, just: str='<<') -> str:
        """
        Show a dict as a two column Markdown table, for looking at it.

        This is a static method. Use it as `print(Daf.dict_to_md(my_da))`. The keys are
        in the first column and the values in the second.

        Args:
            da: The dict.
            cols: The two column names. Default `key` and `value`.
            just: The justification of the two columns.

        Returns:
            The Markdown text.

        Examples:
            >>> print(Daf.dict_to_md({'a': 1, 'b': 'two'}))
            | key | value |
            | :-- | :---- |
            | a   | 1     |
            | b   | two   |
            <BLANKLINE>
        """
        if not cols:
            cols = ['key', 'value']

        return Daf.from_lod_to_cols([da], cols=cols).to_md(just=just)


    #=========================================
    #  Reporting Convenience Methods

    def value_counts_daf(self,
            colname: str,                       # column name to include in the value_counts table
            sort: bool=False,                   # sort values in the category
            reverse: bool=True,                 # reverse the sort
            include_total: bool=False,          #
            omit_nulls: bool=False,             # set to true if '' should be omitted.
            ) -> 'Daf':
        """
        Make a Daf that lists each value of a column and its count.

        The columns are the name of the column, and `counts`. There is no keyfield.
        With `include_total` a last row holds the total.

        Args:
            colname: The column to count.
            sort: If True, order by count.
            reverse: With `sort`, True puts the most common first.
            include_total: If True, add a row with the total.
            omit_nulls: If True, leave out the count of empty cells.

        Returns:
            The new Daf.

        Examples:
            >>> daf = Daf(lol=[['a'], ['b'], ['a']], cols=['g'])
            >>> daf
            <BLANKLINE>
            | g  |
            | -: |
            |  a |
            |  b |
            |  a |
            <BLANKLINE>
            %% daf rows=3; cols=1; keyfield=''; name=''
            <BLANKLINE>
            >>> daf.value_counts_daf('g', sort=True)
            <BLANKLINE>
            | g  | counts |
            | -: | -----: |
            |  a |      2 |
            |  b |      1 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            <BLANKLINE>
        """

        value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)

        if omit_nulls:
            daf_utils.safe_del_key(value_counts_di, '')

        value_counts_daf = Daf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

        if include_total:
            value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf.col('counts'))})

        return value_counts_daf


DafIterRtype = TypeVar('DafIterRtype', Dict[str, Any], KeyedList, list)


class DafIterator(Generic[DafIterRtype]):
    """ Generic in the row shape it produces (dict/KeyedList/list), so iter_dict()/iter_klist()/
        iter_list() below can each promise the narrower Iterator[X] they actually construct,
        instead of every DafIterator instance claiming the full T_ma | list regardless of
        which rtype it was actually built with. """
    def __init__(self, this_daf: Daf, rtype: Type[DafIterRtype] = dict):  # type: ignore[assignment]
        # every real caller (iter_dict/iter_klist/iter_list below) passes rtype explicitly;
        # mypy just can't verify a single concrete default satisfies a constrained TypeVar.
        if rtype is not list and this_daf.lol and not this_daf.hd:
            raise KeysDisabledError(
                "Iterating the rows as dicts or KeyedList objects needs column names. This Daf has none. "
                "Call set_cols() to name them, or use iter_list().")
        self.this_daf = this_daf
        self.rtype: Type[DafIterRtype] = rtype
        self._index = 0
        # the hd of the Daf, shared by every KeyedList row of this loop.
        self._hd: Dict[str, int] = this_daf.hd

    def __iter__(self) -> 'DafIterator[DafIterRtype]':
        return self

    def __next__(self) -> DafIterRtype:
        if self._index < len(self.this_daf.lol):
            row = self.this_daf.lol[self._index]
            self._index += 1

            if self.rtype is dict:
                return cast(DafIterRtype, dict(zip(self.this_daf.hd.keys(), row)))

            elif self.rtype == KeyedList:
                return cast(DafIterRtype, KeyedList(self._hd, row))

            elif self.rtype is list:
                return cast(DafIterRtype, row)

            else:
                raise NotImplementedError(f"Unknown return type: {self.rtype}")
        else:
            self._index = 0
            raise StopIteration



class _IndirectRowView:
    """
    Row adapter that exposes keys from an indirect column as if they were normal columns.

    Lookup order:
        1. explicit row fields
        2. keys inside the indirect column dict
        3. return '' (Daf null sentinel) if not found

    Allows concise rows (values stored in an indirect column) to behave
    like explicit rows for operations such as select_where() and split_where().

    No copying occurs; the wrapper holds references to the original row
    (dict or KeyedList) and the indirect dict.
    """

    def __init__(self, row: T_ma, indirect_col: str | None = None):
        self.row = row
        self.indirect = (
            daf_utils.get_indirect_da(row, indirect_col)
            if indirect_col else None
        )

    def __getitem__(self, key: Hashable) -> Any:

        if key in self.row:
            return self.row[key]     # type: ignore[index]  # keys are strings

        if self.indirect and key in self.indirect:
            return self.indirect[key]

        return ''

    def get(self, key: Hashable, default: Any='') -> Any:

        if key in self.row:
            return self.row.get(key, default)     # type: ignore[call-overload]  # keys are strings

        if self.indirect:
            return self.indirect.get(key, default)     # type: ignore[call-overload]  # keys are strings

        return default


    def keys(self) -> Iterable[Hashable]:
        if not self.indirect:
            return self.row.keys()

        return list(self.row.keys()) + [
            k for k in self.indirect.keys() if k not in self.row
            ]


    def values(self) -> Iterable[Any]:
        if not self.indirect:
            return self.row.values()

        return (self[key] for key in self.keys())


    def items(self) -> Iterable[Tuple[Hashable, Any]]:
        if not self.indirect:
            return self.row.items()

        return ((key, self[key]) for key in self.keys())


def register_daf_namespace(name: str) -> Callable[[type], type]:
    """
    Add a class of methods to every Daf, under one name, from a package outside daffodil.

    This is how an add-on gives Daf new methods without a change to daffodil. Decorate a class
    whose constructor takes the Daf. Then `daf.<name>` is an object of that class for that Daf, and
    its methods work on it. Every Daf has it, a selection or a copy included, since it is on the class.
    pandas, xarray and polars add methods the same way.

    An add-on is active once its package is imported. Nothing loads add-ons by itself.

    A name already in use raises ValueError: a method or attribute of Daf, or another add-on.
    Registering the same class again, as when its module is reloaded, is allowed.

    An add-on for Daf is refused on a SqlDaf, whose rows are in SQLite, not in memory.
    `register_sqldaf_namespace()` in daffodil.sqldaf adds a SqlDaf version, under the same name or another.

    Type checkers and editors do not see a name added at run time. For checked code, an add-on can
    be used through its class, as `HistNamespace(daf).update_with_history(new_daf)`.

    Args:
        name: The attribute name, such as `hist`. It must be a public identifier.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: The name is not a public identifier, or it is already in use on Daf.

    Examples:
        >>> @register_daf_namespace('totals')
        ... class Totals:
        ...     def __init__(self, daf):
        ...         self._daf = daf
        ...     def of(self, col):
        ...         return sum(self._daf.col(col))
        >>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |   3 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.totals.of('qty'), daf[0:1].totals.of('qty')
        (13, 3)
        >>> del Daf.totals
    """
    instance_attrs: T_ls = list(vars(Daf()))
    return daf_namespace.register_namespace(name, Daf, instance_attrs)


# Profiling mode: with the environment variable DAFFODIL_PROFILE=1, count how Daf is used.
# See daffodil/lib/daf_profile.py. Nothing is wrapped unless it is set.
import daffodil.lib.daf_profile as daf_profile      # noqa: E402
daf_profile.start_from_env(Daf)
