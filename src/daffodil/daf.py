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
#import numpy as np         # moved to uses of numpy
#from typing_extensions import deprecated       # can't get this to import correctly
from pathlib import Path

# no longer need the following due to using pytest
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from daffodil.lib.daf_types import T_ls, T_lola, T_di, T_loda, T_da, T_li, T_dtype_dict, \
                            T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_ds, T_lb, T_rli, \
                            T_ta, T_lor, T_kva, T_donpa, T_npa, T_lsi, T_cs, T_ca, T_ma  # noqa: F401

import daffodil.lib.daf_utils    as daf_utils
import daffodil.lib.daf_md       as md
import daffodil.lib.daf_pandas   as daf_pandas
import daffodil.lib.daf_pdf      as daf_pdf
import daffodil.lib.daf_schema   as daf_schema

from daffodil.keyedlist import KeyedList
from daffodil.keyedlist import KeyedIndex

import typing
from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable, Generic, TypeVar  # noqa: F401
from collections.abc import Iterable, Collection, Sequence, Iterator, Hashable    # noqa: F401


#T_Daf = Type['Daf']
# No T_daf alias -- mypy does not treat a plain string-valued module attribute as an implicit
# forward-reference type alias (confirmed directly; needs either a real class object, only
# available after Daf is defined below, or a `TypeAlias`-annotated assignment, which needs
# Python 3.10+ and this project targets >=3.9). Using the 'Daf' forward-reference string
# literally at each use site instead avoids the whole issue.

T_dodaf = Dict[str, 'Daf']

logs = daf_utils                # alias
_COPY_LEVELS: Dict[str, int] = {'shallow': 0, 'sortable': 1, 'editable': 2, 'deep': 3}   # for copy().

NULL = ''                       # instead of var == '' use var is NULL

# global
_use_keyedindex_for_hd = False

# define a sentinel object to express a missing item where None is a valid value.
from daffodil.lib.daf_utils import _MISSING

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
    It is rebuilt after the rows change.

    A missing value is `NULL`, which is the empty string. It prints as nothing.

    Rows are returned as dicts or as [KeyedList][daffodil.keyedlist.KeyedList]
    objects. See [retmode][daffodil.daf.Daf.retmode] and
    [itermode][daffodil.daf.Daf.itermode].

    Examples:
        >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
        >>> d.columns()
        ['id', 'v']
        >>> d.select_record(2)
        {'id': 2, 'v': 'b'}
        >>> d.shape()
        (2, 2)
    """

    RETMODE_OBJ  = 'obj'
    RETMODE_VAL  = 'val'

    ITERMODE_DICT = 'dict'
    ITERMODE_KEYEDLIST = 'keyedlist'


    def __init__(self,
            lol:        T_lola|None         = None,     # used to initialize the data array.
            hd:         T_di|None           = None,     # used to initialize the hd array. If used, then cols not needed.
            kd:         Dict[Union[str, int], int]|None = None,  # used to initialize the kd array if no keyfield is set.
                                                        # keys can be int too (an int-valued single-column keyfield), not just str.
            cols:       T_cs|None           = None,     # Optional column names to use.
            dtypes:     T_dtype_dict|None   = None,     # Optional dtype_dict describing the desired type of each column.
                                                        #   also used to define column names if provided and cols not provided.
            schema:     type | Daf |None    = None,     # Optional schema class used to define columns and defaults.
                                                        # can also be a daffodil table specifying the schema.
            keyfield:   Union[str, int, T_ta, T_la]  = '',  # A field of the columns to be used as a key.
                                                            # can be set even if columns not set yet.
                                                            # can be tuple or list of colnames, and then they are used as tuple keys.
            name:       str                 = '',       # An optional name of the Daf array.
            use_copy:   bool                = False,    # If True, make a deep copy of the lol data.
            disp_cols:  T_cs | None         = None,     # Optional list of strings to use for display, if initialized.

            retmode:    str                 = 'obj',    # default retmode
            itermode:   str                 = 'dict',   # default itermode, either 'dict' or 'keyedlist'
            attrs:      Optional[T_da]      = None,     # arbitrary additional attributes.
        ):
        """
        Create a Daf from rows, column names and options.

        Every argument is optional, so `Daf()` makes an empty table. The usual call
        gives the rows and the column names.

        The rows are not copied. The `lol` list you pass in becomes the data of the
        Daf, so changing one changes the other. The same holds for `hd`, `kd` and
        `attrs`. Pass `use_copy=True` to deep copy `lol` first.

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
            dtypes: Type for each column, used when converting from strings.
            schema: A `@schemaclass` or a schema Daf that supplies columns and defaults.
            keyfield: Column, or tuple or list of columns, whose values identify rows. A name that is not a column is
                stored without an error, and then key lookups find nothing. See `set_keyfield()`.
            name: Free text name of this Daf.
            use_copy: If True, deep copy `lol` instead of adopting it.
            disp_cols: Column names to show when the Daf is printed.
            retmode: Whether a one cell result is returned as a Daf or a bare value.
            itermode: Whether iteration yields dicts or KeyedList objects.
            attrs: Free form dict of extra information. Adopted, not copied.

        Raises:
            TypeError: `disp_cols` is not a list, a tuple or None.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> d.num_rows()
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
            >>> for row in daf:
            ...     row['y'] = 0
            >>> daf
            | x | y |
            | -: | -: |
            | 1 | 2 |
            | 3 | 4 |
            %% daf rows=2; cols=2; keyfield=''; name=''
            >>> daf.itermode = 'keyedlist'
            >>> for row in daf:
            ...     row['y'] = 0
            >>> daf
            | x | y |
            | -: | -: |
            | 1 | 0 |
            | 3 | 0 |
            %% daf rows=2; cols=2; keyfield=''; name=''
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

    def __iter__(self) -> Iterator[Union[Dict[str, Any], KeyedList]]:
        """
        Loop over the rows, as in `for row in daf`.

        Each row is a dict or a KeyedList, depending on itermode. For plain lists, use
        `iter_list()`.

        Returns:
            Iterator: The rows, as dicts or KeyedList objects.
        """
        return self._default_iterator()


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
            >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> list(d.iter_dict())
            [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
            >>> for row in d.iter_dict():
            ...     row['y'] = 0
            >>> d
            | x | y |
            | -: | -: |
            | 1 | 2 |
            | 3 | 4 |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        return DafIterator(self, dict)


    def iter_klist(self) -> Iterator[KeyedList]:
        """
        Loop over the rows as [KeyedList][daffodil.keyedlist.KeyedList] objects, whatever the
        itermode is.

        See [KeyedList][daffodil.keyedlist.KeyedList] for how these rows work.

        Returns:
            Iterator[KeyedList]: Each row as a KeyedList.

        Examples:
            >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> [row['x'] for row in d.iter_klist()]
            [1, 3]
            >>> for row in d.iter_klist():
            ...     row['y'] = 0
            >>> d
            | x | y |
            | -: | -: |
            | 1 | 0 |
            | 3 | 0 |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        return DafIterator(self, KeyedList)


    def iter_list(self) -> Iterator[list]:
        """
        Loop over the rows as the plain lists that the Daf stores.

        Values are read by position, not by column name. These are the Daf's own row lists,
        not copies, so assigning to one changes the Daf.

        Returns:
            Iterator[list]: Each row as a list.

        Examples:
            >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
            >>> [row[0] for row in d.iter_list()]
            [1, 3]
            >>> for row in d.iter_list():
            ...     row[1] = 9
            >>> d
            | x | y |
            | -: | -: |
            | 1 | 9 |
            | 3 | 9 |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        return DafIterator(self, list)


    # def __next__(self) -> Union[Dict[str, int], KeyedList]:
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
            >>> a = Daf(lol=[[1]], cols=['x'])
            >>> a == Daf(lol=[[1]], cols=['x'])
            True
            >>> a == Daf(lol=[[1]], cols=['x'], keyfield='x')
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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> 2 in d, 5 in d
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
            >>> d = Daf(cols=['x'], lol=[[1], [2], [3]])
            >>> d.len(), len(d)
            (3, 3)
        """
        return self.num_rows()


    def is_rectangular(self) -> bool:
        """
        Check whether every row has the same length.

        With column names, every row must be as long as the number of names. Without
        names, every row must be as long as the first row. An empty Daf is rectangular.

        This looks at every row. Use it when you do not trust the source of the data.
        `num_cols()` only samples the first rows.

        Returns:
            True if all rows have the expected length.

        Examples:
            >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).is_rectangular()
            False
        """
        if not self.lol:
            return True

        target_len = len(self.hd) if self.hd else len(self.lol[0])

        return all(len(row) == target_len for row in self.lol)


    def force_rectangular(self) -> 'Daf':
        """
        Pad short rows with empty strings, in place.

        Use this when a source leaves off trailing empty cells. An xlsx file read by
        `xlsx_to_csv()` does that. The target width is the number of column names. With
        no names, it is the length of the longest row.

        Rows that are too long are not cut. That would hide damage, such as an
        unquoted comma that split a value. They raise an error instead.

        Returns:
            This Daf, which has been changed.

        Raises:
            ValueError: A row is longer than the number of columns. Nothing is changed.

        Examples:
            >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).force_rectangular()
            | a | b |
            | -: | -: |
            | 1 | 2 |
            | 3 |   |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        if not self.lol:
            return self

        target_len = len(self.hd) if self.hd else max((len(row) for row in self.lol), default=0)

        long_rows = [(i, len(row)) for i, row in enumerate(self.lol) if len(row) > target_len]
        if long_rows:
            raise ValueError(
                    f"force_rectangular: {len(long_rows)} row(s) exceed the target width of "
                    f"{target_len} -- likely real data corruption (e.g. an unquoted comma "
                    f"splitting a value), not the xlsx trailing-cell omission this pads for. "
                    f"(row index, actual length): {long_rows[:10]}")

        for row in self.lol:
            if len(row) < target_len:
                row.extend([''] * (target_len - len(row)))

        return self


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
            level:       Union[str, bool] = 'shallow',   # 'shallow' | 'sortable' | 'editable' | 'deep'
            deep:        bool = False,          # same as level='deep'.
            for_sorting: bool = False,          # same as level='sortable'.
            ) -> Daf:
        """
        Make a copy of the Daf, sharing as little or as much as you choose.

        A copy costs more the less it shares. Pick the cheapest `level` that is safe for
        what you do next. The table gives the cost for 200,000 rows of 50 columns.

        | Level      | New in the copy                                   | Cost            |
        |------------|---------------------------------------------------|-----------------|
        | `shallow`  | only `attrs`                                      | 0.0001 s        |
        | `sortable` | the row list, `hd`, `dtypes`, and the key index   | 0.002 s, 2 MB   |
        | `editable` | `sortable`, plus a new list for every row         | 0.8 s, 93 MB    |
        | `deep`     | everything that is a container                    | 2.4 s, 107 MB   |

        With `shallow`, the copy shares the row list, the rows, `hd` and `dtypes` with the
        original. Reading is safe. Sorting with `sort_by_colname()` is safe too, because it
        builds a new row list. Appending a row, sorting the row list in place or dropping
        a column reaches the original, and can leave it with a key index that is wrong.

        With `sortable`, the copy has its own row list, so you can append, extend, insert,
        remove, sort or reverse rows, rename columns, change the keyfield or dtypes, and
        use `drop_cols()`. The key index is cleared on the copy and is rebuilt on first use.
        The rows are still shared. Changing values reaches the original, whichever way you do it. That makes a
        selection a live view for editing. Adding a column does not reach the original.
        `insert_col()`, `insert_icol()`, `insert_idx_col()` and `assign_col()` with a new name
        copy shared rows first.

        With `editable`, each row is a new list. You can also add columns and change cells.
        The cells themselves are shared. That is safe for text and numbers, and not for a
        list or dict held in a cell.

        With `deep`, nothing that can be changed is shared. Text and numbers are not copied,
        because they cannot change. A list or dict held in a cell is copied.

        The older arguments `deep=True` and `for_sorting=True` still work. They give the
        level `deep` and `sortable`. A higher level wins if you give both. A call like
        `copy(True)` still means `deep`.

        Args:
            level: How much to copy. One of `shallow`, `sortable`, `editable` or `deep`.
            deep: Same as level `deep`.
            for_sorting: Same as level `sortable`.

        Returns:
            The new Daf.

        Raises:
            ValueError: `level` is not one of the four names.

        Examples:
            >>> d = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'])
            >>> _ = d.copy().append([3, 'c'])
            >>> d.num_rows()
            3
            >>> d = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'])
            >>> s = d.copy('sortable')
            >>> _ = s.append([3, 'c'])
            >>> d.num_rows()
            2
            >>> e = d.copy('editable')
            >>> e[0, 'v'] = 'X'
            >>> d.iloc(0)
            {'id': 2, 'v': 'b'}
        """

        if isinstance(level, bool):     # an old call, copy(True), meant deep.
            deep, level = deep or level, 'shallow'

        if level not in _COPY_LEVELS:
            raise ValueError(f"copy: level must be one of {list(_COPY_LEVELS)}, not {level!r}")

        rank = _COPY_LEVELS[level]
        if for_sorting:
            rank = max(rank, _COPY_LEVELS['sortable'])
        if deep:
            rank = _COPY_LEVELS['deep']

        if rank == _COPY_LEVELS['deep']:
            return copy.deepcopy(self)  # Fully independent copy

        new_instance = copy.copy(self)  # Shallow copy

        new_instance.attrs = copy.deepcopy(self.attrs)  # Ensure metadata is copied but not linked

        if rank >= _COPY_LEVELS['sortable']:
            new_instance.lol = list(self.lol)
            new_instance.hd = dict(self.hd) if type(self.hd) is dict else copy.deepcopy(self.hd)
            if isinstance(self.dtypes, dict):
                new_instance.dtypes = dict(self.dtypes)

            new_instance._invalidate_kd()

        if rank >= _COPY_LEVELS['editable']:
            new_instance.lol = [list(row) for row in self.lol]

        return new_instance

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


    @staticmethod
    def isin(listlike1: Union[T_da, T_la], listlike2: Union[T_da, T_la]) -> T_lb:
        """
        Make a list of True and False, one per item of the first collection. Deprecated.

        This is deprecated. Use a list comprehension, such as `[item in names for item in items]`,
        which does the same, or `select_where()` with a function. It was an early attempt to match
        the `isin()` of pandas. Daffodil does not use it, and it will be removed. `select_where()`
        shows how to test values against a set in one pass.

        An item is True if it is found in the second collection. This is a static
        method, so call it as `Daf.isin(a, b)`. It is handy for picking or leaving out
        columns by name.

        Do not use the list of bools as a column selector, as in `my_daf[:, mask]`. A list of bools
        is read as a list of positions, where False is 0 and True is 1, so the wrong columns are
        chosen, and some more than once. To keep or leave out columns by name, make a list of the
        names first, as in the example.

        Args:
            listlike1: The items to test, in order.
            listlike2: The collection to look in.

        Returns:
            A list of bools, as long as `listlike1`.

        Examples:
            >>> Daf.isin(['a', 'b', 'c'], ['b'])
            [False, True, False]
            >>> [item in ['b'] for item in ['a', 'b', 'c']]
            [False, True, False]
            >>> d = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
            >>> omit = Daf.isin(d.columns(), ['b'])
            >>> d[:, [name for name, drop in zip(d.columns(), omit) if not drop]].columns()
            ['a', 'c']
        """

        """ creates a boolean mask (list of bools) for each item in list1 which is in list2

            this can be used particularly for omitting columns, like:

                my_daf[:, ~my_daf.columns().isin(colnames_to_omit_list)]

            can also be used to select columns

                my_daf[:, my_daf.columns().isin(colnames_to_keep_list)]

            but this is easier done by providing the list directly

                my_daf[:, colnames_to_keep_list]

            as long as the colnames are not numbers, because then the indexing will
            assume they are column numbers. So this can be a workaround if the colnames
            are numbers and using them directly can be confusing, but mainly it is used
            to exclude columns. Can be also used for rows, but it is best to use
            direct selection if possible.

            This will directly select rows with the keys selected.

                my_daf[rowkeys_to_keep_list]

            But can also select with a boolean mask, but it is not as efficient.

                my_daf[my_daf.keys().isin(rowkeys_to_keep_list)]

            However, that may be good if you just want to exclude rows

                my_daf[~my_daf.keys().isin(rowkeys_to_keep_list)]

        """
        searchable2: Union[Dict[Any, Any], T_la]
        if isinstance(listlike2, list) and len(listlike1) > 10 and len(listlike2) > 30:
            searchable2 = dict.fromkeys(listlike2)
        else:
            searchable2 = listlike2

        bool_mask_lb = [col in searchable2 for col in listlike1]

        return bool_mask_lb


    def calc_cols(self,
            include_cols: Optional[Iterable]=None,
            exclude_cols: Optional[Iterable]=None,
            include_types: Optional[List[Type]]=None,
            exclude_types: Optional[List[Type]]=None,
           ) -> Iterable:
        """
        Work out a list of column names from rules.

        Use it to choose the columns for `apply` or `reduce`. The rules are applied
        in this order: include by name, exclude by name, include by type and exclude
        by type. The result keeps the order of the Daf. A single name may be given
        as a string.

        With a group by operation, leave the group by columns out of the list.

        Args:
            include_cols: Keep only these column names.
            exclude_cols: Drop these column names.
            include_types: Keep only columns whose dtype is in this list.
            exclude_types: Drop columns whose dtype is in this list.

        Returns:
            The selected column names.

        Raises:
            RuntimeError: A type rule is given and no dtypes are set.

        Examples:
            >>> d = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
            >>> d.calc_cols(exclude_cols='b')
            ['a', 'c']
        """

        """ this method helps to calculate the columns to be specified for a apply or reduce operation.
            Can use any combination of listing columns to be included, or excluded by name,
                or included by type.
            If using a groupby function, the cols spec should not include the groupby column(s)
        """


        # start with all cols.
        selected_cols: List[str] = list(self.hd.keys())  # change to columns()
        if not selected_cols:
            return []

        if include_cols:
            if isinstance(include_cols, str):
                include_cols = [include_cols]
            # if len(include_cols) > 10:
                # include_cols_dict = dict.fromkeys(include_cols)
                # selected_cols = [col for col in selected_cols if col in include_cols_dict]
            selected_cols = [col for col in selected_cols if col in include_cols]

        if exclude_cols:
            if isinstance(exclude_cols, str):
                exclude_cols = [exclude_cols]
            # if len(exclude_cols) > 10:
                # exclude_cols_dict = dict.fromkeys(exclude_cols)
                # selected_cols = [col for col in selected_cols if col not in exclude_cols_dict]
            selected_cols = [col for col in selected_cols if col not in exclude_cols]

        if include_types:
            if not self.dtypes:
                raise RuntimeError("calc_cols(): include_types/exclude_types requires dtypes to be set")

            if not isinstance(include_types, list):
                include_types = [include_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) in include_types]

        if exclude_types:
            if not self.dtypes:
                raise RuntimeError("calc_cols(): include_types/exclude_types requires dtypes to be set")

            if not isinstance(exclude_types, list):
                exclude_types = [exclude_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) not in exclude_types]

        return selected_cols

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


    def rename_cols(self, from_to_dict: T_ds) -> 'Daf':
        """
        Rename columns in place.

        Names that are not in the mapping stay as they are. Names in the mapping that
        are not in the Daf are ignored. The dtypes are renamed too.

        The keyfield is cleared, even if its column was not renamed. Call
        `set_keyfield()` afterwards to turn key lookups back on.

        Args:
            from_to_dict: Maps old names to new names.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 2]], cols=['a', 'b'])
            >>> d.rename_cols({'b': 'c'}).columns()
            ['a', 'c']
        """

        # unit tests exist

        self.hd         = {from_to_dict.get(col, col):idx for idx, col in enumerate(self.hd.keys())}
        if self.dtypes:
            self.dtypes = {from_to_dict.get(col, col):typ for col, typ in self.dtypes.items()}
        self._invalidate_kd()
        self.keyfield   = ''

        return self


    def set_cols(self, new_cols: Optional[T_ls]=None, sanitize_cols: bool=True, unnamed_prefix: str='col') -> 'Daf':
        """
        Set the column names, in place.

        The new names are given by position, so the first name goes to the first
        column. Without a list, the names are A, B, C and so on, like a spreadsheet.

        With `sanitize_cols` on, a repeated name gets a suffix, so `['a', 'a']` becomes
        `['a', 'a_1']`. An empty name becomes the prefix and its position, like `col2`. The
        prefix is short, because these names are printed. Names that come from parsing a header, in the
        constructor and in `from_md()`, use `Unnamed` instead, which says that there is no name.
        The dtypes are renamed by position as well.

        The keyfield is cleared. Call `set_keyfield()` afterwards to turn key lookups
        back on.

        Args:
            new_cols: The names, in order. If None, spreadsheet names are made.
            sanitize_cols: If True, make the names valid and unique.
            unnamed_prefix: The start of a name made for an empty one.

        Returns:
            This Daf, which has been changed.

        Raises:
            AttributeError: The number of names is not the number of columns.

        Examples:
            >>> Daf(lol=[[1, 2, 3]]).set_cols().columns()
            ['A', 'B', 'C']
            >>> Daf(lol=[[1, 2, 3]]).set_cols(['a', 'a', '']).columns()
            ['a', 'a_1', 'col2']
        """

        num_cols = self.num_cols() or len(self.hd)

        if new_cols is None:
            new_cols = daf_utils._generate_spreadsheet_column_names_list(num_cols)

        elif sanitize_cols:
            new_cols = daf_utils._sanitize_cols(new_cols, unnamed_prefix=unnamed_prefix)

        if num_cols and len(new_cols) != num_cols:
            raise AttributeError("Length of new_cols not the same as existing cols")

        # Renaming columns always resets the keyfield to '' rather than attempting to remap it
        # to the new name. Field renaming is rare, and the caller is expected to explicitly
        # re-set the correct keyfield afterward (via set_keyfield()) rather than relying on
        # automatic repair -- which would otherwise need to track which old column name
        # corresponded to which new one, an error-prone correspondence to maintain silently.
        self.keyfield = ''

        self._invalidate_kd()

        # set new cols to the hd
        self._cols_to_hd(new_cols)

        # convert dtypes dict to use the new names.
        if self.dtypes:
            self.dtypes = dict(zip(new_cols, self.dtypes.values()))

        return self


    #===========================
    # keyfield

    def keys(self, 
            *,
            silent_error: bool=True, 
            astype: str = 'list',           # 'list' | 'view'
            ) -> Union[T_la, T_kva]:

        """
        Return the row keys.

        The keys are the values of the keyfield column, in row order. With a composite
        keyfield each key is a tuple. The key index is built here if it is needed.

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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> d.keys()
            [1, 2]
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
            keyfield: Union[str, T_ta, T_la]='', 
            *,
            silent_error: bool=True,
            force_kd_rebuild: bool=False,
            ) -> 'Daf':
        """
        Choose the column, or columns, that identify the rows.

        Give a column name, or a tuple or list of names for a composite key. A
        composite key is a tuple of those values. An empty keyfield turns key lookups
        off. The key index is built later, when it is first needed.

        The Daf must have column names first. With none, nothing happens.

        A name that is not a column is stored anyway unless `silent_error` is False.
        Then a `KeyError` is raised. A key that is not unique is not checked. A lookup
        finds the last row that has it.

        A Daf that has column names and no rows can have a keyfield. It applies to the rows that are
        added later.

        Args:
            keyfield: Column name, or a tuple or list of names. Empty to turn off.
            silent_error: If False, raise an error for a name that is not a column.
            force_kd_rebuild: If True, build the key index now.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: The keyfield is not a column and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
            >>> d.set_keyfield('id').keys()
            [1, 2]
            >>> d.set_keyfield(['id', 'v']).keys()
            [(1, 'a'), (2, 'b')]
            >>> e = Daf(cols=['id', 'v']).set_keyfield('id')
            >>> e.keyfield
            'id'
            >>> e.append({'id': 7, 'v': 'x'}).keys()
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

        if not self._is_keyfield_valid(keyfield):
            if not silent_error:
                raise KeyError
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
    def _build_kd(col_idx: Union[int, T_li], lol: T_lola) -> Dict[Union[str, int], int]:
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
            kd = cast(Dict[Union[str, int], int], Daf._build_hd(key_col))
            #kd = {key: index for index, key in enumerate(key_col)}
        else:
            col_idx_list = col_idx
            kd = cast(Dict[Union[str, int], int], Daf._build_hd(Daf(lol=lol)[:, col_idx_list].to_lota()))
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


    def _is_keyfield_valid(self, keyfield: Union[str, int, T_ta, T_la]='') -> bool:
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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> d.get_existing_keys([1, 5, 2])
            [1, 2]
        """

        # unit tested

        self. _rebuild_kd_if_invalidated()

        return [key for key in keylist if key in self._kd]

    #===========================

    apply_schema = daf_schema._apply_schema
    default_record = daf_schema._default_record
    attach_schema = daf_schema._attach_schema


    #===========================
    # dtypes

    def set_dtypes(self,
            default_type: Type = str,
            typ_to_cols_dict: Optional[Dict[Type, T_ls]] = None,
            ) -> 'Daf':

        """
        Set the dtype of every column, from a default and a few exceptions.

        Use it when most columns have one type and a few differ. The result is stored
        in `dtypes`. This does not convert any data. Call `apply_dtypes()` for that.

        If most columns differ, assign a dict to `dtypes` yourself.

        Args:
            default_type: The type of every column that is not listed.
            typ_to_cols_dict: Maps a type to the names of the columns of that type.

        Returns:
            This Daf, which has been changed.

        Raises:
            NotImplementedError: The Daf has no column names.

        Examples:
            >>> d = Daf(lol=[['1', '2', 'x']], cols=['a', 'b', 'c'])
            >>> d.set_dtypes(str, {int: ['a', 'b']}).dtypes
            {'a': <class 'int'>, 'b': <class 'int'>, 'c': <class 'str'>}
        """

        """ set dtypes from default and dol where the key of the dict is the type,
            and the list is the column names of that type.
            Useful if most columns are the same type and there are only a few exceptions.

            Otherwise, simply set my_daf.dtypes = dtypes dict, and then
            apply_dtypes() when reading files.
            writing files will generally automatically flatten.

            This function does NOT apply the types.

            @@TODO May want to provide a simpler function that sets a type to a set of columns.
        """

        if not self.hd:
            raise NotImplementedError ("self.hd must be defined to use .set_dtypes()")

        dtypes: T_dtype_dict = {}

        col_to_typ_dict = daf_utils.invert_dol_to_dict(typ_to_cols_dict or {})

        for colname in self.hd:

            if colname in col_to_typ_dict:
                dtypes[colname] = col_to_typ_dict[colname]
            else:
                dtypes[colname] = default_type

        self.dtypes = dtypes

        return self



    def apply_dtypes(self, *,
            dtypes:         Optional[T_dtype_dict]=None,
            unflatten:      bool=True,
            from_str:       bool=True,
            default_type:   Type=str,
            silent_error:   bool=False,
            ) -> 'Daf':
        """
        Convert the columns to their dtypes, in place.

        A CSV file is read as text, because that is the fastest way to load it. Call
        this method to turn the columns you need into numbers, lists and so on. It
        changes the cells where they are and does not make a new table. Columns you
        leave out are not touched.

        The `dtypes` argument is a dict that maps a column name to a type, or one type
        for all columns. It may hold more columns than the Daf. If it is given, it
        replaces `dtypes` of the Daf. If neither is set, nothing happens. With no
        columns defined, the names are taken from the dtypes.

        Types must be plain types such as `int`, `float`, `bool`, `str`, `list`,
        `dict`, `tuple` or `set`. Annotations such as `List[str]` do not work.

        By default, columns of type `str` are skipped. The cells are assumed to be text
        already. Pass `from_str=False` when they may hold other values. Then each cell
        is converted with `str()`. Columns of type `list` or `dict` are read from
        their text. Pass `unflatten=False` to leave them as text.

        A cell that cannot be converted to an `int` or a `float` keeps its text, so a bad
        value is still there to be found, as with `list` and `dict`. An empty cell stays
        empty. No error is raised here. A later step, such as a sum or a sort, may raise
        one. Whole number text of any size is converted exactly. Text with a decimal point
        or an exponent is converted to an `int` by way of a float, which cuts the decimal part.

        This method does the common conversions and keeps them simple. For your own
        rules, convert the columns yourself and then say what the types are. Use
        `apply_to_col()` for one column, or `apply_in_place()` for several. Your
        function can raise, collect the bad values, or use any default. Then set
        `dtypes`. This does not convert anything. See the second example.

        Args:
            dtypes: Maps column names to types, or a single type for all columns.
            unflatten: If True, read list and dict columns from their text.
            from_str: If True, the cells are text, so `str` columns are not converted.
            default_type: The type to use for a column that has no dtype.
            silent_error: If False, raise an error when a column has no dtype.

        Returns:
            This Daf, which has been changed.

        Raises:
            ValueError: A column has no dtype and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c'])
            >>> d.apply_dtypes(dtypes={'a': int, 'b': float, 'c': str})
            | a |  b  | c |
            | -: | --: | -: |
            | 1 | 2.5 | x |
            %% daf rows=1; cols=3; keyfield=''; name=''
            >>> Daf(lol=[['x', '']], cols=['a', 'b']).apply_dtypes(dtypes={'a': int, 'b': int})
            | a | b |
            | -: | -: |
            | x |   |
            %% daf rows=1; cols=2; keyfield=''; name=''

            Your own conversion, here one that records the values that fail:

            >>> bad = []
            >>> def to_int(value):
            ...     try:
            ...         return int(value)
            ...     except ValueError:
            ...         bad.append(value)
            ...         return ''
            >>> d = Daf(lol=[['1'], ['x'], ['3']], cols=['n'])
            >>> _ = d.apply_to_col('n', to_int)
            >>> d.dtypes = {'n': int}
            >>> d
            | n |
            | -: |
            | 1 |
            |   |
            | 3 |
            %% daf rows=3; cols=1; keyfield=''; name=''
            >>> bad
            ['x']
        """

        """ convert columns of daf array to the datatypes specified in self.dtypes or in passed parameter.
                dtypes can be a dict where each column may have a different type, or it can be a single type.
                dtypes may provide desired new types for only some of the columns.
                types in dtypes must be origin types, like bool, str, int, float, list, dict, tuple, set,
                    and not type annotations such as List[str] etc.
            columns (self.hd) must be defined.
            unflatten: unflatten from pyon or json to list or dict types
            if from_str is True, (default) assumes that data starts as str, such as when read from csv
                If the csv is first read, data is all delivered as str.
                Columns specified as str type are not converted.

            Note, this converts types in place, and does not create a new array.

            if no dtypes is passed and self.dtypes is not defined, do nothing.

            Unless silent_error is True, will check for columns consistency against the dtypes.
            Note: dtypes can be a SUPERSET of the columns. I.e. a single dtypes definition can be
                    used for a number of tables that differ only in which columns are included.
        """

        if dtypes:
            self.dtypes = dtypes

        if not self.dtypes:
            return self

        # this adopts the hd from dtypes if not already defined.
        if not self.hd and self.dtypes and isinstance(self.dtypes, dict):
            # self.dtypes, not the (possibly still-None if the caller didn't pass dtypes=)
            # local dtypes param -- the guard above checks self.dtypes specifically.
            self._cols_to_hd(self.dtypes.keys())
            # currently will not overwrite existing cols.
            # (i.e. pass partial dtypes will alter only those cols)

        if (
            not silent_error
            and isinstance(self.dtypes, dict)
            and self.hd
            and not set(self.hd).issubset(self.dtypes)
            ):
            missing = sorted(set(self.hd) - set(self.dtypes))
            raise ValueError(
                "dtypes mismatch. check the schema:\n"
                f"missing dtypes for cols={missing}\n"
                f"dtypes keys={sorted(self.dtypes)},\n"
                f"cols={sorted(self.hd)}"
            )
        if not self.lol or not self.lol[0]:
            # this can sometimes happen, no worries.
            return self

        # first calculate all columns to consider, that are not str or str and unflatten
        cols = self.hd.keys()
        for col in cols:
            if isinstance(self.dtypes, dict):           # this is the normal case, where each column is defined.
                if col not in self.dtypes:
                    desired_type = default_type         # type not specified for a column, but it exists, use default_type
                    self.dtypes[col] = default_type     # make sure self.dtypes is updated.
                desired_type = self.dtypes[col]
            else:
                desired_type = self.dtypes              # only one type is defined.

            if (    desired_type is str and from_str or
                    desired_type in [list, dict] and not unflatten
                ):
                continue

            # update this column if needed.
            icol = self.hd[col]

            # look up the conversion once for the column, not for each cell.
            convert = daf_utils.get_converter(desired_type)

            for row_la in self.lol:
                row_la[icol] = convert(row_la[icol])

        return self

    def flatten(self, convert_bool_to_int: bool=True, use_pyon: bool = True) -> 'Daf':
        """
        Turn list and dict cells into text, in place.

        You rarely need this. `to_csv_buff()` and `to_csv_file()` write every cell as
        text already, so a separate pass is not needed.

        Only the columns whose dtype is `list` or `dict` are changed. Each cell
        becomes its `str()` text. A column of dtype `bool` becomes 0 and 1. Without
        dtypes nothing happens.

        Args:
            convert_bool_to_int: If True, write bool columns as 0 and 1.
            use_pyon: Kept for compatibility. Only True is supported.

        Returns:
            This Daf, which has been changed.

        Raises:
            ValueError: `use_pyon` is False.

        Examples:
            >>> d = Daf(lol=[[[1, 2], True]], cols=['a', 'b'], dtypes={'a': list, 'b': bool})
            >>> d.flatten()
            |   a    | b |
            | -----: | -: |
            | [1, 2] | 1 |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """
        if not use_pyon:
            raise ValueError("flatten(): use_pyon=False is not supported. Cells are flattened to PYON.")

        if not self.lol or not self.lol[0] or not self.dtypes:
            # this can sometimes happen, no worries.
            # when daf is constructed internally and has no list or dict types, there is no need to flatten.
            # in this case, the self.dtypes need not be initialized or used until the table is read.
            return self

        if not self.hd:     # pragma: no cover
            # should not be the case. Logic error.
            raise RuntimeError("flatten(): Daf has data and dtypes but no header")

        # first calculate all columns to consider
        cols = self.hd.keys()
        for col in cols:
            if isinstance(self.dtypes, dict):
                if col not in self.dtypes:
                    continue
                desired_type = self.dtypes[col]
            else:
                desired_type = self.dtypes

            if desired_type in (list, dict):

                icol = self.hd[col]

                for irow in range(len(self.lol)):
                    self.lol[irow][icol] = f"{self.lol[irow][icol]}"

            if convert_bool_to_int and desired_type is bool:
                # this should be rare!

                icol = self.hd[col]

                for irow in range(len(self.lol)):

                    self.lol[irow][icol] = int(bool(self.lol[irow][icol]))

        return self

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


    def strip(self, chrs: str=' ') -> 'Daf':
        """
        Remove characters from both ends of every text cell, in place.

        Each character in `chrs` is removed on its own, so `'()"'` removes any mix of
        parentheses and quotes. Cells that are not text, and empty cells, are skipped.

        Args:
            chrs: The characters to remove.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> Daf(lol=[[' a ', 3, '("x")']], cols=['p', 'q', 'r']).strip(' ()"')
            | p | q | r |
            | -: | -: | -: |
            | a | 3 | x |
            %% daf rows=1; cols=3; keyfield=''; name=''
        """
        """ remove leading or trailing characters in the string chrs from each str value in the array.
            modifies in-place.  Ignores non str values.

            each character in chrs treated seperately, such as strip('"()') removes quotes and parens.
        """

        for row_la in self.lol:
            for icol in range(len(row_la)):
                if isinstance(row_la[icol], str) and row_la[icol]:
                    row_la[icol] = row_la[icol].strip(chrs)

        return self


    def _safe_tofloat(val: Any) -> Union[float]:
        """
        Safely convert value to float.

        Args:
            val: Value to convert.

        Returns:
            Union[float, str]: Converted float or original value on failure.

        Note:
            Internal use.    
        """
        try:
            return float(val)
        except ValueError:
            return 0.0

    #===========================
    # schema support
    
    default_record = daf_schema._default_record


    #===========================
    # initializers

    def clone_empty(self, lol: Optional[T_lola]=None, cols: Optional[T_ls]=None, name:str='') -> 'Daf':
        """
        Make a new Daf with the same layout and no rows.

        The new Daf has the same column names, keyfield and dtypes. The dtypes dict is
        copied, and the `attrs` are deep copied. The name, the key index and the
        display settings are not carried over. Set them on the new Daf if you need them.

        Give `lol` to fill the new Daf with rows. They are adopted, not copied.

        Args:
            lol: Rows for the new Daf. If None, it has no rows.
            cols: Column names to use instead of the existing ones.
            name: The name of the new Daf.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> c = d.clone_empty()
            >>> c.columns(), c.keyfield, c.num_rows()
            (['id', 'v'], 'id', 0)
        """
        """
        Create a new empty Daf instance from self, adopting column names but not data.

        - Adopts `keyfield` but does not adopt `kd` or `attrs` (metadata is not propagated).
        - If `lol` is provided as an argument, it is used in the new Daf.
        - This method creates a fresh instance with the same structure but no metadata.

        If metadata needs to be propagated, it must be done manually:
            new_daf.attrs = daf.attrs.copy()

        Returns:
            A new `Daf` instance with the same column structure but no data (or using passed data lol).
        """
        if self is None:
            return Daf()

        new_cols = cols if cols else self.columns()

        # the initialization below automatically invalidates kd for lazy rebuilding.
        new_daf = Daf(cols=new_cols, lol=lol, keyfield=self.keyfield, dtypes=copy.copy(self.dtypes), name=name)

        # Ensure metadata attributes are deeply copied
        new_daf.attrs = copy.deepcopy(self.attrs)

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
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> d.set_lol([[5, 'q'], [6, 'r']]).keys()
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

    # ==== Python lod (list of dictionaries)
    @classmethod
    def from_lod(
            cls,
            records_lod:    T_loda,                             # List[List[Any]] to initialize the lol data array.
            *,
            keyfield:       Union[str, int, T_ta, T_la] = '',   # set a new keyfield or set no keyfield.
            dtypes:         T_dtype_dict | None         = None, # set the data types for each column.
            name:           str                         = '',   # Optional name of the daffodil instance.
            cols:           T_ls | None                 = None, # Optionally use cols to define column names
            ) -> 'Daf':
        """
        Make a Daf from a list of dicts, one dict for each row.

        Without `cols` or `dtypes`, the column names are the keys of the first dict. A
        later dict that lacks a key gets NULL there. A later dict with a key that the
        first dict does not have raises `ValueError`, because that value would be
        lost. The error names the keys. Give `cols` with all the columns you want to
        avoid it. Empty dicts and items that are not dicts are skipped.

        If `cols` is given, those are the columns. Otherwise the keys of `dtypes` are
        the columns. In both cases any other keys are left out, because you chose
        the columns.

        Args:
            records_lod: The rows, as dicts.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            dtypes: Type for each column. When given, it also selects the columns.
            name: Name of the new Daf.
            cols: Column names to use, in order.

        Returns:
            The new Daf.

        Raises:
            ValueError: A dict has a key that is not a column of the first dict, and
                neither `cols` nor `dtypes` is given.

        Examples:
            >>> Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3}])
            | a | b |
            | -: | -: |
            | 1 | 2 |
            | 3 |   |
            %% daf rows=2; cols=2; keyfield=''; name=''
            >>> Daf.from_lod([{'a': 1, 'b': 2}], cols=['b', 'a'])
            | b | a |
            | -: | -: |
            | 2 | 1 |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """

        """ Create Daf instance from loda type, adopting dict keys as column names
            Generally, all dicts in records_lod should be the same OR the first one must have all keys
                and others can be missing keys.
            However, if dtypes is provided, it will be used to establish the columns.

            my_daf = Daf.from_lod(sample_lod)
        """
        # test exists in test_daf.py


        if dtypes is None:
            dtypes = {}

        if cols is None:
            cols = []

        if not records_lod:
            return cls(cols=cols, keyfield=keyfield, dtypes=dtypes)

        if cols or dtypes:
            # the caller chose the columns, so keys that are not columns are left out on purpose.
            if not cols:
                cols = list(dtypes.keys())

            lol = [list(daf_utils.set_cols_da(record_da, cols).values())
                    for record_da in records_lod if record_da and isinstance(record_da, dict)]

        else:
            # the columns come from the first record. A later record may lack some of them, which
            # are then NULL. A key that is not a column would be lost, so stop and say so.
            # Checking the keys of each record against a set of the columns is done in C, and
            # building the row directly avoids making a dict for each record.
            cols = list(records_lod[0].keys())
            colset = set(cols)

            lol = []
            for record_da in records_lod:
                if record_da and isinstance(record_da, dict):
                    if not record_da.keys() <= colset:
                        extra_keys = [key for key in record_da if key not in colset]
                        raise ValueError(
                            f"from_lod: a record has keys that are not columns of the first record: "
                            f"{extra_keys[:5]}. Pass cols= with all the columns that you want.")
                    lol.append([record_da.get(col, NULL) for col in cols])

        # following invalidates kd for lazy rebuilding.
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)


    # ==== Python lot (list of tuples)
    @classmethod
    def from_lot(
            cls,
            records_lot:    T_lota,                             # List of tuples to initialize the lol data array.
            cols:           Optional[T_ls]              = None, # Optional column names to use.
            dtypes:         Optional[T_dtype_dict]      = None, # Optional dtype_dict describing the desired type of each column.
                                                                #   also used to define column names if provided and cols not provided.
            keyfield:       Union[str, int, T_ta, T_la] = '',   # A field of the columns to be used as a key.
            name:           str                         = '',   # Optional name of the daffodil instance.
            ) -> 'Daf':
        """
        Make a Daf from a list of tuples, one tuple for each row.

        Without `cols` the Daf has no column names, as with `Daf(lol=...)`. Call
        `set_cols()` to give it spreadsheet names such as `A` and `B`, or to name them. A
        Daf with no names cannot give its rows as dicts or KeyedList objects.

        Args:
            records_lot: The rows, as tuples.
            cols: Column names. They must be as many as the items in each tuple.
            dtypes: Type for each column.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            name: Name of the new Daf.

        Returns:
            The new Daf.

        Raises:
            ValueError: A tuple does not have as many items as there are columns.

        Examples:
            >>> Daf.from_lot([(1, 'a'), (2, 'b')], cols=['id', 'v'])
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            %% daf rows=2; cols=2; keyfield=''; name=''
            >>> Daf.from_lot([(1, 'a')]).columns()
            []
            >>> Daf.from_lot([(1, 'a')]).set_cols().columns()
            ['A', 'B']
        """
        """
        Create Daf instance from LOT (list of tuples), adopting given column names or generating default ones.

        Args:
            records_lot (List[Tuple[Any, ...]]):                    The LOT data.
            columns (Optional[List[str]]):                          Column names for the tuples. If None, default names will be generated.
            keyfield (Union[str, int, Tuple[Any, ...], List[Any]]): Keyfield for the Daf instance.
            dtypes (Optional[Dict[str, type]]):                     Data types for each column.

        Returns:
            Daf: A new Daf instance with data populated from lot.

        Example:
            records_lot = [(1, 'Alice', 30), (2, 'Bob', 25)]
            columns = ['id', 'name', 'age']
            my_daf = Daf.from_lot(records_lot, cols=cols)
        """
        if not records_lot:
            return cls(keyfield=keyfield, dtypes=dtypes)

        if dtypes is None:
            dtypes = {}

        # Check for mismatch between column names and tuple length, or between the tuples.
        num_cols = len(cols) if cols is not None else len(records_lot[0])
        if any(len(row) != num_cols for row in records_lot):
            raise ValueError("Each tuple in records_lot must have the same number of elements as the columns.")

        # Convert LOT to LOL (list of lists) for Daf
        lol = [list(row) for row in records_lot]

        # following invalidates kd for lazy rebuilding.
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)



    def to_lod(self) -> T_loda:
        """
        Make a list of dicts, one dict for each row.

        The dicts are new, but the values in them are the same objects as in the Daf.
        An empty Daf gives an empty list. See `iter_dict()` to go through rows one at
        a time without building the whole list.

        Returns:
            The rows as dicts.

        Examples:
            >>> Daf(lol=[[1, 'a']], cols=['id', 'v'])
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """
        # test exists in test_daf.py

        if not self:
            return []

        if not self.hd:
            raise KeysDisabledError("to_lod(): the Daf has no column names. Call set_cols() to name them.")

        cols = self.columns()
        result_lod = [dict(zip(cols, la)) for la in self.lol]

        return result_lod


    # ==== Python dod (dict of dict)
    @classmethod
    def from_dod(
            cls,
            dod:            T_doda,         # Dict(str, Dict(str, Any))
            keyfield:       str='rowkey',   # The keyfield will be set to the keys of the outer dict.
                                            # this will set the preferred name. Defaults to 'rowkey'
            dtypes:         Optional[T_dtype_dict]=None     # optionally set the data types for each column.
            ) -> 'Daf':

        """
        Make a Daf from a dict of dicts, where the outer key names the row.

        A dict of dicts usually does not repeat the row key inside each row. A Daf
        table always has it as a column. If the inner dicts lack the `keyfield`
        column, it is added from the outer keys. The new Daf has that keyfield.

        Use `to_dod()` to go back.

        Args:
            dod: A dict that maps a row key to a dict of that row.
            keyfield: The column that holds the outer key.
            dtypes: Type for each column.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf.from_dod({'r0': {'x': 1}, 'r1': {'x': 2}})
            >>> d
            | rowkey | x |
            | -----: | -: |
            |     r0 | 1 |
            |     r1 | 2 |
            %% daf rows=2; cols=2; keyfield='rowkey'; name=''
        """

        """ a dict of dict (dod) structure is very similar to a Daf table, but there is a slight difference.
            A dod structure will have a first key which indexes to a specific dict.
            The key in that dict is likely not also found in the "value" dict of the first level, but it might be.

            a Daffodil table always has the keys of the outer dict as items in each table.
            Thus dod1 = {'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2, ... },
                         'row_1': {'rowkey': 'row_1', 'data1': 11, ... },
                         ...
                         }
            is fully compatible because it has a first item which is the rowkey.
            If a dod is passed that does not have this column, then it will be created.
            The 'keyfield' parameter should be set to the name of this column.

            A typical dod does not have the row key as part of the data in each row, such as:

             dod2 = {'row_0': {'data1': 1, 'data2': 2, ... },
                     'row_1': {'data1': 11, ... },
                     ...
                    }

            If dod2 is passed, it will be convered to dod1 and then converted to daf instance.

            A Daf table is able 1/3 the size of an equivalent dod. because the column keys are not repeated.

            use to_dod() to recover the original form by setting 'remove_rowkeys'=True if the row keys are
            not required in the dod.

        """
        # following invalidates kd for lazy rebuilding.
        return cls.from_lod(daf_utils.dod_to_lod(dod, keyfield=keyfield), keyfield=keyfield, dtypes=dtypes)


    def to_dod(
            self,
            remove_keyfield:    bool=True,      # by default, the keyfield column is removed.
            ) -> T_doda:
        """
        Make a dict of dicts, where the keyfield value names each row.

        The keyfield column is left out of the inner dicts by default. Pass
        `remove_keyfield=False` to keep it there too. The Daf must have a keyfield.
        Without one, a `KeysDisabledError` is raised, unless the Daf has no rows.

        Args:
            remove_keyfield: If True, do not repeat the key inside each inner dict.

        Returns:
            A dict that maps each key to a dict of that row.

        Raises:
            KeysDisabledError: The Daf has no keyfield.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> d.to_dod()
            {1: {'v': 'a'}}
            >>> d.to_dod(remove_keyfield=False)
            {1: {'id': 1, 'v': 'a'}}
        """

        """ a dict of dict structure is very similar to a Daf table, but there is a slight difference.
            a Daf table always has the keys of the outer dict as items in each table.
            Thus dod1 = {'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2, ... },
                         'row_1': {'rowkey': 'row_1', 'data1': 11, ... },
                         ...
                         }
            If a dod is passed that does not have this column, then it will be created.
            The 'keyfield' parameter should be set to the name of this column.

            A typical dod does not have the row key as part of the data in each row, such as:

             dod2 = {'row_0': {'data1': 1, 'data2': 2, ... },
                     'row_1': {'data1': 11, ... },
                     ...
                    }

            If remove_keyfield=True (default) dod2 will be produced, else dod1.

        """
        if self and not self.keyfield:
            raise KeysDisabledError("to_dod(): the Daf has no keyfield.")

        # lod_to_dod() does a single dict-key lookup per row (da[keyfield]), which only makes
        # sense for a simple str keyfield, not daf's own broader composite (tuple/list) keyfield
        # support -- this method is only meaningful when self.keyfield is a plain str.
        return daf_utils.lod_to_dod(self.to_lod(), keyfield=cast(str, self.keyfield), remove_keyfield=remove_keyfield)


    # ==== cols_dol
    @classmethod
    def from_cols_dol(
            cls,
            cols_dol: T_dola,
            keyfield: str='',
            dtypes: Optional[T_dtype_dict]=None,
            ) -> 'Daf':
        """
        Make a Daf from a dict of lists, where each list is a column.

        All the lists must be as long as each other. If one differs in length from the
        first, a `ValueError` is raised, because the rows could not be made without
        losing values or guessing them.

        Args:
            cols_dol: Maps a column name to the list of its values.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            dtypes: Type for each column.

        Returns:
            The new Daf.

        Raises:
            ValueError: A list has a different length from the first.

        Examples:
            >>> d = Daf.from_cols_dol({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> d
            | A | B |
            | -: | -: |
            | 1 | 4 |
            | 2 | 5 |
            | 3 | 6 |
            %% daf rows=3; cols=2; keyfield=''; name=''
        """
        """ Create Daf instance from cols_dol type, adopting dict keys as column names
            and creating columns from each value (list)

            my_daf = Daf.from_cols_dol({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9])

            produces:
                my_daf.columns() == ['A', 'B', 'C']
                my_daf.lol == [[1,4,7], [2,5,8], [3,6,9]]


        """
        if dtypes is None:
            dtypes = {}

        if not cols_dol:
            # following invalidates kd for lazy rebuilding.
            return cls(keyfield=keyfield, dtypes=dtypes)

        cols = list(cols_dol.keys())

        num_rows = len(cols_dol[cols[0]])
        for col in cols:
            if len(cols_dol[col]) != num_rows:
                raise ValueError(
                    f"from_cols_dol: column '{col}' has {len(cols_dol[col])} values, "
                    f"but column '{cols[0]}' has {num_rows}.")

        # zip turns the columns into rows in C.
        lol = [list(row) for row in zip(*cols_dol.values())]

        # following invalidates kd for lazy rebuilding.
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)


    def to_cols_dol(self) -> dict:
        """
        Make a dict of lists, one list of values for each column.

        Returns:
            A dict that maps each column name to the list of its values.

        Examples:
            >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_cols_dol()
            {'id': [1, 2], 'v': ['a', 'b']}
        """

        """ convert daf to dictionary of lists of values, where key is the
            column name, and the list are the values in that column.
        """
        result_dol: Dict[str, list] = {colname: [] for colname in self.columns()}

        for row_da in self:
            for key, val in row_da.items():
                result_dol[key].append(val)     # type: ignore[index]  # a row's keys are column names (str)

        return result_dol


    def to_attrib_dict(self) -> dict:
        """
        Make a dict with the column names and the rows.

        This is deprecated. It keeps the rows as they are, not copied. Use
        `to_lod()` or `to_cols_dol()` instead.

        Returns:
            A dict with the keys `cols` and `lol`.

        Examples:
            >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_attrib_dict()
            {'cols': ['id', 'v'], 'lol': [[1, 'a']]}
        """
        return {'cols': self.columns(), 'lol': self.lol}



    @classmethod
    def from_lod_to_cols(
            cls,
            lod:        T_loda,
            cols:       Optional[List]=None,
            keyfield:   str='',
            dtypes:     Optional[T_dtype_dict]=None
            ) -> 'Daf':
        """
        Make a Daf in which each dict becomes a column, not a row.

        The keys of the dicts become the first column, and each dict adds one more
        column after it. Use this to compare several results side by side, such as
        the same features measured in several tries. It is a transpose of
        `from_lod()`.

        Without `cols`, the first column is named `key` and the others `A`, `B` and
        so on. The names in `cols` include the first column. The keyfield is not set
        unless `keyfield` is given.

        Args:
            lod: The dicts. They should all have the same keys.
            cols: Column names, starting with the name of the column of keys.
            keyfield: Column whose values identify rows.
            dtypes: Type for each column of the dicts, before the change.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf.from_lod_to_cols([{'A': 1, 'B': 2}, {'A': 4, 'B': 5}], cols=['Feature', 'T1', 'T2'])
            >>> d
            | Feature | T1 | T2 |
            | ------: | -: | -: |
            |       A |  1 |  4 |
            |       B |  2 |  5 |
            %% daf rows=2; cols=3; keyfield=''; name=''
        """
        r""" Create Daf instance from a list of dictionaries to be placed in columns
            where each column shares the same keys in the first column of the array.
            This transposes the data from rows to columns and adds the new 'cols' header,
            while adopting the keys as the keyfield. dtypes is applied to the columns
            transposition and then to the rows.

            If no 'cols' parameter is provided, then it will be the name 'key'
            followed by normal spreadsheet column names, like 'A', 'B', ...

            Creates a daf where the first column are the keys from the dicts,
            and each subsequent column are each of the values of the dicts.

            my_daf = Daf.from_coldicts_lod(
                cols = ['Feature', 'Try 1', 'Try 2', 'Try 3'],
                lod =       [{'A': 1, 'B': 2, 'C': 3},          # data for Try 1
                             {'A': 4, 'B': 5, 'C': 6},          # data for Try 2
                             {'A': 7, 'B': 8, 'C': 9} ]         # data for Try 3

            produces:
                my_daf.columns() == ['Feature', 'Try 1', 'Try 2', 'Try 3']
                my_daf.lol ==        [['A',       1,       4,       7],
                                             ['B',       2,       5,       8],
                                             ['C',       3,       6,       9]]

            This format is useful for producing reports of several tries
            with different values for the same attributes placed in columns,
            particularly when there are many features that need to be compared.
            Columns are defined directly from cols parameter.

        """
        if dtypes is None:
            dtypes = {}

        if cols is None:
            cols = []

        if not lod:
            # following invalidates kd for lazy rebuilding.
            return cls(keyfield=keyfield, dtypes=dtypes, cols=cols)

        # the following will adopt the dictionary keys as cols.
        # note that dtypes applies to the columns in this orientation.
        # following invalidates kd for lazy rebuilding.
        rows_daf = cls.from_lod(lod, dtypes=dtypes)

        # this transposes the data, and puts the keys of the dicts in the first column. The keys are
        # taken from the dicts and not from the column names of rows_daf, because those names are
        # made valid and unique, and a key such as '' would come back as 'Unnamed1'. The keys are data here.
        # The column names are those given, or else ['key', 'A', 'B', ...]
        if not cols:
            cols = ['key'] + daf_utils._generate_spreadsheet_column_names_list(num_cols=len(lod))

        cols_lol = [[key] + list(values) for key, values in zip(lod[0].keys(), zip(*rows_daf.lol))]

        return cls(lol=cols_lol, cols=cols, keyfield=keyfield)


    #==== Excel
    @classmethod
    def from_excel_buff(
            cls,
            excel_buff: bytes,
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            ) -> 'Daf':
        """
        Make a Daf from the bytes of an xlsx file.

        The first sheet is turned into CSV by the `xlsx2csv` package. The CSV is then
        read like any CSV text. All values start as text, so give `dtypes` or call
        `apply_dtypes()` to convert them.

        Short rows are not padded here. `xlsx2csv` pads them. See `xlsx_to_csv()`.

        Args:
            excel_buff: The bytes of the xlsx file.
            keyfield: Column whose values identify rows.
            dtypes: Type for each column.
            noheader: If True, the first row is data, not column names.
            user_format: If True, skip comment lines and blank lines.
            unflatten: If True, read list and dict columns from their text.

        Returns:
            The new Daf.

        Examples:
            >>> import io, xlsxwriter
            >>> buff = io.BytesIO()
            >>> workbook = xlsxwriter.Workbook(buff, {'in_memory': True})
            >>> sheet = workbook.add_worksheet()
            >>> for irow, row in enumerate([['id', 'v'], [1, 'a'], [2, 'b']]):
            ...     for icol, value in enumerate(row):
            ...         _ = sheet.write(irow, icol, value)
            >>> workbook.close()
            >>> d = Daf.from_excel_buff(buff.getvalue(), keyfield='id')
            >>> d
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> Daf.from_excel_buff(buff.getvalue(), dtypes={'id': int, 'v': str})
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """

        # from utilities import xlsx_utils

        csv_buff = daf_utils.xlsx_to_csv(excel_buff)

        # following invalidates kd for lazy rebuilding.
        my_daf  = cls.from_csv_buff(
                        csv_buff,
                        keyfield    = keyfield,         # field to use as unique key, if not ''
                        dtypes      = dtypes,           # dictionary of types to apply if set.
                        noheader    = noheader,         # if True, do not try to initialize columns in header dict.
                        user_format = user_format,      # if True, preprocess the file and omit comment lines.
                        unflatten   = unflatten,        # unflatten fields that are defined as dict or list.
                        )

        return my_daf

    #==== CSV
    @classmethod
    def from_csv(cls, source: str | Path, **kwargs: Any) -> 'Daf':
        r"""
        Read a CSV file, a web address or an S3 object into a Daf.

        The source may be a path, a `Path`, an `http` or `https` address, or an
        `s3://bucket/key` name. The `requests` and `boto3` packages are imported only
        when they are needed.

        Every value is read as text. Give `dtypes`, or call `apply_dtypes()`, to
        convert them. The other keyword arguments, such as `keyfield`, are those of
        `from_csv_buff()`. The length of each row is not checked. See `from_csv_buff()`.

        Args:
            source: A file path, an `http` address or an `s3://` name.
            **kwargs: Passed on to `from_csv_buff()`.

        Returns:
            The new Daf.

        Raises:
            RuntimeError: The download fails, a needed package is missing, or the
                local file cannot be read or parsed. The message says which.

        Examples:
            >>> import os, tempfile
            >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
            >>> with open(path, 'w') as f:
            ...     n = f.write('id,v\n1,a\n')
            >>> Daf.from_csv(path)
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """
        """
        Load a CSV file from a local file, URL, or S3 path into a Daf array.

        - Supports streaming for large files.
        - `requests` and `boto3` are only imported if needed.

        - All data is imported as str types. use apply_dtypes() to convert data types.
        - Does not set the keyfield,

        Args:
            source (str): File path, URL (http/https), or S3 path (s3://bucket/key).
            **kwargs: Additional arguments passed to from_csv_buff().

        Returns:
            Daf: A Daf array loaded from the CSV.
        """
        if isinstance(source, Path):  # Convert Path object to string
            source = str(source)

        if source.startswith('http'):  # Handle HTTP(S) URLs
            try:
                import requests  # Import only if needed
                response = requests.get(source, stream=True)
                response.raise_for_status()
                data_stream = (line.decode("utf-8") for line in response.iter_lines() if line)
            except ImportError:
                raise RuntimeError("Missing `requests` module. Install it via `pip install requests`.")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download CSV from {source}: {e}")

        elif source.startswith("s3://"):  # Handle S3 Paths with streaming
            try:
                import boto3  # Import only if needed
                s3 = boto3.client("s3")
                bucket, key = source[5:].split("/", 1)  # Extract bucket and key
                obj = s3.get_object(Bucket=bucket, Key=key)
                data_stream = (line.decode("utf-8") for line in obj["Body"].iter_lines() if line)
            except ImportError:
                raise RuntimeError("Missing `boto3` module. Install it via `pip install boto3`.")
            except Exception as e:
                raise RuntimeError(f"Failed to read CSV from S3: {e}")

        else:  # Assume local file with streaming
            try:
                with open(source, "r", encoding="utf-8") as f:  # Use `with` to ensure closure
                    return cls.from_csv_buff(csv_buff=f, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to read local file: {e}")

        # following invalidates kd for lazy rebuilding.
        new_daf = cls.from_csv_buff(csv_buff=data_stream, **kwargs)

        return new_daf

    # STILL ACTIVE -- use when we know the source is a buffer.
    @classmethod
    def from_csv_buff(
            cls,
            csv_buff: Union[bytes, str, Iterator[str]], # Can now accept iterators directly
            keyfield: str='',                           # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,        # dictionary of types to apply if set.
            noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
            sep: str=',',                               # field separator.
            unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,          # include only the columns specified. noheader must be false.
            name: str = '',                             # name attribute of the Daf array created.
            ) -> 'Daf':
        r"""
        Make a Daf from CSV text, bytes or an iterator of lines.

        The first row holds the column names, unless `noheader` is True. Quoted fields
        may hold commas. Empty rows at the end are dropped. Cells are text, unless
        `dtypes` is given. Then the cells are converted, and list and dict columns
        are read from their text unless `unflatten` is False.

        Empty text, or a source with no rows, gives an empty Daf with no columns, as
        `from_md()` does. Check `len()` of the result if an empty source would be an error.

        A `keyfield` that is not a column of the file is stored without an error, and key
        lookups then find nothing. Check the names, or call `set_keyfield()` with
        `silent_error=False`.

        The length of each row is not checked, so that a read costs no more than it must.
        A row with a missing cell, or with an extra one from an unquoted comma, is kept as
        it is, and a later call may fail with an error that does not mention it. If you do
        not trust the source, call `is_rectangular()`. It looks at every row. To pad short
        rows, call `force_rectangular()`. It raises `ValueError` for a row that is too long.

        Args:
            csv_buff: The CSV, as text, bytes or an iterator of lines.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            dtypes: Type for each column.
            noheader: If True, the first row is data, and the Daf has no column names.
            user_format: If True, skip comment lines and blank lines.
            sep: The character that separates fields.
            unflatten: If True, read list and dict columns from their text.
            include_cols: Keep only these columns, in this order. The cells of the other
                columns are never kept, so a wide file costs little memory.
            name: Name of the new Daf.

        Returns:
            The new Daf.

        Raises:
            KeyError: A name in `include_cols` is not in the header of the file.
            ValueError: `include_cols` is given with `noheader=True`, because the names
                of the columns come from the header.

        Examples:
            >>> d = Daf.from_csv_buff('id,v\n1,a\n2,"b,c"\n')
            >>> d
            | id |  v  |
            | -: | --: |
            |  1 |   a |
            |  2 | b,c |
            %% daf rows=2; cols=2; keyfield=''; name=''
            >>> Daf.from_csv_buff('id,v\n1,a\n', dtypes={'id': int, 'v': str})
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
            >>> d = Daf.from_csv_buff('a,b,c\n1,2,3\n4,5,6\n', include_cols=['c', 'a'])
            >>> d
            | c | a |
            | -: | -: |
            | 3 | 1 |
            | 6 | 4 |
            %% daf rows=2; cols=2; keyfield=''; name=''
            >>> d = Daf.from_csv_buff('id,v\n1,a\n2\n')
            >>> d.is_rectangular()
            False
            >>> d.force_rectangular()
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 |   |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """

        """
        Convert CSV data in a buffer (string, bytes, or iterator) to a daf object

        Note: probably just use from_csv() and then apply_dtypes(), remove 'unflatten',
            however, in the future of a more optimized conversion is written, then python
            types can be created as the csv is scanned rather than scanning again.

        - Fully supports streaming CSVs (does not require full file in memory).
        - Directly reads data into a list of lists (LoL) from an iterator.

        Args:
            csv_buff: Union[bytes, str, Iterator[str]], # Can now accept iterators directly
            keyfield: str='',                           # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,        # dictionary of types to apply if set.
            noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
            sep: str=',',                               # field separator.
            unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,          # include only the columns specified. noheader must be false.
            name: str = '',                             # name attribute of the Daf array created.

        Returns:
            Daf: The loaded Daf array.
        """

        if include_cols and noheader:
            raise ValueError("from_csv_buff: include_cols needs the header row to find the columns, so noheader must be False.")

        # in the case of bytes, this will set up a conversion of the stream.
        # Converts bytes into a file-like object without reading everything at once.
        if isinstance(csv_buff, bytes):
            csv_buff = io.TextIOWrapper(io.BytesIO(csv_buff), encoding="utf-8")

        # the following will be able to stream from the source and convert directly to lol.
        data_lol = daf_utils.buff_csv_to_lol(csv_buff, user_format=user_format, sep=sep, include_cols=include_cols, dtypes=dtypes)

        # Remove trailing empty list rows sometimes caused by trailing newlines
        while data_lol and data_lol[-1] == []:
            data_lol.pop()

        cols = []
        if not noheader and data_lol:
            cols = data_lol.pop(0)        # return the first item and shorten the list. No rows gives an empty Daf.

        # following invalidates kd for lazy rebuilding.
        my_daf = cls(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes, name=name)

        # the following will act nicely if there is no dtypes defined.
        my_daf.apply_dtypes(unflatten=unflatten)

        return my_daf

    # DEPRECATED
    @classmethod
    def from_csv_file(
            cls,
            filepath: str,                          # The CSV filename
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            sep: str=',',                           # field separator.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,      # include only the columns specified. noheader must be false.
            name: str = '',                         # name attribute of the Daf array created.
            ) -> 'Daf':                             # New daf instance.
        r"""
        Read a CSV file into a Daf. Deprecated, use `from_csv()`.

        This now calls `from_csv()`, so it reads the file as UTF-8, and a file that cannot be
        read raises `RuntimeError`. It used to print a message and return None, and to read
        the file with the encoding of the machine.

        Args:
            filepath: Path of the file.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            dtypes: Type for each column.
            noheader: If True, the first row is data, and the Daf has no column names.
            user_format: If True, skip comment lines and blank lines.
            sep: The character that separates fields.
            unflatten: If True, read list and dict columns from their text.
            include_cols: Keep only these columns, in this order. The cells of the other
                columns are never kept, so a wide file costs little memory.
            name: Name of the new Daf.

        Returns:
            The new Daf.

        Raises:
            RuntimeError: The file cannot be read or parsed.

        Examples:
            >>> import os, tempfile
            >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
            >>> with open(path, 'w') as f:
            ...     n = f.write('id,v\n1,a\n')
            >>> Daf.from_csv_file(path)
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """
        """ Read a csv file directly into a daf array in memory, per arguments.

            Args:
                filepath: str,                          # The CSV filename
                keyfield: str='',                       # field to use as unique key, if not ''
                dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
                noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
                user_format: bool=False,                # if True, preprocess the file and omit comment lines.
                sep: str=',',                           # field separator.
                unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
                include_cols: Optional[T_ls]=None,      # include only the columns specified. noheader must be false.
                name: str = '',                         # name attribute of the Daf array created.
            Returns
                New daf instance

        """

        # following invalidates kd for lazy rebuilding.
        return Daf.from_csv(
            filepath,                               # The CSV file.
            keyfield    = keyfield,                 # field to use as unique key, if not ''
            dtypes      = dtypes,                   # dictionary of types to apply if set.
            noheader    = noheader,                 # if True, do not try to initialize columns in header dict.
            user_format = user_format,              # if True, preprocess the file and omit comment lines.
            sep         = sep,                      # field separator.
            unflatten   = unflatten,                # unflatten fields that are defined as dict or list.
            include_cols    = include_cols,         # include only the columns specified. noheader must be false.
            name        = name,                     # name attribute of the Daf array created.
            )


    def to_csv_file(
            self,
            file_path:          str | Path = '',
            line_terminator:    Optional[str]=None,
            include_header:     bool=True,
            #append_if_exists:   bool=False,
            ) -> str:
        r"""
        Write the Daf to a CSV file.

        The file has a header row of column names unless `include_header` is False.
        Each cell is written as text, so lists and dicts are written in their `str()`
        form. A NULL cell is written as nothing. The default line ending is `\r\n`.

        Args:
            file_path: Where to write the file.
            line_terminator: The line ending. If None, `\r\n` is used.
            include_header: If True, write the column names first.

        Returns:
            The path that was written.

        Examples:
            >>> import os, tempfile
            >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
            >>> d.to_csv_file(path) == path
            True
        """
        if isinstance(file_path, Path):  # Convert Path object to string
            file_path = str(file_path)

        buff = self.to_csv_buff(
                line_terminator=line_terminator,
                include_header=include_header,
                )

        type(self).buff_to_file(buff, file_path=file_path, fmt='.csv') #, append=append_mode)

        return file_path


    def to_csv_buff(
            self,
            line_terminator: Optional[str]=None,
            include_header: bool=True,
            ) -> T_buff:
        r"""
        Make CSV text from the Daf.

        The text can be saved to a file or uploaded. There is no need to call
        `flatten()` first. Each cell is written with `str()`, so a list or dict is
        written as its Python text, with single quotes. The bool `True` is written as
        `True`. A NULL cell is written as nothing. The text is not JSON.

        A dict whose keys are not text, such as `{1: 'a'}`, is written with its keys as they are.

        Args:
            line_terminator: The line ending. If None, `\r\n` is used.
            include_header: If True, write the column names first.

        Returns:
            The CSV text.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
            >>> d.to_csv_buff(line_terminator='\n')
            'id,v\n1,a\n'
            >>> Daf(lol=[[{1: 'a'}]], cols=['x']).to_csv_buff(line_terminator='\n')
            "x\n{1: 'a'}\n"
        """
        """ this function writes the daf array to a csv buffer, including the header if include_header==True.
            The buffer can be saved to a local file or uploaded to a storage service like s3.

            There is no need to call .flatten() first. This function flattens any nested
                    objects to PYON, according to __repr__ for that object.
                This differs from pure JSON format because:
                    1. it uses single-quotes instead of double quotes around strings.
                    2. it allows non-str keys in dicts.
                    3. it encodes True/False as 'True'/'False' instead of 'true'/'false'

        """

        if line_terminator is None:
            line_terminator = '\r\n'

        f = io.StringIO(newline = '')           # Use newline='' to ensure consistent line endings

        csv_writer = csv.writer(f, lineterminator=line_terminator)
        if include_header:
            csv_writer.writerow(self.columns())     # Write the header row
        csv_writer.writerows(self.lol)              # Write the data rows

        buff = f.getvalue()
        f.close()

        return buff


    @staticmethod
    def buff_to_file(buff: T_buff, file_path: str | Path, fmt:str='.csv') -> str:
        r"""
        Write text or bytes to a file.

        This is a static method, so call it as `Daf.buff_to_file(buff, path)`. It is
        the helper that `to_csv_file()` uses.

        Args:
            buff: The text or bytes to write.
            file_path: Where to write.
            fmt: The file format, such as `.csv`.

        Returns:
            The path that was written.

        Examples:
            >>> import os, tempfile
            >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
            >>> Daf.buff_to_file('id,v\n1,a\n', path) == path
            True
            >>> open(path).read()
            'id,v\n1,a\n'
        """
        # write_buff_to_fp() does str-only operations on file_path (.startswith('s3'), a regex
        # substitution) that would raise AttributeError on a real Path object -- str() it here
        # rather than widening write_buff_to_fp() to silently accept something it can't handle.
        return daf_utils.write_buff_to_fp(buff, str(file_path), fmt=fmt)

    #==== Directory capture

    @classmethod
    def from_directory(
            cls,
            source: str | Path,
            schema: type | None = None,
            recursive: bool = True,
            file_pat: str | None = None,
            include_dirs: bool = False,
        ) -> 'Daf':
        r"""
        Make a Daf that lists the files in a folder.

        Each row describes one file. The columns are the path, the folder, the name,
        the name without its extension, the extension, the size in bytes, and the
        modified and changed times. On Linux and macOS the `ctime` is the time of the last
        change to the file's metadata, such as its permissions, not the time it was
        created. Paths use `/` on every system.

        With `include_dirs=True`, the folders are listed too, each before the files in the
        same folder. A folder has an `is_dir` of 1, a size of 0 and no extension. Without
        it, no folder is listed and `is_dir` is always 0.

        The `schema` chooses the columns of the result. A `@schemaclass` that lists
        only some of the fields above keeps only those. Other columns of the schema
        get their defaults. The new Daf has no keyfield. Files that cannot be read
        are skipped.

        Only local folders are supported.

        Args:
            source: The folder to list.
            schema: A `@schemaclass` that chooses the columns. If None, the built in one is used.
            recursive: If True, list files in all sub folders. Otherwise only the folder itself.
            file_pat: A regular expression. Only names that match it are listed. Case is ignored.
            include_dirs: If True, list the folders as well as the files.

        Returns:
            The new Daf.

        Examples:
            >>> import os, tempfile
            >>> folder = tempfile.mkdtemp()
            >>> os.mkdir(os.path.join(folder, 'sub'))
            >>> for name in ('b.txt', 'a.csv', os.path.join('sub', 'c.csv')):
            ...     with open(os.path.join(folder, name), 'w') as f:
            ...         n = f.write('hello')
            >>> d = Daf.from_directory(folder)
            >>> sorted(d.col('basename')), d.col('size')
            (['a.csv', 'b.txt', 'c.csv'], [5, 5, 5])
            >>> Daf.from_directory(folder, recursive=False).col('basename')
            ['b.txt', 'a.csv']
            >>> sorted(Daf.from_directory(folder, file_pat=r'\.csv$').col('basename'))
            ['a.csv', 'c.csv']
            >>> sorted(Daf.from_directory(folder, include_dirs=True).col('basename'))
            ['a.csv', 'b.txt', 'c.csv', 'sub']
        """

        import os
        import re

        from daffodil.lib.schemaclass import schemaclass, SchemaBase

        #=========================
        # default schema
        #=========================

        if schema is None:

            @schemaclass
            class FilesystemSchema(SchemaBase):

                filepath:      str = ''
                dirpath:       str = ''
                basename:      str = ''
                rootname:      str = ''
                extension:     str = ''

                size:          int = 0

                mtime:         float = 0.0
                ctime:         float = 0.0

                is_dir:        int = 0

            schema = FilesystemSchema

        #=========================
        # initialize daf
        #=========================

        daf_obj = cls(schema=schema)

        #=========================
        # helper
        #=========================

        def normalize_path(path: str) -> str:

            return path.replace('\\', '/')

        #=========================
        # traversal
        #=========================

        walker: Iterable[Tuple[str, List[str], List[str]]]
        if recursive:

            walker = os.walk(str(source))

        else:

            full_source = str(source)

            names_ls = os.listdir(full_source)
            dirname_ls = [name for name in names_ls if os.path.isdir(os.path.join(full_source, name))]
            dirname_set = set(dirname_ls)
            basename_ls = [name for name in names_ls if name not in dirname_set]

            walker = [
                (
                    full_source,
                    dirname_ls,
                    basename_ls,
                )
            ]

        #=========================
        # process files
        #=========================

        for dirpath, dirname_ls, basename_ls in walker:

            dirpath = normalize_path(dirpath)

            entries_lot: List[Tuple[str, int]] = [(name, 1) for name in dirname_ls] if include_dirs else []
            entries_lot += [(name, 0) for name in basename_ls]

            for basename, is_dir in entries_lot:

                if file_pat:

                    if not re.search(file_pat, basename, flags=re.I):

                        continue

                filepath = normalize_path(
                    os.path.join(dirpath, basename)
                )

                try:

                    stat_result = os.stat(filepath)

                except Exception:

                    continue

                if is_dir:
                    rootname, extension = basename, ''
                else:
                    rootname, extension = os.path.splitext(basename)

                row_da = daf_obj.default_record()

                row_da['filepath']  = filepath
                row_da['dirpath']   = dirpath
                row_da['basename']  = basename
                row_da['rootname']  = rootname
                row_da['extension'] = extension

                row_da['size']      = 0 if is_dir else stat_result.st_size

                row_da['mtime']     = stat_result.st_mtime
                row_da['ctime']     = stat_result.st_ctime

                row_da['is_dir']    = is_dir

                daf_obj.append(row_da)

        return daf_obj

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
    @classmethod
    def from_numpy(cls, npa: Any, keyfield:str='', cols:Optional[T_la]=None, name:str='') -> 'Daf':
        """
        Make a Daf from a NumPy array.

        The values become plain Python values. A two dimensional array gives one row
        for each row of the array. A one dimensional array gives a single row.

        NumPy arrays hold one type. If the array mixes numbers and text, NumPy has
        already turned everything into text before this method sees it.

        Args:
            npa: The NumPy array.
            keyfield: Column, or tuple or list of columns, whose values identify rows.
            cols: Column names.
            name: Name of the new Daf.

        Returns:
            The new Daf.

        Examples:
            >>> import numpy as np
            >>> Daf.from_numpy(np.array([[1, 2], [3, 4]]), cols=['a', 'b'])
            | a | b |
            | -: | -: |
            | 1 | 2 |
            | 3 | 4 |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        """
        Convert a Numpy dataframe to daf object
        The resulting Python list will contain Python native types, not NumPy types.

        Numpy arrays are homogeneous, meaning all elements in a numpy array
        must have the same data type. If you attempt to create a numpy array
        with elements of different data types, numpy will automatically cast
        them to a single data type that can accommodate all elements. This can
        lead to loss of information if the original data types are different.
        For example, if you try to create a numpy array with both integers and
        strings, numpy will cast all elements to a common data type, such as Unicode strings.

        """
        if npa.ndim == 1:
            lol = [npa.tolist()]
        else:
            lol = npa.tolist()

        # following invalidates kd for lazy rebuilding.
        return cls(cols=cols, lol=lol, keyfield=keyfield, name=name)


    def to_numpy(self) -> T_npa:
        """
        Make a NumPy array of the rows.

        Column names, the keyfield and any dtypes are not kept. NumPy picks one type
        for the whole array. If the Daf mixes numbers and text, every value becomes
        text.

        Returns:
            The NumPy array.

        Examples:
            >>> Daf(lol=[[1, 2.5]], cols=['a', 'b']).to_numpy().tolist()
            [[1.0, 2.5]]
        """
        """
        Convert the core array of a Daf object to numpy.
        Note: does not convert any column names if they exist.
        Keyfield lookups are lost, if they are defined.

        When you create a NumPy array with a specific data type
        (e.g., int32, float64), NumPy will attempt to coerce or
        cast the elements to the specified data type. The rules
        for type casting follow a hierarchy where more general
        types are converted to more specific types.

        examples:
           if dtype is np.int32 and there are some float values, they will be truncated.
           if dtype is np.int32 and there are some string values, they will be converted to an integer if possible.
           if dtype is float64 and there are some integer values, they will be csst to float64 type.
           if casting is not possible, it will raise an error.

        """

        import numpy as np
        return np.array(self.lol)

    def to_donpa(self, colnames: Optional[T_ls]=None, default: Any = _MISSING) -> T_donpa:
        """
        Make a dict of NumPy arrays, one array for each column.

        This is a light form of a DataFrame. The arrays can be used in vector
        operations, such as `donpa['D_pct'] = donpa['D_votes'] / donpa['RV_total']`.
        Each column gets its own type, so numbers and text can be mixed.

        Args:
            colnames: The columns to include. If None, all columns.
            default: A value that replaces each NULL, None and NaN cell, in the arrays only.
                The Daf is not changed. It applies to every column in `colnames`, so
                give only the numeric columns, or a text column gets it too.

        Returns:
            A dict that maps each column name to an array.

        Notes:
            A column that mixes numbers and NULL becomes a text array, because NumPy
            has one type for an array. Give a numeric `default` to keep it numeric.
            A column with a `default` is also faster to make than one with blanks and
            no default, because the array is built from numbers.

        Raises:
            ColumnNotFoundError: A name in `colnames` is not a column. This is a `KeyError`,
                and also a `RuntimeError`.

        Examples:
            >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_donpa(['id'])['id'].tolist()
            [1, 2]
            >>> d = Daf(lol=[[1, 5], [2, ''], [None, 7]], cols=['n', 'm'])
            >>> d.to_donpa(['m'])['m'].tolist()
            ['5', '', '7']
            >>> d.to_donpa(['n', 'm'], default=0)['m'].tolist()
            [5, 0, 7]
            >>> d
            |  n   | m |
            | ---: | -: |
            |    1 | 5 |
            |    2 |   |
            | None | 7 |
            %% daf rows=3; cols=2; keyfield=''; name=''
        """
        """
        Convert specified columns of the Daffodil table to a dict of NumPy arrays (donpa).

        Parameters:
            colnames:
                List of column names to include. If None, include all columns.
            default:
                Optional replacement for missing values ('' or None).
                If specified, all missing values in the selected columns will be replaced
                with this value before conversion to NumPy arrays.

        Returns:
            Dict[str, np.ndarray], where each array corresponds to a 1D column vector.

        """
        if colnames is None:
            colnames = self.columns()

        import numpy as np

        if default is _MISSING:
            return {col: np.array(self.col(col)) for col in colnames}

        # replace the missing cells as the column is read, so that NumPy sees only numbers.
        donpa = {}
        for col in colnames:
            donpa[col] = np.array([default if (val is NULL or val is None or val != val) else val
                                    for val in self.col(col)])

        return donpa


    #==== Googlesheets

    @classmethod
    def from_googlesheet(cls, spreadsheet_id: str, sheetname: str = 'Sheet1', *, service_account_file: str) -> 'Daf':
        """
        Read a Google Sheet into a Daf. This is not implemented yet.

        It raises `NotImplementedError`. The method is a placeholder for the interface, which
        takes the path of a service account file. An earlier draft read the sheet with the
        Google API, but it could not be tested here, and it had a placeholder path in its source.
        The draft is in the history of the repository, in the commit e9e69fd.

        Args:
            spreadsheet_id: The ID of the Google Sheet.
            sheetname: The name of the sheet.
            service_account_file: The path of the Google service account file with the credentials.

        Raises:
            NotImplementedError: Always.

        Examples:
            >>> Daf.from_googlesheet('some-id', service_account_file='key.json')
            Traceback (most recent call last):
                ...
            NotImplementedError: from_googlesheet() is not implemented yet.
        """
        raise NotImplementedError("from_googlesheet() is not implemented yet.")

    def to_googlesheet(self, spreadsheet_id: str, sheetname: str = 'Sheet1', *, service_account_file: str) -> 'Daf':
        """
        Write the Daf to a Google Sheet. This is not implemented yet.

        It raises `NotImplementedError`. The method is a placeholder for the interface, which
        takes the path of a service account file. An earlier draft wrote the rows with the Google
        API, but it could not be tested here, and it had a placeholder path in its source. The draft
        is in the history of the repository, in the commit e9e69fd.

        Args:
            spreadsheet_id: The ID of the Google Sheet.
            sheetname: The name of the sheet.
            service_account_file: The path of the Google service account file with the credentials.

        Raises:
            NotImplementedError: Always.

        Examples:
            >>> Daf(cols=['x'], lol=[[1]]).to_googlesheet('some-id', service_account_file='key.json')
            Traceback (most recent call last):
                ...
            NotImplementedError: to_googlesheet() is not implemented yet.
        """
        raise NotImplementedError("to_googlesheet() is not implemented yet.")

    #===========================
    # JSON compatible representation.

    # there are some limitations to JSON encoding, as it does not correctly allow
    # for non-str keys of dictionaries. The kd can be corrected if the keyfield is
    # defined and there is a dtypes that defines the type of that column, so it can
    # be corrected if it is not a str. Also, the dtypes dict cannot contain values
    # that are types.

    def to_json(self, concise: bool=True) -> str:
        """
        Make JSON text that holds the whole Daf.

        The text holds the rows, the column names, the dtypes, the keyfield, the name,
        the attrs and the display columns. With `concise=True`, the parts that are
        empty are left out. Use `from_json()` to read it back.

        Types are written by name. Only `int`, `float`, `str`, `bool`, `list` and `dict`
        are read back as types. Other names, such as `date`, come back as text.

        Cells must be JSON values. A tuple comes back as a list, and a set raises a
        `TypeError`. A NaN is written as `NaN`, which some JSON readers reject.

        Args:
            concise: If True, leave out the parts that are empty.

        Returns:
            The JSON text.

        Examples:
            >>> Daf(lol=[[1]], cols=['a']).to_json()
            '{"lol": [[1]], "hd": {"a": 0}}'
        """
        # Convert data types to string representations, rather than type objects.
        #   isinstance(v, type) -- Checks if the value is a Python type object (int, float, str, etc.).
        #   v.__name__          -- the type’s name as a string
        #   else str(v)         -- if the type is already a string, for example.
        
        dtypes_str = {
            k: (v.__name__ if isinstance(v, type) else str(v))
                for k, v in (self.dtypes or {}).items()
            }
        # Serialize Daf object to a JSON-compatible dictionary
        daf_dict = {
            'name':         self.name,
            'lol':          self.lol,
            'hd':           self.hd,
            'dtypes':       dtypes_str,
            'keyfield':     self.keyfield,
            'attrs':        self.attrs,
            'disp_cols':    self.disp_cols,
        }
        if concise:
            # strip out empty items.
            # change this if bool or int items are added.
            daf_dict = {k: v for k, v in daf_dict.items() if v}
        
        # Convert dictionary to JSON string
        return json.dumps(daf_dict)
        

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

    @classmethod
    def from_json(cls, json_str: str) -> 'Daf':
        """
        Make a Daf from JSON text made by `to_json()`.

        Args:
            json_str: The JSON text.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> Daf.from_json(d.to_json()) == d
            True
        """
        # Deserialize JSON string into a Daf object
        daf_dict = json.loads(json_str)

        # Convert string representations of data types back to actual types
        dtypes = {k: cls.TYPE_MAP.get(v, v) 
                    for k, v in daf_dict.get('dtypes', {}).items()
                }

        # following invalidates kd for lazy rebuilding.
        return cls(lol          = daf_dict.get('lol', []),
                    hd          = daf_dict.get('hd', {}),
                    dtypes      = dtypes,
                    keyfield    = daf_dict.get('keyfield', ''),
                    name        = daf_dict.get('name', ''),
                    attrs       = daf_dict.get('attrs', {}),
                    disp_cols   = daf_dict.get('disp_cols', []),
                    )

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

    def append(self,
            data_item:  Union['Daf', T_loda, T_da, T_la, KeyedList, None] = None,
            respect_kd: bool = False,
            *,
            lol:        Optional[T_lola] = None,
            la:         Optional[T_la] = None,
            ) -> 'Daf':
        """
        Add one row, or several, to the end of the Daf.

        This is the general way to add data. What it does depends on what you give it.
        Give exactly one of `data_item`, `lol` and `la`.

        A dict or a [KeyedList][daffodil.keyedlist.KeyedList] is one row. It is placed
        by column name, so its keys may be in any order. A missing key gets NULL. A key
        that is not a column is dropped.

        A list of values is one row, in column order. A short list is padded with NULL.
        A list with more values than there are columns raises `ValueError`. With no
        columns defined, the list is added as it is. A list of lists given as
        `data_item` is therefore one row whose cells are lists, which fits only if there
        are enough columns.

        To say what you mean, use the keywords. `lol` is several rows, each a list in
        column order, as in `append(lol=[[2, 'b'], [3, 'c']])`. `la` is one row, even if
        its items are lists, as in `append(la=[[2, 'b'], [3, 'c']])`. Each row of `lol`
        follows the length rule above.

        A list of dicts is several rows. See `extend()`.

        A Daf is several rows. See `concat()`. Its columns must match.

        An empty dict, list or Daf adds nothing.

        By default the keyfield is not checked, so a key that is already present is
        added again. This keeps appending fast. Pass `respect_kd=True` to replace the
        row that has the same key instead. That looks the key up on every call, so
        it costs more when you add many rows one at a time.

        A list you pass in is added as the row itself, not as a copy.

        None, an empty dict, an empty list and an empty Daf add nothing.

        Args:
            data_item: The row or rows to add.
            respect_kd: If True, replace the row that has the same key. If False, the default, add it.
            lol: Several rows, each a list of values in column order.
            la: One row, as a list of values in column order. Its items are not read as rows.

        Returns:
            This Daf, which has been changed.

        Raises:
            TypeError: More than one of `data_item`, `lol` and `la` is given, or `data_item` is not a supported type.
            ValueError: A list has more values than there are columns.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> d.append({'v': 'b', 'id': 2})
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> d.append([3, 'c'])
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            |  3 | c |
            %% daf rows=3; cols=2; keyfield='id'; name=''
            >>> d.append(lol=[[4, 'd'], [5, 'e']])
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
            |  3 | c |
            |  4 | d |
            |  5 | e |
            %% daf rows=5; cols=2; keyfield='id'; name=''
            >>> d.append({'id': 2, 'v': 'new'}, respect_kd=True)
            | id |  v  |
            | -: | --: |
            |  1 |   a |
            |  2 | new |
            |  3 |   c |
            |  4 |   d |
            |  5 |   e |
            %% daf rows=5; cols=2; keyfield='id'; name=''
            >>> e = Daf(lol=[[1, 'a']], cols=['id', 'v'])
            >>> e.append(None).append({}).append([])
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """

        """ general append method can handle appending one record as T_da or T_la, many records as T_loda or T_daf
            if the data item is None or empty, do not append.
        """
        # test exists in test_daf.py for all three cases

        diagnose = False

        if (data_item is not None) + (lol is not None) + (la is not None) > 1:
            raise TypeError("append(): give only one of data_item, lol and la.")

        if lol is not None:
            return self.extend(lol=lol, respect_kd=respect_kd)

        if la is not None:
            if not la:
                return self
            return self._append_row_la(la, respect_kd)

        if not data_item:
            return self

        if diagnose:
            start_time = time.time()
            logs.sts(f"{logs.prog_loc()} starting append.", 3)

        if isinstance(data_item, (dict, KeyedList)):
            self.record_append(data_item, respect_kd=respect_kd)

        elif isinstance(data_item, list):
            if isinstance(data_item[0], dict):
                # lod type
                self.extend(data_item, respect_kd=respect_kd)
            else:
                self._append_row_la(data_item, respect_kd)

        elif isinstance(data_item, Daf):  # type: ignore
            self.concat(data_item, respect_kd=respect_kd)

        else:
            raise TypeError(f"append(): data_item must be a dict, KeyedList, list or Daf, not {type(data_item).__name__}.")

        if diagnose:
            logs.sts(f"{logs.prog_loc()} append done. Elapsed: {(time.time() - start_time):.10f}", 3)

        return self


    def _append_row_la(self, row_la: T_la, respect_kd: bool) -> 'Daf':
        """
        Add one row that is given as a list of values in column order. Internal use.

        A list with more values than the columns raises `ValueError`. A short list is padded
        with NULL. With no columns defined, the list is added as it is.
        """
        if self.hd:
            if len(row_la) > len(self.hd):
                raise ValueError(f"append(): the list has {len(row_la)} values for {len(self.hd)} columns.")
            # columns are defined, and keyfield might also be defined
            # create a dict.
            da = dict(zip(self.hd.keys(), row_la))
            self.record_append(da, respect_kd=respect_kd)  # <-- this takes care of respecing the row kd (invalidating)
        else:
            # no columns defined, therefore just append to lol.
            self.lol.append(row_la)

        return self


    def concat(self, other_instance: 'Daf', respect_kd: bool=False) -> 'Daf':
        """
        Add the rows of another Daf to the end of this one.

        Use this to combine two tables with the same columns. By default every row is
        appended, so a key that exists in both tables appears twice afterwards. Pass
        `respect_kd=True` to get an upsert instead.

        Args:
            other_instance: Daf whose rows are added. It is not changed.
            respect_kd: If False, the default, append every row without looking at the keys. If True,
                replace the row that has the same key and append the rows with new keys.

        Returns:
            This Daf, which has been changed in place. The rows are deep copies.

        Raises:
            KeyError: The column names of the two Daf instances differ.
            ValueError: With `respect_kd=True`, both Daf instances have a keyfield and the
                keyfields differ.

        Notes:
            The columns must be equal, including their order.
            With `respect_kd=True`, this Daf's keyfield decides the key. If it has no
            keyfield, the rows are simply appended.
            A key repeated in the other Daf replaces the earlier row, so the last one wins.
            If the other Daf is empty, nothing happens.

        Examples:
            >>> a = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
            >>> b = Daf(lol=[[2, 'b2'], [3, 'b3']], cols=['id', 'v'])
            >>> a.concat(b)
            | id | v  |
            | -: | -: |
            |  1 | a1 |
            |  2 | a2 |
            |  2 | b2 |
            |  3 | b3 |
            %% daf rows=4; cols=2; keyfield='id'; name=''
            >>> a = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
            >>> a.concat(b, respect_kd=True)
            | id | v  |
            | -: | -: |
            |  1 | a1 |
            |  2 | b2 |
            |  3 | b3 |
            %% daf rows=3; cols=2; keyfield='id'; name=''
        """

        if not other_instance:
            return self

        if not self.lol and not self.hd:
            self.hd = copy.deepcopy(other_instance.hd)
            self.lol = copy.deepcopy(other_instance.lol)
            self.keyfield = other_instance.keyfield
            self._invalidate_kd()    # use lazy kd rebuilding

            return self

        # Fields must match exactly!
        if self.hd != other_instance.hd:
            _, missing_list, extra_list, _ = daf_utils.compare_lists(
                work_list=self.hd,
                ref_list=other_instance.hd,
                req_list=None,
            )

            error_str = (f"columns mismatch: this_instance ({self.name}):\n({list(self.hd.keys())}) \n"
                  f"other_instance:\n({list(other_instance.hd.keys())})\n"
                  f"missing_list:\n{missing_list}\n"
                  f"extra_list:\n{extra_list}\n")
            raise KeyError(error_str)

        if respect_kd and self.keyfield:
            if other_instance.keyfield and other_instance.keyfield != self.keyfield:
                raise ValueError(f"concat: keyfield mismatch: {self.keyfield!r} versus {other_instance.keyfield!r}")

            self._rebuild_kd_if_invalidated()
            kd = self._kd
            lol = self.lol
            if isinstance(self.keyfield, (str, int)):
                key_idx = self.hd[cast(str, self.keyfield)]
                for rec_la in other_instance.lol:
                    rec_la = copy.deepcopy(rec_la)
                    keyval = rec_la[key_idx]
                    if keyval in kd:
                        lol[kd[keyval]] = rec_la
                    else:
                        kd[keyval] = len(lol)
                        lol.append(rec_la)
            else:
                key_idxs = [self.hd[cast(str, key)] for key in self.keyfield]
                for rec_la in other_instance.lol:
                    rec_la = copy.deepcopy(rec_la)
                    keyval = tuple(rec_la[i] for i in key_idxs)
                    if keyval in kd:
                        lol[kd[keyval]] = rec_la  # type: ignore[index]
                    else:
                        kd[keyval] = len(lol)  # type: ignore[index]  # composite key is a tuple
                        lol.append(rec_la)
            return self

        # Append deep-copied rows to avoid referencing issues
        for rec_la in other_instance.lol:
            self.lol.append(copy.deepcopy(rec_la))

        self._invalidate_kd()    # use lazy kd rebuilding

        return self

    def extend(self,
            records_lod:    Optional[T_loda] = None,
            respect_kd:     bool = False,
            *,
            lol:            Optional[T_lola] = None,
            ) -> 'Daf':
        """
        Append several records, given as a list of dicts or as a list of lists.

        Give a list of dicts as `records_lod`. Each dict is a row, placed by column name,
        as in `append()`. A Daf with no columns takes them from the first dict. An empty
        list, or a list that holds one empty dict, adds nothing.

        Give a list of lists as `lol`, as in `extend(lol=[[2, 'b'], [3, 'c']])`. Each list is a row, in
        column order. A short list is padded with NULL. A list with more values than there
        are columns raises `ValueError`, and then no row is added. A Daf with no columns
        takes the lists as they are. The keyword says that these are several rows, and not
        one row whose cells are lists. For that, see `append(la=...)`.

        Without `respect_kd`, a key that already exists is added again. Use
        `respect_kd=True` when the keyfield must stay unique.

        Args:
            records_lod: The records, as dicts.
            respect_kd: If True, replace the row that has the same key. If False, the default, add it.
            lol: The rows, as lists of values in column order.

        Returns:
            This Daf, which has been changed.

        Raises:
            TypeError: Both `records_lod` and `lol` are given, or a row of `lol` is not a list.
            ValueError: A row of `lol` has more values than there are columns.

        Examples:
            >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
            >>> d.extend([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'new'}], respect_kd=True)
            | id |  v  |
            | -: | --: |
            |  1 | new |
            |  2 |   b |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> d.extend(lol=[[3, 'c'], [4]])
            | id |  v  |
            | -: | --: |
            |  1 | new |
            |  2 |   b |
            |  3 |   c |
            |  4 |     |
            %% daf rows=4; cols=2; keyfield='id'; name=''
        """

        if records_lod is not None and lol is not None:
            raise TypeError("extend(): give only one of records_lod and lol.")

        if lol is not None:
            return self._extend_lol(lol, respect_kd)

        if not records_lod or len(records_lod) == 1 and not records_lod[0]:
            # edge case of one record which is empty.
            return self

        if not self.lol and not self.hd:
            # new daf, adopt structure of lod.
            # but there is only one header and data is lol
            # this saves space.
            self.hd = {col_name: index for index, col_name in enumerate(records_lod[0].keys())}
            self.lol = [list(record_da.values()) for record_da in records_lod]
            self._invalidate_kd() # use lazy kd building.
            return self

        for record_da in records_lod:
            self.record_append(record_da, respect_kd=respect_kd)

        if not respect_kd:
            self._invalidate_kd() # use lazy kd building.

        return self


    def _extend_lol(self, lol: T_lola, respect_kd: bool) -> 'Daf':
        """
        Add several rows that are lists of values in column order. Internal use.

        Every row is checked before any row is added. A short row is padded with NULL.
        """
        if not lol:
            return self

        num_cols = len(self.hd)

        for row_la in lol:
            if not isinstance(row_la, list):
                raise TypeError(f"extend(): each row of lol must be a list, not {type(row_la).__name__}.")
            if num_cols and len(row_la) > num_cols:
                raise ValueError(f"extend(): a row has {len(row_la)} values for {num_cols} columns.")

        if respect_kd and self.keyfield and num_cols:
            for row_la in lol:
                self.record_append(dict(zip(self.hd, row_la)), respect_kd=True)
            return self

        if num_cols:
            lol = [row_la + [NULL] * (num_cols - len(row_la)) if len(row_la) < num_cols else row_la for row_la in lol]

        self.lol.extend(lol)
        self._invalidate_kd()       # use lazy kd building.

        return self


    def record_append(self, record: Union[T_da, KeyedList], respect_kd: bool=True) -> 'Daf':
        """
        Add one row that is given as a dict or a KeyedList.

        This is the single row case of `append()`. The row is placed by column name,
        a missing key gets NULL, and a key that is not a column is dropped. An empty
        Daf takes its columns from the first row. An empty record adds nothing.

        With a keyfield, the row that has the same key is replaced, and a new key is
        added at the end. This is the default here, unlike `append()`. Pass
        `respect_kd=False` to always add. The key index is kept up to date, so adding
        many rows one at a time stays fast.

        A plain list is not accepted. Use `append()` for that.

        Args:
            record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
            respect_kd: If True, the default, replace the row that has the same key. Otherwise add it.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> d.record_append({'id': 2, 'v': 'new'})
            | id |  v  |
            | -: | --: |
            |  1 |   a |
            |  2 | new |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> d.record_append({'id': 2, 'v': 'dup'}, respect_kd=False)
            | id |  v  |
            | -: | --: |
            |  1 |   a |
            |  2 | new |
            |  2 | dup |
            %% daf rows=3; cols=2; keyfield='id'; name=''
        """
            # test exists in test_daf.py

        if not record:
            return self

        if not self.lol and not self.hd:
            # new daf, adopt structure of da or keyedlist.

            if isinstance(record, KeyedList):
                # for keyedlist, simply adopt the hd and the list as first row.
                # self.hd is a real dict everywhere else in this class -- record.hd is a
                # KeyedIndex; .to_dict() gives the equivalent {key: position} dict.
                self.hd = cast(Dict[str, int], record.hd.to_dict())
                self.lol = [record.values()]

            elif isinstance(record, dict):
                self.hd = type(self)._build_hd(record.keys())
                self.lol = [list(record.values())]

            self._invalidate_kd()    # use lazy kd rebuilding
            # self._rebuild_kd()   # functions only if the keyfield is set.
            return self

        # check if fields match exactly.
        reorder = False
        # if isinstance(record, KeyedList):
        #     if record.hd != self.hd:
        #         reorder = True
        # el
        if list(self.hd.keys()) != list(record.keys()):
            reorder = True

        if reorder:
            # construct a dict with exactly the cols specified.
            # defaults to '' at this point.
            # this works for both dict and KeyedList
            rec_la = [record.get(col, '') for col in self.hd]
        # not reordering, slightly different
        elif isinstance(record, KeyedList):
            rec_la = record.values()        # returns a list.
        elif isinstance(record, dict):
            rec_la = list(record.values())  # must copy into a list.
        else:
            # any other mapping with keys in column order.
            rec_la = [record[col] for col in self.hd]

        if self.keyfield and respect_kd:
            keyval = self._get_keyval(record)

            self._rebuild_kd_if_invalidated()

            if keyval in self._kd:
                self.lol[self._kd[keyval]] = rec_la
            else:
                self.lol.append(rec_la)
                if isinstance(self._kd, dict):
                    self._kd[keyval] = len(self.lol) - 1
                else:
                    self._kd.append(keyval)
        else:
            # no keyfield is set, or not respect_kd, just append to the end.
            self.lol.append(rec_la)
            self._invalidate_kd()    # use lazy kd rebuilding only if keyfield != ''

        return self



    def _basic_append(self, row: Union[KeyedList, Dict[Any, Any], list]) -> 'Daf':
        # --- list ---
        if isinstance(row, list):
            if self.hd:
                assert len(row) == len(self.hd)
            self.lol.append(row)
            return self

        # --- KeyedList ---
        if isinstance(row, KeyedList):
            if not self.hd:
                if _use_keyedindex_for_hd:
                    # _use_keyedindex_for_hd is currently False -- self.hd staying a real dict
                    # (Dict[str, int]) everywhere else in this class is still the real contract;
                    # this branch is scaffolding for a not-yet-completed migration, not live code.
                    self.hd = row.hd  # type: ignore[assignment]  # already KeyedIndex
                else:
                    # equivalent to (and simpler/faster than) dict(zip(row.hd, range(len(row.hd))))
                    # -- KeyedIndex.to_dict() already returns exactly this {key: position} dict.
                    self.hd = cast(Dict[str, int], row.hd.to_dict())
            self.lol.append(row._values)
            return self

        # --- dict ---
        if isinstance(row, dict):
            if not self.hd:
                keys = list(row.keys())

                if _use_keyedindex_for_hd:
                    # see the KeyedList branch above -- same not-yet-completed migration flag.
                    self.hd = KeyedIndex(keys)  # type: ignore[assignment]
                else:
                    self.hd = dict(zip(keys, range(len(keys))))

                self.lol.append(list(row.values()))
                return self

            # hd exists → align order
            self.lol.append([row.get(col, '') for col in self.hd])
            return self

        raise TypeError("Unsupported row type for basic_append")


    #=========================
    # remove records per keyfield; drop cols

    def remove_key(self, keyval: Optional[Union[str, int, T_la, T_ta]], silent_error: bool=False) -> 'Daf':
        """
        Make a new Daf without the row that has the given key.

        This is deprecated. Use `select_krows(key, inverse=True)`, which does the same.

        This does not remove the row from this Daf. It returns a new Daf that leaves
        the row out, and this Daf is unchanged. Keep the result, as in
        `d = d.remove_key(2)`.

        The new Daf shares the surviving rows with this Daf. Changing a cell in one
        changes it in the other. Call `copy()` with `level='editable'` if you need rows that
        are independent.

        A tuple is a range of keys, as in `remove_key((2, 3))`. With a composite keyfield, a
        tuple as long as the keyfield, with no tuples inside it, is one key, as in
        `remove_key(('a', 1))`. A tuple of key tuples is a range, and a list of key tuples
        is a list of keys.

        Args:
            keyval: The key of the row to leave out.
            silent_error: If True, a key that is not found is ignored.

        Returns:
            The new Daf. It has the same keyfield.

        Raises:
            KeysDisabledError: The Daf has no keyfield and no key index.
            KeyError: The key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
            >>> d.remove_key(1)
            | id | v |
            | -: | -: |
            |  2 | b |
            %% daf rows=1; cols=2; keyfield='id'; name=''
            >>> d.num_rows()
            2
            >>> c = Daf(lol=[['a', 1], ['a', 2]], cols=['g', 'n'], keyfield=('g', 'n'))
            >>> c.remove_key(('a', 1))
            | g | n |
            | -: | -: |
            | a | 2 |
            %% daf rows=1; cols=2; keyfield='('g', 'n')'; name=''
        """

        # test exists in test_daf.py
        if not self.keyfield and not self._kd:
            raise self._no_keys_error('remove_key')

        if (isinstance(keyval, tuple) and isinstance(self.keyfield, (tuple, list))
                and len(keyval) == len(self.keyfield)
                and not any(isinstance(item, (tuple, list)) for item in keyval)):
            keyval = [keyval]       # a composite key, not a range.

        return self.select_krows(krows=keyval, inverse=True, silent_error=silent_error)


    def remove_keylist(self, keylist: T_ls, silent_error: bool=False) -> 'Daf':
        """
        Make a new Daf without the rows that have the given keys.

        This is deprecated. Use `select_krows(keys, inverse=True)`, which does the same.

        This does not remove the rows from this Daf. It returns a new Daf that leaves
        them out, and this Daf is unchanged. See `remove_key()` for how the rows are
        shared.

        Args:
            keylist: The keys of the rows to leave out.
            silent_error: If True, keys that are not found are ignored.

        Returns:
            The new Daf. It has the same keyfield.

        Raises:
            KeysDisabledError: The Daf has no keyfield and no key index.
            KeyError: A key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')
            >>> d.remove_keylist([1, 3])
            | id | v |
            | -: | -: |
            |  2 | b |
            %% daf rows=1; cols=2; keyfield='id'; name=''
            >>> d2 = Daf(lol=[[1, 'a', 0], [2, 'b', 1]], cols=['p', 'q', 'r'], keyfield=['p', 'q'])
            >>> d2.remove_keylist([(1, 'a')])
            | p | q | r |
            | -: | -: | -: |
            | 2 | b | 1 |
            %% daf rows=1; cols=3; keyfield='['p', 'q']'; name=''
        """
        # test exists in test_daf.py

        if not self.keyfield and not self._kd:
            raise self._no_keys_error('remove_keylist')

        return self.select_krows(krows=keylist, inverse=True, silent_error=silent_error)


    #===========================
    # indexing

    def __getitem__(self,
            slice_spec:   Union[slice, int, str, T_li, T_ls, range, T_lor,
                                Tuple[  Union[slice, int, str, T_li, T_ls, range, T_lor, Tuple[str, str]],
                                        Union[slice, int, str, T_li, T_ls, range, T_lor, Tuple[str, str]]]],
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
        makes new rows, so the result is independent. The keyfield and dtypes carry
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
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d[1]
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
            >>> d[1:]
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d[:, 'v']
            | v |
            | -: |
            | a |
            | b |
            | c |
            %% daf rows=3; cols=1; keyfield=''; name=''
            >>> d[[2, 0], ['n', 'id']]
            | n  | id |
            | -: | -: |
            | 30 |  3 |
            | 10 |  1 |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> d[1, 'n'].to_value()
            20
            >>> d[(1, 2), :]
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  2 | b | 20 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
        """
        irows, icols = self._parse_selectors(slice_spec)

        if icols is None:
            ret_daf = self.select_irows(irows=irows)
        else:
            ret_daf = self.select_irows(irows=irows).select_icols(icols=icols)

        return ret_daf._adjust_return_val(self.retmode)


    def __setitem__(self,
            slice_spec:   Union[slice, int, str, range, T_li, T_ls, T_lb, T_lor,
                                Tuple[  Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]],
                                        Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]]]],
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

        If you change a keyfield cell, the key index is rebuilt when it is next needed.

        Args:
            slice_spec: A row selector, or a tuple of a row selector and a column selector.
            value: The value, list, dict or Daf to assign.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d[1, 'v'] = 'z'
            >>> d[:, 'n'] = [1, 2, 3]
            >>> d
            | id | v | n |
            | -: | -: | -: |
            |  1 | a | 1 |
            |  2 | z | 2 |
            |  3 | c | 3 |
            %% daf rows=3; cols=3; keyfield='id'; name=''
            >>> d[0] = {'v': 'q'}
            >>> d.iloc(0)
            {'id': '', 'v': 'q', 'n': ''}
        """
        irows, icols = self._parse_selectors(slice_spec)
        return self.set_irows_icols(irows=irows, icols=icols, value=value)

    def _parse_selectors(
            self,
            slice_spec: Union[
                slice,
                int,
                str,
                range,
                T_li,
                T_ls,
                T_lb,
                T_lor,
                Tuple[
                    Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]],
                    Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]],
                ],
            ],
        ) -> Tuple[
            Union[int, slice, range, T_li],
            Union[int, slice, range, T_li] | None,
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
            irows: Union[int, slice, range, T_li] = list(range(len(self)))

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
            irows = cast(Union[int, slice, range, T_li], row_spec)

        else:
            raise TypeError(f"Invalid row selector: {row_spec}")

        # --- COLS ---
        if not col_provided or col_spec == slice(None):
            icols: Optional[Union[int, slice, range, T_li]] = None

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
            icols = cast(Union[int, slice, range, T_li], col_spec)

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


    def set_irows_icols(self,
            irows: Union[slice, int, range, T_li, Iterable, None],
            icols: Union[slice, int, range, T_li, None],
            value: Any) -> 'Daf':
        """
        Set values at row positions and column positions, in place.

        This is what `my_daf[rows, cols] = value` calls, after the selectors are turned
        into positions. `irows` and `icols` are an integer, a slice, a range or a list.
        If `icols` is None, the whole row is set. If `irows` is None, nothing is
        set. Use `slice(None)` for all rows.

        What is set depends on the value.

            a single value    fills every selected cell. A str or bytes is a single value.
            a list            fills a selection in order. For several whole rows, each
                              row becomes a copy of the list.
            a dict            sets a row. The cells of columns that the dict lacks become NULL.
            a Daf             is copied as a block, from its top left corner.

        Nothing is checked, and no error is raised for a size mismatch. A smaller source
        fills the top left of the selection and leaves the rest. A larger one fills the
        selection and the extra values are ignored. Check the sizes first if it matters.

        Args:
            irows: The row positions.
            icols: The column positions.
            value: The value, list, dict or Daf to set.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'])
            >>> d.set_irows_icols([0, 1], [1, 2], 'Z')
            | id | v | n  |
            | -: | -: | -: |
            |  1 | Z |  Z |
            |  2 | Z |  Z |
            |  3 | c | 30 |
            %% daf rows=3; cols=3; keyfield=''; name=''
            >>> d.set_irows_icols(2, None, ['x', 'y', 'z']).iloc(2)
            {'id': 'x', 'v': 'y', 'n': 'z'}
        """
        """ set rows and cols in given daf.

            irows, icols: can be either a slice, int, or list of integers. These
                            refer to row/col indices that are inherent in the lol structure.

            mutates existing daf
        """
        if icols is None:
            icols = []
        if irows is None:
            irows = []

        tot_num_cols = self.num_cols()
        tot_num_rows = self.num_rows()

        # irows accepts a bare Iterable per this method's own signature, but len_rowcol_spec
        # only actually measures slice/int/range/list (silently returns 0 for anything else,
        # e.g. a generator) -- in every real caller, irows/icols is one of those four by here.
        num_irows = daf_utils.len_rowcol_spec(cast(Union[slice, int, range, T_li, None], irows), tot_num_rows)
        num_icols = daf_utils.len_rowcol_spec(icols, tot_num_cols)

        if num_irows == 1 and isinstance(irows, int):
            irows = [irows]
        if num_icols == 1 and isinstance(icols, int):
            icols = [icols]

        if isinstance(irows, slice):
            irows = daf_utils.slice_to_range(irows, len(self))
        if isinstance(icols, slice):
            icols = daf_utils.slice_to_range(icols, self.num_cols())

        # By this point every branch above has normalized irows/icols down to range or list[int]
        # only -- slice was just converted to range, None to [], and (unlike icols, which has no
        # equivalent conversion) a lone int irows/icols was already wrapped in a single-item list
        # a few lines up (num_irows/num_icols == 1 and isinstance(..., int)). The declared param
        # type is wider (Iterable, T_li, ...) to accept what callers may pass in, not what
        # remains once this normalization runs.
        irows = cast(Union[range, T_li], irows)
        icols = cast(Union[range, T_li], icols)

        # special case when cols not specified.
        if num_irows == 1 and num_icols == 0:

            irow = irows[0]

            if isinstance(value, list):
                self.lol[irow] = value
            elif isinstance(value, dict):
                self.assign_record_irow(irow, record=value)
            elif isinstance(value, type(self)):
                self.lol[irow] = list(value.lol[0])
            else:
                # set the same value in the row for all columns.
                self.lol[irow] = [value] * len(self.lol[irow])

        elif num_irows == 1 and num_icols == 1:

            irow = irows[0]
            icol = icols[0]

            if isinstance(value, dict):
                self.assign_record_irow(irow, record=value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 1:
                # frequently, we will have a list generated from a selection of a column, and it it has only one value
                # it needs to be entered in the array location, but not as a list.
                # place a list with only one item in a cell can't be done this way:

                # my_daf[0,0] = [4]   # this will insert 4 not [4]

                self.lol[irow][icol] = value[0]
            else:
                self.lol[irow][icol] = value

        elif num_irows > 1 and num_icols == 0:

            if isinstance(value, dict):
                for irow in irows:
                    self.assign_record_irow(irow, record=value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # a str or bytes is a single value, not a sequence of values.
                # each row gets its own copy, so the rows do not share one list.
                for irow in irows:
                    self.lol[irow] = list(value)
            elif isinstance(value, type(self)):
                for source_row, irow in zip(value.lol, irows):
                    self.lol[irow] = list(source_row)
            else:
                # set the same value in the row for all columns.
                for irow in irows:
                    self.lol[irow] = [value] * len(self.lol[irow])

        elif num_irows > 0 and num_icols == 1:

            icol = icols[0]

            if irows is None:
                irows = range(len(self.lol))

            if isinstance(value, dict):
                # this is the same as cols=0 bc dict updates the corresponding cols.
                for irow in irows:
                    self.assign_record_irow(irow, record=value)

            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for source_val, irow in zip(value, irows):
                    self.lol[irow][icol] = source_val

            elif isinstance(value, type(self)):
                for source_row, irow in zip(value.lol, irows):
                    self.lol[irow][icol] = source_row[0]

            else:
                # set the same value in the row for all selected columns.
                for irow in irows:
                    self.lol[irow][icol] = value
        else:
            if irows is None:
                irows = range(len(self.lol))

            if isinstance(value, dict):
                # this is the same as cols=0 bc dict updates the corresponding cols.
                for irow in irows:
                    self.assign_record_irow(irow, record=value)

            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # the same list of values is applied to each selected row.
                for irow in irows:
                    for source_val, icol in zip(value, icols):
                        self.lol[irow][icol] = source_val

            elif isinstance(value, type(self)):
                for source_row, irow in zip(value.lol, irows):
                    for source_val, icol in zip(source_row, icols):
                        self.lol[irow][icol] = source_val

            else:
                # set the same value in the row for all selected columns.
                for irow in irows:
                    for icol in icols:
                        self.lol[irow][icol] = value
        return self


    def krows_to_irows(self,
            krows:          Union[slice, str, T_la, int, Tuple[Any, Any], T_lota, Iterable, None],
            inverse:        bool = False,
            silent_error:   bool = False,
            ) -> Union[slice, int, T_li, range]:
        """
        Turn row keys into row positions.

        This is the step that lets `my_daf['r2']` work. Call it when you need the
        positions themselves. The Daf must have a keyfield, or a key index passed in
        as `kd`.

        The keys may be a key, a list of keys, or a tuple that gives an inclusive range
        of keys. A range is made from positions, so the keys must be in the same order
        as the rows.

        The first lookup builds the key index, by reading the keyfield column. In one test with
        200,000 rows that took 0.05 s, and a later lookup took about 2 microseconds. Adding or
        removing rows clears the index, and the next lookup builds it again.

        Args:
            krows: A key, a list of keys, or a tuple that gives a range of keys.
            inverse: If True, return the positions of the rows that are not selected.
            silent_error: If True, keys that are not found are ignored.

        Returns:
            The row positions.

        Raises:
            KeysDisabledError: There is no keyfield and no key index, or the keyfield is not a column.
            KeyError: A key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.krows_to_irows('r2')
            [1]
            >>> d.krows_to_irows(['r3', 'r1'])
            [2, 0]
            >>> d.krows_to_irows(('r1', 'r2'))
            slice(0, 2, 1)
            >>> d.krows_to_irows(['r1'], inverse=True)
            [1, 2]
            >>> d.krows_to_irows(['zz'], silent_error=True)
            []
            >>> d.krows_to_irows(['zz'])
            Traceback (most recent call last):
                ...
            KeyError: 'zz'
        """
        """
        If the keyfield is set, then the rows can be selected by providing a
        krows parameter that will index the rows by using values in the keyfield column.
        The keyfield column is read as a list and then converted to a dictionary that
        provides the indexes of the row for each value in that column. Lookups using
        this method are very fast but there is overhead to reading the column and
        creating the dictionary. Therefore, set keyfield to '' to disable row key lookups.
        
        raises KeysDisabledError if keyfield is not set.
        """
        if not self.keyfield and not self._kd:
            # if keyfield is unset, kd may still be initialized manually.

            # `raise` was missing here -- constructed but never raised, so this docstring's own
            # "raises KeysDisabledError if keyfield is not set" silently didn't happen; every
            # other identical message elsewhere in this file (e.g. select_krows()) does raise.
            raise self._no_keys_error('krows_to_irows')
            # if inverse:
                # return range(len(self))
            # else:
                # return []

        self._rebuild_kd_if_invalidated()    # Only rebuilds when keyfield is set and `_kd` is empty.

        if not self._kd and self.keyfield and not self._is_keyfield_valid():
            raise KeysDisabledError(self._keyfield_not_a_column_message())

        return type(self).gkeys_to_idxs(
                    keydict         = self._kd,
                    gkeys           = krows,
                    inverse         = inverse,
                    silent_error    = silent_error,
                    axis            = 'rowkeys',     # for error message only
                    name            = self.name,
                    )

    def kcols_to_icols(self,
            kcols: Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], Iterable, None] = None,
            inverse: bool = False,
            silent_error: bool=False,
            ) -> Union[slice, int, T_li, range, None]:
        """
        Turn column names into column positions.

        This is the step that lets `my_daf[:, 'v']` work. A name that looks like an
        integer is still read as a name.

        Args:
            kcols: A name, a list of names, or a tuple that gives an inclusive range of names.
            inverse: If True, return the positions of the columns that are not selected.
            silent_error: If True, names that are not found are ignored.

        Returns:
            The column positions. With no column names the result is empty, or all
            positions if `inverse` is True.

        Raises:
            KeyError: A name is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.kcols_to_icols('v')
            [1]
            >>> d.kcols_to_icols(['n', 'id'])
            [2, 0]
            >>> d.kcols_to_icols(('id', 'v'))
            slice(0, 2, 1)
            >>> d.kcols_to_icols('v', inverse=True)
            [0, 2]
            >>> d.kcols_to_icols('zz')
            Traceback (most recent call last):
                ...
            KeyError: 'zz'
        """
        """
        If cols are defined is set, then the cols can be selected by providing a
        kcols parameter that will index the cols by using values in the header dict hd.
        This forces an attempt to use the spec as a column name even if it may look
        like an integer.
        """
        if not self.hd:
            if inverse:
                return range(self.num_cols())
            else:
                return []

        return type(self).gkeys_to_idxs(
                    # self.hd's keys (Dict[str, int]) are a subset of what gkeys_to_idxs accepts
                    # (Dict[str|int, int]) -- dict is invariant in its key type for mypy, so this
                    # narrower-is-fine relationship needs a cast to type-check.
                    keydict         = cast(Dict[Union[str, int], int], self.hd),
                    gkeys           = kcols,
                    inverse         = inverse,
                    silent_error    = silent_error,
                    axis            = 'colnames',     # for error message only
                    name            = self.name,
                    )

    @staticmethod
    def gkeys_to_idxs(
            keydict:    Dict[Union[str, int], int],
            gkeys:      Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], T_lota, Iterable, None] = None,
            inverse:    bool = False,
            silent_error: bool=False,
            axis:       str='rowkeys',              # used for status messages only.
            name:       str='unspecified',          # used for status messages only.
            ) -> Union[slice, int, T_li, range]:
        """
        Turn keys into positions, using a dict of key to position.

        This is the step under `krows_to_irows()` and `kcols_to_icols()`. It is a
        static method. It is for internal use.

        A key gives a list with one position. A list of keys gives their positions in
        that order. A tuple of two keys is an inclusive range, and `None` in it means
        the start or the end. A tuple of one key means that key to the end. A slice is
        taken as positions. With `inverse`, the positions that are not selected are
        returned.

        Args:
            keydict: Maps each key to its position.
            gkeys: The keys.
            inverse: If True, return the positions that are not selected.
            silent_error: If True, keys that are not found are ignored.
            axis: A label for error messages.
            name: A label for error messages.

        Returns:
            The positions, as a list or a slice.

        Raises:
            KeysDisabledError: The dict is empty, as it is for a Daf with no rows.
            KeyError: A key is not found and `silent_error` is False.
            TypeError: The keys are None, or a tuple of a length other than 1 or 2.

        Examples:
            >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ['c', 'a'])
            [2, 0]
            >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ('a', 'b'))
            slice(0, 2, 1)
        """
        """
        If keydict is defined, then the idxs can be selected by providing a
        gkeys parameter that will index the keydict, and return either a
        slice, int, T_li, or None, which will index the range.
        """

        # --- no key system ---
        if not keydict:
            raise KeysDisabledError(f"gkeys_to_idxs(): the {axis} index is empty, so there is nothing to look up. A Daf with no rows has no keys.")

        # --- invalid selector ---
        if gkeys is None:
            raise TypeError("None is not a valid key selector")

        # --- empty selection ---
        idxs: Union[slice, T_li]
        if isinstance(gkeys, (list,dict,tuple)) and not gkeys:
            idxs = []

        elif isinstance(gkeys, (str, int)):
            idxs = []
            gkey = gkeys

            # For the following, see https://github.com/raylutz/daffodil/issues/6
            try:
                idxs.append(keydict[gkey])
            except KeyError:
                if not silent_error:
                    # logs.sts(f"{logs.prog_loc()} Cannot find key '{gkey}' in {axis} in dataframe '{name}'", 3)
                    raise

        elif isinstance(gkeys, tuple):

            n = len(gkeys)

            if n == 0:
                raise TypeError("Empty tuple selector is invalid")

            # --- (start,) → start to end ---
            if n == 1:
                start_key = gkeys[0]

                if start_key is None:
                    raise TypeError("Tuple (None,) is ambiguous")

                start_idx = keydict[start_key]
                stop_idx  = len(keydict)

            # --- (start, stop) ---
            elif n == 2:
                start_key, stop_key = gkeys

                # (None, stop) → beginning to stop
                if start_key is None:
                    start_idx = 0
                else:
                    start_idx = keydict[start_key]

                # (start, None) → start to end
                if stop_key is None:
                    stop_idx = len(keydict)
                else:
                    stop_idx = keydict[stop_key] + 1   # inclusive

            else:
                raise TypeError("Tuple selector must have length 1 or 2")

            idxs = slice(start_idx, stop_idx, 1)


        elif isinstance(gkeys, Iterable) and not isinstance(gkeys, (str, bytes, dict)):     # can be list of integer or strings (or anything hashable)
            idxs = []
            for one_gkey in gkeys:   # renamed from gkey -- distinct from the str|int `gkey` above
                # For the following, see https://github.com/raylutz/daffodil/issues/6
                try:
                    # gkeys may iterate to a tuple or other non-str/int element (per its Union
                    # type); keydict only has str|int keys, so such a lookup simply KeyErrors
                    # below and is handled the same as any other missing key.
                    idxs.append(keydict[cast(Union[str, int], one_gkey)])
                except KeyError:
                    if not silent_error:
                        # logs.sts(f"{logs.prog_loc()} Cannot find key '{one_gkey}' in {axis} in dataframe '{name}'", 3)
                        # breakpoint()
                        raise


        elif isinstance(gkeys, slice):
            # slice.start/.stop/.step are ordinal positions into keydict's current iteration
            # order (same convention as integer indices elsewhere) -- NOT keys to look up.
            # For arbitrary (e.g. string) keys there is no well-defined "next key" to support
            # exclusive-stop slicing the way integer positions do; use the (start_key, stop_key)
            # tuple form (inclusive of stop_key) for genuine key-range selection instead.
            #
            # Note: this branch is only reachable via the explicit select_krows()/select_kcols()
            # method calls. The [] / __getitem__/__setitem__ syntax (_parse_selectors) routes ANY
            # slice object straight to the plain-index path without ever calling krows_to_irows/
            # kcols_to_icols/gkeys_to_idxs at all -- regardless of whether the slice's bounds are
            # ints or strings. So `daf['a':'c']` does NOT do key-based slicing (it errors trying
            # to use 'a'/'c' as literal list-slice bounds); only daf.select_krows(slice(...)) /
            # daf.select_kcols(slice(...)) reach this code.
            n = len(keydict)
            start = gkeys.start if gkeys.start is not None else 0
            stop  = gkeys.stop  if gkeys.stop  is not None else n
            step  = gkeys.step  if gkeys.step  is not None else 1
            idxs = slice(start, stop, step)


        if inverse:
            n = len(keydict)
            if isinstance(idxs, slice):
                this_range = daf_utils.slice_to_range(idxs, n)
                idxs = [idx for idx in range(n) if idx not in this_range]
            else:
                idxs = [idx for idx in range(n) if idx not in idxs]

        return idxs


    def select_krows(self,
            krows:          Union[slice, str, T_la, int, T_lota, Tuple[Any, Any], Iterable, None],
            inverse:        bool=False,
            silent_error:   bool=False,
            ) -> 'Daf':
        """
        Select rows by key. Rows with those keys are kept, or dropped if `inverse` is True.

        This is the same as `my_daf[keys]`, with a choice to drop rows and to ignore
        keys that are not found. It works with integer keys, which brackets cannot.
        The result is a shallow new Daf. It is a new Daf with a new list of rows, and
        each row is the same list as in this Daf. Use `copy()` if you need to change
        them independently.

        A bare tuple means an inclusive range of keys, so `(1, 2)` is the rows from key
        1 through key 2. For a composite keyfield, give a list of tuples.

        Args:
            krows: A key, a list of keys, or a tuple that gives a range of keys.
            inverse: If True, drop the selected rows and keep the others.
            silent_error: If True, keys that are not found are ignored.

        Returns:
            The new Daf.

        Raises:
            KeysDisabledError: The Daf has no keyfield and no key index.
            KeyError: A key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_krows([3, 1])
            | id | v | n  |
            | -: | -: | -: |
            |  3 | c | 30 |
            |  1 | a | 10 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d.select_krows([1, 2], inverse=True)
            | id | v | n  |
            | -: | -: | -: |
            |  3 | c | 30 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
            >>> d.select_krows([1, 9], silent_error=True)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
        """
        self._rebuild_kd_if_invalidated()

        if not self.keyfield and not self._kd:
            raise self._no_keys_error('select_krows')

        irows = self.krows_to_irows(
            krows = krows,
            inverse = inverse,
            silent_error = silent_error,
            )
        new_daf = self.select_irows(irows)

        return new_daf


    def select_kcols(self,
            kcols:          Union[slice, str, T_la, int, Tuple[Any, Any], None],
            inverse:        bool=False,
            flip:           bool=False,
            silent_error:   bool=False,
            ) -> 'Daf':
        """
        Select columns by name. Columns with those names are kept, or dropped if `inverse` is True.

        This is the same as `my_daf[:, names]`, with more choices. The result is
        a new Daf with new rows, in the order of the names you give. With `flip=True`
        the selected columns become rows, and the result has no column names and no
        keyfield. This costs no more than selecting the columns.

        Selecting columns copies data, so it is not cheap. For `apply` and `reduce`,
        use their `cols` argument instead.

        Args:
            kcols: A name, a list of names, or a tuple that gives a range of names.
            inverse: If True, drop the named columns and keep the others.
            flip: If True, turn the selected columns into rows.
            silent_error: If True, names that are not found are ignored.

        Returns:
            The new Daf.

        Raises:
            KeysDisabledError: The Daf has no column names.
            KeyError: A name is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_kcols(['n', 'id'])
            | n  | id |
            | -: | -: |
            | 10 |  1 |
            | 20 |  2 |
            | 30 |  3 |
            %% daf rows=3; cols=2; keyfield='id'; name=''
            >>> d.select_kcols('v', inverse=True).columns()
            ['id', 'n']
            >>> d.select_kcols('v', flip=True)
            | A | B | C |
            | -: | -: | -: |
            | a | b | c |
            %% daf rows=1; cols=3; keyfield=''; name=''
        """
        if not self.hd:
            raise KeysDisabledError("select_kcols requires hd.")

        icols = self.kcols_to_icols(
            kcols = kcols,
            inverse = inverse,
            silent_error = silent_error,
            )
        return self.select_icols(icols, flip=flip)


    def select_irows(self, irows: Union[slice, int, T_li, range, T_lor, Iterable, None], inverse: bool=False, invert: bool=False) -> 'Daf':
        """
        Select rows by position. Those rows are kept, or dropped if `inverse` is True.

        This is the same as `my_daf[rows]`, with a choice to drop rows. It is cheap.
        The result is a shallow new Daf. It is a new Daf with a new list of rows, and
        each row is the same list as in this Daf, so changing a cell in the result
        changes it here too. This is also so when the selection is empty and `inverse`
        is True, which keeps all the rows. The keyfield, dtypes and column names carry over.

        Args:
            irows: A position, a slice, a range, a list of positions, or a list of ranges.
            inverse: If True, drop the selected rows and keep the others.
            invert: The old name of `inverse`, still accepted.

        Returns:
            The new Daf.

        Raises:
            IndexError: A single position is out of range.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_irows(1)
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
            >>> d.select_irows(1, inverse=True)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d.select_irows(1, invert=True)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d.select_irows(slice(1, None))
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
        """
        """ select rows from daf and return a new instance.
            This is an efficient opeation. The array in the new instance
            uses references to selected rows in the original array.

            irows: can be either a slice, int, range, or list of integers. These
                    refer to row indices that are inherent in the lol structure.

            returns a new daf instance cloned from the original.
        """
        inverse = inverse or invert     # invert is the old name.

        row_sliced_lol = self.lol

        no_rows_specified = bool(not isinstance(irows, int) and not irows)

        if not self.lol or no_rows_specified and not inverse:
            return self.clone_empty()

        if no_rows_specified:
            if inverse:
                return self.clone_empty(lol=list(self.lol))     # drop nothing: a shallow new Daf, rows shared.
            else:
                return self.clone_empty()

        if isinstance(irows, int):
            if not inverse:
                # simple single row selection
                try:
                    row_sliced_lol = [self.lol[irows]]
                except IndexError:
                    raise
            else:
                # build a fresh list rather than mutating self.lol in place (row_sliced_lol was
                # an alias for self.lol, so .pop() here previously destructively removed the row
                # from the original Daf too). Normalize negative indices the way list.pop() does.
                actual_idx = irows if irows >= 0 else len(self.lol) + irows
                row_sliced_lol = [row for i, row in enumerate(self.lol) if i != actual_idx]

        elif irows and isinstance(irows, list):

            if daf_utils.is_list_of_type(irows, int):
                # is_list_of_type() isn't a TypeGuard -- confirmed at runtime that every element
                # is int, mypy just can't narrow irows: T_li | T_lor from that call itself.
                irows_li = cast(T_li, irows)
                if not inverse:
                    # short-circuits on the first mismatch rather than materializing/comparing
                    # two full-length lists, so a non-natural-order irows (the common case)
                    # bails out almost immediately rather than doing O(num_rows) work regardless
                    # -- matters for large arrays (e.g. 500K+ rows) where this check needs to
                    # stay cheap even when it doesn't end up taking the fast path.
                    if len(irows_li) == len(self.lol) and all(irow == i for i, irow in enumerate(irows_li)):
                        row_sliced_lol = self.lol
                    else:
                        row_sliced_lol = [self.lol[i] for i in irows_li]
                else:
                    irows_iter: Iterable[int]
                    if len(irows_li) > 10:
                        irows_iter = dict.fromkeys(irows_li)
                    else:
                        irows_iter = irows_li
                    row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows_iter]

            elif daf_utils.is_list_of_type(irows, range):
                rows_lor = cast(T_lor, irows)    # just a name change
                if not inverse:
                    row_sliced_lol = [self.lol[i] for irange in rows_lor for i in irange]
                else:
                    row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if not any(i in r for r in rows_lor)]

        elif irows and isinstance(irows, (range, Iterable)):
            irows_ii = cast(Iterable[int], irows)
            if not inverse:
                row_sliced_lol = [self.lol[i] for i in irows_ii]
            else:
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows_ii]

        elif isinstance(irows, slice):
            slice_spec = irows
            if not inverse:
                row_sliced_lol = self.lol[slice_spec]
            else:
                slice_range = daf_utils.slice_to_range(slice_spec, len(self.lol))
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in slice_range]

        if row_sliced_lol is self.lol:
            row_sliced_lol = list(self.lol)     # a shallow new Daf: its own row list, rows shared.

        new_daf = self.clone_empty(lol=row_sliced_lol)

        return new_daf


    def select_icols(self, icols: Union[slice, int, T_li, range, T_lor, None], flip: bool=False) -> 'Daf':
        """
        Select columns by position. Those columns are kept, in the order given.

        This is the same as `my_daf[:, cols]`. It makes new rows, so it is not cheap.
        For `apply` and `reduce`, use their `cols` argument instead. A slice works as
        it does for a Python list. The keyfield and dtypes carry over if their columns
        are kept. With `flip=True` the columns become rows, and the result has no
        column names, no dtypes and no keyfield.

        With `flip=True` the columns are turned into rows as they are selected. That costs less than
        selecting them and then calling `transpose()`. In one test with 5 of 50 columns and 20,000
        rows it took 0.003 s, against 0.034 s.

        Args:
            icols: A position, a slice, a range, a list of positions, or a list of ranges.
            flip: If True, turn the selected columns into rows.

        Returns:
            The new Daf.

        Raises:
            IndexError: A position is beyond the end of some row.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_icols([2, 0])
            | n  | id |
            | -: | -: |
            | 10 |  1 |
            | 20 |  2 |
            | 30 |  3 |
            %% daf rows=3; cols=2; keyfield='id'; name=''
            >>> d.select_icols(slice(-2, None)).columns()
            ['v', 'n']
            >>> d.select_icols([0, 1], flip=True)
            | A | B | C |
            | -: | -: | -: |
            | 1 | 2 | 3 |
            | a | b | c |
            %% daf rows=2; cols=3; keyfield=''; name=''
        """

        """ select cols from daf and return a new instance.
            This is not an efficient operation and can normally be avoided except when:
                reading/writing data, then columns may need to be dropped.
                exporting a portion of the array to NumPy, for example.

            instead, use the cols parameter to select the columns included
                in operations like apply() and reduce()

            icols: can be either a slice, int, range, list of integers, or list of ranges. These
                    refer to column indices that are inherent in the lol structure.

            flip: if True, then the columns selected are turned into rows.
                    This is a transposition operation. Otherwise, the columns selected
                    remain as columns in a new daf instance. There is no additional cost
                    to transpose if it is done when the columns are selected.

            returns a new daf instance cloned from the original,
                with cols, keyfield, and dtypes adjusted to be reasonable.
                If flip=True, then colnames=[], dtypes={} and keyfield='' (inactive)

        """

        orig_cols = list(self.hd.keys())    # may be an empty list if colnames not defined.
        orig_dtypes = self.dtypes or {}
        sliced_cols = []

        if isinstance(icols, int):
            # simple single col selection
            icol = icols
            if not flip:
                col_sliced_lol = [[row[icol]] for row in self.lol]
                if orig_cols:
                    sliced_cols = [orig_cols[icol]]
            else: # flip
                col_sliced_lol = [row[icol] for row in self.lol]

        elif isinstance(icols, slice):
            slice_spec = icols
            icols_range = range(*slice_spec.indices(self.num_cols()))   # same rule as a Python list slice.

            if not icols_range:
                return Daf()

            if not flip:
                try:
                    col_sliced_lol = [[row[icol] for icol in icols_range]
                                            for row in self.lol]
                except IndexError as exc_info:
                    raise IndexError("select_icols(): column slice exceeds the length of some rows") from exc_info

                if orig_cols:
                    sliced_cols = [orig_cols[icol] for icol in icols_range]
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols_range]

        elif (daf_utils.is_list_of_type(icols, int) or
                isinstance(icols, list) and not icols or
                isinstance(icols, range)):
            # list of integers or range -- is_list_of_type() isn't a TypeGuard, confirmed at
            # runtime, mypy just can't narrow icols from that call itself.
            icols = cast(Union[T_li, range], icols)
            if not flip:
                try:
                    # short-circuits on the first mismatch rather than materializing/comparing
                    # two full-length lists, so a non-natural-order icols (the common case)
                    # bails out almost immediately rather than doing O(num_cols) work regardless.
                    if len(icols) == self.num_cols() and all(icol == i for i, icol in enumerate(icols)):
                        col_sliced_lol = self.lol
                    elif len(icols) == 0:
                        col_sliced_lol = []
                    else:
                        col_sliced_lol = [[row[icol] for icol in icols]
                                            for row in self.lol]
                except IndexError:
                    logs.sts("Columns specified don't exist, array may have uneven rows.", 3)
                    raise

                if orig_cols:
                    sliced_cols = [orig_cols[icol] for icol in icols]

            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols]

        # this part needs to be tested!
        elif isinstance(icols, list) and icols and daf_utils.is_list_of_type(icols, range):
            # list of ranges:
            cols_lor = cast(T_lor, icols) # name change only.
            if not flip:
                # Flatten ranges into individual column indices
                col_sliced_lol = [[row[icol] for irange in cols_lor for icol in irange]
                                        for row in self.lol]
                if orig_cols:
                    # Also handle orig_cols, slicing them similarly
                    sliced_cols = [orig_cols[icol] for irange in cols_lor for icol in irange]

            else:  # flip
                # Flip logic, slice columns and then rows, based on ranges
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for irange in cols_lor for icol in irange]
        else:
            if not flip:
                return self
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in range(self.num_cols())]
                sliced_cols = []    # perflint-reviewed (use-tuple-over-list)

        # fix up the dtypes and reset the keyfield if it is no longer in the daf.
        if sliced_cols:
            new_dtypes = {col: orig_dtypes[col] for col in sliced_cols if col in orig_dtypes}
            if self.keyfield and isinstance(self.keyfield, str):
                new_keyfield = self.keyfield if self.keyfield in sliced_cols else ''
            else:
                new_keyfield = ''

        else:
            new_dtypes = {}
            new_keyfield = ''

        new_daf = Daf(  cols=sliced_cols,
                        lol=col_sliced_lol,
                        keyfield=new_keyfield,
                        dtypes=new_dtypes,
                        )

        return new_daf

    #=============================================
    # the following methods might be absorbed into the above.
    #

    def select_record(self, key: Union[str, int, T_ta], silent_error: bool=True) -> T_da:
        """
        Get one row as a dict, by its key.

        If the key is not found, the answer is an empty dict, unless `silent_error` is
        False. Then a `KeyError` is raised. A typo in a key gives an empty dict, so
        check it, or pass `silent_error=False`. An empty Daf gives an empty dict.
        For a composite keyfield the key is a tuple.

        Args:
            key: The key of the row.
            silent_error: If False, raise an error when the key is not found.

        Returns:
            The row as a dict, or an empty dict.

        Raises:
            KeysDisabledError: The Daf has rows but no keyfield and no key index.
            KeyError: The key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_record(2)
            {'id': 2, 'v': 'b', 'n': 20}
            >>> d.select_record(9)
            {}
        """
        """ Select one record from daf using the key and return as a single T_da dict.

            returns {} if not self
            assertion break if keyfield not defined.
            if key not found, return {} if silent_error, else raise KeyError exception.

            TODO Update for returning KeyedList
            should return daf unless .to_dict() is used?
        """

        if not self:
            return {}

        self._rebuild_kd_if_invalidated()

        if not self.keyfield and not self._kd:
            raise self._no_keys_error('select_record')

        if key in self._kd:
            return self._basic_get_record(self._kd[key])

        if silent_error:
            return {}

        raise KeyError(key)


    def _basic_get_record(self, irow: int, include_cols: Optional[T_ls]=None) -> T_da:
        """
        Retrieve a row as a dictionary.

        Args:
            irow: Row index.
            include_cols: Optional subset of columns.

        Returns:
            Dict: Row data.
        """

        """ return a record at irow as dict
            include only "include_cols" if it is defined
            note, this requires that hd is defined.

            TODO Update to return KeyedList as an option.
        """
        if not self.hd:
            raise KeysDisabledError("Getting a row as a dict needs column names. This Daf has none. Call set_cols() to name them.")

        if include_cols:
            return {col:self.lol[irow][self.hd[col]] for col in include_cols if col in self.hd}
        else:
            return dict(zip(self.hd, self.lol[irow]))


    # def select_irows(self, irows_li: T_li) -> 'Daf':
        # """ Select multiple records from daf using row indexes and create new daf.

        # """

        # selected_daf = self.clone_empty()

        # for row_idx in irows_li:
            # record = self._basic_get_record(row_idx)

            # selected_daf.append(record)

        # return selected_daf


    def select_records_daf(self, keys_ls: Union[T_ls, Iterable], inverse:bool=False, silent_error: bool=False) -> 'Daf':
        """
        Select several rows by key and return them as a Daf.

        This is `select_krows()` with a friendlier answer for an empty list of keys.
        No keys gives an empty Daf, or all the rows if `inverse` is True.

        The rows are shared with this Daf, as in `select_krows()`. The new Daf has its own
        row list, also with no keys and `inverse` True, so adding a row to one does not
        add it to the other.

        Args:
            keys_ls: The keys of the rows.
            inverse: If True, drop the selected rows and keep the others.
            silent_error: If True, keys that are not found are ignored.

        Returns:
            The new Daf.

        Raises:
            KeysDisabledError: The Daf has no keyfield and no key index.
            KeyError: A key is not found and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.select_records_daf(['r3', 'r1'])
            | id | v | n  |
            | -: | -: | -: |
            | r3 | a | 30 |
            | r1 | a | 10 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d.select_records_daf([])
            %% daf rows=0; cols=0; keyfield='id'; name=''
            >>> kept = d.select_records_daf([], inverse=True)
            >>> kept == d
            True
            >>> _ = kept.append(['r4', 'd', 40])
            >>> d.num_rows(), kept.num_rows()
            (3, 4)
            >>> kept[0, 'v'] = 'Q'
            >>> d.iloc(0)['v']
            'Q'
        """

        """ Select multiple records from daf using the keys and return as a single daf.
            If inverse is true, select records that are not included in the keys.
            silent_error = True: do not raise an error if any keys are not found.

            This function requires that a keyfield or manual kd exists, otherwise raises KeysDisabledError.
        """
        # rudamentary special cases:
        if not keys_ls:
            if inverse:
                return self.clone_empty(lol=list(self.lol))
            else:
                return self.clone_empty()

        return self.select_krows(krows=keys_ls, inverse=inverse, silent_error=silent_error)


    def irow_la(self, irow: int) -> T_la:
        """
        Get one row as a list, by position.

        The list is the row itself, not a copy. Changing it changes the Daf. Use
        `iloc()` with `rtype='list'` for a copy.

        Args:
            irow: The row position.

        Returns:
            The row.

        Raises:
            IndexError: The position is out of range.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.irow_la(1)
            ['r2', 'b', '']
            >>> row = d.irow_la(0)
            >>> row[1] = 'Z'
            >>> d.iloc(0)
            {'id': 'r1', 'v': 'Z', 'n': 10}
            >>> d.irow_la(9)
            Traceback (most recent call last):
                ...
            IndexError: list index out of range
        """
        return self.lol[irow]


    def to_value(self,
        # irow:       int=0,
        # icol:       int=0,
        default:    Any = _MISSING,
        astype:     Optional[Union[Callable, str]]=None,
        ) -> Any:
        """
        Get the one value of a Daf that has one row and one column.

        Use it on the result of a selection, as in `my_daf[1, 'n'].to_value()`.

        Args:
            default: Returned if the Daf is not one cell. Without it, an error is raised.
            astype: A type or function to convert the value with.

        Returns:
            The value.

        Raises:
            ValueError: The Daf is not one cell and no default is given.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d[1, 'n'].to_value()
            20
        """

        num_rows, num_cols = self.shape()

        if num_rows == 1 and num_cols == 1:
            return daf_utils.astype_value(self.lol[0][0], astype)

        if default is not _MISSING:
            return default

        raise ValueError(f"to_value() requires 1×1, got shape {self.shape()}")


    def to_list(self,
        # irow:       Optional[int]=None,   # select a row
        # icol:       Optional[int]=None,   # or column.
        unique:     bool=False,           # reduce to unique values
        flatten:    bool=False,           # if items is the list are lists, combine them into one list.
        omit_nulls: bool=False,           # omit items that are empty strings (nulls).
        default:    Any = _MISSING,       # use this value instead if value is None or '' or NAN (default can be None)
        astype:     Optional[Union[Callable, str, type]]=None,
        ) -> list:
        """
        Get the values of a Daf that has one row or one column, as a list.

        Use it on the result of a selection, as in `my_daf[:, 'v'].to_list()`. A column
        is read from the Daf, so use `col()` if you only need the list. An empty
        Daf gives an empty list. A Daf with more than one row and more than one column
        is not accepted.

        Args:
            unique: If True, leave out repeated values and keep the order.
            flatten: If True, join items that are lists into one list.
            omit_nulls: If True, leave out the empty values.
            default: Replaces NULL, None and NaN values. This may itself be None.
            astype: A type or function to convert each value with.

        Returns:
            The list.

        Raises:
            ValueError: The Daf has more than one row and more than one column.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d[:, 'v'].to_list()
            ['a', 'b', 'c']
            >>> d[1].to_list()
            [2, 'b', 20]
        """
        """ return data from a daf array as a list
            defaults to the most obvious list if irow and icol not specified.
                from irow 0, if num_rows is 1 and num_cols >= 1
                from icol 0, if num_rows >= 1 and num_cols == 1
            otherwise, choose irow or icol specified.
                if irow, specified, ignore icol.
                if irow=None and icol specified, then use icol.
            Note:
                If neither irow nor icol is specified, and the table has more than one row and more than one column,
                then the result is an empty list. Use irow or icol explicitly to avoid ambiguity.
        """

        num_rows, num_cols = self.shape()

        if num_rows == 1 and num_cols >= 1:
            # single row, return as list.
            result_la = self.lol[0]

        elif num_rows > 1 and num_cols == 1:
            # single column result as a list.
            result_la = self.icol(0)
        elif num_rows == num_cols == 0:
            result_la = []
        else:
            raise ValueError("to_list() requires a 1D Daf (single row or single column)")

        if flatten:
            new_list = []       # perflint-reviewed (use-tuple-over-list)
            for sublist in result_la:
                if isinstance(sublist, list):
                    new_list += sublist
                else:
                    new_list += [sublist]
            result_la = new_list

        if unique and result_la:
            result_la = list(dict.fromkeys(result_la))

        if omit_nulls and any(val is NULL for val in result_la):
            result_la = [val for val in result_la if val is not NULL]

        if default is not _MISSING and any(val is NULL or val is None for val in result_la):
            filtered_result_la = []
            for val in result_la:
                if val is None or val is NULL or val != val:  # val != val catches NaN
                    filtered_result_la.append(default)
                else:
                    filtered_result_la.append(val)
            result_la = filtered_result_la

        return daf_utils.astype_la(result_la, astype)   # returns result_la if astype is None.


    def to_lota(self,
        ) -> T_lota:
        """
        Make a list of tuples, one tuple for each row.

        This is handy for making composite keys.

        Returns:
            The rows as tuples.

        Examples:
            >>> Daf(lol=[[1, 'a']], cols=['id', 'v'])
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield=''; name=''
        """
        """ return data from a daf array as a list of tuple of any (List[Tuple[Any...]])
            This is convenient for creating compound keys
        """

        lota = []

        for la in self.lol:
            lota.append(tuple(la))

        return lota



    def to_dict(self, 
            # irow: int=0, 
            # include_cols: Optional[T_ls]=None,
            ) -> T_da:
        """
        Get the one row of a Daf as a dict.

        Use it on the result of a selection, as in `my_daf[1].to_dict()`. A column is
        not turned into a dict. Use `to_list()` for that. An empty Daf gives an empty
        dict. A Daf with no column names raises `KeysDisabledError`.

        Returns:
            The row, as a dict that maps column names to values.

        Raises:
            KeysDisabledError: The Daf has a row and no column names.
            ValueError: The Daf has more than one row.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d[1].to_dict()
            {'id': 2, 'v': 'b', 'n': 20}
        """
        """ alias for iloc
            Note that this does not convert a column to a dict. Use to_list to convert a column.
            test exists in test_daf.py
        """
        if len(self) > 1:
            raise ValueError("Ambiguous 2dim array.")

        return cast(T_da, self.iloc(irow=0, include_cols=None))


    def to_klist(self, irow: int=0) -> KeyedList:
        """
        Get one row as a [KeyedList][daffodil.keyedlist.KeyedList].

        The KeyedList shares the row and the column names with the Daf, so it costs
        little. A negative position counts from the end. A position that is out of range
        raises `IndexError`. A Daf with no rows gives an empty KeyedList.

        Args:
            irow: The row position.

        Returns:
            The row.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.to_klist(1)['v']
            'b'
        """
        return cast(KeyedList, self.iloc(irow, rtype='klist'))


    def irow(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """
        Get one row as a dict, by position.

        This is `iloc()` with the default `rtype`.

        Args:
            irow: The row position.
            include_cols: Only these columns are included.

        Returns:
            The row as a dict. A negative position counts from the end. A Daf with no rows
            gives an empty dict.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.irow(1, include_cols=['n'])
            {'n': 20}
        """
        return cast(T_da, self.iloc(irow, include_cols))


    def _get_kidx(self) -> KeyedIndex:
        """ An index of the column names, kept so that one-row KeyedLists can share it.

            It is rebuilt when hd is replaced or its length changes. Changing the names in hd
            in place, without changing its length, is not detected.
        """
        cache = getattr(self, '_kidx_cache', None)
        if cache is None or cache[0] is not self.hd or cache[1] != len(self.hd):
            cache = (self.hd, len(self.hd), KeyedIndex(cast(dict, self.hd)))
            self._kidx_cache = cache
        return cache[2]


    def iloc(self, irow: int=0, include_cols: Optional[T_ls]=None, rtype: str='dict') -> Union[T_ma, T_la]:
        """
        Get one row by position, as a dict, a KeyedList or a list.

        A negative position counts from the end, so `-1` is the last row. A position that
        is out of range raises `IndexError`. A Daf with no rows, and a row with no cells,
        give an empty dict, or an empty KeyedList. A dict or a KeyedList needs column
        names, so a Daf that has rows and no column names raises `KeysDisabledError`. Name
        them with `set_cols()`, or ask for `rtype='list'`.

        Args:
            irow: The row position.
            include_cols: Only these columns are included. This applies to a dict.
            rtype: `dict` for a new dict, `klist` for a KeyedList that shares the row, or `list` for a copy of the row.

        Returns:
            The row.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows.
            KeysDisabledError: `rtype` is `dict` or `klist`, and the Daf has rows and no column names.
            ValueError: `rtype` is not one of the three names.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.iloc(1)
            {'id': 2, 'v': 'b', 'n': 20}
            >>> d.iloc(1, rtype='list')
            [2, 'b', 20]
            >>> d.iloc(-1)
            {'id': 3, 'v': 'c', 'n': 30}
        """
        """ Select one record from daf using the idx and return as a single T_da dict
            test exists in test_daf.py

            rtype can be 'dict', 'klist', or 'list'  <-- should be astype

        """
        if self.lol:
            if irow < 0:
                irow += len(self.lol)           # count from the end, as a list does.
            if irow < 0 or irow >= len(self.lol):
                raise IndexError(f"iloc: row position {irow} is out of range for {len(self.lol)} rows.")

        if not self.lol or not self.lol[irow]:
            if rtype == 'dict':
                return {}
            else:
                return KeyedList()

        if rtype in ('klist', 'dict') and not self.hd:
            raise KeysDisabledError(
                "Getting a row as a dict or a KeyedList needs column names. This Daf has none. "
                "Call set_cols() to name them, or ask for rtype='list'.")

        if rtype == 'klist':
            return KeyedList(self._get_kidx(), self.lol[irow])

        elif rtype == 'dict':
            return self._basic_get_record(irow, include_cols)

        elif rtype == 'list':
            return list(self.lol[irow])

        raise ValueError(f"iloc: unrecognized rtype '{rtype}'")


    # def select_by_dict_to_lod(self, selector_da: T_da, expectmax: int=-1, inverse: bool=False) -> T_loda:
        # """ Select rows in daf which match the fields specified in d, returning lod
            # test exists in test_daf.py

            # DEPRECATE, use select_by_dict().to_lod()
        # """

        # result_lod = self.select_by_dict(selector_da=selector_da, expectmax=expectmax, inverse=inverse).to_lod()

        # return result_lod


    def select_by_dict(
            self,
            selector_da:    T_da,
            expectmax:      int=-1,
            inverse:        bool=False,
            keyfield:       Union[str, int, T_ta]='',
            ) -> 'Daf':
        """
        Select the rows that match every field of a dict.

        A row matches if each key of `selector_da` is a column whose value in that row
        equals the value given. With `inverse=True` the rows that do not match are
        returned. The cells are compared by position, so no row is turned into a dict.
        A Daf with rows and no column names raises `KeysDisabledError`. The new Daf is a shallow new Daf, a live view of this one. It has its own
        row list, and its rows are the same lists as the rows of this Daf. Changing a value in
        it changes this Daf. Adding a column with `insert_col()` or `insert_idx_col()` does not,
        because those copy shared rows first. Use `copy('editable')` for rows of your own.

        Args:
            selector_da: The column names and the values they must have.
            expectmax: If this is not -1 and more rows match, raise `LookupError`.
            inverse: If True, return the rows that do not match.
            keyfield: The keyfield of the new Daf. If empty, the keyfield of this Daf.

        Returns:
            The new Daf.

        Raises:
            LookupError: More than `expectmax` rows match.
            KeysDisabledError: The Daf has rows and no column names.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_by_dict({'v': 'b'})
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
            >>> d.select_by_dict({'v': 'b'}, inverse=True)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
        """

        """ Selects rows in daf which match the fields specified in selector_da
            and return new daf, with keyfield set according to 'keyfield' argument.
        """
        # test exists in test_daf.py

        # the cells are compared by position, so no row is turned into a dict or a KeyedList.
        if self.lol and not self.hd:
            raise KeysDisabledError("select_by_dict(): this Daf has no column names. Call set_cols() to name them.")

        hd = self.hd
        if any(col not in hd for col in selector_da):
            result_lol = list(self.lol) if inverse else []          # an unknown column matches nothing.
        else:
            pairs = [(hd[col], val) for col, val in selector_da.items()]
            if not pairs:
                result_lol = [] if inverse else list(self.lol)      # an empty selector matches every row.
            elif len(pairs) == 1:
                icol, val = pairs[0]
                result_lol = [row_la for row_la in self.lol if (row_la[icol] == val) is not inverse]
            else:
                icol, val = pairs[0]
                rest = pairs[1:]
                result_lol = [row_la for row_la in self.lol
                              if (row_la[icol] == val and all(row_la[i] == v for i, v in rest)) is not inverse]

        if expectmax != -1 and len(result_lol) > expectmax:
            raise LookupError(f"select_by_dict(): {len(result_lol)} rows match, more than expectmax={expectmax}.")
            # breakpoint() #perm
            # pass

        new_keyfield = keyfield or self.keyfield

        # following invalidates kd for future lazy rebuild.
        daf = Daf(cols=self.columns(), lol=result_lol, keyfield=new_keyfield, dtypes=self.dtypes)

        return daf


    def select_first_row_by_dict(self, selector_da: T_da, inverse:bool=False) -> T_ma:
        """
        Get the first row that matches every field of a dict.

        The matching rule is that of `select_by_dict()`. With `inverse=True` it is the
        first row that does not match.

        Args:
            selector_da: The column names and the values they must have.
            inverse: If True, find the first row that does not match.

        Returns:
            The row, as a dict or a KeyedList according to `itermode`. An empty dict if none matches.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_first_row_by_dict({'v': 'b'})
            {'id': 2, 'v': 'b', 'n': 20}
            >>> d.select_first_row_by_dict({'v': 'zz'})
            {}
        """

        """ Selects the first row in daf which matches the fields specified in selector_da
            and returns that row. Else returns {}.
            Use inverse to find the first row that does not match.
        """

        # test exists in test_daf.py

        for d2 in self:
            if inverse ^ daf_utils.is_d1_in_d2(d1=selector_da, d2=d2):
                return d2

        return {}


    def select_where(self, where: Callable, indirect_col: Optional[str]=None) -> 'Daf':
        """
        Select the rows for which a function is true.

        The function gets each row, as a [KeyedList][daffodil.keyedlist.KeyedList].
        Read cells by column name, as in `row['n']`. Values are used as stored, so
        convert text first if the Daf was read from a CSV.

        With `indirect_col`, a name that is not a column is looked up in the dict
        held in that column.

        If the test only compares columns to values, use `select_by_dict()`. It
        compares the cells by position and does not call a function for each row. For
        200,000 rows by 50 columns it took 0.012 s, against 0.15 s for this method.

        The new Daf shares the selected rows with this one. The keyfield and dtypes
        carry over.

        To test a value against a list, a set or another table, write the test in the function.
        There is no need to build a list of bools first, as the deprecated `isin()` did. Build a set
        of the values before the call, so that each lookup is fast and the set is built once. The
        function can use `and`, `or`, `not` and any other Python. It is called once for each row, so
        for a test on one column of a large Daf it is not the fastest way. A comprehension over
        `col()`, followed by `select_irows()`, is faster. In one test with 200,000 rows and 1,000
        values, `select_where()` took 0.11 s and the comprehension took 0.02 s.

        Args:
            where: A function that takes a row and returns True to keep it.
            indirect_col: A column that holds a dict, to read names that are not columns from.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_where(lambda row: row['n'] > 10)
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30], [4, 'a', 40]], cols=['id', 'v', 'n'])
            >>> keep = {'a', 'c'}
            >>> d.select_where(lambda row: row['v'] in keep)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            |  4 | a | 40 |
            %% daf rows=3; cols=3; keyfield=''; name=''
            >>> d.select_where(lambda row: row['v'] not in keep)
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            %% daf rows=1; cols=3; keyfield=''; name=''
            >>> d.select_where(lambda row: row['v'] in keep and row['n'] > 15)
            | id | v | n  |
            | -: | -: | -: |
            |  3 | c | 30 |
            |  4 | a | 40 |
            %% daf rows=2; cols=3; keyfield=''; name=''
            >>> d.select_where(lambda row: row['id'] % 2 == 0 or row['v'] == 'c')
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            |  3 | c | 30 |
            |  4 | a | 40 |
            %% daf rows=3; cols=3; keyfield=''; name=''

            The values can come from another Daf. Make the set once, outside the function:

            >>> other = Daf(lol=[['a'], ['c']], cols=['v'], keyfield='v')
            >>> other_keys = set(other.keys())
            >>> d.select_where(lambda row: row['v'] in other_keys)
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            |  4 | a | 40 |
            %% daf rows=3; cols=3; keyfield=''; name=''

            For a large Daf, the faster form picks the positions from the column:

            >>> d.select_irows([irow for irow, v in enumerate(d.col('v')) if v in keep])
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  3 | c | 30 |
            |  4 | a | 40 |
            %% daf rows=3; cols=3; keyfield=''; name=''
        """
        """
        Select rows in Daf based on the provided where condition
        
        # Example Usage

            result_daf = original_daf.select_where(lambda row: bool(int(row['colname']) > 5))

        if indirect_col is set to a column that exists in the daf array, then any 
            column references will be first tried in the non-indirect columns and if not found,
            then the indirect column will be attempted.

        """
        # unit test exists.

        result_lol: T_lola = []

        # --- fast path: no indirect ---
        if not indirect_col:
            for row_kl in self.iter_klist():
                if where(row_kl):
                    result_lol.append(row_kl.values())

        # --- indirect path ---
        else:
            for row_kl in self.iter_klist():
                row = _IndirectRowView(row_kl, indirect_col)
                if where(row):
                    result_lol.append(row_kl.values())

        daf = Daf(cols=self.columns(), lol=result_lol, keyfield=self.keyfield, dtypes=self.dtypes)

        return daf


    def select_where_idxs(self, where: Callable) -> T_li:
        """
        Get the positions of the rows for which a function is true.

        The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], whatever the
        `itermode` is, as in `select_where()`. Read cells by column name, as in `row['n']`.
        For a test that only compares columns to values, `select_by_dict()` is faster.

        The positions are a snapshot. Inserting, removing or sorting rows makes them point
        to other rows. Use them right away. To keep a reference to a row for longer, keep its key.

        Args:
            where: A function that takes a row and returns True to keep it.

        Returns:
            The row positions.

        Raises:
            KeysDisabledError: The Daf has rows and no column names.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_where_idxs(lambda row: row['n'] > 10)
            [1, 2]
        """

        """
        Select rows in Daf based on the provided where condition
        return list of indexes.

        # Example Usage
            result_daf = original_daf.select_where(lambda row: int(row['colname']) > 5)

        Could use .to_index() approach instead?

        """
        # unit test exists.

        return [idx for idx, row_kl in enumerate(self.iter_klist()) if where(row_kl)]


    def remove_dups(self, keyfield: Union[str, T_ta, T_la]='') -> Tuple['Daf', 'Daf']:  # unique_daf, duplicates_daf
        """
        Split the rows into those with a unique key and those with a repeated key.

        Only the key columns are compared, not the whole row. For each key, the last row
        is kept as the unique one. The earlier rows with that key are the duplicates.
        The unique rows are in the order in which their keys first appear.

        Without `keyfield`, the keyfield of this Daf is used. If there is none, a
        `KeysDisabledError` is raised. This Daf is not changed. Both results share their
        rows with it, and both have the key columns as their keyfield.

        Args:
            keyfield: The column, or tuple or list of columns, that identifies a row.
                If empty, the keyfield of this Daf.

        Returns:
            A tuple of the Daf of unique rows and the Daf of duplicate rows.

        Raises:
            KeysDisabledError: There is no `keyfield` and this Daf has none.
            KeyError: A key column is not found.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'v'])
            >>> unique, dups = d.remove_dups('id')
            >>> unique
            | id | v |
            | -: | -: |
            |  1 | c |
            |  2 | b |
            %% daf rows=2; cols=2; keyfield='id'; name=''
            >>> dups
            | id | v |
            | -: | -: |
            |  1 | a |
            %% daf rows=1; cols=2; keyfield='id'; name=''
        """
        """
        If it is known that duplicates may exist in the array with respect to keyfield,
        remove records that have the same keyfield and return two daf arrays,
        uniques_daf, dups_daf.

        Please note that this does NOT compare all components, only the keyfields.

        Returns unique_daf array with keyfield set and duplicates_daf, which may have repeats.

        """
        key_cols: Any = keyfield or self.keyfield
        if not key_cols:
            raise KeysDisabledError("remove_dups: give a keyfield, or set one on the Daf.")

        # a key index built here, so this Daf keeps its own keyfield and key index.
        if isinstance(key_cols, (str, int)):
            kd = type(self)._build_kd(self.hd[key_cols], self.lol)  # type: ignore[index]  # an int keyfield is a column name here
        else:
            kd = type(self)._build_kd([self.hd[cast(str, col)] for col in key_cols], self.lol)

        # irows of the last record of each key. These are the unique records.
        unique_irows = list(kd.values())

        unique_daf = self.select_irows(irows=unique_irows, inverse=False)
        unique_daf.keyfield = key_cols

        # all the other records are duplicates.
        dups_daf = self.select_irows(irows=unique_irows, inverse=True)
        dups_daf.keyfield = key_cols

        return unique_daf, dups_daf


    def split_where(
            self, 
            where: Callable,
            *,
            indirect_col: Optional[str] = None,
            ) -> Tuple['Daf', 'Daf']:
        """
        Split the rows in two, by a function that is true or false for each row.

        The function gets each row, as in `select_where()`. Both new Daf instances
        share their rows with this one, so changing a cell in one changes it here
        too. The keyfield and dtypes carry over to both.

        Args:
            where: A function that takes a row and returns True or False.
            indirect_col: A column that holds a dict, to read names that are not columns from.

        Returns:
            A tuple of the Daf of the rows where the function is true and the Daf of the others.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> big, small = d.split_where(lambda row: row['n'] > 10)
            >>> big
            | id | v | n  |
            | -: | -: | -: |
            |  2 | b | 20 |
            |  3 | c | 30 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> small
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
        """
        """
        Select rows in Daf based on the provided where condition,
        and split into two Daf objects, for True and False.

        Note: this does a shallow copy of the lines and any changes to
        the split objects will change the original daf object.

        # Example Usage

            true_daf, false_daf = original_daf.split_where(lambda row: bool(int(row['colname']) > 5))

        """
        true_lol = []
        false_lol = []

        if indirect_col:

            for klist in self.iter_klist():

                row = _IndirectRowView(klist, indirect_col)

                if where(row):
                    true_lol.append(klist._values)
                else:
                    false_lol.append(klist._values)

        else:    

            for klist in self.iter_klist():
                if where(klist):
                    true_lol.append(klist._values)
                else:
                    false_lol.append(klist._values)

        true_daf = Daf(cols=self.columns(), lol=true_lol, keyfield=self.keyfield, dtypes=self.dtypes)
        false_daf = Daf(cols=self.columns(), lol=false_lol, keyfield=self.keyfield, dtypes=self.dtypes)

        return true_daf, false_daf


    def col(self, 
            colname:        str,
            *,
            unique:         bool=False, 
            omit_nulls:     bool=False, 
            silent_error:   bool=False,
            astype:         Optional[Union[Callable, str, type]]=None,
            indirect_col:   Optional[str]=None,
            default:        Optional[Any]='',
            ) -> list:
        """
        Get one column as a list, by name.

        This does not make a Daf first, as `my_daf[:, 'v'].to_list()` does.

        With `indirect_col`, a name that is not a column is read from the dict held in
        that column, row by row. A row that lacks it gets `default`.

        Args:
            colname: The column name.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out the empty values.
            silent_error: If True, a column that is not found gives an empty list.
            astype: A type or function to convert each value with.
            indirect_col: A column that holds a dict, to read the name from.
            default: The value for a row that lacks the name. It is used only with `indirect_col`.

        Returns:
            The values of the column.

        Raises:
            ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
                `KeyError`, and also a `RuntimeError`, which this method raised in earlier versions.
            RuntimeError: The name is empty.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.col('v')
            ['a', 'b', 'c']
            >>> d.col('n', astype=str)
            ['10', '20', '30']
        """

        """ alias for col_to_la()
            can also use column ranges and then transpose()
            test exists in test_daf.py
            silent_error: if colname not found, return []

            Can use my_daf[:, colname].to_list(unique=unique, omit_nulls=omit_nulls)
        """
        return self.col_to_la(colname, 
                unique          = unique, 
                omit_nulls      = omit_nulls, 
                silent_error    = silent_error,
                astype          = astype,
                indirect_col    = indirect_col,
                default         = default,
                )


    def col_to_la(self, 
            colname:        str, 
            *, 
            unique:         bool=False, 
            omit_nulls:     bool=False, 
            silent_error:   bool=False,     # no error if column not found.
            astype:         Optional[Union[Callable, str, type]]=None,
            indirect_col:   Optional[str]=None,
            default:        Optional[Any]='',
        ) -> list:
        """
        Get one column as a list, by name.

        This does the same as `col()`. See that method.

        Args:
            colname: The column name.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out the empty values.
            silent_error: If True, a column that is not found gives an empty list.
            astype: A type or function to convert each value with.
            indirect_col: A column that holds a dict, to read the name from.
            default: The value for a row that lacks the name. It is used only with `indirect_col`.

        Returns:
            The values of the column.

        Raises:
            ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
                `KeyError`, and also a `RuntimeError`, which this method raised in earlier versions.
            RuntimeError: The name is empty.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.col_to_la('v')
            ['a', 'b', 'a']
            >>> d.col_to_la('v', unique=True)
            ['a', 'b']
            >>> d.col_to_la('n', omit_nulls=True)
            [10, 30]
            >>> d.col_to_la('n', astype=str)
            ['10', '', '30']
            >>> d.col_to_la('zz', silent_error=True)
            []
        """

        if not colname:
            raise RuntimeError("colname is required.")

        # normal column    
        if colname in self.hd:
            icol = self.hd[colname]
            result_la = self.icol_to_la(icol, unique=unique, omit_nulls=omit_nulls)
            result_la = daf_utils.astype_la(result_la, astype)

        elif indirect_col and indirect_col in self.hd:
            result_la = []

            for row_da in self:
                val = daf_utils.get_indirect_val(row_da, indirect_col, colname, default=default)

                if omit_nulls and val is NULL:
                    continue

                # this added for numpy compatibility.
                if val is NULL and default is not NULL:
                    val = default

                val = daf_utils.astype_value(val, astype)
                result_la.append(val)
            if unique:
                result_la = list(dict.fromkeys(result_la))

        else:
            if silent_error:
                return []
            raise ColumnNotFoundError(colname)

        return result_la


    def icol(self, icol: int) -> list:
        """
        Get one column as a list, by position.

        A negative position counts from the end, so `-1` is the last column. A position
        that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

        Args:
            icol: The column position.

        Returns:
            The values of the column.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.icol(1)
            ['a', 'b', 'c']
            >>> d.icol(-1)
            [10, 20, 30]
        """

        return self.icol_to_la(icol)


    def icol_to_la(self, icol: int, unique: bool=False, omit_nulls: bool=False) -> list:
        """
        Get one column as a list, by position, with options.

        A negative position counts from the end, so `-1` is the last column. A position
        that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

        Args:
            icol: The column position.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out the empty values.

        Returns:
            The values of the column.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows.

        Examples:
            >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

            >>> d.icol_to_la(1)
            ['a', 'b', 'a']
            >>> d.icol_to_la(-1)
            [10, '', 30]
            >>> d.icol_to_la(1, unique=True)
            ['a', 'b']
            >>> d.icol_to_la(5)
            Traceback (most recent call last):
                ...
            IndexError: icol: column position 5 is out of range for 3 columns.
        """

        if not self:
            return []

        num_cols = self.num_cols()
        if icol < 0:
            icol += num_cols                    # count from the end, as a list does.
        if icol < 0 or icol >= num_cols:
            raise IndexError(f"icol: column position {icol} is out of range for {num_cols} columns.")

        if omit_nulls:
            result_la = [la[icol] for la in self.lol if la[icol] is not NULL]
        else:
            result_la = [la[icol] for la in self.lol]

        if unique:
            result_la = list(dict.fromkeys(result_la))

        return result_la


    def drop_cols(self, exclude_cols: Optional[T_ls]=None) -> 'Daf':
        """
        Remove columns from this Daf, in place.

        The rows are rebuilt without those columns, so this copies all the data. Avoid
        it for large tables. Use the `cols` argument of `apply` and `reduce`, or
        `select_kcols()` to get a new Daf, instead.

        The column names and dtypes are updated. A name that is not a column is ignored.
        If a column of the keyfield is dropped, the keyfield is cleared, as in the other
        methods that remove a key column. Set it again with `set_keyfield()`. With no
        names, nothing happens.

        Args:
            exclude_cols: The names of the columns to remove.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.drop_cols(['v']).columns()
            ['id', 'n']
        """

        if exclude_cols:
            keep_idxs_li: T_li = [self.hd[col] for col in self.hd if col not in exclude_cols]

        else:
            return self

        for irow, la in enumerate(self.lol):
            la = [la[idx] for idx in keep_idxs_li]
            self.lol[irow] = la

        old_cols = list(self.hd.keys())
        new_cols = [old_cols[idx] for idx in keep_idxs_li]
        self._cols_to_hd(new_cols)

        if self.dtypes:
            kept_cols = {old_cols[idx] for idx in keep_idxs_li}
            new_dtypes = {col: typ for col, typ in self.dtypes.items() if col in kept_cols}
            self.dtypes = new_dtypes

        # a keyfield that lost a column can no longer find its rows, so clear it, as the other
        # methods that remove a key column do.
        key_cols = [self.keyfield] if isinstance(self.keyfield, (str, int)) else list(self.keyfield or [])
        if key_cols and any(col in exclude_cols for col in key_cols):
            self.keyfield = ''
            self._kd = {}       # _invalidate_kd() leaves the index alone when there is no keyfield.

        return self



    def select_cols(self,
            cols: Optional[T_ls]=None,
            exclude_cols: Optional[T_ls]=None,
            ) -> 'Daf':
        """
        Make a new Daf with only some columns, chosen by name.

        The columns come in the order of `cols`, as with `select_kcols()` and
        `my_daf[:, cols]`. A name given twice is used once. A name that is not a column
        raises `KeyError`. With no `cols`, all columns are kept, in the order of this
        Daf. Then `exclude_cols` leaves out the columns you name. A name in
        `exclude_cols` that is not a column is ignored.

        This copies data, so it is not cheap. For `apply` and `reduce`, use their
        `cols` argument instead. The keyfield carries over if its column is kept. A Daf
        with no column names gives rows with no columns.

        Args:
            cols: The names of the columns to keep, in the order you want. If empty, all columns.
            exclude_cols: The names of the columns to leave out.

        Returns:
            The new Daf.

        Raises:
            KeyError: A name in `cols` is not a column.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.select_cols(['n', 'id']).columns()
            ['n', 'id']
            >>> d.select_cols(exclude_cols=['v']).columns()
            ['id', 'n']
        """

        if not self.hd:
            return Daf(lol=[[] for _ in self.lol])

        if isinstance(cols, str):
            cols = [cols]
        if isinstance(exclude_cols, str):
            exclude_cols = [exclude_cols]
        exclude_set = set(exclude_cols) if exclude_cols else set()

        if cols:
            for col in cols:
                self.hd[col]                # a KeyError for a name that is not a column.
            new_cols = [col for col in dict.fromkeys(cols) if col not in exclude_set]
        else:
            new_cols = [col for col in self.hd if col not in exclude_set]

        idxs = [self.hd[col] for col in new_cols]

        # select from the array and create a new object.
        new_lol = [[la[idx] for idx in idxs] for la in self.lol]

        if self.dtypes:
            dtypes = {col: self.dtypes[col] for col in new_cols if col in self.dtypes}
        else:
            dtypes = None

        new_keyfield = self.keyfield \
            if self.keyfield and isinstance(self.keyfield, str) and self.keyfield in new_cols else ''

        return Daf(lol=new_lol, cols=new_cols, dtypes=dtypes, keyfield=new_keyfield)


    # def from_selected_cols(self, cols: Optional[T_ls]=None, exclude_cols: Optional[T_ls]=None) -> 'Daf':
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

    def assign_record(self, record: T_da) -> 'Daf':
        """
        Put one row in the Daf by its key, replacing a row that has the same key.

        The row is a dict. If a row with that key exists, the whole row is replaced.
        The cells of columns that the dict lacks become NULL. If the key is new, the
        row is added at the end. This is an upsert for one row. For that, `append()`
        with `respect_kd=True` also works. Use `update_by_keylist()` to change only
        some cells.

        Args:
            record: The row, as a dict. It must have the keyfield.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeysDisabledError: The Daf has no keyfield.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.assign_record({'id': 2, 'v': 'new'})
            >>> d
            | id |  v  | n  |
            | -: | --: | -: |
            |  1 |   a | 10 |
            |  2 | new |    |
            |  3 |   c | 30 |
            %% daf rows=3; cols=3; keyfield='id'; name=''
        """

        if not self.keyfield:
            raise KeysDisabledError("assign_record(): the Daf needs a keyfield to read the key from the record.")

        # test if valid keyfield, produce an error if not.
        self._is_keyfield_valid()

        keyval = self._get_keyval(record)

        self._rebuild_kd_if_invalidated()

        if keyval in self._kd:
            # assign the record, normalize the fields.
            self.lol[self._kd[keyval]] = [record.get(col, '') for col in self.hd]
        else:
            # otherwise add it, and invalidate kd.
            self.append(record)

        return self


    def assign_record_irow(self, irow: Optional[int]=None, record: Optional[T_da]=None) -> 'Daf':
        """
        Put one row in the Daf by position, replacing the row there.

        The row is a dict. The whole row is replaced, and the cells of columns that
        the dict lacks become NULL. Use `update_record_irow()` to change only some cells.

        With the default position, `None`, the row is added at the end. So it is with a
        position beyond the last row, and with any position when the Daf has no rows.
        A negative position counts from the end, as in a list, so `-1` replaces the last
        row. This is also what `my_daf[-1] = {...}` does.

        Args:
            irow: The row position. None, the default, adds the row at the end.
            record: The row, as a dict. If None, nothing happens.

        Returns:
            This Daf, which has been changed.

        Raises:
            IndexError: A negative position is before the first row, for a Daf that has rows.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.assign_record_irow(1, {'v': 'q'})
            >>> d.iloc(1)
            {'id': '', 'v': 'q', 'n': ''}
            >>> _ = d.assign_record_irow(-1, {'id': 9})
            >>> d.iloc(-1)
            {'id': 9, 'v': '', 'n': ''}
            >>> _ = d.assign_record_irow(record={'id': 10})
            >>> d.iloc(-1), len(d)
            ({'id': 10, 'v': '', 'n': ''}, 4)
        """

        if record is None:
            return self

        if irow is not None and irow < 0 and self.lol:
            irow += len(self.lol)               # count from the end, as a list does.
            if irow < 0:
                raise IndexError(f"assign_record_irow(): row position {irow - len(self.lol)} is before the first row.")

        if irow is None or irow < 0 or irow >= len(self.lol):
            self.append(record)
        else:
            #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)
            self.lol[irow] = [record.get(col, '') for col in self.hd]

        return self


    #@deprecated("Use 'my_daf[keylist] = record' syntax")
    def update_by_keylist(self, keylist: Optional[T_ls]=None, record: Optional[T_da]=None) -> 'Daf':
        """
        Change some cells in the rows that have the given keys.

        Only the columns that are keys of the dict are changed. Other cells keep
        their values. A key that is not found is skipped. This is a bulk form of
        `my_daf[key, colname] = value`. The row positions do not change, so the key
        index stays valid.

        Args:
            keylist: The keys of the rows to change.
            record: The new values, as a dict of column name and value. Names that are not columns are ignored.

        Returns:
            This Daf, which has been changed. With no keyfield, rows, keys or record, nothing happens.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.update_by_keylist([1, 3, 9], {'v': 'q'})
            | id | v | n  |
            | -: | -: | -: |
            |  1 | q | 10 |
            |  2 | b | 20 |
            |  3 | q | 30 |
            %% daf rows=3; cols=3; keyfield='id'; name=''
        """

        if record is None or not self.lol or not self.hd or not self.keyfield or not keylist:
            return self

        self._rebuild_kd_if_invalidated()

        for key in keylist:
            if key in self._kd:
                self.update_record_irow(self._kd[key], record)

        # no need to invalidate kd because key is unchanged by definition.

        return self


    def update_record_irow(self, irow: int=-1, record: Optional[T_da]=None) -> 'Daf':
        """
        Change some cells in the row at a position.

        Only the columns that are keys of the dict are changed. Other cells keep their
        values. A negative position counts from the end, as in a list, so the default,
        `-1`, is the last row. A position that is out of range raises `IndexError`, and
        nothing is changed. A Daf with no rows or no columns, and a record of `None`, change nothing.

        Args:
            irow: The row position. -1, the default, is the last row.
            record: The new values, as a dict of column name and value. Names that are not columns are ignored.

        Returns:
            This Daf, which has been changed.

        Raises:
            IndexError: The position is out of range, for a Daf that has rows and columns.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.update_record_irow(1, {'v': 'q'})
            >>> d.iloc(1)
            {'id': 2, 'v': 'q', 'n': 20}
            >>> _ = d.update_record_irow(record={'n': 99})
            >>> d.iloc(-1)
            {'id': 3, 'v': 'c', 'n': 99}
        """

        if record is None or not self.lol or not self.hd:
            return self

        position = irow
        if irow < 0:
            irow += len(self.lol)               # count from the end, as a list does.

        if irow < 0 or irow >= len(self.lol):
            raise IndexError(f"update_record_irow(): row position {position} is out of range for {len(self.lol)} rows.")

        for colname, val in record.items():
            if colname in self.hd:
                self.lol[irow][self.hd[colname]] = record[colname]          # perflint-reviewed (loop-invariant-statement)

            # icol = self.hd.get(colname, -1)
            # if icol >= 0:
                # self.lol[irow][icol] = record_da[colname]

        return self


    def assign_icol(
            self,
            icol: int=-1,
            col_la: Optional[T_la]=None,
            default: Any=''
            ) -> 'Daf':
        """
        Fill a column by position with the values of a list.

        A list that is too short is filled out with `default`. With no list, every cell
        gets `default`. With `icol=-1` a new column is added at the right. If the Daf has
        column names, the new column gets the next spreadsheet name, such as `C`, made
        unique. Use `insert_col()` to add a column with a name of your own.

        Args:
            icol: The column position. -1 adds a column at the right.
            col_la: The values, one for each row.
            default: The value for rows that the list does not reach.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.assign_icol(1, ['x', 'y'], default='D')
            >>> d.col('v')
            ['x', 'y', 'D']
        """
        # from utilities import daf_utils

        if self.lol and (icol < 0 or icol >= len(self.lol[0])):
            self._own_rows()        # a column is added. Rows shared with another Daf must not grow.

        self.lol = daf_utils.assign_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)

        if self.hd and self.lol and len(self.lol[0]) > len(self.hd):
            # a column was added at the right. Name it, so the names match the data.
            self._cols_to_hd(list(self.hd) + [self._new_colname()])

        return self

    def _no_keys_error(self, who: str) -> 'KeysDisabledError':
        """
        Make the error for a key lookup on a Daf that has no keyfield and no key index.

        A Daf with a key index (kd) and no keyfield can look up keys, so both must be missing. Internal use.
        """
        return KeysDisabledError(f"{who}(): key lookups are disabled, as the keyfield is not set and there is no key index (kd).")


    def _keyfield_not_a_column_message(self) -> str:
        """
        Say why key lookups fail when the keyfield is set but is not made of columns of this Daf.

        Internal use. It is called only on the failure path.
        """
        if not self.hd:
            return (f"Key lookups are disabled: the keyfield {self.keyfield!r} is set, but this Daf has no column names. "
                    f"Call set_cols() to name them.")
        return (f"Key lookups are disabled: the keyfield {self.keyfield!r} is not a column of this Daf. "
                f"The columns are {list(self.hd)}. Use set_keyfield() to choose one.")


    def _own_rows(self) -> None:
        """
        Give this Daf its own copy of the rows if any row is shared with another Daf.

        Call this before writing into rows. It is a lazy copy. Nothing is copied when the rows are not shared.
        Internal use.
        """
        if daf_utils.rows_are_shared(self.lol):
            self.lol = [list(row_la) for row_la in self.lol]


    def _new_colname(self) -> str:
        """
        Make a name for a new column that has no name: the next spreadsheet name, such as `C`.

        If that name is taken, a suffix is added, such as `C_1`. Internal use.
        """
        base = daf_utils._calculate_single_column_name(len(self.hd))
        name = base
        suffix = 0
        while name in self.hd:
            suffix += 1
            name = f"{base}_{suffix}"
        return name



    def insert_icol(
            self,
            icol:       int=-1,
            col_la:     Optional[T_la]=None,
            colname:    str='',
            default:    Any=''
            ) -> 'Daf':
        """
        Insert a column at a position and move the later columns right.

        A list that is too short is filled out with `default`. With `icol=-1`, or a
        position beyond the last column, the column is added at the right. Give
        `colname` to name it. Without a name, if the Daf has column names, the column
        gets the next spreadsheet name, such as `C`, made unique. The dtypes are not
        changed. Use `set_keyfield()` if the column is to be the keyfield.

        If the rows are shared with another Daf, as after a selection, this Daf first
        takes its own copies of the rows. The other Daf is not changed.

        Args:
            icol: The column position. -1 adds the column at the right.
            col_la: The values, one for each row.
            colname: The name of the new column.
            default: The value for rows that the list does not reach.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.insert_icol(1, ['x', 'y', 'z'], colname='new').columns()
            ['id', 'new', 'v', 'n']
        """

        # from utilities import daf_utils

        self._own_rows()        # the insert must not change the rows of another Daf.

        self.lol = daf_utils.insert_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)

        if self.hd and not colname:
            colname = self._new_colname()       # the names must match the data.

        if colname:
            if not self.hd:
                self.hd = {}
            if icol < 0 or icol >= len(self.hd):
                icol = len(self.hd)
            hl = list(self.hd.keys())
            hl.insert(icol, colname)
            self.hd = {k: idx for idx, k in enumerate(hl)}

        return self


    def insert_irow(self, irow: Optional[int]=None, row: Optional[Union[T_la, T_da]]=None, default: Any='') -> 'Daf':
        """
        Insert a row at a position and move the later rows down.

        The row is a list of values, or a dict that is placed by column name. A short
        list is filled out with `default`. With the default position, `None`, or with a
        position beyond the last row, the row is added at the end. A negative position,
        such as `-1`, also adds it at the end. This differs from `list.insert()`, which
        puts the item before the last one. Prefer `None`. The key index is rebuilt when
        it is next needed.

        Args:
            irow: The row position. None, the default, adds the row at the end.
            row: The row, as a list or a dict.
            default: The value for cells that a short list does not reach.

        Returns:
            This Daf, which has been changed.

        Raises:
            TypeError: The row is not a list or a dict. This includes the default, None.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.insert_irow(1, {'id': 9, 'v': 'z'})
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  9 | z |    |
            |  2 | b | 20 |
            |  3 | c | 30 |
            %% daf rows=4; cols=3; keyfield='id'; name=''
        """

        # from utilities import daf_utils

        if isinstance(row, list):

            row_la = row

        elif isinstance(row, dict):

            row_da = row
            # create normalize list
            row_la = [row_da.get(col, '') for col in self.hd]

        else:
            raise TypeError(f"insert_irow(): row must be a list or a dict, not {type(row).__name__}.")

        self.lol = daf_utils.insert_row_in_lol_at_irow(irow=-1 if irow is None else irow, row_la=row_la, lol=self.lol, default=default)

        self._invalidate_kd()    # use lazy kd rebuilding
        #self._rebuild_kd()
        return self


    def assign_col(self, colname: str, la: Optional[T_la]=None, default: Any='') -> 'Daf':
        """
        Fill a column by name with the values of a list, or add it if it is new.

        This is `my_daf[:, colname] = values`, and it also adds a column. A list that
        is too short is filled out with `default`. With no list, every cell gets
        `default`. If the column is the keyfield, the key index is rebuilt when it is
        next needed.

        Args:
            colname: The column name.
            la: The values, one for each row.
            default: The value for rows that the list does not reach.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.assign_col('w', default=0).columns()
            ['id', 'v', 'n', 'w']
            >>> d.col('w')
            [0, 0, 0]
        """

        if colname in self.hd:
            self.assign_icol(self.hd[colname], la, default)

        else:
            self.insert_col(
                colname = colname,
                col_la = la,
                #icol: int=-1,
                default = default,
                )
        if self.keyfield == colname:
            self._invalidate_kd()    # use lazy kd rebuilding

        return self

    def insert_col(
            self,
            colname:    str,                    # name of the col
            col_la:     Optional[T_la]=None,    # column to insert
            icol:       int=-1,                 # insert at end by default
            default:    Any='',
            ) -> 'Daf':

        """
        Add a named column at a position, or overwrite it if the name exists.

        A list that is too short is filled out with `default`. With no list, every cell
        gets `default`, so this can add a constant column. If the name already exists
        the column is overwritten, and `icol` is ignored. An empty name does nothing.
        Use `set_keyfield()` if the column is to be the keyfield.

        When a column is added and the rows are shared with another Daf, as after a
        selection, this Daf first takes its own copies of the rows. The other Daf is
        not changed.

        Args:
            colname: The name of the column.
            col_la: The values, one for each row.
            icol: The column position. -1 adds the column at the right.
            default: The value for rows that the list does not reach.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.insert_col('w', ['x', 'y', 'z'], icol=1).columns()
            ['id', 'w', 'v', 'n']
            >>> d.insert_col('k', default=5).col('k')
            [5, 5, 5]
        """

        if not colname:
            return self
        if not col_la:
            col_la = []             # perflint-reviewed (use-tuple-over-list)

        if colname in self.hd:
            # column already exists. ignore icol, overwrite data.
            self.assign_col(colname, col_la, default)

        else:
            self.insert_icol(icol=icol, col_la=col_la, colname=colname, default=default) #, keyfield=keyfield)

        return self


    def insert_idx_col(self, colname: str='idx', icol:int=0, startat:int=0) -> 'Daf':
        """
        Insert a column of row numbers.

        This is `insert_col()`, so rows shared with another Daf are copied first and
        the other Daf is not changed.

        Args:
            colname: The name of the new column.
            icol: The column position.
            startat: The number of the first row.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.insert_idx_col().col('idx')
            [0, 1, 2]
        """

        num_rows = len(self)
        col_la = list(range(startat, startat + num_rows))

        self.insert_col(colname, col_la, icol)

        return self


    def set_col_irows(self, colname: str, irows: T_li, val: Any) -> 'Daf':
        """
        Set one value in the given rows of a named column.

        This is deprecated. Use `my_daf[irows, colname] = value`, which does the same. A
        column name that is not found raises `KeyError`. Row positions that are out of
        range are skipped.

        Args:
            colname: The column name.
            irows: The row positions.
            val: The value to set.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: The column name is not found.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.set_col_irows('v', [0, 2], 'Z').col('v')
            ['Z', 'b', 'Z']
        """

        icol = self.hd[colname]

        self.set_icol_irows(icol, irows, val)

        return self


    def set_icol(self, icol: int, val: Any) -> 'Daf':

        """
        Set one value in every row of a column, by position.

        This is `my_daf[:, icol] = value`.

        Args:
            icol: The column position.
            val: The value to set.

        Returns:
            This Daf, which has been changed.

        Raises:
            IndexError: The position is beyond the end of a row.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.set_icol(1, 'Z').col('v')
            ['Z', 'Z', 'Z']
        """


        for irow in range(len(self.lol)):
            self.lol[irow][icol] = val

        return self


    def set_icol_irows(self, icol: int, irows: T_li, val: Any) -> 'Daf':
        """
        Set one value in the given rows of a column, by position.

        This is `my_daf[irows, icol] = value`. Row positions that are out of range
        are skipped.

        Args:
            icol: The column position.
            irows: The row positions.
            val: The value to set.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.set_icol_irows(1, [0, 9, -1], 'Z')
            >>> d.col('v')
            ['Z', 'b', 'c']
        """

        for irow in irows:
            if irow >= len(self.lol) or irow < 0:
                continue

            self.lol[irow][icol] = val

        return self


    #=========================
    # find/replace

    def find_replace(self, find_pat: str, replace_val: Any) -> 'Daf':
        """
        Replace every cell that matches a pattern, in place.

        Each cell is converted to text and searched with the regular expression
        `find_pat`. If it matches anywhere in the text, the whole cell is replaced by
        `replace_val`. It is not a substitution inside the text. Every column is
        searched, including numbers, and the key index is rebuilt when it is next
        needed.

        Args:
            find_pat: A regular expression.
            replace_val: The value that replaces a matching cell.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> _ = d.find_replace(r'^[ab]$', 'HIT')
            >>> d.col('v')
            ['HIT', 'HIT', 'c']
        """

        for row_la in self.lol:
            for i, value in enumerate(row_la):
                if bool(re.search(find_pat, str(value))):
                    row_la[i] = replace_val

        self._invalidate_kd()

        return self


    def replace_in_columns(
        self,
        cols: Optional[T_lsi],
        find_values: Optional[List[Any]] = None,
        replacement: Any = _MISSING
    ) -> 'Daf':
        """
        Replace listed values with another value, in some columns, in place.

        A cell is replaced when it equals any value in `find_values`. A typical use
        is `['', None]` to fill empty cells. Columns may be given by name or by
        position, or all columns if `cols` is None. With no `find_values` nothing
        happens. The key index is rebuilt if the keyfield may have changed.

        Args:
            cols: Column names or positions. If None, all columns.
            find_values: The values to look for.
            replacement: The value to put in their place. This is required.

        Returns:
            This Daf, which has been changed.

        Raises:
            ValueError: No `replacement` is given.
            KeyError: A column name is not found.
            TypeError: A column is neither a name nor a position.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> d.replace_in_columns(['v'], ['a', 'c'], '-').col('v')
            ['-', 'b', '-']
        """

        if find_values is None:
            return self

        if replacement is _MISSING:
            raise ValueError("replace_in_columns() requires an explicit `replacement` value.")

        if cols is None:
            # process the entire array
            target_indices = list(range(self.num_cols()))
        else:
            target_indices = []
            for col in cols:
                if isinstance(col, int):
                    target_indices.append(col)
                elif isinstance(col, str):
                    if col in self.hd:
                        target_indices.append(self.hd[col])
                    else:
                        raise KeyError(f"Column name '{col}' not found in header.")
                else:
                    raise TypeError(f"Column specifier must be str or int, got {type(col).__name__}")

        for row in self.lol:
            for col_idx in target_indices:
                if row[col_idx] in find_values:
                    row[col_idx] = replacement

        if self.keyfield:
            if not isinstance(self.keyfield, str):
                # composite keyfield (tuple/list of columns) -- conservatively always invalidate,
                # since checking whether any individual component column was touched is more work
                # than it's worth here.
                self._invalidate_kd()
            else:
                keyfield_idx = self.hd.get(self.keyfield)
                if keyfield_idx is None or keyfield_idx in target_indices:
                    self._invalidate_kd()

        return self


    #=========================
    # split and grouping

    def split_daf_into_ranges(self, chunk_ranges: List[Tuple[int, int]]) -> List['Daf']:
        """
        Split the rows into several Daf instances, by position ranges.

        Each range is `(start, end)`, and `end` is not included. The new Daf instances
        share their rows with this one.

        Args:
            chunk_ranges: The ranges of row positions.

        Returns:
            A list of Daf instances, one for each range.

        Examples:
            >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
            >>> parts = d.split_daf_into_ranges([(0, 2), (2, 3)])
            >>> parts[0]
            | id | v | n  |
            | -: | -: | -: |
            |  1 | a | 10 |
            |  2 | b | 20 |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> parts[1]
            | id | v | n  |
            | -: | -: | -: |
            |  3 | c | 30 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
        """

        chunks_lodaf = [self.select_irows(list(range(start, end))) for start,end in chunk_ranges]
        #chunks_lodaf = [self[start:end] for start, end in chunk_ranges]
        return chunks_lodaf



    def split_daf_into_chunks_lodaf(self, max_chunk_size: int) -> List['Daf']:
        """
        Split the rows evenly into Daf instances of at most a given size.

        The sizes are as equal as possible, so some chunks are smaller than the
        maximum. None is larger. The chunks share their rows with this Daf.

        Args:
            max_chunk_size: The most rows in one chunk.

        Returns:
            A list of Daf instances.

        Examples:
            >>> d = Daf(lol=[[i] for i in range(7)], cols=['a'])
            >>> [len(part) for part in d.split_daf_into_chunks_lodaf(3)]
            [3, 2, 2]
        """
        # from utilities import daf_utils

        chunk_sizes_list = daf_utils.calc_chunk_sizes(num_items=len(self), max_chunk_size=max_chunk_size)
        chunk_ranges = daf_utils.convert_sizes_to_idx_ranges(chunk_sizes_list)
        chunks_lodaf = self.split_daf_into_ranges(chunk_ranges)
        return chunks_lodaf


    #=========================
    #   sort

    def sort_by_colname(self, colname:str, *, reverse: bool=False, length_priority: bool=False, as_str: bool=False) -> 'Daf':
        """
        Sort the rows by one column, in place.

        Make a copy first if you need the original order. An empty cell sorts before
        a cell with content, unlike a spreadsheet. Values in the column must be
        comparable with each other. A column that mixes None or text with numbers
        raises `TypeError`, which names the column. Use `as_str=True` to sort by the text
        of each value instead.

        With `length_priority`, a shorter text sorts before a longer one, so numbers
        that are stored as text sort as numbers. Without it, `'10'`, `'100'`, `'8'`
        are in text order. `length_priority` needs text, so for real numbers use it
        with `as_str=True`. Then whole numbers sort in numeric order, and negative and
        decimal numbers sort in text order.

        Args:
            colname: The column to sort by.
            reverse: If True, sort from high to low.
            length_priority: If True, sort by length first, then by value.
            as_str: If True, sort by the text of each value, and None sorts as an empty cell.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: The column name is not found.
            TypeError: The column holds values that cannot be compared, and `as_str` is False.

        Examples:
            >>> d = Daf(lol=[['10'], ['99'], ['8'], ['100']], cols=['a'])
            >>> d.sort_by_colname('a', length_priority=True).col('a')
            ['8', '10', '99', '100']
            >>> d = Daf(lol=[[10], [None], [2], ['']], cols=['n'])
            >>> d.sort_by_colname('n', as_str=True, length_priority=True).col('n')
            [None, '', 2, 10]
        """
        if not self or len(self) <= 1:
            return self

        colidx = self.hd[colname]

        try:
            self.lol = daf_utils.sort_lol_by_col(self.lol, colidx, reverse=reverse, length_priority=length_priority, as_str=as_str)
        except TypeError as exc_info:
            raise TypeError(
                f"sort_by_colname(): column '{colname}' holds values that cannot be compared, such as numbers "
                f"with text or None. Use as_str=True to sort by the text of the values, or convert the column "
                f"with apply_dtypes().") from exc_info

        self._invalidate_kd()    # use lazy kd rebuilding
        return self


    def sort_by_colnames(self, colnames:T_ls, reverse: bool=False, length_priority: bool=False, as_str: bool=False) -> 'Daf':
        """
        Sort the rows by several columns, in place.

        The first column is the main sort key. The others break ties. See
        `sort_by_colname()` for the rules of ordering, for `length_priority` and for `as_str`.
        Calling `sort_by_colname()` for each column, last column first, gives the same
        order.

        Args:
            colnames: The columns to sort by, main key first.
            reverse: If True, sort from high to low.
            length_priority: If True, sort by length first, then by value.
            as_str: If True, sort by the text of each value, and None sorts as an empty cell.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: A column name is not found.
            TypeError: A column holds values that cannot be compared, and `as_str` is False.

        Examples:
            >>> d = Daf(lol=[[2, 'b'], [1, 'z'], [1, 'a']], cols=['p', 'q'])
            >>> d.sort_by_colnames(['p', 'q'])
            | p | q |
            | -: | -: |
            | 1 | a |
            | 1 | z |
            | 2 | b |
            %% daf rows=3; cols=2; keyfield=''; name=''
        """
        if not self or len(self) <= 1:
            return self

        colidxs = [self.hd[colname] for colname in colnames]

        try:
            self.lol = daf_utils.sort_lol_by_cols(self.lol, colidxs, reverse=reverse, length_priority=length_priority, as_str=as_str)
        except TypeError as exc_info:
            raise TypeError(
                f"sort_by_colnames(): one of the columns {list(colnames)} holds values that cannot be compared, "
                f"such as numbers with text or None. Use as_str=True to sort by the text of the values, or convert "
                f"the columns with apply_dtypes().") from exc_info

        #self._rebuild_kd()
        self._invalidate_kd()    # use lazy kd rebuilding
        return self


    #=========================
    #   apply formulas

    def apply_formulas(self, formulas_daf: 'Daf') -> 'Daf':
        """
        Fill cells from spreadsheet like formulas, in place.

        `formulas_daf` is a Daf of the same shape. Each cell holds a Python
        expression as text. An empty cell is skipped. The result of each expression is
        stored in the same cell of this Daf. The formulas are evaluated again and again
        until no cell changes, so a cell may use the result of another. A circular
        set of formulas raises `RuntimeError` after 100 passes.

        In a formula, `$d` is this Daf, `$r` is the row number of the cell and `$c` is
        its column number. References are absolute unless you build them from `$r` and
        `$c`. So `sum($d[$r, :$c])` is the sum of the cells to the left in the same row.

        Other examples:

            $d[14,20]+$d[15,25]       the sum of two cells
            max(0,$d[($r-1),$c])      the cell above, but not below 0
            $d[($r-1),$c] * 0.15      15 percent of the cell above

        Warning: the formulas are run with `eval()`. Never use formulas from a source
        you do not trust.

        An error in a formula prints the cell and the formula, and is raised again. The
        `retmode` of this Daf is restored, and the key index is rebuilt when it is next
        needed. The cells that were already changed stay changed.

        Args:
            formulas_daf: The formulas, with the same shape as this Daf.

        Returns:
            This Daf, which has been changed.

        Raises:
            RuntimeError: The shapes differ, or the formulas never settle.

        Examples:
            >>> d = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
            >>> f = Daf(cols=['A', 'B', 'C'], lol=[
            ...     ['', '', 'sum($d[$r,:$c])'],
            ...     ['', '', 'sum($d[$r,:$c])'],
            ...     ['', '', 'sum($d[$r,:$c])'],
            ...     ['sum($d[:$r,$c])', 'sum($d[:$r,$c])', 'sum($d[:$r,$c])']])
            >>> _ = d.apply_formulas(f)
            >>> d
            | A  | B  | C  |
            | -: | -: | -: |
            |  1 |  2 |  3 |
            |  4 |  5 |  9 |
            |  7 |  8 | 15 |
            | 12 | 15 | 27 |
            %% daf rows=4; cols=3; keyfield=''; name=''
        """

        # TODO: This algorithm is not optimal. Ideally, a dependency tree would be formed and
        # cells modified in from those with no dependencies to those that depend on others.
        # This issue will not become a concern unless the number of formulas is substantial.

        if not self:
            return self

        if self.shape() != formulas_daf.shape():
            raise RuntimeError("apply_formulas requires data arrays of the same shape.")

        lol_changed = True     # must evaluate at least once.
        loop_limit = 100
        loop_count = 0

        # the following deals with $d, $r, $c in the formulas
        parsed_formulas_daf = formulas_daf._parse_formulas()

        # we must use RETMODE_VAL for formulas to work easily for users.
        prior_retmode = self.retmode
        self.retmode = self.RETMODE_VAL

        try:
            while lol_changed:
                lol_changed = False
                loop_count += 1
                if loop_count > loop_limit:
                    raise RuntimeError("apply_formulas is resulting in excessive evaluation loops.")      # perflint-reviewed (loop-invariant-statement)

                for irow in range(len(self.lol)):
                    for icol in range(self.num_cols()):
                        cell_formula = parsed_formulas_daf.lol[irow][icol]          # perflint-reviewed (loop-invariant-statement)
                        if not cell_formula:
                            # no formula provided -- do nothing
                            continue
                        try:
                            new_value = eval(cell_formula)
                        except Exception as exc_info:
                            print(f"Error in formula for cell [{irow},{icol}]: '{cell_formula}': '{exc_info}'")
                            raise

                        if new_value != self.lol[irow][icol]:
                            # update the value in the array, and set lol_changed flag
                            self.lol[irow][icol] = new_value
                            lol_changed = True
                        else:
                            continue
        finally:
            self.retmode = prior_retmode
            self._invalidate_kd()       # cells may have changed, even if there was an error.

        #self._rebuild_kd()

        return self


    def _parse_formulas(self) -> 'Daf':

        # start with unparsed formulas
        parsed_formulas = copy.deepcopy(self)

        for irow in range(len(self.lol)):
            for icol in range(self.num_cols()):
                proposed_formula = self.lol[irow][icol]
                if not proposed_formula:
                    # no formula provided.
                    continue

                proposed_formula = proposed_formula.replace('$d', 'self')
                proposed_formula = proposed_formula.replace('$c', str(icol))
                proposed_formula = proposed_formula.replace('$r', str(irow))    # perflint-reviewed (loop-invariant-statement)

                parsed_formulas[irow,icol] = proposed_formula

        return parsed_formulas



    def cols_to_dol(self, colname1: str, colname2: str) -> T_dola:
        """
        Make a lookup from the values of one column to the values of another.

        For each value in `colname1`, the result lists the different values that appear
        with it in `colname2`, in the order first seen. Use it to see how two columns
        relate. The values must be hashable. If a name is not a column, or the Daf is
        empty, the result is empty.

        Args:
            colname1: The column of keys.
            colname2: The column of values.

        Returns:
            A dict that maps each value of the first column to a list of values of the second.

        Examples:
            >>> d = Daf(lol=[['a', 'b'], ['b', 'd'], ['a', 'f'], ['b', 'd']], cols=['c1', 'c2'])
            >>> d.cols_to_dol('c1', 'c2')
            {'a': ['b', 'f'], 'b': ['d']}
        """

        if colname1 not in self.hd or colname2 not in self.hd or not self.lol:
            return {}

        colidx1 = self.hd[colname1]
        colidx2 = self.hd[colname2]


        # first work with dict of dict for speed.
        result_dadn: Dict[Any, Dict[Any, None]] = {}

        for la in self.lol:
            val1 = la[colidx1]
            val2 = la[colidx2]
            if val1 not in result_dadn:
                result_dadn[val1] = {val2: None}
            elif val2 not in result_dadn[val1]:
                result_dadn[val1][val2] = None
            # otherwise, it is already in the result.

        # now convert dadn to dola

        result_dola = {k: list(d.keys()) for k, d in result_dadn.items()}

        return result_dola


    def insert_dif_row(self,
            irow1: int,
            irow2: Optional[int]=None,
            irow_insert: Optional[int]=None,
            cols: Optional[T_ls]=None,
            ) -> 'Daf':

        """
        Insert a row that holds the difference of two rows.

        The difference is the first row minus the second row, for the numeric columns
        you name. Other columns of the new row are empty. A cell that is empty counts
        as 0. By default the second row is the one after the first, and the new row goes
        between them. Columns that hold text must not be in `cols`.

        Args:
            irow1: The position of the first row.
            irow2: The position of the second row. If None, the row after the first.
            irow_insert: Where to insert the new row. If None, at `irow2`.
            cols: The columns to subtract. If None, all columns.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
            >>> d.insert_dif_row(0)
            | a  | n  |
            | -: | -: |
            |  1 | 10 |
            | -2 | -4 |
            |  3 | 14 |
            |  6 | 20 |
            %% daf rows=4; cols=2; keyfield=''; name=''
        """

        if irow2 is None:
            irow2 = irow1 + 1
        if irow_insert is None:
            irow_insert = irow2
        if cols is None:
            cols = self.columns()

        # calculate the difference.
        diff_result_da = type(self).diff_da(self[irow1].to_dict(), self[irow2].to_dict(), keys=cols)

        # this function handles normalizing the columns before insertion
        self.insert_irow(irow=irow_insert,  row=diff_result_da)

        return self


    def insert_dif_rows(self,
            irows_rli: Optional[T_rli]=None,
            cols: Optional[T_ls]=None,
            offset: int=0
            ) -> 'Daf':

        """
        Insert a difference row after each of several rows.

        For each position, a row is inserted that holds that row minus the next row,
        as in `insert_dif_row()`. Do not list the last row. By default all rows are
        used. With `offset=1` the new row goes after the next row, not between them.

        Args:
            irows_rli: The positions of the first rows. If None, every row but the last.
            cols: The columns to subtract. If None, all columns.
            offset: 0 inserts between the two rows. 1 inserts after the second.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
            >>> d.insert_dif_rows()
            | a  | n  |
            | -: | -: |
            |  1 | 10 |
            | -2 | -4 |
            |  3 | 14 |
            | -3 | -6 |
            |  6 | 20 |
            %% daf rows=5; cols=2; keyfield=''; name=''
        """

        reversed_irows_rli: Union[range, T_li]
        if irows_rli is None:
            reversed_irows_rli = range(len(self) - 2, -1, -1)
        else:
            reversed_irows_rli = sorted(irows_rli, reverse=True)

        for irow in reversed_irows_rli:

            self.insert_dif_row(irow1=irow, irow_insert=irow + 1 + offset, cols=cols)

        return self

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
            >>> a = Daf(lol=[[1, 'x'], [2, 'y']], cols=['id', 'v'], keyfield='id')
            >>> o = Daf(lol=[[1, 'P'], [2, 'Q']], cols=['id', 'w'], keyfield='id')
            >>> a.annotate_daf(o, {'v': 'w'})
            | id | v |
            | -: | -: |
            |  1 | P |
            |  2 | Q |
            %% daf rows=2; cols=2; keyfield='id'; name=''
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

            rowkey = my_rec_klist[my_keyfield]

            other_rec = other_daf.select_record(rowkey)

            for my_field, other_field in my_to_other_dict.items():
                my_rec_klist[my_field] = other_rec[other_field]

        # keys still valid

        return self


    #===============================
    # apply and reduce

    def apply(
            self,
            # called as func(self, **kwargs) for by='table', func(row, **kwargs) for by='row'.
            # by='table' is a pure passthrough of func's return value (any type -- see
            # test_daf_misc.py, where it returns a bare int), so func's return type isn't
            # pinned down here.
            func:       Callable[..., Any],
            by:         str='row',
            keylist:    Optional[Union[T_la, T_lota]]=None,     # list of keys of rows to include (DEPRECATE?)
            **kwargs:   Any,
                # kwargs may commonly include:
                # cols: Optional[T_la]=None,                    # columns included in the apply operation.
            ) -> "Daf":
        """
        Apply a function to each row and collect the results in a new Daf.

        The function gets a row and returns the new row as a dict. If it returns an
        empty dict or None, that row is left out. The new Daf takes its columns from
        the first row returned. It has no keyfield. This Daf is not changed.

        With `by='table'` the function gets the whole Daf, and its result is returned
        as it is. Use that to run any function on the table. `by='col'` is not
        supported.

        The function gets the rows as dicts, or as KeyedList objects if `itermode` is
        `keyedlist`. The extra keyword arguments are passed on to it. To work on only
        some rows, give the `keylist`, or select them first.

        Args:
            func: The function to apply. It takes a row, or the Daf, and the keyword arguments.
            by: `row` to apply to each row, or `table` to apply to the whole Daf.
            keylist: Keys of the rows to include. All rows if None. This may be removed.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            The new Daf, or the result of the function for `by='table'`.

        Raises:
            NotImplementedError: `by` is `col`, or is not recognized.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.apply(lambda row: {'g': row['g'].upper(), 'z': row['y'] * 2})
            | g | z  |
            | -: | -: |
            | A | 20 |
            | B | 40 |
            | A | 60 |
            %% daf rows=3; cols=2; keyfield=''; name=''
            >>> d.apply(lambda row: row if row['y'] > 15 else None)
            | g | x | y  |
            | -: | -: | -: |
            | b | 2 | 20 |
            | a | 3 | 30 |
            %% daf rows=2; cols=3; keyfield=''; name=''
        """
        if by == 'table':
            # by contract (see docstring) func returns a 'Daf' when by='table', though this
            # is a pure passthrough and isn't actually enforced (a test exercises it with a
            # func returning a bare int).
            return func(self, **kwargs)  # type: ignore[no-any-return]

        result_daf = Daf()

        if by == 'row':
            if keylist is None:
                keylist = []                        # perflint-reviewed

            keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)

            self._rebuild_kd_if_invalidated()

            for row in self:
                # self.keyfield here always names a single field (composite/multi-col keyfields
                # aren't used for this row-membership check).
                if self.keyfield and keylist_or_dict and row[cast(str, self.keyfield)] not in keylist_or_dict:
                    continue
                transformed_row = func(row, **kwargs)
                result_daf.append(transformed_row)          # Will not append an empty row.

        elif by == 'col':
            # this is not working yet, don't know how to handle cols, for example.
            raise NotImplementedError

            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                transformed_col = func(col_la, **kwargs)
                result_daf.insert_icol(icol, transformed_col)
        else:
            raise NotImplementedError

        # Rebuild the internal data structure (if needed)
        # result_daf._rebuild_kd()
        result_daf._invalidate_kd()   # use lazy rebuilding of kd.

        return result_daf


    @staticmethod
    def update_row(row: T_ma, da: T_da) -> T_ma:
        """
        Update a row with the items of a dict, and return the row.

        This is a small helper to use inside `apply()`, as in
        `d.apply(lambda row: Daf.update_row(row, {'z': 0}))`. It is a static method.

        Args:
            row: The row, as a dict. It is changed.
            da: The items to put in the row.

        Returns:
            The same row.

        Examples:
            >>> Daf.update_row({'a': 1}, {'b': 2})
            {'a': 1, 'b': 2}
        """

        row.update(da)
        return row


    def apply_in_place(
            self,
            # called as func(row, **kwargs) -- see the apply() comment on Callable[...] above.
            func:       Callable[..., Union[T_ma, None]],
            by:         str='row',
            rowkeys:    Union[T_la, T_lota] | None=None,  # list of rowkeys to include.
                        # the above changed from keylist to avoid confusion with KeyedList
            **kwargs:   Any,
            ) -> 'Daf':
        """
        Apply a function to each row and store the results in this Daf.

        With `by='row'` the function gets each row as a dict. It must return a dict.
        Its values are written back by column name. A column that the dict lacks
        keeps its value, and a key that is not a column is ignored. The order of
        the keys does not matter, and the row keeps its length.

        With `by='row_klist'` the function gets each row as a
        [KeyedList][daffodil.keyedlist.KeyedList]. It changes the row and returns
        nothing, so no dict is built for each row. The cost of `row` grows with the
        number of columns, because each row is turned into a dict of all its columns.
        The cost of `row_klist` does not. In a test that changed one column, the two
        took the same time at about 6 columns. At 1,000 columns and 2,000 rows,
        `row_klist` took 0.005 s and `row` took 0.245 s. With 3 columns and 200,000
        rows, `row` was faster, 0.25 s against 0.30 s. Add or delete no keys in the
        KeyedList, because that changes the length of the row, not the columns.

        The key index is rebuilt when it is next needed. Use `apply()` to get a new Daf.

        Args:
            func: The function to apply to each row. It takes a row and the keyword arguments.
            by: `row` or `row_klist`.
            rowkeys: Keys of the rows to include. All rows if None. The Daf needs a keyfield.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            This Daf, which has been changed.

        Raises:
            ValueError: With `by='row'` the function returned None.
            NotImplementedError: `by` is not `row` or `row_klist`.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> _ = d.apply_in_place(lambda row: {**row, 'y': row['y'] + 1})
            >>> d.col('y')
            [11, 21, 31]
            >>> _ = d.apply_in_place(lambda row: row.__setitem__('y', 0), by='row_klist')
            >>> d.col('y')
            [0, 0, 0]
        """
        if rowkeys is None:
            rowkeys = []
        # here we create a dict if the rowkeys to search are numerous.
        rowkeys_list_or_dict = rowkeys if (not rowkeys or
                                        isinstance(rowkeys, dict)
                                        or len(rowkeys) < 30
                                    ) else dict.fromkeys(rowkeys)

        if by == 'row':

            self._rebuild_kd_if_invalidated()

            hd = self.hd
            for row_la, row_da in zip(self.lol, self):
                if rowkeys_list_or_dict and self.keyfield and row_da[cast(str, self.keyfield)] not in rowkeys_list_or_dict:
                    continue
                transformed_row_da = func(row_da, **kwargs)
                if transformed_row_da is None:
                    raise ValueError("apply_in_place: func must return a row for by='row' (None is only valid for by='row_klist')")
                # write the returned values back by column name. The row keeps its length and its
                # column order. A name that is not a column is ignored. A column that is not
                # returned keeps its value.
                for colname, val in transformed_row_da.items():
                    icol = hd.get(colname)  # type: ignore[call-overload]  # a row key from a KeyedList is Hashable
                    if icol is not None:
                        row_la[icol] = val

        elif by == 'row_klist':

            for idx, row_klist in enumerate(self.iter_klist()):
                if rowkeys_list_or_dict and self.keyfield and row_klist[cast(str, self.keyfield)] not in rowkeys_list_or_dict:
                    continue

                # func should return nothing and instead mutate row_klist, which will mutate the row in the array.
                func(row_klist, **kwargs)

        else:
            raise NotImplementedError

        # Rebuild the internal data structure (if needed)
        # note, this should not be necessary. apply_in_place should not modify the keyfield column.
        self._invalidate_kd()   # use lazy rebuilding of kd.

        # self._rebuild_kd()

        return self


    # def reduce(
            # self,
            # func: Callable[[T_da, T_da], Union[T_da, T_la]],
            # by: str='row',
            # cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> Union[T_da, T_la]:
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


    def manifest_apply(
            self,
            func: Callable[[T_da, Optional[T_la]], Tuple[T_da, 'Daf']],    # function to apply according to 'by' parameter
            load_func: Callable[[T_ma], 'Daf'],            # optional function to load data for each manifest entry, defaults to local file system
            save_func: Callable[[T_da, 'Daf'], str],       # optional function to save data for each manifest entry, defaults to local file system
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the apply operation.
             **kwargs: Any,
            ) -> "Daf":
        """
        Run a function on each chunk that a manifest lists, and save the results.

        A manifest is a Daf in which each row describes one chunk of data. For each
        row, `load_func` loads the chunk as a Daf. `func` is applied to it with
        `by='table'`, so it gets the loaded Daf and the keyword `cols`. It returns a
        tuple of a dict that describes the result chunk and the new Daf. `save_func`
        saves the new Daf. The result manifest has one row for each dict.

        Args:
            func: Gets a loaded Daf. Returns a dict that describes the result and the new Daf.
            load_func: Loads the chunk that a manifest row describes.
            save_func: Saves a new Daf, given the dict that describes it.
            by: Must be `table`. The default is `row`, which applies `func` to each row, so always pass `by='table'`.
            cols: Passed to `func` as the keyword `cols`.
            **kwargs: Keyword arguments passed on to `func`.

        Returns:
            The manifest of the result chunks.

        Examples:
            >>> chunks = {'a': Daf(cols=['n'], lol=[[1], [2]]), 'b': Daf(cols=['n'], lol=[[10]])}
            >>> saved = []
            >>> def load(spec):
            ...     return chunks[spec['chunk']]
            >>> def double(daf, cols=None):
            ...     new = Daf(cols=['n'], lol=[[row['n'] * 2] for row in daf])
            ...     return {'rows': len(new)}, new
            >>> def save(spec, daf):
            ...     saved.append(daf.to_lod())
            ...     return 'saved'
            >>> manifest = Daf(cols=['chunk'], lol=[['a'], ['b']])
            >>> result = manifest.manifest_apply(double, load, save, by='table')
            >>> result
            | rows |
            | ---: |
            |    2 |
            |    1 |
            %% daf rows=2; cols=1; keyfield=''; name=''
            >>> saved
            [[{'n': 2}, {'n': 4}], [{'n': 20}]]
        """

        result_manifest_daf = Daf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Daf table
            loaded_daf = load_func(chunk_spec)

            # Apply the function to the loaded Daf. apply()'s by='table' mode is a pure
            # passthrough of func's return value, so at by='table' this really returns
            # func's declared Tuple[T_da, 'Daf'] rather than apply()'s nominal 'Daf'.
            # No caller/test currently exercises manifest_apply -- untested.
            result_chunk_spec, transformed_daf = cast(Tuple[T_da, 'Daf'], loaded_daf.apply(func, by=by, cols=cols, **kwargs))

            # Save the resulting Daf table
            save_func(result_chunk_spec, transformed_daf)

            # Update the manifest with information about the resulting chunk
            result_manifest_daf.append(result_chunk_spec)

        return result_manifest_daf


    def manifest_reduce(
            self,
            # func is passed through to Daf.reduce() (row/reduction/cols/**kwargs), not called
            # directly here, so its exact arity isn't pinned down at this level.
            func: Callable[..., Any],
            load_func: Optional[Callable[[T_ma], 'Daf']] = None,
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> T_da:
        """
        Reduce the chunks that a manifest lists into one row.

        Each chunk is loaded with `load_func` and reduced with `reduce()`. The
        reductions are put in a Daf, and that is reduced again with the same function.
        This works for functions such as `sum_da()` that can be combined in this way.

        Args:
            func: The reduction function. See `reduce()`.
            load_func: Loads the chunk that a manifest row describes. It is required.
            by: How the function is applied. See `reduce()`.
            cols: The columns to reduce. All columns if None.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            The reduced row, as a dict.

        Raises:
            ValueError: No `load_func` is given.

        Examples:
            >>> store = {'c1': Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b']), 'c2': Daf(lol=[[10, 20]], cols=['a', 'b'])}
            >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
            >>> manifest.manifest_reduce(Daf.sum_da, load_func=lambda spec: store[spec['chunk']])
            {'a': 14, 'b': 26}
        """
        if load_func is None:
            raise ValueError("manifest_reduce: load_func is required")

        # collects one reduction per chunk; its columns come from the chunks, not the manifest.
        first_reduction_daf = type(self)()

        for chunk_spec in self:
            # Load the specified Daf table
            loaded_daf = load_func(chunk_spec)

            # Apply the function to the loaded Daf
            reduction_ma = loaded_daf.reduce(func, by=by, cols=cols, **kwargs)

            first_reduction_daf.append(reduction_ma)

        final_reduction_ma = first_reduction_daf.reduce(func, by=by, cols=cols, **kwargs)

        return cast(T_da, final_reduction_ma)


    def manifest_process(
            self,
            # called as func(chunk_spec, **kwargs) -- see the apply() comment on Callable[...] above.
            func: Callable[..., T_da],   # function to run for each hunk specified by the manifest
            **kwargs: Any,
            ) -> 'Daf':                                    # records describing metadata of each hunk
        """
        Call a function once for each chunk that a manifest lists.

        The function gets the manifest row as a dict, and does its own loading and
        saving. It returns a dict of information about what it did. These dicts are
        collected as the rows of the result.

        Args:
            func: Gets a manifest row and returns a dict of results.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            A Daf with one row for each returned dict.

        Examples:
            >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
            >>> manifest.manifest_process(lambda spec: {'chunk': spec['chunk'], 'seen': True})
            | chunk | seen |
            | ----: | ---: |
            |    c1 | True |
            |    c2 | True |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """

        result_daf = Daf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Daf table
            # Apply the function to the loaded Daf
            result_da = func(chunk_spec, **kwargs)

            # Update the manifest with information about the resulting chunk
            result_daf.append(result_da)

        return result_daf


    def _cols_scope(self, cols: Optional[T_cs]) -> Tuple[List[str], List[int], bool]:
        """
        Work out which columns a grouping keeps: their names, their positions, and whether that is all of them.

        With no `cols`, all columns are kept. A name that is not a column raises `KeyError`. Internal use.
        """
        if not cols:
            return list(self.hd), list(range(len(self.hd))), True

        names = list(cols)
        idxs = [self.hd[name] for name in names]

        return names, idxs, names == list(self.hd)


    def _reduce_scope(self, by: str, reduce_cols: Optional[T_cs], kwargs: Dict[str, Any]) -> Optional[T_ls]:
        """
        Work out the columns that the groups need for a reduction, or None for all columns.

        Only a reduction by row, or by sparse row, looks at just the columns in `reduce_cols`.
        For a sparse row the `indirect_col` is kept as well. A name that is not a column is
        left out, as `reduce()` does. If none are columns, all columns are kept. Internal use.
        """
        if not reduce_cols or by not in ('row', 'sparse_row'):
            return None

        names = [col for col in reduce_cols if col in self.hd]

        indirect_col = kwargs.get('indirect_col')
        if names and indirect_col and indirect_col in self.hd and indirect_col not in names:
            names.append(indirect_col)

        return names or None


    def _new_group_daf(self, rows: T_lola, names: T_ls, all_cols: bool) -> 'Daf':
        """
        Make the Daf for one group, with the layout of this Daf, or of the columns kept.

        The keyfield is kept only if every column of it is kept. The dtypes are cut to the
        columns kept. Internal use.
        """
        group_daf = self.clone_empty(lol=rows, cols=names)

        if not all_cols:
            if self.dtypes:
                group_daf.dtypes = {col: typ for col, typ in self.dtypes.items() if col in names}

            key_cols = [self.keyfield] if isinstance(self.keyfield, (str, int)) else list(self.keyfield or [])
            if not all(key_col in names for key_col in key_cols):
                group_daf.keyfield = ''

        return group_daf


    def groupby(
            self,
            colname: str='',
            colnames: Optional[T_ls]=None,
            omit_nulls: bool=False,         # do not group to values in column that are null ('')
            cols: Optional[T_cs]=None,
            ) -> Union[Dict[str, 'Daf'], Dict[Tuple[str, ...], 'Daf']]:

        """
        Split the Daf into several Daf instances, one for each value of a column.

        The result is a dict. Each key is a value found in the column, in the order
        first seen. Each value is a Daf of the rows that have it, with all columns, or
        only the columns in `cols`. The rows are new lists, so changing a cell in a group
        does not change this Daf.

        Use `cols` when you need only a few of many columns. The other columns are never
        copied, and this Daf is not changed. It is much faster for a wide table.

        With several columns, as a list, the keys are tuples of their values. See
        `groupby_cols()`. With `omit_nulls`, rows that have an empty value in the
        column are left out.

        Args:
            colname: The column to group by.
            colnames: Several columns to group by. Use this or `colname`.
            omit_nulls: If True, leave out rows that have an empty value.
            cols: The columns that each group keeps, in this order. If None, all columns.
                A keyfield is kept only if all its columns are kept.

        Returns:
            A dict that maps each value, or tuple of values, to a Daf.

        Raises:
            KeyError: The group column, or a name in `cols`, is not a column.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> groups = d.groupby('g')
            >>> list(groups)
            ['a', 'b']
            >>> groups['a']
            | g | x | y  |
            | -: | -: | -: |
            | a | 1 | 10 |
            | a | 3 | 30 |
            %% daf rows=2; cols=3; keyfield=''; name=''
            >>> d.groupby('g', cols=['y'])['a']
            | y  |
            | -: |
            | 10 |
            | 30 |
            %% daf rows=2; cols=1; keyfield=''; name=''
        """

        if isinstance(colname, list) and not colnames:
            return self.groupby_cols(colnames=colname, cols=cols)
        elif colnames and not colname:
            if len(colnames) > 1:
                return self.groupby_cols(colnames=colnames, cols=cols)
            else:
                colname = colnames[0]
                # can continue below.

        if not self.lol:
            return {}

        names, idxs, all_cols = self._cols_scope(cols)
        group_idx = self.hd[colname]

        groups: Dict[Any, T_lola] = {}

        for row_la in self.lol:
            fieldval = row_la[group_idx]
            if omit_nulls and fieldval is NULL:
                continue

            group_lol = groups.get(fieldval)
            if group_lol is None:
                group_lol = groups[fieldval] = []

            group_lol.append(list(row_la) if all_cols else [row_la[idx] for idx in idxs])

        return {fieldval: self._new_group_daf(group_lol, names, all_cols) for fieldval, group_lol in groups.items()}


    def groupby_cols(self, colnames: T_ls, cols: Optional[T_cs]=None) -> Dict[Tuple[str, ...], 'Daf']:

        """
        Split the Daf by the values of several columns.

        The result is a dict. Each key is a tuple of the values in the columns, even
        for one column. Each value is a Daf of the rows that have them, with all
        columns, or only the columns in `cols`. With all columns, the rows are not
        copied. With `cols`, they are new lists of the columns kept, and the other
        columns are never copied.

        Args:
            colnames: The columns to group by.
            cols: The columns that each group keeps, in this order. If None, all columns.
                A keyfield is kept only if all its columns are kept.

        Returns:
            A dict that maps each tuple of values to a Daf.

        Raises:
            KeyError: A name in `colnames` or `cols` is not a column.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> list(d.groupby_cols(['g']))
            [('a',), ('b',)]
            >>> d.groupby_cols(['g'], cols=['y'])[('a',)]
            | y  |
            | -: |
            | 10 |
            | 30 |
            %% daf rows=2; cols=1; keyfield=''; name=''
        """

        names, idxs, all_cols = self._cols_scope(cols)
        group_idxs = [self.hd[colname] for colname in colnames]

        groups: Dict[Tuple[Any, ...], T_lola] = {}

        for row_la in self.lol:
            fieldval_tuple = tuple([row_la[idx] for idx in group_idxs])

            group_lol = groups.get(fieldval_tuple)
            if group_lol is None:
                group_lol = groups[fieldval_tuple] = []

            group_lol.append(row_la if all_cols else [row_la[idx] for idx in idxs])

        return {fieldval_tuple: self._new_group_daf(group_lol, names, all_cols) for fieldval_tuple, group_lol in groups.items()}


    def group_where(
            self,
            where: Callable[[Any], Any],
            *,
            indirect_col: Optional[str] = None,
        ) -> Dict[Any, 'Daf']:

        """
        Group the rows by the result of a function.

        The function gets each row and returns a key, or a list of keys, or None. A row
        goes into the group of each key. A list of keys puts the same row in several
        groups. None leaves the row out. The groups are Daf instances in a dict. The rows
        are not copied, so changing a cell in a group changes it here too.

        Args:
            where: The function. It gets a row and returns None, a key, or an iterable of keys.
            indirect_col: A column that holds a dict, to read names that are not columns from.

        Returns:
            A dict that maps each key to a Daf.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> groups = d.group_where(lambda row: [row['g'], 'all'])
            >>> {key: len(group) for key, group in groups.items()}
            {'a': 2, 'all': 3, 'b': 1}
        """

        # --- helpers ---

        def _is_iterable(val: Any) -> bool:
            return isinstance(val, Iterable) and not isinstance(val, (str, bytes))

        # --- iterator selection ---

        if indirect_col:
            # reuse existing indirect semantics via wrapper
            def _iter_rows() -> Iterator['_IndirectRowView']:
                for row_kl in self.iter_klist():
                    yield _IndirectRowView(row_kl, indirect_col)
            row_iter: Iterator[Any] = _iter_rows()
        else:
            # prefer KeyedList iteration for zero-copy
            row_iter = self.iter_klist()

        # --- grouping ---

        dodaf: Dict[Any, 'Daf'] = {}

        for row in row_iter:

            keys = where(row)

            if keys is None:
                continue

            if not _is_iterable(keys):
                keys = [keys]
            else:
                # if iterable but empty, skip early
                # (avoid constructing list unless needed)
                try:
                    iterator = iter(keys)
                    first = next(iterator)
                except StopIteration:
                    continue
                # reconstruct iterable including first element
                keys = (k for k in (first, *iterator))

            for key in keys:

                sub_daf = dodaf.get(key)
                if sub_daf is None:
                    sub_daf = Daf(name=str(key), keyfield=self.keyfield)
                    dodaf[key] = sub_daf

                # append original KeyedList row (no copy)
                # if indirect view was used, unwrap to underlying row
                if isinstance(row, _IndirectRowView):
                    sub_daf._basic_append(row.row)
                else:
                    sub_daf._basic_append(row)


        return dodaf



    def groupby_cols_reduce(
            self,
            groupby_colnames: T_ls,
            # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
            # manifest_reduce comment on Callable[...] above.
            func: Callable[..., Any],
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,               # columns included in the reduce operation.
            diagnose: bool = False,
            **kwargs: Any,
            ) -> 'Daf':

        """
        Group the rows by several columns and reduce each group to one row.

        Use it to total the numbers for each combination of a few identifying columns.
        The result has one row for each combination. It has the group columns first and
        then the `reduce_cols`. It has no keyfield. For each group, `func` is used as in
        `reduce()`.

        With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
        columns, and the `indirect_col` if there is one. A function that reads another
        column of the row will not find it.

        For example, group by gender, religion and zip code, and sum the counts of
        several causes in each group. The number of rows is the number of combinations
        that occur.

        Args:
            groupby_colnames: The columns that identify a group.
            func: The reduction function. See `reduce()`.
            by: How the function is applied. See `reduce()`.
            reduce_cols: The columns to reduce.
            diagnose: If True, print progress messages.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            The Daf with one row for each group.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['y'])
            | g | y  |
            | -: | -: |
            | a | 40 |
            | b | 20 |
            %% daf rows=2; cols=2; keyfield=''; name=''
        """
        # unit test exists.
        """
            
            This can be commonly used when some colnames are important for grouping, while others
            contain values or numeric data that can be reduced.
            
            For example, consider the data table with the following columns:
            
            gender, religion, zipcode, cancer, covid19, gun, auto
            
            The data can be first grouped by the attribute columns gender, religion, zipcode, and then
            then prevalence of difference modes of death can be summed. The result is a daf with one
            row per unique combination of gender, religion, zipcode. Say we consider just M/F, C/J/I, 
            and two zipcodes 90001, and 90002, this would result in the following rows, where the 
            values in paranthesis are the reduced values for each of the numeric columns, such as the sum.
            
            In general, the number of rows is reduced to the product of number of unique values in each column
            grouped. In this case, there are 2 genders, 3 religions, and 2 zipcodes, resulting in
            2 * 3 * 2 = 12 rows.
            
            groupby_colnames = ['gender', 'religion', 'zipcode']
            reduce_colnames  = ['cancer', 'covid19', 'gun', 'auto']
            
            grouped_and_summed = data_table.groupby_cols_reduce(
                groupby_colnames=['gender', 'religion', 'zipcode'], 
                func = sum_np(),
                by='table',                                     # determines how the func is applied.
                reduce_cols = reduce_colnames,                  # columns included in the reduce operation.
                )

            
            cols = ['gender', 'religion', 'zipcode', 'cancer', 'covid19', 'gun', 'auto']
            lol = [
            ['M', 'C', 90001,  1,  2,  3,  4],
            ['M', 'C', 90001,  5,  6,  7,  8],
            ['M', 'C', 90002,  9, 10, 11, 12],
            ['M', 'C', 90002, 13, 14, 15, 16],
            ['M', 'J', 90001,  1,  2,  3,  4],
            ['M', 'J', 90001, 13, 14, 15, 16],
            ['M', 'J', 90002,  5,  6,  7,  8],
            ['M', 'J', 90002,  9, 10, 11, 12],
            ['M', 'I', 90001, 13, 14, 15, 16],
            ['M', 'I', 90001,  1,  2,  3,  4],
            ['M', 'I', 90002,  4,  3,  2,  1],
            ['M', 'I', 90002,  9, 10, 11, 12],
            ['F', 'C', 90001,  4,  3,  2,  1],
            ['F', 'C', 90001,  5,  6,  7,  8],
            ['F', 'C', 90002,  4,  3,  2,  1],
            ['F', 'C', 90002, 13, 14, 15, 16],
            ['F', 'J', 90001,  4,  3,  2,  1],
            ['F', 'J', 90001,  1,  2,  3,  4],
            ['F', 'J', 90002,  8,  7,  6,  5],
            ['F', 'J', 90002,  1,  2,  3,  4],
            ['F', 'I', 90001,  8,  7,  6,  5],
            ['F', 'I', 90001,  5,  6,  7,  8],
            ['F', 'I', 90002,  8,  7,  6,  5],
            ['F', 'I', 90002, 13, 14, 15, 16],
            ]

            result_lol = [
            ['M', 'C', 90001,  6,  8, 10, 12],
            ['M', 'C', 90002, 21, 24, 26, 18],
            ['M', 'J', 90001, 14, 16, 18, 20],
            ['M', 'J', 90002, 14, 16, 18, 20],
            ['M', 'I', 90001, 14, 16, 18, 20],
            ['M', 'I', 90002, 13, 13, 13, 13],
            ['F', 'C', 90001,  9,  9,  9,  9],
            ['F', 'C', 90002, 17, 17, 17, 17],
            ['F', 'J', 90001,  5,  5,  5,  5],
            ['F', 'J', 90002,  9,  9,  9,  9],
            ['F', 'I', 90001, 13, 13, 13, 13],
            ['F', 'I', 90002, 21, 21, 21, 21],
            ]
            
            This reduction can then be further grouped and summed to create reports or to allow for 
            comparison based on any combination of the subgroups.
            
            
        """

        # divide up the table into groups where each group has a unique set of values in groupby_colnames
        # breakpoint() #temp

        if diagnose:  # pragma: no cover
            daf_utils.sts(f"Starting groupby_cols() of {len(self):,} records.", 3)

        scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
        grouped_tdodaf = self.groupby_cols(groupby_colnames, cols=scope)

        if diagnose:  # pragma: no cover
            daf_utils.sts(f"Total of {len(grouped_tdodaf):,} groups. Reduction starting.", 3)

        result_daf = Daf(cols=groupby_colnames + (reduce_cols or []))

        for coltup, this_daf in grouped_tdodaf.items():

            if not this_daf:
                # nothing found with this combination of groupby cols.
                continue

            # apply the reduction function. by='row'/'table' (the only modes that make sense
            # here, to combine with the groupby cols below) always reduce to a single record.
            reduction_da = cast(T_ma, this_daf.reduce(func, by=by, cols=reduce_cols, **kwargs))

            # add back in the groupby cols
            for idx, groupcolname in enumerate(groupby_colnames):
                reduction_da[groupcolname] = coltup[idx]

            result_daf.append(reduction_da)

        if diagnose:  # pragma: no cover
            daf_utils.sts(f"Reduction completed: {len(result_daf):,} records.", 3)

        return result_daf


    def groupby_reduce(
            self,
            colname:        str,
            # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
            # manifest_reduce comment on Callable[...] above.
            func:           Callable[..., Any], # function reduces one grouped daf to one record.
            by:             str='row',                                  # determines how the func is applied.
            reduce_cols:    T_cs | None=None,                           # columns included in the reduce operation.
            diagnose:       bool=False,
            **kwargs:       Any,
            ) -> 'Daf':
        """
        Group the rows by one column and reduce each group to one row.

        The result has one row for each value in the column, and its keyfield is that
        column. The columns in `reduce_cols` hold the reduced values. Other columns are
        empty. For each group, `func` is used as in `reduce()`.

        With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
        columns, and the `indirect_col` if there is one. The other columns are never copied,
        which is much faster for a wide table. A function that reads another column of the
        row will not find it. The result still has all the columns of this Daf, and the
        columns that are not reduced are empty.

        Args:
            colname: The column to group by.
            func: The reduction function. See `reduce()`.
            by: How the function is applied. See `reduce()`.
            reduce_cols: The columns to reduce. All except `colname` if None.
            diagnose: If True, print progress messages.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            The Daf with one row for each group.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y'])
            | g | x | y  |
            | -: | -: | -: |
            | a |   | 40 |
            | b |   | 20 |
            %% daf rows=2; cols=3; keyfield='g'; name=''
        """

        if diagnose:
            logs.sts(f"{logs.prog_loc()} starting groupby '{colname}' operation", 3)
        # groupby(colname=<str>) (colnames not passed) always takes the Dict[str, 'Daf']
        # branch, never the tuple-keyed Dict[Tuple[str, ...], 'Daf'] one.
        scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
        grouped_dodaf = cast(T_dodaf, self.groupby(colname, cols=scope))
        result_daf = Daf.reduce_dodaf_to_daf(
            func            = func,             # function reduces one grouped daf to one record.
            colname         = colname,
            grouped_dodaf   = grouped_dodaf,
            reduce_cols     = list(reduce_cols) if reduce_cols is not None else None,
            by              = by,
            diagnose        = diagnose,
            all_cols        = self.columns() if scope else None,    # the rows keep all the columns, as for whole groups.
            **kwargs,
            )
        return result_daf


    def multi_groupby(
            self,
            groupby_colnames:   T_cs,
            colnames:           T_cs | None=None,
            omit_nulls:         bool=False,         # do not group to values in column that are null ('')
            ) -> Dict[str, Dict[str, 'Daf']]:   # result_dododaf

        """
        Group the rows by each of several columns, one column at a time.

        The result is a dict of dicts. The first key is the column. The second key is a
        value in that column. Each innermost value is a Daf of the rows that have it,
        with all columns, or only the columns in `colnames`.
        This is not a grouping by combinations. Use `groupby_cols()` for that.
        The groups are not reduced. The rows are new lists, so changing a cell in a group
        does not change this Daf.

        Use `colnames` when you need only a few of many columns. The other columns are never
        copied, and this Daf is not changed. It is much faster for a wide table.

        Args:
            groupby_colnames: The columns to group by, one at a time.
            colnames: The columns that each group keeps, in this order. If None, all columns.
                A keyfield is kept only if all its columns are kept.
            omit_nulls: If True, leave out rows that have an empty value.

        Returns:
            A dict that maps each column to a dict of value and Daf.

        Raises:
            KeyError: A group column, or a name in `colnames`, is not a column.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> groups = d.multi_groupby(['g', 'x'])
            >>> list(groups), list(groups['g'])
            (['g', 'x'], ['a', 'b'])
            >>> d.multi_groupby(['g'], colnames=['y'])['g']['a']
            | y  |
            | -: |
            | 10 |
            | 30 |
            %% daf rows=2; cols=1; keyfield=''; name=''
        """

        if isinstance(groupby_colnames, str):
            groupby_colnames = [groupby_colnames]

        if not self.lol:
            return {}

        names, idxs, all_cols = self._cols_scope(colnames)
        group_idxs = [(col, self.hd[col]) for col in groupby_colnames]

        grouped: Dict[str, Dict[Any, T_lola]] = {}

        for row_la in self.lol:
            for col, group_idx in group_idxs:
                col_groups = grouped.get(col)
                if col_groups is None:
                    col_groups = grouped[col] = {}

                fieldval = row_la[group_idx]
                if omit_nulls and fieldval is NULL:
                    continue

                group_lol = col_groups.get(fieldval)
                if group_lol is None:
                    group_lol = col_groups[fieldval] = []

                group_lol.append(list(row_la) if all_cols else [row_la[idx] for idx in idxs])

        return {col: {fieldval: self._new_group_daf(group_lol, names, all_cols)
                            for fieldval, group_lol in col_groups.items()}
                        for col, col_groups in grouped.items()}


    @staticmethod
    def reduce_dodaf_to_daf(
            colname:        str,
            # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
            # manifest_reduce comment on Callable[...] above.
            func:           Callable[..., Any],
            grouped_dodaf:  T_dodaf,
            reduce_cols:    Optional[T_la]=None,    # columns included in the reduce operation, None = all except for colname.
            diagnose:       bool=False,
            all_cols:       Optional[T_ls]=None,    # the columns of the result, if the groups hold only some of them.
            **kwargs:       Any,
            ) -> 'Daf':
        """
        Reduce each Daf in a dict to one row, and join the rows in a Daf.

        This is the second half of `groupby_reduce()`. The dict maps the values of a
        column to Daf instances. Each Daf is reduced with `reduce()`. The value is
        stored in the column `colname` of the row. The result has `colname` as its keyfield.

        Args:
            colname: The column that holds the value of each group.
            func: The reduction function. See `reduce()`.
            grouped_dodaf: A dict that maps each value to a Daf.
            reduce_cols: The columns to reduce. All columns except `colname` if None.
            diagnose: If True, print progress messages.
            all_cols: The columns of the result, in order. Use it when each group holds only
                some columns, so that the rows have the same columns as for whole groups. The
                columns that are not reduced are then empty.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            The Daf with one row for each group.

        Examples:
            >>> groups = {'a': Daf(lol=[[1], [2]], cols=['n']), 'b': Daf(lol=[[3]], cols=['n'])}
            >>> result = Daf.reduce_dodaf_to_daf('g', Daf.sum_da, groups)
            >>> result
            | n | g |
            | -: | -: |
            | 3 | a |
            | 3 | b |
            %% daf rows=2; cols=2; keyfield='g'; name=''
        """

        if diagnose:
            logs.sts(f"{logs.prog_loc()} Grouped into {len(grouped_dodaf)} groups.", 3)

            # display the first three daf arrays and the last

            for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):
                if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                    logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)


        result_daf = Daf(keyfield = colname)

        for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):

            # maybe remove colname from cols here

            if diagnose:
                logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)
            # by defaults to 'row' (the only mode used here), which always reduces to a
            # single record.
            reduction_da = cast(T_ma, this_daf.reduce(func, cols=reduce_cols, **kwargs))
                    # def reduce(
                            # self,
                            # func: Callable[[T_da, T_da], Union[T_da, T_la]],
                            # by: str='row',
                            # cols: Optional[Iterable]=None,                  # columns included in the reduce operation.
                            # initial_da: Optional[T_da]=None,
                            # **kwargs: Any,
                            # ) -> Union[T_da, T_la]:

            if diagnose:
                logs.sts(f"Post reduction: {Daf.from_lod([cast(T_da, reduction_da)])=}", 3)

            # add colname:colval to the dict, as it is removed by the reduction func.
            reduction_da[colname] = colval

            if all_cols:
                reduction_da = {col: reduction_da.get(col, NULL) for col in all_cols}

            if diagnose:
                logs.sts(f"Post add colname:colval to the dict: {Daf.from_lod([cast(T_da, reduction_da)])=}", 3)

            # this will also maintain the kd.
            result_daf.append(reduction_da)

        if diagnose:
            logs.sts(f"{logs.prog_loc()} result_daf:\n\n{result_daf}", 3)

        return result_daf


    def multi_groupby_reduce(
            self,
            colnames:       T_cs,
            # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
            # manifest_reduce comment on Callable[...] above.
            func:           Callable[..., Any],  # function reduces one grouped daf to one record.
            by:             str='row',                                  # determines how the func is applied.
            reduce_cols:    T_cs | None=None,                           # columns included in the reduce operation.
            diagnose:       bool=False,
            **kwargs:       Any,
            ) -> Dict[str, 'Daf']:
        """
        Group by each of several columns and reduce each group to one row.

        This is `multi_groupby()` followed by `groupby_reduce()` for each column. The
        result is a dict. Each key is a column. Each value is a Daf with one row for
        each value of that column, and that column as its keyfield.

        With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
        columns, and the `indirect_col` if there is one. A function that reads another
        column of the row will not find it. The result still has all the columns of this
        Daf, and the columns that are not reduced are empty.

        Args:
            colnames: The columns to group by, one at a time.
            func: The reduction function. See `reduce()`.
            by: How the function is applied. See `reduce()`.
            reduce_cols: The columns to reduce.
            diagnose: If True, print progress messages.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            A dict that maps each column to its Daf.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.multi_groupby_reduce(['g'], Daf.sum_da, reduce_cols=['y'])['g']
            | g | x | y  |
            | -: | -: | -: |
            | a |   | 40 |
            | b |   | 20 |
            %% daf rows=2; cols=3; keyfield='g'; name=''
        """

        if diagnose:
            logs.stsloc(f"starting multi-groupby '{colnames}' operation", 3)
        scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
        multi_grouped_dododaf = self.multi_groupby(colnames, colnames=scope)

        result_dodaf: T_dodaf = {}
        reduce_cols_la = list(reduce_cols) if reduce_cols is not None else None
        result_cols = self.columns() if scope else None         # the rows keep all the columns, as for whole groups.

        for colname, grouped_dodaf in multi_grouped_dododaf.items():

            result_dodaf[colname] = Daf.reduce_dodaf_to_daf(
                func            = func,         # function reduces one grouped daf to one record.
                colname         = colname,
                grouped_dodaf   = grouped_dodaf,
                reduce_cols     = reduce_cols_la,
                diagnose        = diagnose,
                all_cols        = result_cols,
                **kwargs,
                )
        if diagnose:
            logs.stsloc(f"Grouped into {len(result_dodaf)} groups.", 3)

            for group_num, (colval, this_daf) in enumerate(result_dodaf.items()):
                if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                    logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)

        return result_dodaf


    def apply_colwise(
        self,
        target_col: str,
        func: Callable[[T_da], Any],
        *,
        default: Any = 0,
    ) -> "Daf":
        """
        Compute one column from the other columns of each row, in place.

        The function gets each row, as a dict, and returns the value for `target_col`.
        If the function raises an error for a row, that row gets `default`. If
        `target_col` is not a column, it is added at the right, and the rows are
        filled with `default` first. Use it for a ratio or a total of other columns.

        Args:
            target_col: The column to store the result in.
            func: A function that takes a row and returns the value.
            default: The value for a row where the function raises an error.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 2], [3, 0]], cols=['a', 'b'])
            >>> d.apply_colwise('ratio', lambda row: row['a'] / row['b'], default=-1.0)
            | a | b | ratio |
            | -: | -: | ----: |
            | 1 | 2 |   0.5 |
            | 3 | 0 |  -1.0 |
            %% daf rows=2; cols=3; keyfield=''; name=''
        """

        if target_col not in self.columns():
            self.insert_col(target_col, default=default)

        def row_op(row: T_da) -> T_da:
            try:
                row[target_col] = func(row)
            except Exception:
                row[target_col] = default

            return row

        self.apply_in_place(row_op)

        return self


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
    #         cols: Optional[T_la]=None,
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
    #         cols: Optional[T_la]=None,
    #         **kwargs: Any,
    #         ) -> T_da:
    #     # this one to investigate why daf_sum is so slow!
    #
    #     return self.reduce(func=Daf.sum_da3, by=by, cols=cols, **kwargs)


    # def reduce2(
            # self,
            # func: Callable[[T_da, T_da], Union[T_da, T_la]],
            # by: str='row',
            # cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> Union[T_da, T_la]:
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


    def daf_sum(
            self,
            by:             str = 'row',                    # row|col|table|sparse_row
            cols:           T_cs | None = None,
            indirect_col:   str = '',                       # indirect_col is required for lol array.
            **kwargs:       Any,
            ) -> T_ma:

        # for the default by='row' (the only mode used by any caller/test), reduce() always
        # returns a single record.
        """
        Add up the columns, using `reduce()` and `sum_da()`.

        Cells that cannot be added, such as text, are skipped. A column that is not
        in `cols` is empty in the result.

        Args:
            by: How the function is applied. See `reduce()`.
            cols: The columns to sum. All if None.
            indirect_col: A column that holds a dict. Required for `sparse_row`.
            **kwargs: Keyword arguments passed on to `reduce()`.

        Returns:
            A dict with a total for each column.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.daf_sum(cols=['y'])
            {'g': '', 'x': '', 'y': 60}
        """

        return cast(T_ma, self.reduce(func=Daf.sum_da, by=by, cols=cols, indirect_col=indirect_col, **kwargs))


    def reduce(
            self,
            # real reduction functions (sum_da, count_values_da, ...) take further keyword-only
            # params beyond the row/reduction pair (cols, astype, omit_nulls, is_sparse,
            # diagnose, ...) passed through via **kwargs below, so the fixed 2-arg Callable
            # forms can't express this.
            func:           Callable[..., Any],
            by:             str = 'row',                                # row|col|table|sparse_row
            cols:           T_cs|None=None,                    # columns included in the reduce operation.
            initial_da:     T_ma|None=None,
            indirect_col:   str = '',                                   # indirect_col is required for lol array.
            silent_error:   bool = False,                               # if True, skip rows where func raises.
            **kwargs:       Any,
            ) -> Union[T_ma, T_la]:
        """
        Combine all the rows into one result, using a function.

        The function gets each row, and the result so far, and returns the new result.
        A sum or a count is a reduction. With `by='row'`, the function is called as
        `func(row, result, cols=cols, **kwargs)`. The result starts as 0 for each of
        the columns, or as `initial_da`. It returns a dict with every column. Columns
        that are not in `cols` are empty.

        With `by='col'` the function gets each column as a list and the result so far
        as a list. With `by='table'` it gets this Daf and `cols`, and its result is
        returned as it is. With `by='sparse_row'`, rows are read from the dict in
        `indirect_col`, and the result starts as `initial_da` or an empty dict.

        An error in the function stops the reduction. With `silent_error=True` the
        row is skipped. The result may then be missing rows, with no sign of it. An
        empty Daf gives an empty dict for `by='row'`.

        To reduce part of the table, select the rows or columns first, or give `cols`.

        Args:
            func: The function that combines. For example `Daf.sum_da`.
            by: `row`, `col`, `table` or `sparse_row`.
            cols: The columns included in the reduction. All columns if None.
            initial_da: The result to start from, instead of zeros.
            indirect_col: A column that holds a dict. Required for `sparse_row`.
            silent_error: If True, a row for which the function raises is skipped.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            A dict for `row`, `table` and `sparse_row`. A list for `col`.

        Raises:
            ValueError: `sparse_row` is used with no `indirect_col`.
            NotImplementedError: `by` is not recognized.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.reduce(Daf.sum_da, cols=['x', 'y'])
            {'g': '', 'x': 6, 'y': 60}
        """
        if by == 'table':
            reduction_ma = func(self, cols, **kwargs)
            return reduction_ma

        if by == 'row':

            if not self:
                return {}

            #allcols = self.hd.keys()

            cols_iter: Iterable[str]
            if cols is None:
                cols_iter = self.hd.keys()
            elif isinstance(cols, str):
                cols_iter = [cols]                              # perflint-reviewed (use-tuple-over-list)
            # elif isinstance(cols, list) and len(cols) > 10:
                # cols_iter = dict.fromkeys(cols)
            else:
                cols_iter = {col: None for col in self.hd.keys() if col in cols}

            if initial_da is not None:
                reduction_ma = initial_da
            else:
                reduction_ma = dict.fromkeys(cols_iter, 0)

            for row_ma in self:
                try:
                    reduction_ma = func(row_ma, reduction_ma, cols=cols_iter, **kwargs)
                except Exception:
                    if not silent_error:
                        raise

                # def count_values_da(row_da: T_da, reduction_da: T_da, cols: Iterable, omit_nulls: bool=False) -> T_dodi:
                # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

            # normalize the result so it contains all columns
            result_ma = {key: reduction_ma.get(key,'') for key in self.hd.keys()}

            return result_ma

        elif by == 'col':
            reduction_la: T_la = []                   # perflint-reviewed (use-tuple-over-list)
            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                reduction_la = func(col_la, reduction_la, **kwargs)
            return reduction_la

        elif by == 'sparse_row':
            if not indirect_col:
                raise ValueError("indirect_col must be specified for 'sparse_row' reduction and lol array.")

            reduction_ma = initial_da or {}

            for row_ma in self:
                indirect_ma = daf_utils.get_indirect_da(row_ma, indirect_col)
                # indirect_val = row_da[indirect_col]
                # if isinstance(indirect_val, str):
                #     indirect_da = daf_utils.safe_convert_json_to_obj(indirect_val)
                # else:
                #     indirect_da = indirect_val
                try:
                    reduction_ma = func(indirect_ma, reduction_ma, cols=cols, is_sparse=True, **kwargs)
                except Exception:
                    if not silent_error:
                        raise

                # def count_values_da(row_da: T_da, reduction_da: T_da, cols: Iterable, omit_nulls: bool=False) -> T_dodi:
                # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

            # # normalize the result so it contains all columns
            # result_da = {key: reduction_da.get(key,'') for key in self.hd.keys()}

            return reduction_ma



        else:
            raise NotImplementedError
        return [] # for mypy only.


    @staticmethod
    def sum_da( row_da:         T_ma,                       # the current row from the daf array.
                reduction_da:   T_ma,                       # an accumulated result. Must be initialized for all columns in cols.
                *,
                cols:           T_cs | None=None,    # defines the active columns. Can be a list, keys(), range, or slice
                astype:         Type | None=None,        # a type like int, float, str to cast the value if it is not that type. Optional.
                is_sparse:      bool=False,
                diagnose:       bool=False
                ) -> T_ma:  # result_ma -- same object (and type) as reduction_da, returned in place
        """
        Add the values of a row to a running total. Use it with `reduce()`.

        This is a static method. A value that cannot be added, such as text or an empty
        cell, is skipped. The total is changed and returned. With `astype`, each value
        is first converted to that type.

        Args:
            row_da: The current row.
            reduction_da: The running total. It is changed.
            cols: The columns to add. All if None.
            astype: A type to convert each value to before adding.
            is_sparse: If True, the row may hold only some of the columns.
            diagnose: Not used.

        Returns:
            The running total.

        Examples:
            >>> Daf.sum_da({'x': 1, 'y': 2}, {'x': 10, 'y': 0}, cols=['x', 'y'])
            {'x': 11, 'y': 2}
        """
        # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

        diagnose = diagnose
        #nan_indicator = ''

        # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
            # if col not in cols:                 # this check is not needed in the version below.
                # continue                        # 251 vs 207.

        if cols is None or is_sparse:
            # iterate by cols in row_da
            for col, val in row_da.items():

                if cols and col not in cols:
                    continue

                try:
                    # note that the "+ 0" in the following expression is important so that
                    # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                    # This will only invoke the exception when encountering non-numeric data, and does
                    # not require an initial check

                    reduction_da[col] = reduction_da.get(col, 0) + val + 0     # type: ignore[index, call-overload]

                #except Exception:
                except (ValueError, TypeError):
                    continue

            return reduction_da

        else:
            for col in cols:

                value = row_da.get(col, 0)
                # if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
                # if isinstance(value, str):        # this makes the loop take 42% longer (294)
                # if isinstance(value, str) and value == '':  # same (294)
                # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
                # if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
                # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
                # if isinstance(value, str):          # this makes the loop take 50% longer (305)
                                                    # but is needed to disallow concatenating strings.
                    # continue
                # if isinstance(value, (int, float, np.int32, np.int64)):
                    # (indent)

                # the try/except below is the most time efficient way to handle this while still
                # allowing for astype and nan values. (212 ms for 1000x1000 array)
                # Please note that the cols value is determined
                # prior to entering the function and must contain an iterable, even if all columns
                # are specified.

                # writing this loop the other way around, by going through all columns and skipping those not
                # mentioned in cols is also very inefficient.

                # 213 for the version below, which seems like it should be fastest.
                # but it is slightly less advantageous because initial assignment is inside the try/except block.

                # try:
                    # if astype:
                        # value = row_da[col]
                        # if astype==int and isinstance(value, (str, float, bool)):
                            # value = int(float(value))
                        # elif astype==float and isinstance(value, (str, int, bool)):
                            # value = float(value)
                        # elif astype==str and isinstance(value, (float, int, bool)):
                            # value = str(value)
                        # accum_da[col] += value
                    # else:
                        # accum_da[col] += row_da[col]
                # except Exception:
                    # continue

                # this one measured at 230
                # value = row_da[col]
                # try:
                    # if astype:
                        # if astype==int and isinstance(value, (str, float, bool)):
                            # value = int(float(value))
                        # elif astype==float and isinstance(value, (str, int, bool)):
                            # value = float(value)
                        # elif astype==str and isinstance(value, (float, int, bool)):
                            # value = str(value)

                    # accum_da[col] += value

                # except Exception:
                    # continue

                # this one measured at 209 with all cols and no astype.
                if astype:
                    #value = row_da.get(col, 0)
                    try:
                        if astype is int and isinstance(value, (str, float, bool)):
                            value = int(float(value or 0))
                        elif astype is float and isinstance(value, (str, int, bool)):
                            value = float(value or 0)
                        elif astype is str and isinstance(value, (float, int, bool)):
                            value = str(value)

                        reduction_da[col] += value

                    except ValueError:
                        continue

                else:
                    try:
                        # note that the "+ 0" in the following expression is important so that
                        # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                        # This will only invoke the exception when encountering non-numeric data, and does
                        # not require an initial check

                        reduction_da[col] = reduction_da.get(col, 0) + value + 0

                    #except Exception:
                    except (ValueError, TypeError):
                        continue



            return reduction_da


    # sum_da2 / sum_da3 -- investigatory variants used to find out why daf_sum() (sum_da) was
    # slow. The key finding: comparing `value == ''` instead of `value is NULL` made the loop
    # take ~10x longer -- Python implements '' as a singleton (like None), so the very fast
    # `is` comparison works and was adopted into sum_da and other NULL comparisons throughout
    # daf.py. Commented out rather than deleted, as a record of that investigation/finding.

#     @staticmethod
#     def sum_da2(row_da: T_da,                   # the current row from the daf array.
#                 accum_da: T_da,                 # an accumulated result. Must be initialized for all columns in cols.
#                 cols: Iterable,                 # defines the active columns. Can be a list, keys(), range, or slice
#                 astype: Optional[Type]=None,    # a type like int, float, str to cast the value if it is not that type. Optional.
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
#                 astype: Optional[Type]=None,    # a type like int, float, str to cast the value if it is not that type. Optional.
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
        Count how often each value occurs, in each column, using `reduce()`.

        Args:
            by: How the function is applied. See `reduce()`.
            cols: The columns to count. All if None.

        Returns:
            A dict that maps each column to a dict of value and count. A column that is
            not in `cols` is empty.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.daf_valuecount(cols=['g'])['g']
            {'a': 2, 'b': 1}
        """

        # by='row' (the default, and the only mode exercised) always reduces to a single record.
        return cast(T_ma, self.reduce(func=type(self).count_values_da, by=by, cols=cols))


    def groupsum_daf(
            self,
            colname:        str,
            by:             str='row',                      # determines how the func is applied.
            reduce_cols:    T_cs | None = None,             # columns included in the reduce operation.
            # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                            # current operation sets keyfield to colname.
            ) -> 'Daf':

        """
        Group by a column and add up the other columns of each group.

        This is `groupby_reduce()` with `sum_da()`.

        Args:
            colname: The column to group by.
            by: How the function is applied. See `reduce()`.
            reduce_cols: The columns to add.

        Returns:
            The Daf with one row for each group.

        Examples:
            >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
            >>> d.groupsum_daf('g', reduce_cols=['y'])
            | g | x | y  |
            | -: | -: | -: |
            | a |   | 40 |
            | b |   | 20 |
            %% daf rows=2; cols=3; keyfield='g'; name=''
        """

        result_daf = self.groupby_reduce(colname=colname, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)

        return result_daf


    def multi_groupsum(
            self,
            colnames:       T_cs | None = None,             # colnames to group over individually
            by:             str         ='row',             # determines how the func is applied.
            reduce_cols:    T_cs | None = None,             # columns included in the reduce operation.
            # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                            # current operation sets keyfield to colname.
            ) -> Dict[str, 'Daf']:

        """
        Group by each of several columns, one at a time, and add up the columns.

        This is `multi_groupby_reduce()` with `sum_da()`.

        Args:
            colnames: The columns to group by. These are required.
            by: How the function is applied. See `reduce()`.
            reduce_cols: The columns to add.

        Returns:
            A dict that maps each column to a Daf with one row for each value.

        Raises:
            ValueError: No `colnames` are given.

        Examples:
            >>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
            >>> sums = d.multi_groupsum(colnames=['g', 'k'], reduce_cols=['n'])
            >>> sums['g']
            | g | k | n |
            | -: | -: | -: |
            | a |   | 3 |
            | b |   | 3 |
            %% daf rows=2; cols=3; keyfield='g'; name=''
            >>> sums['k']
            | g | k | n |
            | -: | -: | -: |
            |   | x | 4 |
            |   | y | 2 |
            %% daf rows=2; cols=3; keyfield='k'; name=''
            >>> d.multi_groupsum()
            Traceback (most recent call last):
                ...
            ValueError: multi_groupsum: colnames is required
        """

        if colnames is None:
            raise ValueError("multi_groupsum: colnames is required")

        result_dodaf = self.multi_groupby_reduce(colnames=colnames, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)

        return result_dodaf


    def set_col2_from_col1_using_regex_select(self, col1: str, col2: str='', regex: str='') -> 'Daf':

        r"""
        Fill a column with the part of another column that a regex selects, in place.

        `regex` must have one pair of parentheses around the part to keep. A cell that
        does not match gives an empty cell. Give `regex` as a keyword. Without `col2`,
        `col1` is changed. If `col2` is not a column, it is added at the right.

        Args:
            col1: The column to read.
            col2: The column to write. Defaults to `col1`.
            regex: A regular expression with one group.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'ab12'], [2, 'cd34']], cols=['id', 's'])
            >>> _ = d.set_col2_from_col1_using_regex_select('s', regex=r'(\d+)')
            >>> d.col('s')
            ['12', '34']
        """

        # from utilities import daf_utils

        def set_row_col2_from_col1_using_regex_select(row_da: T_da, col1: str, col2: str, regex: str) -> T_da:
            row_da[col2] = daf_utils.safe_regex_select(regex, row_da[col1])
            return row_da

        if not col2:
            col2 = col1

        if col2 not in self.hd and col1 in self.hd:
            self.insert_col(col2)       # a new column, so the result has a place to go.

        self.apply_in_place(lambda row_da: set_row_col2_from_col1_using_regex_select(row_da, col1, col2, regex))

        return self


    def apply_replace_regex(self, col: str, col2: str='', replace_regex: str='') -> 'Daf':

        r"""
        Change a column with a pattern of the form `/find/replace/`, in place.

        The pattern has three parts that are separated by `/`. The first is a regular
        expression. The second is what replaces the part it finds. The groups it
        finds can be used as `\1`. A pattern with an empty second part removes the text.

            /find//                       remove
            /find/replace/                replace
            /pre(select)post/\1/          keep only the selected part
            /pre(select)post/a\1b/        keep it, with new text around it

        The result goes to `col2`, or to `col` if there is no `col2`. A column that is
        not found does nothing. If `col2` is not a column, it is added at the right.

        Args:
            col: The column to read.
            col2: The column to write. Defaults to `col`.
            replace_regex: The pattern.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 'ab12']], cols=['id', 's'])
            >>> d.apply_replace_regex('s', replace_regex='/ab//').col('s')
            ['12']
        """

        # from utilities import daf_utils

        def set_row_col2_from_col_using_regex_replace(row_ma: T_ma, col: str, col2: str, replace_regex: str) -> T_ma:
            if col in row_ma:
                row_ma[col2] = daf_utils.safe_regex_replace(regex=replace_regex, s=row_ma[col])
            return row_ma

        if not col2:
            col2 = col

        if col2 not in self.hd and col in self.hd:
            self.insert_col(col2)       # a new column, so the result has a place to go.

        self.apply_in_place(lambda row_ma: set_row_col2_from_col_using_regex_replace(row_ma, col, col2, replace_regex))

        return self


    def alter_daf_per_setting(
            self,
            settingsdict:           T_da,
            setting_name:           str,
            setting_select_dict:    dict,
            silent_error:           bool = False,   # if True, a missing setting does nothing
            ) -> 'Daf':

        """
        Change this Daf with the replace patterns found in a settings dict.

        The setting is a dict, or a list of dicts. Each has a `spec_name`, a `colname`
        and a `replace_regex`. The ones whose fields match `setting_select_dict` are
        used. For each, `apply_replace_regex()` changes the column. Use it to give
        different edits to different files, such as fixing ids in one source.

        A setting that is None or empty changes nothing. A name that is not in the
        settings dict raises `KeyError`, unless `silent_error` is True.

        Args:
            settingsdict: A dict of settings.
            setting_name: The key of the setting in the dict.
            setting_select_dict: Selects which specs apply, such as `{'spec_name': 'file1.csv'}`.
            silent_error: If True, a missing setting changes nothing.

        Returns:
            This Daf, which has been changed.

        Raises:
            KeyError: The setting is missing and `silent_error` is False.

        Examples:
            >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
            >>> spec = {'spec_name': 'a.zip', 'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}
            >>> d.alter_daf_per_setting({'fix': [spec]}, 'fix', {'spec_name': 'a.zip'}).col('ballot_id')
            ['14_1', '05_2']
        """

        if silent_error:
            setting_lod = settingsdict.get(setting_name)
        else:
            setting_lod = settingsdict[setting_name]

        if not setting_lod:
            return self     # do nothing.

        if isinstance(setting_lod, dict):
            setting_lod = [setting_lod]                 # perflint-reviewed (use-tuple-over-list)

        setting_daf = Daf.from_lod(setting_lod)

        alter_specs_daf = setting_daf.select_by_dict(setting_select_dict)

        return self.alter_daf_per_alter_specs_daf(
            alter_specs_daf     = alter_specs_daf,     # daf containing cols: colname, replace_regex
            )



    def alter_daf_per_alter_specs_daf(
            self,                       # daf to alter
            alter_specs_daf: 'Daf',     # daf containing cols: colname, replace_regex
            ) -> 'Daf':
        """
        Change this Daf with the replace patterns listed in a Daf.

        Each row of `alter_specs_daf` has a `colname` and a `replace_regex`. The
        pattern is applied to the column with `apply_replace_regex()`. Give it only
        the rows that apply, for example by selecting on `spec_name` first.

        Args:
            alter_specs_daf: The specs, with the columns `colname` and `replace_regex`.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
            >>> specs = Daf.from_lod([{'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}])
            >>> d.alter_daf_per_alter_specs_daf(specs).col('ballot_id')
            ['14_1', '05_2']
        """
        for alter_spec_da in alter_specs_daf:

            self.apply_replace_regex(col=alter_spec_da['colname'], replace_regex=alter_spec_da['replace_regex'])

        return self



    def apply_to_col(self, col: str, func: Callable, **kwargs: Any) -> 'Daf':

        """
        Replace each value of a column by the result of a function, in place.

        Args:
            col: The column name.
            func: A function that takes a value and returns the new value.
            **kwargs: Keyword arguments passed on to the function with each value.

        Returns:
            This Daf, which has been changed.

        Examples:
            >>> d = Daf(lol=[[1, 5], [2, 6]], cols=['a', 'b'])
            >>> _ = d.apply_to_col('b', lambda value: value * 2)
            >>> d.col('b')
            [10, 12]
            >>> _ = d.apply_to_col('b', lambda value, factor: value * factor, factor=10)
            >>> d.col('b')
            [100, 120]
        """

        if kwargs:
            self[:, col] = [func(value, **kwargs) for value in self.col(col)]
        else:
            self[:, col] = list(map(func, self.col(col)))

        if col == self.keyfield or not isinstance(self.keyfield, str):
            self._invalidate_kd()

        return self

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
    # def sum_da(row_da: T_da, accum_da: T_da, cols: Optional[T_la]=None, astype:Type=int, diagnose:bool=False) -> T_da:     # result_da
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


    @staticmethod
    def diff_da(d1_da: T_ma, d2_da: T_ma, keys: T_ls | str | None=None) -> T_da:     # result_da
        """
        Subtract one dict from another, for the keys you name.

        This is a static method. A key that is missing, or an empty value, counts as
        0. Keys that you do not name are left out. The values must be numbers.

        Args:
            d1_da: The first dict.
            d2_da: The dict to subtract.
            keys: The key or keys to include. If None, the result is empty.

        Returns:
            A dict of the differences.

        Examples:
            >>> Daf.diff_da({'a': 5, 't': 'x'}, {'a': 2, 't': 'y'}, keys=['a'])
            {'a': 3}
        """

        keys_ca: T_ls

        if isinstance(keys, str):
            keys_ca = [keys]                      # perflint-reviewed (use-tuple-over-list)
        else:
            keys_ca = keys or []

        # the 'or 0' part handles null string. Always a plain dict comprehension,
        # regardless of whether d1_da/d2_da are dict or KeyedList.
        result_da = {key: (d1_da.get(key, 0) or 0) - (d2_da.get(key, 0) or 0)
                        for key in keys_ca}

        return result_da


    @staticmethod
    def count_values_da(
            row_da:         T_ma,
            # needs a dict-based accumulator (see test_daf_reduction.py header), but despite the
            # T_dodi ("dict of dict of int") name, values may also be a list or dict copied
            # straight from row_da (see the 'val is a list'/'val is a dict' branches below).
            reduction_da:   T_da,
            cols:           T_cs,
            *,
            omit_nulls: bool=False,

            ) -> T_da:
        """
        Add one row to running counts of the values in each column. Use it with `reduce()`.

        This is a static method. The counts are a dict that maps each column to a dict
        of value and count. The counts are changed and returned. A cell that holds
        a list is collected into a list. A cell that holds a dict is added to the
        counts for that column with `sum_da()`. Use `omit_nulls` to skip empty cells.

        Args:
            row_da: The current row.
            reduction_da: The running counts. They are changed.
            cols: The columns to count.
            omit_nulls: If True, empty cells are not counted.

        Returns:
            The running counts.

        Examples:
            >>> Daf.count_values_da({'g': 'a'}, {}, ['g'])
            {'g': {'a': 1}}
        """

        # if cols is None:
            # cols_dict = {}
        # else:
            # cols_dict = dict.fromkeys(cols)

        if not reduction_da:
            reduction_da = {}

        result_dodi = reduction_da

        for col in cols:

            val = row_da[col]

            if omit_nulls and val is NULL:
                continue

            if val and isinstance(val, list):
                # val is a list of values. Copy it, so later appends don't mutate the source row.
                if col not in result_dodi:
                    result_dodi[col] = list(val)
                else:
                    result_dodi[col].append(val)
                continue

            if val and isinstance(val, dict):
                # val is a dict of values determined in another pass.
                # Copy it, since sum_da() below accumulates into result_dodi[col] in place.
                if col not in result_dodi:
                    result_dodi[col] = dict(val)
                else:
                    result_dodi[col] = Daf.sum_da(val, result_dodi[col])
                continue

            if col not in result_dodi or not result_dodi[col]:
                result_dodi[col] = {val: 1}
            elif val not in result_dodi[col]:
                result_dodi[col][val] = 1
            else:
                result_dodi[col][val] += 1

        return result_dodi


    @staticmethod
    def sum_dodis(this_dodi: T_dodi, accum_dodi: T_dodi) -> None:
        """
        Add one dict of dicts of numbers into another, in place.

        This is a static method. For each key, the numbers of the inner dicts are
        added with `sum_da()`. A key that is new is stored as it is, not copied.

        Args:
            this_dodi: The counts to add.
            accum_dodi: The running totals. They are changed.

        Examples:
            >>> total = {'c': {'x': 2, 'y': 1}}
            >>> Daf.sum_dodis({'c': {'x': 1}}, total)
            >>> total
            {'c': {'x': 3, 'y': 1}}
        """

        for key, this_di in this_dodi.items():
            if key in accum_dodi:
                Daf.sum_da(this_di, accum_dodi[key])
            else:
                accum_dodi[key] = this_di



    # def classify_by_logic_spec(self, groupcol, input_da: T_da) -> T_da: # groupname

        # for logic_spec_da in self:




    #===============================================
    # functions not following apply or reduce pattern

    # this function does not use reduction approach.
    def sum(
            self,
            colnames_ls:    T_cs | None = None,    # parameter name now inconsistent.
            numeric_only:   bool = False,
            ) -> dict: # sums_di
        """
        Total the columns, and return a dict of the totals.

        Each total starts as a float. Empty cells are skipped. A cell that is text
        that is not a number raises `ValueError`, so give `colnames_ls` to leave
        those columns out. With `numeric_only`, and dtypes of `int` or `float`, only
        those columns are totaled, and a cell that is not a number counts as 0. The
        totals are converted to the dtypes, if the Daf has them.

        Args:
            colnames_ls: The columns to total. All if None.
            numeric_only: If True, total only the columns with an `int` or `float` dtype.

        Returns:
            A dict that maps each column name to its total.

        Raises:
            ValueError: A cell cannot be converted to a number.

        Examples:
            >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum()
            {'x': 3.0, 'y': 30.0}
        """


        if colnames_ls is None:
            cleaned_colnames_cs: T_cs = self.hd.keys()
        elif not (numeric_only and self.dtypes):
            cleaned_colnames_cs = {col:None for col in colnames_ls if col in self.hd.keys()}
        else:
            cleaned_colnames_cs = {col:None for col in colnames_ls if col in self.hd and self.dtypes.get(col) in [int, float]}

        sums_d = dict.fromkeys(cleaned_colnames_cs, 0.0)

        for colname in cleaned_colnames_cs:
            colidx = self.hd[colname]
            for la in self.lol:
                val = la[colidx]
                if val:
                    if numeric_only:
                        sums_d[colname] += Daf._safe_tofloat(val)    # perflint-reviewed
                    else:
                        sums_d[colname] += float(val)        # perflint-reviewed

        sums_d = daf_utils.set_dict_dtypes(sums_d, dtypes=self.dtypes)

        return sums_d


    def sum_np(
            self,
            colnames_ls: T_cs | None = None,
            ) -> dict: # sums_di

        """
        Total the columns with NumPy, and return a dict of the totals.

        This needs NumPy. A blank, `None` or NaN cell counts as 0, as it does in `sum()`.
        Use `colnames_ls` to pass only the columns that hold numbers. A column that holds
        text raises `TypeError` and names the column.

        Args:
            colnames_ls: The columns to total. All if None.

        Returns:
            A dict that maps each column name to its total. An empty Daf gives an empty dict.

        Raises:
            KeyError: A name in `colnames_ls` is not a column.
            TypeError: A column holds text or other values that cannot be added.

        Examples:
            >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum_np()
            {'x': 3, 'y': 30}
            >>> Daf(lol=[[1, 10], [2, '']], cols=['x', 'y']).sum_np()
            {'x': 3, 'y': 10}
        """
        # unit tests exist

        import numpy as np

        if not self:
            return {}

        colnames_ls = self.columns() if colnames_ls is None else list(colnames_ls)

        for colname in colnames_ls:
            self.hd[colname]        # a KeyError for a name that is not a column.

        donpa = self.to_donpa(colnames_ls, default=0)

        sums_d = {}
        for colname in colnames_ls:
            try:
                sums_d[colname] = np.sum(donpa[colname]).item()
            except TypeError as exc_info:
                raise TypeError(
                    f"sum_np(): column '{colname}' holds text or other values that cannot be added."
                    ) from exc_info

        return sums_d


    def valuecounts_for_colname(
            self,
            colname:    str,
            sort:       bool=False,
            reverse:    bool=True,
            omit_nulls: bool=False,
            ) -> T_di:
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
            >>> d = Daf(lol=[['a'], ['b'], ['a'], ['']], cols=['g'])
            >>> d.valuecounts_for_colname('g', sort=True, omit_nulls=True)
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
            ) -> T_dodi:
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
            >>> d = Daf(lol=[['a', 'x'], ['b', 'x']], cols=['g', 'h'])
            >>> d.valuecounts_for_colnames_ls()
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
            ) -> T_di:
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
            >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
            >>> d.valuecounts_for_colname_selectedby_colname('g', 'h', 'x')
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
            ) -> T_dodi:

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
            >>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
            >>> d.valuecounts_for_colnames_ls_selectedby_colname(['k'], 'g', 'a')
            {'k': {'x': 1, 'y': 1}}
            >>> d.valuecounts_for_colnames_ls_selectedby_colname(['g', 'k'], 'g', 'a')
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
            ) -> T_dodi:
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
            >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
            >>> d.valuecounts_for_colname1_groupedby_colname2('g', 'h')
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
            >>> d = Daf(lol=[[1], [3]], cols=['n'])
            >>> d.gen_stats_daf([('n', int, '', 'scalar')])['n']['mean']
            2
        """

        info_dod = {}

        # from utilities import daf_utils

        for col_def_ta in col_def_lot:
            col_name, col_dtype, col_format, col_profile = col_def_ta

            col_data_la = self.col(col_name)                                       # perflint-reviewed (loop-invariant-statement)

            info_dod[col_name] = daf_utils.list_stats(col_data_la, profile=col_profile)     # perflint-reviewed (loop-invariant-statement)

        return info_dod


    def transpose(self, new_keyfield:str='', new_cols:Optional[T_la]=None, include_header:bool = False) -> 'Daf':
        """
        Turn rows into columns and columns into rows.

        The result has one row for each column of this Daf. With `include_header=True`,
        the first column of the result holds the column names of this Daf, and the
        names given in `new_cols` or the default names must then include that column.
        The default cols are `A`, `B` and so on, one for each row of this Daf. With
        `include_header=True` they start with `key`. If you pass `new_cols`, give one
        name for each row of this Daf, plus one with `include_header`. The data is
        copied.

        Args:
            new_keyfield: The keyfield of the result.
            new_cols: The cols of the result.
            include_header: If True, the column names become the first column.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
            >>> d.transpose(include_header=True)
            | key | A | B |
            | --: | -: | -: |
            |  id | 1 | 2 |
            |   v | a | b |
            %% daf rows=2; cols=3; keyfield=''; name=''
            >>> d.transpose().columns()
            ['A', 'B']
            >>> d.transpose(include_header=True).columns()
            ['key', 'A', 'B']
        """
        import numpy as np

        if not new_cols:
            new_cols = daf_utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol))
            if include_header:
                new_cols = ['key'] + new_cols

        # transpose the array
        # new_lol = [list(row) for row in zip(*self.lol)]

        # the following leverages the use of the transposition operation in numpy 
        # over an array of references to python objects. Faster and concise.

        npao = np.array(self.lol, dtype=object)
        npaoT = npao.T
        new_lol = npaoT.tolist()

        # instead of adding column for the header row, consider initializing kd from hd and hd from kd.

        # hd, kd = self._kd, self.hd
        # new_keyfield = ''

        if include_header:
            # add a new first column which will be the old column names row.
            # from utilities import daf_utils

            new_lol = daf_utils.insert_col_in_lol_at_icol(icol=0, col_la=self.columns(), lol=new_lol)

        # following invalidates kd for lazy rebuilding.
        return Daf(lol=new_lol, name=self.name, keyfield=new_keyfield, cols=new_cols, use_copy=True)


    def derive_join_translator(
        self,
        other_daf: 'Daf',                               # The other Daf instance to join with
        shared_fields: Optional[T_ls] = None,           # Columns shared between tables that do not require renaming
                                                        # keyfields do not need to be added here.
        omit_other_cols: Optional[T_ls]=None,           # cols to omit from other (use instead of shared fields)
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
            >>> a = Daf(lol=[[1, 'x']], cols=['id', 'name'], keyfield='id')
            >>> b = Daf(lol=[[1, 'y']], cols=['id', 'name'], keyfield='id')
            >>> a.derive_join_translator(b).col('resolved_colname')
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
            | resolved_colname | source_name | source_colname | is_keyfield |
            | ---------------: | ----------: | -------------: | ----------: |
            |               id |           L |             id |        True |
            |                v |           L |              v |       False |
            |                w |           R |              w |       False |
            %% daf rows=3; cols=4; keyfield='resolved_colname'; name=''
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

        Both need a keyfield, and it must be a single column. A row of this Daf is
        joined with the row of `other_daf` that has the same key.

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
            >>> a = Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id')
            >>> b = Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id')
            >>> a.join(b)
            | id | name  | salary |
            | -: | ----: | -----: |
            |  1 | Alice |     50 |
            %% daf rows=1; cols=3; keyfield='id'; name=''
            >>> a.join(b, how='left')
            | id | name  | salary |
            | -: | ----: | -----: |
            |  1 | Alice |     50 |
            |  2 |   Bob |        |
            %% daf rows=2; cols=3; keyfield='id'; name=''
            >>> a.join(b, how='left', fill=None)
            | id | name  | salary |
            | -: | ----: | -----: |
            |  1 | Alice |     50 |
            |  2 |   Bob |   None |
            %% daf rows=2; cols=3; keyfield='id'; name=''
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
        result_daf = Daf(cols=resolved_colnames, name=name, keyfield=self.keyfield)
        keyfield = self.keyfield   # okay to set now with lazy kd generation.

        # Helper function to fetch a record by key, with silent error
        def fetch_record(daf: 'Daf', mykey: Union[str, int]) -> T_da:
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

    def wide_to_narrow(self,
            id_cols: T_ls      ,                # identify the record but are not used to identify values
            varname_colname: str = 'varname',   # narrow format provides varname for each value col.
            value_colname: str = 'value',       # column of values from each value column
            ) -> 'Daf':
        """
        Turn columns into rows. This is called melt or unpivot.

        Each column that is not an id column gives one row for each row of this Daf.
        The row holds the id values, the name of the column, and its value.

        Args:
            id_cols: The columns that identify a row. They are kept as they are.
            varname_colname: The name of the new column that holds the old column names.
            value_colname: The name of the new column that holds the values.

        Returns:
            The new Daf.

        Raises:
            TypeError: `id_cols` is not a list.

        Examples:
            >>> d = Daf(lol=[['x', 1, 2], ['y', 3, 4]], cols=['id', 'a', 'b'])
            >>> d.wide_to_narrow(['id'])
            | id | varname | value |
            | -: | ------: | ----: |
            |  x |       a |     1 |
            |  x |       b |     2 |
            |  y |       a |     3 |
            |  y |       b |     4 |
            %% daf rows=4; cols=3; keyfield=''; name=''
        """
        if not isinstance(id_cols, list):
            raise TypeError("id_cols must be a list")

        narrow_daf = Daf(cols= id_cols + [varname_colname, value_colname])

        value_cols = [colname for colname in self.columns() if colname not in id_cols]

        for row_da in self:
            # accept id values from original wide rows.
            new_row = {col: row_da[col] for col in id_cols}

            # add a new row for each column
            for col in value_cols:
                new_row[varname_colname] = col
                new_row[value_colname] = row_da[col]
                narrow_daf.append(new_row)

        return narrow_daf


    def narrow_to_wide(self,
            id_cols:        T_cs,
            varname_col:    str = 'variable',
            value_col:      str = 'value',
            wide_cols:      T_cs | None = None,
            ) -> 'Daf':
        """
        Turn rows into columns. This is called pivot or spread.

        The rows of one id may be anywhere in the table. The result has one row for
        each id, in the order in which the ids first appear. The columns are the id
        columns and then the names found in `varname_col`, in the order first seen. An
        id that has no value for a name gets NULL. If an id has the same name more
        than once, the last value is kept.

        With `wide_cols`, those are the columns after the ids, in that order. A
        name that is not listed is left out.

        An empty Daf gives an empty Daf.

        Args:
            id_cols: The columns that identify a row.
            varname_col: The column whose values become the new column names.
            value_col: The column whose values fill the new columns.
            wide_cols: The names to use as columns, in order. If None, all names found.

        Returns:
            The new Daf.

        Examples:
            >>> d = Daf(lol=[['x', 'a', 1], ['y', 'a', 3], ['x', 'b', 2], ['y', 'b', 4]], cols=['id', 'variable', 'value'])
            >>> d.narrow_to_wide(['id'])
            | id | a | b |
            | -: | -: | -: |
            |  x | 1 | 2 |
            |  y | 3 | 4 |
            %% daf rows=2; cols=3; keyfield=''; name=''
            >>> d.narrow_to_wide(['id'], wide_cols=['b', 'a'])
            | id | b | a |
            | -: | -: | -: |
            |  x | 2 | 1 |
            |  y | 4 | 3 |
            %% daf rows=2; cols=3; keyfield=''; name=''
        """
        if not self.lol:
            return Daf()

        hd          = self.hd
        id_idxs     = [hd[col] for col in id_cols]
        var_idx     = hd[varname_col]
        val_idx     = hd[value_col]

        rows_dod: Dict[Tuple[Any, ...], Dict[Any, Any]] = {}
        names_d: Dict[Any, None] = {}

        for row_la in self.lol:
            id_tup = tuple([row_la[idx] for idx in id_idxs])
            wide_row_d = rows_dod.get(id_tup)
            if wide_row_d is None:
                wide_row_d = rows_dod[id_tup] = {}
            var_name = row_la[var_idx]
            wide_row_d[var_name] = row_la[val_idx]
            names_d[var_name] = None

        names_ls = list(wide_cols) if wide_cols is not None else list(names_d)

        wide_lol = [list(id_tup) + [wide_row_d.get(name, NULL) for name in names_ls]
                    for id_tup, wide_row_d in rows_dod.items()]

        return Daf(lol=wide_lol, cols=list(id_cols) + names_ls)


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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> print(d.md_daf_table_snippet(), end='')
            | x | y |
            | -: | -: |
            | 1 | a |
            | 2 | b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name=''
            An empty Daf has no rows and no columns, so only the summary line shows. This is so even if it has column names:

            >>> Daf(cols=['x', 'y'])
            %% daf rows=0; cols=0; keyfield=''; name=''
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
            disp_cols:      T_cs | None=None,    # use these column names instead of those defined in daf.
            header:         T_cs | None=None,    # use this header instead.
            ) -> str:
                
        """
        Make a Markdown table of the Daf.

        Without limits the whole table is written. With `max_rows` or `max_cols`, the
        first and last are kept, and the middle is replaced by `...`. Text longer
        than `max_text_len` is shortened by cutting out its middle. `just` has one
        character for each column: `<` left, `^` center, `>` right. The default is
        right. With no column names, `A`, `B` and so on are used. With `include_summary`,
        the Markdown can be read back with `from_md()`, as text.

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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
            >>> print(d.to_md())
            | id | v |
            | -: | -: |
            |  1 | a |
            |  2 | b |
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
            disp_cols:      Optional[T_ls]=None, # use these column names instead of those defined in daf.
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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> print(d.to_md_cols(), end='')
            | x | 1 | 2 |
            | y | a | b |
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
            >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
            >>> d.daf_to_lol_summary()
            [['x', 'y'], [1, 'a'], [2, 'b']]
            >>> big = Daf(lol=[[i, i] for i in range(20)], cols=['x', 'y'])
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

        # no limits, return summary.
        if not max_rows and not max_cols:
            return result_lol

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
    def dict_to_md(da: T_da, cols: Optional[T_ls]=None, just: str='<<') -> str:
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
            >>> d = Daf(lol=[['a'], ['b'], ['a']], cols=['g'])
            >>> d.value_counts_daf('g', sort=True)
            | g | counts |
            | -: | -----: |
            | a |      2 |
            | b |      1 |
            %% daf rows=2; cols=2; keyfield=''; name=''
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
        instead of every DafIterator instance claiming the full Union[T_ma, list] regardless of
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
        # one index of the column names, shared by every KeyedList row of this loop.
        self._kidx: Optional[KeyedIndex] = KeyedIndex(cast(dict, this_daf.hd)) if rtype == KeyedList else None

    def __iter__(self) -> 'DafIterator[DafIterRtype]':
        return self

    def __next__(self) -> DafIterRtype:
        if self._index < len(self.this_daf.lol):
            row = self.this_daf.lol[self._index]
            self._index += 1

            if self.rtype is dict:
                return cast(DafIterRtype, dict(zip(self.this_daf.hd.keys(), row)))

            elif self.rtype == KeyedList:
                return cast(DafIterRtype, KeyedList(self._kidx, row))

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

    def __init__(self, row: T_ma, indirect_col: Optional[str] = None):
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
