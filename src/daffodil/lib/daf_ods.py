# daf_ods.py
"""
Read one sheet of an OpenDocument spreadsheet, an .ods file, into a list of rows.

Only the content of the cells is read. Styles, comments, links and formulas are left out.
The file is read with the standard library: `zipfile` opens it and `xml.etree.ElementTree`
parses its `content.xml`, so no package is needed. `Daf.from_ods()` uses this.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from daffodil.lib.daf_types import T_la, T_lola, T_ls

_TABLE  = '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}'
_TEXT   = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'
_OFFICE = '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}'

_SHEET    = _TABLE + 'table'
_ROW      = _TABLE + 'table-row'
_CELL     = _TABLE + 'table-cell'
_COVERED  = _TABLE + 'covered-table-cell'
_REP_COLS = _TABLE + 'number-columns-repeated'
_REP_ROWS = _TABLE + 'number-rows-repeated'
_PARA     = _TEXT + 'p'
_SPACES   = _TEXT + 's'
_TAB      = _TEXT + 'tab'
_BREAK    = _TEXT + 'line-break'
_NOTE     = _OFFICE + 'annotation'

# A number, date or boolean cell keeps its value in an attribute. The text shown may be
# rounded or formatted, such as $168.50 for 168.5, so the value is read instead.
_VALUE_ATTRS: Dict[str, str] = {
    'float':      _OFFICE + 'value',
    'currency':   _OFFICE + 'value',
    'percentage': _OFFICE + 'value',
    'date':       _OFFICE + 'date-value',
    'time':       _OFFICE + 'time-value',
    'boolean':    _OFFICE + 'boolean-value',
}
_VALUE_TYPE = _OFFICE + 'value-type'


def _para_text(elem: ET.Element) -> str:
    """ The text of a paragraph, with runs of spaces, tabs and line breaks put back, and markup left out. """
    parts_ls: T_ls = [elem.text or '']
    for child in elem:
        if child.tag == _SPACES:
            parts_ls.append(' ' * int(child.get(_TEXT + 'c', '1')))
        elif child.tag == _TAB:
            parts_ls.append('\t')
        elif child.tag == _BREAK:
            parts_ls.append('\n')
        elif child.tag != _NOTE:                    # a comment is not part of the cell.
            parts_ls.append(_para_text(child))      # a span or a link: keep its text.
        parts_ls.append(child.tail or '')
    return ''.join(parts_ls)


def _cell_value(cell: ET.Element) -> str:
    """ The content of one cell, as text. Several paragraphs are joined with a newline. """
    value_attr = _VALUE_ATTRS.get(cell.get(_VALUE_TYPE, ''))
    if value_attr is not None:
        value = cell.get(value_attr)
        if value is not None:
            return value
    return '\n'.join(_para_text(para) for para in cell if para.tag == _PARA)


def ods_sheet_names(source: str | Path | bytes) -> T_ls:
    """ The names of the sheets of an .ods file, in order. """
    return [sheet.get(_TABLE + 'name', '') for sheet in _sheets(source)]


def _sheets(source: str | Path | bytes) -> List[ET.Element]:
    zip_source = io.BytesIO(source) if isinstance(source, bytes) else source
    with zipfile.ZipFile(zip_source) as zip_file:
        root = ET.fromstring(zip_file.read('content.xml'))
    return list(root.iter(_SHEET))


def ods_to_lol(source: str | Path | bytes, sheet: int | str = 0) -> T_lola:
    """
    Read one sheet of an .ods file into a list of rows of text.

    Every cell is text. A number, date or boolean cell gives its stored value, such as
    `168.5` or `2026-09-01`, not the text shown. A cell with several lines gives them
    joined with a newline. An empty cell, and a cell hidden under a merged cell, give
    NULL. Rows are padded with NULL to the width of the widest row. LibreOffice leaves
    off empty cells at the end of a row, so without this most rows would be short.

    A blank row inside the table is kept, as a row of NULL, so the rows still line up
    with the rows of the spreadsheet. The blank rows after the last row with content
    are left out.

    Args:
        source: The path of the file, or its bytes.
        sheet: The position of the sheet, from 0, or its name. The first sheet by default.

    Returns:
        The rows, each a list of text.

    Raises:
        IndexError: There is no sheet at that position.
        KeyError: There is no sheet with that name.
    """
    sheets = _sheets(source)
    if isinstance(sheet, int):
        if not -len(sheets) <= sheet < len(sheets):
            raise IndexError(f"ods_to_lol(): there is no sheet {sheet}. The file has {len(sheets)} sheets.")
        table = sheets[sheet]
    else:
        names_ls = [elem.get(_TABLE + 'name', '') for elem in sheets]
        if sheet not in names_ls:
            raise KeyError(f"ods_to_lol(): there is no sheet named {sheet!r}. The sheets are {names_ls}.")
        table = sheets[names_ls.index(sheet)]

    rows_lola: T_lola = []
    num_blank = 0                                   # blank rows not yet added: only those before a row with content are.
    for row in table.iter(_ROW):
        row_la: T_la = []
        for cell in row:
            if cell.tag == _CELL:
                value = _cell_value(cell)
            elif cell.tag == _COVERED:
                value = ''
            else:
                continue
            row_la.extend([value] * int(cell.get(_REP_COLS, '1')))
        while row_la and row_la[-1] == '':
            row_la.pop()                            # the empty cells that fill a row out to the last column.
        num_rows = int(row.get(_REP_ROWS, '1'))
        if row_la:
            rows_lola.extend([] for _ in range(num_blank))
            num_blank = 0
            rows_lola.extend(list(row_la) for _ in range(num_rows))
        else:
            num_blank += num_rows                   # the blank rows at the end may be repeated a million times.

    width = max((len(row_la) for row_la in rows_lola), default=0)
    for row_la in rows_lola:
        if len(row_la) < width:
            row_la.extend([''] * (width - len(row_la)))
    return rows_lola
