# make_ods_fixtures.py -- write the .ods files for tests/test_daf_from_ods.py.
#
# patterns.ods holds the markup that LibreOffice can write, one pattern to a row. A
# reader that uses regular expressions on content.xml gets many of them wrong. The text
# expected from every cell is written to patterns_expected.csv. All names are made up.
#
# layout.ods has two sheets, repeated and blank rows, and cells that keep a value.
#
# Run by hand, from the repo root: uv run --with odfpy python tests/make_ods_fixtures.py

import csv
import sys
from pathlib import Path

from odf.opendocument import OpenDocumentSpreadsheet
from odf.office import Annotation
from odf.table import Table, TableRow, TableCell, CoveredTableCell
from odf.text import P, Span, A, S, LineBreak
from odf.style import Style, TextProperties

COLS = ['PartialDescription', 'Amount', 'Name', 'Account', 'LookupMemo']


def plain(text: str) -> TableCell:
    cell = TableCell(valuetype='string')
    cell.addElement(P(text=text))
    return cell


def number(value: float, shown: str, valuetype: str = 'float') -> TableCell:
    """ A number cell. LibreOffice keeps the value in an attribute and the shown text in <text:p>. """
    attrs = {'valuetype': valuetype, 'value': value}
    if valuetype == 'currency':
        attrs['currency'] = 'USD'
    cell = TableCell(**attrs)
    cell.addElement(P(text=shown))
    return cell


def empty() -> TableCell:
    return TableCell()


def build(out_dir: Path) -> None:
    doc = OpenDocumentSpreadsheet()
    bold = Style(name='T1', family='text')
    bold.addElement(TextProperties(fontweight='bold'))
    doc.automaticstyles.addElement(bold)
    row_style = Style(name='ro1', family='table-row')     # LibreOffice gives every row a style.
    doc.automaticstyles.addElement(row_style)

    table = Table(name='PayeeLookup')
    rows: list = []         # (cells, expected values, what the row tests)

    rows.append(([plain(c) for c in COLS], COLS, 'header'))
    rows.append(([plain('Acme Hardware'), empty(), plain('Acme Hardware'), plain('Repairs'), empty()],
                 ['Acme Hardware', '', 'Acme Hardware', 'Repairs', ''], 'plain cells'))

    # two spaces inside the text, written as <text:s/>.
    p = P(text='Blue')
    p.addElement(S(c=2))
    p.addText('Ridge Gas')
    cell = TableCell(valuetype='string')
    cell.addElement(p)
    rows.append(([cell, empty(), plain('Blue Ridge Gas'), plain('Fuel'), empty()],
                 ['Blue  Ridge Gas', '', 'Blue Ridge Gas', 'Fuel', ''], 'two spaces, <text:s text:c="2"/>'))

    # part of the text bold: <text:span>.
    p = P(text='Zeta ')
    sp = Span(stylename=bold, text='Coffee')
    p.addElement(sp)
    cell = TableCell(valuetype='string')
    cell.addElement(p)
    rows.append(([cell, empty(), plain('Zeta Coffee'), plain('Food:Restaurant'), empty()],
                 ['Zeta Coffee', '', 'Zeta Coffee', 'Food:Restaurant', ''], 'part bold, <text:span>'))

    # a span that holds a run of spaces: <text:span>a<text:s/>b</text:span>.
    p = P()
    sp = Span(stylename=bold, text='Kappa')
    sp.addElement(S(c=1))
    sp.addText('Books')
    p.addElement(sp)
    cell = TableCell(valuetype='string')
    cell.addElement(p)
    rows.append(([cell, empty(), plain('Kappa Books'), plain('Books & Documents'), empty()],
                 ['Kappa Books', '', 'Kappa Books', 'Books & Documents', ''], 'span holding <text:s/>, and & in text'))

    # a link, as autocorrect makes of a web address.
    p = P()
    p.addElement(A(href='http://examplestore.com/', type='simple', text='Examplestore.com'))
    cell = TableCell(valuetype='string')
    cell.addElement(p)
    rows.append(([cell, empty(), plain('Examplestore'), plain('Supplies'), empty()],
                 ['Examplestore.com', '', 'Examplestore', 'Supplies', ''], 'link, <text:a>'))

    # a cell comment: <office:annotation> holds its own <text:p>.
    cell = TableCell(valuetype='string')
    note = Annotation()
    note.addElement(P(text='added 2026-01, check the spelling'))
    cell.addElement(note)
    cell.addElement(P(text='Omega Pharmacy'))
    rows.append(([cell, empty(), plain('Omega Pharmacy'), plain('Medical'), empty()],
                 ['Omega Pharmacy', '', 'Omega Pharmacy', 'Medical', ''], 'cell comment, <office:annotation>'))

    # two lines in one cell: two <text:p>.
    cell = TableCell(valuetype='string')
    cell.addElement(P(text='Delta Electric'))
    cell.addElement(P(text='Co'))
    rows.append(([plain('Delta Electric'), empty(), plain('Delta Electric Co'), plain('Utilities'), cell],
                 ['Delta Electric', '', 'Delta Electric Co', 'Utilities', 'Delta Electric\nCo'], 'two lines in one cell'))

    # a line break inside one paragraph.
    p = P(text='Gamma')
    p.addElement(LineBreak())
    p.addText('Water')
    cell = TableCell(valuetype='string')
    cell.addElement(p)
    rows.append(([plain('Gamma Water'), empty(), plain('Gamma Water'), plain('Utilities'), cell],
                 ['Gamma Water', '', 'Gamma Water', 'Utilities', 'Gamma\nWater'], 'line break, <text:line-break/>'))

    # a curly apostrophe, as autocorrect types it, and a no-break space.
    rows.append(([plain('Sam’s Diner'), empty(), plain('Sam’s Diner'), plain('Food:Restaurant'), empty()],
                 ['Sam’s Diner', '', 'Sam’s Diner', 'Food:Restaurant', ''], 'curly apostrophe, U+2019'))
    rows.append(([plain('Iota Market'), empty(), plain('Iota Market'), plain('Groceries'), empty()],
                 ['Iota Market', '', 'Iota Market', 'Groceries', ''], 'no-break space, U+00A0'))
    rows.append(([plain('Café Lumen'), empty(), plain('Cafe Lumen'), plain('Food:Restaurant'), empty()],
                 ['Café Lumen', '', 'Cafe Lumen', 'Food:Restaurant', ''], 'accented letter, e acute'))

    # amounts: a plain number, and a currency cell shown with $.
    rows.append(([plain('Check Withdrawal'), number(45, '45'), plain('Pat Roe'), plain('Glass Blowing'), empty()],
                 ['Check Withdrawal', '45', 'Pat Roe', 'Glass Blowing', ''], 'amount, a number cell'))
    rows.append(([plain('Check Withdrawal'), number(168.5, '$168.50', 'currency'), plain('Lawn Pros'), plain('Landscaping'), empty()],
                 ['Check Withdrawal', '168.5', 'Lawn Pros', 'Landscaping', ''], 'amount, a currency cell shown as $168.50'))

    # merged cells: the second cell is a <table:covered-table-cell>.
    cell = TableCell(valuetype='string', numbercolumnsspanned=2)
    cell.addElement(P(text='Theta Gym'))
    rows.append(([cell, CoveredTableCell(), plain('Theta Gym'), plain('Fitness'), empty()],
                 ['Theta Gym', '', 'Theta Gym', 'Fitness', ''], 'merged cells, <table:covered-table-cell>'))

    # leading and trailing spaces.
    rows.append(([plain(' Sigma Tires '), empty(), plain('Sigma Tires'), plain('Auto'), empty()],
                 [' Sigma Tires ', '', 'Sigma Tires', 'Auto', ''], 'leading and trailing spaces'))

    rows.append(([plain('Last Rule'), empty(), plain('Last Rule'), plain('Misc'), empty()],
                 ['Last Rule', '', 'Last Rule', 'Misc', ''], 'plain row after the others'))

    for cells, _expected, _what in rows:
        tr = TableRow(stylename=row_style)
        for cell in cells:
            tr.addElement(cell)
        table.addElement(tr)
    doc.spreadsheet.addElement(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_dir / 'patterns.ods'))
    with open(out_dir / 'patterns_expected.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(COLS + ['tests'])
        for _cells, expected, what in rows[1:]:
            writer.writerow(expected + [what])


def build_layout(out_dir: Path) -> None:
    """ Two sheets. The first has repeated rows and cells, blank rows, and typed cells. """
    doc = OpenDocumentSpreadsheet()
    row_style = Style(name='ro1', family='table-row')
    doc.automaticstyles.addElement(row_style)

    def row(*cells: TableCell, repeated: int = 1) -> TableRow:
        attrs = {'stylename': row_style}
        if repeated > 1:
            attrs['numberrowsrepeated'] = repeated
        tr = TableRow(**attrs)
        for cell in cells:
            tr.addElement(cell)
        return tr

    def typed(valuetype: str, attr: str, value: str, shown: str) -> TableCell:
        cell = TableCell(valuetype=valuetype, **{attr: value})
        cell.addElement(P(text=shown))
        return cell

    first = Table(name='First')
    first.addElement(row(plain('id'), plain('when'), plain('ok'), empty(), plain('rate')))
    first.addElement(row(plain('a1'), typed('date', 'datevalue', '2026-09-01', '09/01/26'),
                         typed('boolean', 'booleanvalue', 'true', 'TRUE'), empty(), typed('percentage', 'value', '0.05', '5%')))
    first.addElement(row(plain('same'), plain('x'), repeated=2))            # one row, written twice
    first.addElement(row(empty(), repeated=2))                              # two blank rows inside the table
    cell = TableCell(valuetype='string', numbercolumnsrepeated=3)           # one cell, written three times
    cell.addElement(P(text='rep'))
    first.addElement(row(cell))
    first.addElement(row(plain('wide'), empty(), empty(), empty(), empty(), plain('past the header')))
    first.addElement(row(empty(), repeated=1048000))                        # LibreOffice ends a sheet this way
    doc.spreadsheet.addElement(first)

    second = Table(name='Second')
    second.addElement(row(plain('k'), plain('v')))
    second.addElement(row(plain('b1'), number(2, '2')))
    doc.spreadsheet.addElement(second)

    doc.save(str(out_dir / 'layout.ods'))


if __name__ == '__main__':
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'tests/ods_fixtures')
    build(out_dir)
    build_layout(out_dir)
