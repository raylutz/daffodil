# chunk_combine_check.py
#
# Combine the chunk files of a delegation run in two ways, and compare.
#
# - Today: the text of the chunks is appended into one CSV file, as DB.merge_csv_filepaths_local()
#   does. A later stage then loads that file as a Daf.
# - SqlDaf: the rows of every chunk are streamed into one table, in one transaction.
#   The table stays in the database file, so a later stage opens it instead of loading the CSV.
#
# The chunks are read from a copy. Nothing of AuditEngine's is written.
#
# Usage, with the daffodil_sql venv:
#   python chunk_combine_check.py <chunk dir> <scratch dir> [daf]
# With 'daf', the combined CSV is also loaded as a Daf, for the time and memory of that load.

import os
import re
import csv
import sys
import time
import resource

from typing import Iterator, List

from daffodil.daf import Daf
from daffodil.sqldaf import SqlDaf

CHUNK_RE: re.Pattern = re.compile(r'chunk_(\d+)\.csv$')


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def chunk_paths_of(chunk_dir: str) -> List[str]:
    """ The chunk files, in the order of their numbers. """
    names_ls: List[str] = [name for name in os.listdir(chunk_dir) if CHUNK_RE.search(name)]
    names_ls.sort(key=lambda name: (name[:CHUNK_RE.search(name).start()], int(CHUNK_RE.search(name).group(1))))
    return [os.path.join(chunk_dir, name) for name in names_ls]


def concat_csv(paths_ls: List[str], dest_path: str) -> int:
    """ Append the text of each chunk, less its header, as merge_csv_filepaths_local() does. """
    header_line: str = ''
    num_lines: int = 0
    with open(dest_path, 'w', encoding='utf8') as dest_f:
        for path in paths_ls:
            lines_ls: List[str] = open(path, encoding='utf8').read().splitlines()
            if not lines_ls or not lines_ls[0].strip():
                continue
            if not header_line:
                header_line = lines_ls[0]
                dest_f.write(header_line + '\n')
            elif lines_ls[0] != header_line:
                raise ValueError(f'{path} has another header')
            dest_f.write('\n'.join(lines_ls[1:]) + '\n')
            num_lines += len(lines_ls) - 1
    return num_lines


def chunk_rows(paths_ls: List[str], header_ls: List[str]) -> Iterator[List[str]]:
    """ The rows of every chunk, less the headers. A chunk with another header raises. """
    for path in paths_ls:
        with open(path, newline='', encoding='utf8') as chunk_f:
            reader = csv.reader(chunk_f)
            first_ls: List[str] = next(reader, [])
            if not first_ls or not any(val.strip() for val in first_ls):
                continue
            if first_ls != header_ls:
                raise ValueError(f'{path} has another header')
            yield from reader


def main() -> None:
    chunk_dir, scratch_dir = sys.argv[1], sys.argv[2]
    load_daf: bool = len(sys.argv) > 3 and sys.argv[3] == 'daf'
    paths_ls: List[str] = chunk_paths_of(chunk_dir)
    header_ls: List[str] = next(csv.reader(open(paths_ls[0], newline='', encoding='utf8')))
    print(f'{len(paths_ls)} chunks, {len(header_ls)} columns, start {rss_mb():.0f} MB')

    # Today: one CSV, then the Daf a later stage loads.
    start_f: float = time.perf_counter()
    csv_path: str = os.path.join(scratch_dir, 'combined.csv')
    num_rows: int = concat_csv(paths_ls, csv_path)
    print(f'concat to CSV:            {time.perf_counter() - start_f:6.1f} s, {num_rows} rows')
    if load_daf:
        start_f = time.perf_counter()
        combined_daf: Daf = Daf.from_csv(csv_path)
        print(f'load CSV as a Daf:        {time.perf_counter() - start_f:6.1f} s, {len(combined_daf)} rows, peak {rss_mb():.0f} MB')
        del combined_daf

    # SqlDaf: one table, streamed from the chunks.
    SqlDaf.db_path = os.path.join(scratch_dir, 'combined.db')
    if os.path.exists(SqlDaf.db_path):
        os.remove(SqlDaf.db_path)
    start_f = time.perf_counter()
    rows_sdaf: SqlDaf = SqlDaf('combined', cols=header_ls, lol=chunk_rows(paths_ls, header_ls))
    print(f'stream chunks to a table: {time.perf_counter() - start_f:6.1f} s, {len(rows_sdaf)} rows, '
          f'file {os.path.getsize(SqlDaf.db_path) / 1e6:.0f} MB, peak {rss_mb():.0f} MB')

    # A later stage opens it.
    SqlDaf.close()
    start_f = time.perf_counter()
    reopened_sdaf: SqlDaf = SqlDaf('combined')
    print(f'open the table again:     {time.perf_counter() - start_f:6.3f} s, {len(reopened_sdaf)} rows')

    # Same rows as the CSV.
    start_f = time.perf_counter()
    with open(csv_path, newline='', encoding='utf8') as csv_f:
        reader = csv.reader(csv_f)
        next(reader)
        csv_rows_ls = list(reader)
    table_rows_ls = [['' if val is None else val for val in row] for row in reopened_sdaf.iter_list()]
    print(f'same rows as the CSV, in order: {csv_rows_ls == table_rows_ls}  ({time.perf_counter() - start_f:.1f} s)')


if __name__ == '__main__':
    main()
