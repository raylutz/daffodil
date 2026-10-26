# Selecting and indexing

Each selection is a child: it refers to rows of its parent, and runs no query until it is read.

## Indexing with brackets

::: daffodil.sqldaf.SqlDaf.__getitem__

## By condition

::: daffodil.sqldaf_select._select_where
    options:
      heading: "select_where"

::: daffodil.sqldaf_select._select_where_sql
    options:
      heading: "select_where_sql"

::: daffodil.sqldaf_select._select_by_dict
    options:
      heading: "select_by_dict"

::: daffodil.sqldaf_select._split_where
    options:
      heading: "split_where"

::: daffodil.sqldaf_select._select_first_row_by_dict
    options:
      heading: "select_first_row_by_dict"

## Columns

::: daffodil.sqldaf_select._select_cols
    options:
      heading: "select_cols"

::: daffodil.sqldaf_select._calc_cols
    options:
      heading: "calc_cols"

## By key

::: daffodil.sqldaf.SqlDaf.__contains__

::: daffodil.sqldaf.SqlDaf.keys

::: daffodil.sqldaf.SqlDaf.get_existing_keys

::: daffodil.sqldaf_select._select_krows
    options:
      heading: "select_krows"

::: daffodil.sqldaf_select._select_records_daf
    options:
      heading: "select_records_daf"
