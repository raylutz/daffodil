import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from annotate import apply
L = 'src/daffodil/lib/'
D = "'Daf'"
apply(L+'schemaclass.py', {
  47: ('get_pandas_dtypes_from_schema', {'schema': 'type'}, None),
  99: ('_no_init', {'args': 'Any', 'kwargs': 'Any'}, 'None'),
  212: ('get_pandas_dtypes_from_schema', {'schema': 'type'}, 'Dict[str, Any]'),
})
apply(L+'daf_md.py', {
  132: ('new_window_link', {}, 'str'),
  167: ('md_toc', {}, 'str'),
  254: ('md_lol_table', {}, 'str'),
  310: ('md_cols_lol_table', {}, 'str'),
  426: ('_from_md', {}, D),
  818: ('md_2_html_snippet', {}, 'str'),
  827: ('md_2_html', {}, 'str'),
})
apply(L+'daf_pandas.py', {63: ('_from_pandas_df', {}, D)})
