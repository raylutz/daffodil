import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from annotate import apply
edits = {
  357: ('safe_regex_select', {'flags': 'int'}, None),
  373: ('safe_regex_replace', {'flags': 'int'}, None),
  459: ('convert_type_value', {}, 'Any'),
  801: ('profile_ls_to_lr', {'repeat_startswith': 'str'}, None),
  1061: ('safe_del_key', {}, 'Dict[Any, Any]'),
  1116: ('safe_stdev', {'listlike': 'Iterable[float]'}, 'float'),
  1126: ('safe_mean', {'listlike': 'Iterable[float]'}, 'float'),
  1136: ('beep', {}, 'None'),
  1152: ('error_beep', {}, 'None'),
  1176: ('sts', {'color': 'str'}, None),
  1225: ('stsloc', {'color': 'str'}, None),
  1676: ('is_tuple_of_type_len', {}, 'bool'),
  1685: ('len_slice', {}, 'int'),
  1719: ('slice_to_range', {'slice_obj': 'slice', 'length': 'int'}, 'range'),
  1754: ('_sanitize_cols', {'unnamed_prefix': 'str'}, None),
  1848: ('extract_docstring_parts', {'func': 'Callable'}, None),
  2010: ('unexcelstringify', {'astr': 'str'}, 'str'),
  2033: ('get_indirect_val', {}, 'Any'),
  93: ('default', {'obj': 'Any'}, 'Any'),
  1426: ('byte_line_generator', {}, 'Iterator[str]'),
  1433: ('text_line_generator', {}, 'Iterator[str]'),
}
edits[1870] = ('precheck_csv_cols', {'csv_buff': 'T_buff'}, None)
apply('src/daffodil/lib/daf_utils.py', edits, report=False)
