# pydf_md.py -- markdown generation

# copyright (c) 2024 Ray Lutz

import re
#import copy
import markdown
#import functools

#import pprint
from string import Template

from daffodil.lib.daf_types import T_ls, T_lola, T_da, T_li #, T_doda, T_df, T_lf, T_loda, T_loloda, T_lodolodi, T_ts, T_ds, T_dola 
import daffodil.lib.daf_utils as utils

from typing import List, Dict, Any, Tuple, Optional, Union #, cast
def fake_function(a: Optional[List[Dict[str, Union[int, Tuple[int, Any]]]]] = None) -> Optional[int]:
    return None   # pragma: no cover


# Translation dictionaries for table alignment
left_rule = {'<': ':', '^': ':', '>': '-'}
right_rule = {'<': '-', '^': ':', '>': ':'}

#====================================================================================
#    Markdown Content Retrieval Function

# @functools.lru_cache(maxsize=None)  # Apply caching to load_locale_data
# def load_locale_ds(locale: str) -> T_ds:
    # """ load localized text strings.
        # Use remove_newlines when putting text in the cell of a table, for example.
    # """

    # # Load the appropriate @section@ marked Markdown file based on the locale
    # markdown_content = DB.load_data(dirname='localized_text', name=f"mdtext_{locale}.md", s3flag=False)

    # # Define a regular expression pattern to match section headers
    # pattern = r'^%SECTION_NAME%\s*=\s*"([^"]+)"\s*$'

    # # Use re.split to split the content based on section headers
    # # this provides both the headers and the content between the headers.
    # sections_ls = re.split(pattern, markdown_content, flags=re.M)
    
    # # remove any leading & trailing newlines around the block.
    # locale_ds = {sections_ls[idx]: sections_ls[idx+1] for idx in range(1, len(sections_ls), 2)}
    
    # return locale_ds
    

# def getmd(section_name:str, locale:str='', default:str='', remove_newlines: bool=False):
    # """
    # Markdown Content Retrieval Function

    # This function, getmd, allows you to retrieve Markdown content based on the section name 
    # and the selected locale. It loads the appropriate Markdown file, parses it using an 
    # INI-style structure, and caches the results to avoid unnecessary file access when 
    # called multiple times.

    # Usage:
        # content = md.getmd(section_name, locale)

    # Parameters:
        # section_name (str): The name of the section to retrieve.
        # locale (str): The selected locale for content retrieval.
        # default (str): used if no record is found.

    # Example:
        # locale = 'en_US'
        # section_name = 'Section 1'
        # content = getmd(section_name, locale)
        # print(content)
 
    # # Example usage:
    # content = getmd(section_name='Section 1', locale = 'en_US')
    # print(content)
    
    # Variables can be included in the text in {} curly braces.
    # To substitute, use this syntax:
    # content = md.getmd('section name').format(argbane=argvalue, ...)
 
    # """

    # # commented out for Pydf.
    # # if not locale:
        # # locale = args.argsdict.get('locale', 'en_US') or 'en_US'

    # locale_ds = load_locale_ds(locale)

    # # Retrieve the section content from the locale data
    # if section_name not in locale_ds:
        # print(f"WARN: Can't find section '{section_name}' in locale {locale} text data.")

        # if locale != 'en_US':
            # locale_ds = load_locale_ds('en_US')
            # section_content = locale_ds.get(section_name, default)
        # else:    
            # section_content = default
    # else:
        # section_content = locale_ds[section_name]
        
    # if remove_newlines:
        # section_content = re.sub(r'[\n\r]+', ' ', section_content, flags=re.S)
        # # section_content = re.sub(r'<br>',    ' ', section_content).strip()
        # section_content = re.sub(r'\s\s+',   ' ', section_content)

    # return section_content
    

def mdlink(url_or_s3path: str='', title: str='', new_window: bool=False) -> str:
    if not url_or_s3path:
        return f"{title} (Not Available)"
        
    if url_or_s3path.startswith('s3'):    
        if not title:
            title = utils.safe_basename(url_or_s3path)
            url   = utils.s3path_to_url(url_or_s3path)
    else:
        url = url_or_s3path
        
    if new_window:
        return new_window_link(url, title)
    return f"[{title}]({url})"


def mdlink_s3path(s3path: str='', title: str='', new_window: bool=False) -> str:

    return mdlink(url_or_s3path=s3path, title=title, new_window=new_window)


def new_window_link(url: str='', title: str=''):
    return f'<a href="{url}" target="_blank">{title}</a>'
    

def escape_internal_link(header_text: str) -> str:
    # remove all special characters.
    # convert spaces to '-'
    #
    # to create a link: f"['local text'](#{md.escape_internal_link('header text to link to')})
    #
    # <h2 id="precinct-v-windsor-wds-1-2">Precinct V Windsor Wds 1-2</h2>

    cleaned_text = re.sub(r'[^\w_\s\-]', '', header_text)   # remove all non-word, underscores, spaces and hyphens.
    internal_link = cleaned_text.lower().strip().replace(' ', '-')
    internal_link = re.sub(r'--', '-', internal_link)
    internal_link = re.sub(r'--', '-', internal_link)   # just in case there were overlapping patterns missed above (it happens)
    
    return internal_link
    
    
def md_parse_link(md_link: str) -> Tuple[str, str]:

    # Usage:
    #   text, link = md_parse_link(md_link)

    text = md_link
    link = ''
    if md_link.startswith('['):
        match = re.search(r'^\[([^\]]+)\]\(([^)]+)\)$', md_link)    
        text = match[1]             # type: ignore
        link = match[2]             # type: ignore
        
    return text, link
        

def md_toc(headings_list: T_ls):

    rep = "# Table of Contents\n\n"
    
    for heading in headings_list:
        rep += f"- [{heading}](#{escape_internal_link(heading)})\n"  
    return rep + "\n\n"

# called by Pydf.to_md()

def md_lol_table(
        records_lol:        T_lola, 
        header:             Optional[T_ls]=None, 
        includes_header:    bool=False, 
        align:              Optional[List[Tuple[str, str]]]=None, 
        just:               str='', 
        omit_header:        bool=False, 
        shorten_text:       bool=True, 
        max_text_len:       int=80, 
        smart_fmt:          bool=False,
        include_idx:        bool=False,
        sum_col_idxs:       Optional[T_li]=None,
        ):
    """
    Generate a Doxygen-flavor Markdown table from records.
    This could be called md_lol_table()

    records -- list of list of strings
    header -- True if first line should be the heading
    alignment - List of pairs alignment characters.  The first of the pair
        specifies the alignment of the header, (Doxygen won't respect this, but
        it might look good, the second specifies the alignment of the cells in
        the column.

        Possible alignment characters are:
            '<' = Left align (default for cells)
            '>' = Right align
            '^' = Center (default for column headings)
    """
    if not records_lol:
        return ''

    # reorient data into columns
    # would like to not shorten if shorten_text=False but not sure how to do it here.
    # the reason is that the column is set to the length of the entire string rather than longest line
    
    # first remove the header if there is one.
    if includes_header:
        header = records_lol[0]
        records_lol=records_lol[1:]

    if not records_lol:
        return ''    

    if sum_col_idxs:
        new_row = utils.lol_sum_cols(records_lol, sum_col_idxs=sum_col_idxs)
        records_lol += [new_row]

    try:
        cols_lol = utils.transpose_lol(records_lol)
    except RuntimeError as err:
        print(f"{err}: records:{records_lol}")
        #logs.error_beep()
        import pdb; pdb.set_trace() #perm -- debugging aide
        pass
        
    if include_idx:
        # there is never a header at this point.
        col_idx_ls = [str(idx) for idx in range(len(records_lol))]                        
        cols_lol = [col_idx_ls] + cols_lol
        if header:
            header = ['idx'] + header
        just = '^' + just

    return md_cols_lol_table(cols_lol, header=header, align=align, just=just, 
                omit_header=omit_header, shorten_text=shorten_text, max_text_len=max_text_len, smart_fmt=smart_fmt)


def md_cols_lol_table(
        cols_lol:       T_lola, 
        header:         Optional[T_ls] = None, 
        align:          Optional[List[Tuple[str, str]]] = None, 
        just:           str='', 
        omit_header:    bool = False, 
        shorten_text:   bool = True,
        max_text_len:   Optional[int] = None,
        smart_fmt:      bool = False,
        ):

    """ Use this function when a number of columns of data already exist that should be listed side by side.
        This function does not include the header in the data because it is easier to provide a separate list of strings.
    """

    if not cols_lol:
        return ''
    if max_text_len is None:
        max_text_len = 80
        
    #mutated_cols_lol = copy.deepcopy(cols_lol)
    mutated_cols_lol = cols_lol

    # either get the header or pull it off the columns
    if omit_header:
        pass
    elif header:
        # add header to the columns
        header = [f"{col}".replace('\n', '<br>').replace('\r', '') for col in header]
        for idx, col in enumerate(mutated_cols_lol):
            s = utils.safe_get_idx(header, idx, default='')
            col.insert(0, s)
    else:
        raise RuntimeError ("md header is required if 'omit_header' not specified.")
    
    num_cols = len(mutated_cols_lol)
    max_row = max([len(col) for col in mutated_cols_lol], default=0)
    
    # fill new_cols with blanks to start with.
    new_cols_lol = []
    for _ in range(num_cols):
        new_cols_lol.append([''] * max_row)

    # sanitize cols
    for col_idx, col in enumerate(mutated_cols_lol):
        pass
        for row_idx in range(max_row):
            try:
                val = col[row_idx]
            except Exception:
                val = ''
            if smart_fmt:
                val = utils.smart_fmt(val)
            else:    
                val = f"{val}"
            val = val.replace('\n', '<br>').replace('\r', '')
            
            if shorten_text:
                val = utils.shorten_str_keeping_ends(val, max_text_len)
            new_cols_lol[col_idx][row_idx] = escape_md_table_text(val)

    # cols_lol = new_cols_lol

    # Fill out any missing alignment characters.
    just_tup: Tuple[str, ...] = ()
    if not just and not align:
        just = '^' * num_cols
    if just and not align:
        just_tup = tuple(just)
    if just_tup and not align:    
        align = []
        for just_char in just_tup:
            align.append(('^', just_char))

    extended_align = align if align is not None else []
    extended_align = extended_align[0:num_cols]
    if len(extended_align) < num_cols:
        extended_align += [('^', '<')
                           for i in range(num_cols-len(extended_align)) ]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    # here field width must take into account if the lines have breaks, and if so, then calculate for longest line.
    field_widths = []
    for column in new_cols_lol:
        w_max = 0
        for text in column:
            lines = text.split('<br>')
            for line in lines:
                w = len(line)
                if w > w_max:
                    w_max = w
        field_widths.append(w_max)
    
    # now that the columns have been formatted, remove the header from the columns.
    
    heading_template    = '| ' + ' | '.join([f"{{:{a}{w}}}" for a, w in zip(heading_align, field_widths)]) + ' |'
    row_template        = '| ' + ' | '.join([f"{{:{a}{w}}}" for a, w in zip(cell_align, field_widths)])    + ' |'
    ruling              = '| ' + ' | '.join([left_rule[a] + '-'*(w-2) + right_rule[a] for a, w in zip(cell_align, field_widths)]) + ' |'

    s = ''
    if not omit_header:
        heading = [col.pop(0) for col in new_cols_lol]
        if heading:
            s += heading_template.format(*heading).rstrip() + '\n'
        s += ruling.rstrip() + '\n'
    for row in zip(*new_cols_lol):
        s += row_template.format(*row).rstrip() + '\n'
    return s
        
       
    
def escape_md_table_text(s: str) -> str:
    """ this function escapes text inside a table minimally to allow the markdown table to be built.
    """
    s = s.replace('|', r'\|').replace("\n", '<br>')
    return s
    
    
def escape_raw_text(text: str='') -> str:
    """ Escape raw text so markdown does not recognise any special characters
        Good for regex strings to allow any characters in a raw string.
    """
    
    # summary:
    # regex: 12.338
    # build from list: 4.442
    # build from set: 3.38
    # build from string: 2.225
    
    
    # for example:
    # md.escape_raw_text(r'^(.*)\s*\(.+\)\s*$')
    # r'^\(\.\*\)\\s\*\\\(\.\+\\\)\\s\*$'
    
    if not text:
        return ''
        
    # # character build using list method:
    # # timeit.timeit("md.escape_raw_text(r'^(.*)\s*\(.+\)\s*$')", setup="from utilities import md")
    # # 4.442 secs for 1,000,000 iterations
            
    # spec_char_list = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '/', '|']
    # output_str = ''
    # for ch in text:
        # if ch in spec_char_list:
            # output_str += '\\' + ch
        # else:
            # output_str += ch

    # character build using set
    # 3.38 secs
    # spec_char_set = {'\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '/', '|'}
    # output_str = ''
    # for ch in text:
        # if ch in spec_char_set:
            # output_str += '\\' + ch
        # else:
            # output_str += ch


    # character build using string method
    # 2.225 secs for 1,000,000 iterations
    spec_chars = r'\`*_{}[]()#+-.!/|'
    output_str = ''
    for ch in text:
        if ch in spec_chars:
            output_str += '\\' + ch
        else:
            output_str += ch


    
            
    # regex method
    # 12.338 secs for 1,000,000 iterations
    
    # output_str = re.sub(r'([\\`*_{}\[\]()\#+\-\.!/|])', r'\\\1', text)

    return output_str
    
    
def md_process_template(template_str: str, mapping: T_da):

    # replace occurrences of ${dict_key} with values from mapping (dict)
    
    t = Template(template_str)
    identifiers = t.get_identifiers()   # type: ignore # Returns a list of the valid identifiers in the template, 
                                        # in the order they first appear, ignoring any invalid identifiers.
    mapping_keys_set = set(mapping.keys())
    identifiers_set  = set(identifiers)
    
    if mapping_keys_set != identifiers_set:
        print(f"There is a difference between the identifiers in the template:{identifiers_set} vs. the map:{mapping_keys_set}")
        print(f"The template appears to be {'valid' if t.is_valid() else 'invalid'}.")    # type: ignore
    
    return t.safe_substitute(mapping=mapping)


def md_2_html_snippet(md: str, strip_newlines: bool=True):
    #print("Parsing md to html:\n"+md+"\n")
    snippet = markdown.markdown(md, extensions=['tables','toc'])
    if strip_newlines:
        snippet = snippet.replace('\n', '')
    return snippet
    
    
def md_2_html(title: str, md: str, strip_newlines: bool=False, include_open_image_js: bool=False):

    md_html = md_2_html_snippet(md, strip_newlines=strip_newlines)


    return f"""
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>

<title>{title}</title>

<style type="text/css">
{stylesheet}
</style>
{open_image_js if include_open_image_js else ''}
{identical_values_js}
</head>

<body>{md_html}</body></html>"""

stylesheet = """
body {
  font-family: Helvetica, arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  padding-top: 10px;
  padding-bottom: 10px;
  background-color: white;
  padding: 30px;
  color: #333;
}

body > *:first-child {
  margin-top: 0 !important;
}

body > *:last-child {
  margin-bottom: 0 !important;
}

a {
  color: #4183C4;
  text-decoration: none;
}

a.absent {
  color: #cc0000;
}

a.anchor {
  display: block;
  padding-left: 30px;
  margin-left: -30px;
  cursor: pointer;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
}

h1, h2, h3, h4, h5, h6 {
  margin: 20px 0 10px;
  padding: 0;
  font-weight: bold;
  -webkit-font-smoothing: antialiased;
  cursor: text;
  position: relative;
}

h2:first-child, h1:first-child, h1:first-child + h2, h3:first-child, h4:first-child, h5:first-child, h6:first-child {
  margin-top: 0;
  padding-top: 0;
}

h1:hover a.anchor, h2:hover a.anchor, h3:hover a.anchor, h4:hover a.anchor, h5:hover a.anchor, h6:hover a.anchor {
  text-decoration: none;
}

h1 tt, h1 code {
  font-size: inherit;
}

h2 tt, h2 code {
  font-size: inherit;
}

h3 tt, h3 code {
  font-size: inherit;
}

h4 tt, h4 code {
  font-size: inherit;
}

h5 tt, h5 code {
  font-size: inherit;
}

h6 tt, h6 code {
  font-size: inherit;
}

h1 {
  font-size: 28px;
  color: black;
}

h2 {
  font-size: 24px;
  border-bottom: 1px solid #cccccc;
  color: black;
}

h3 {
  font-size: 18px;
}

h4 {
  font-size: 16px;
}

h5 {
  font-size: 14px;
}

h6 {
  color: #777777;
  font-size: 14px;
}

p, blockquote, ul, ol, dl, li, table, pre {
  margin: 15px 0;
}

hr {
  border: 0 none;
  color: #cccccc;
  height: 4px;
  padding: 0;
}

body > h2:first-child {
  margin-top: 0;
  padding-top: 0;
}

body > h1:first-child {
  margin-top: 0;
  padding-top: 0;
}

body > h1:first-child + h2 {
  margin-top: 0;
  padding-top: 0;
}

body > h3:first-child, body > h4:first-child, body > h5:first-child, body > h6:first-child {
  margin-top: 0;
  padding-top: 0;
}

a:first-child h1, a:first-child h2, a:first-child h3, a:first-child h4, a:first-child h5, a:first-child h6 {
  margin-top: 0;
  padding-top: 0;
}

h1 p, h2 p, h3 p, h4 p, h5 p, h6 p {
  margin-top: 0;
}

li p.first {
  display: inline-block;
}

ul, ol {
  padding-left: 30px;
}

ul :first-child, ol :first-child {
  margin-top: 0;
}

ul :last-child, ol :last-child {
  margin-bottom: 0;
}

dl {
  padding: 0;
}

dl dt {
  font-size: 14px;
  font-weight: bold;
  font-style: italic;
  padding: 0;
  margin: 15px 0 5px;
}

dl dt:first-child {
  padding: 0;
}

dl dt > :first-child {
  margin-top: 0;
}

dl dt > :last-child {
  margin-bottom: 0;
}

dl dd {
  margin: 0 0 15px;
  padding: 0 15px;
}

dl dd > :first-child {
  margin-top: 0;
}

dl dd > :last-child {
  margin-bottom: 0;
}

blockquote {
  border-left: 4px solid #dddddd;
  padding: 0 15px;
  color: #777777;
}

blockquote > :first-child {
  margin-top: 0;
}

blockquote > :last-child {
  margin-bottom: 0;
}

table {
  padding: 0;
}
table tr {
  border-top: 1px solid #cccccc;
  background-color: white;
  margin: 0;
  padding: 0;
}

table tr:nth-child(2n) {
  background-color: #f2f2f2; /* #f8f8f8; */
}

table tr th {
  font-weight: bold;
  border: 1px solid #cccccc;
  /* text-align: left; */
  margin: 0;
  /* padding: 6px 13px; */
  padding: 4px 6px;
}

table tr td {
  border: 1px solid #cccccc;
  /* text-align: left; */
  margin: 0;
  /* padding: 6px 13px; */
  padding: 4px 6px;
}

table tr th :first-child, table tr td :first-child {
  margin-top: 0;
}

table tr th :last-child, table tr td :last-child {
  margin-bottom: 0;
}

img {
  max-width: 100%;
}

span.frame {
  display: block;
  overflow: hidden;
}

span.frame > span {
  border: 1px solid #dddddd;
  display: block;
  float: left;
  overflow: hidden;
  margin: 13px 0 0;
  padding: 7px;
  width: auto;
}

span.frame span img {
  display: block;
  float: left;
}

span.frame span span {
  clear: both;
  color: #333333;
  display: block;
  padding: 5px 0 0;
}

span.align-center {
  display: block;
  overflow: hidden;
  clear: both;
}

span.align-center > span {
  display: block;
  overflow: hidden;
  margin: 13px auto 0;
  text-align: center;
}

span.align-center span img {
  margin: 0 auto;
  text-align: center;
}

span.align-right {
  display: block;
  overflow: hidden;
  clear: both;
}

span.align-right > span {
  display: block;
  overflow: hidden;
  margin: 13px 0 0;
  text-align: right;
}

span.align-right span img {
  margin: 0;
  text-align: right;
}

span.float-left {
  display: block;
  margin-right: 13px;
  overflow: hidden;
  float: left;
}

span.float-left span {
  margin: 13px 0 0;
}

span.float-right {
  display: block;
  margin-left: 13px;
  overflow: hidden;
  float: right;
}

span.float-right > span {
  display: block;
  overflow: hidden;
  margin: 13px auto 0;
  text-align: right;
}

code, tt {
  margin: 0 2px;
  padding: 0 5px;
  white-space: nowrap;
  border: 1px solid #eaeaea;
  background-color: #f8f8f8;
  border-radius: 3px;
}

pre code {
  margin: 0;
  padding: 0;
  white-space: pre;
  border: none;
  background: transparent;
}

.highlight pre {
  background-color: #f8f8f8;
  border: 1px solid #cccccc;
  font-size: 13px;
  line-height: 19px;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 3px;
}

pre {
  background-color: #f8f8f8;
  border: 1px solid #cccccc;
  font-size: 13px;
  line-height: 19px;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 3px;
}

pre code, pre tt {
  background-color: transparent;
  border: none;
}

.thumbnail {
    width: 400px;
    height: auto;
    cursor: pointer;
    }

.form-input {
    background: none;
    border: none;
    }
    
/* The Modal (background) */
.modal {
    display: none; /* Hidden by default */
    position: fixed; /* Stay in place */
    z-index: 1; /* Sit on top */
    padding-top: 100px; /* Location of the box */
    left: 0;
    top: 0;
    width: 100%; /* Full width */
    height: 100%; /* Full height */
    overflow: auto; /* Enable scroll if needed */
    background-color: rgb(0,0,0); /* Fallback color */
    background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
    }
    
/* Modal Content */
.modal-content {
    background-color: #fefefe;
    margin: auto;
    padding: 20px;
    border: 1px solid #888;
    width: 80%;
    }
    
/* The Close Button */
.modal-close {
    color: #aaaaaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    }
    
.modal-close:hover,
.modal-close:focus {
    color: #000;
    text-decoration: none;
    cursor: pointer;
    }
    
.width-100{
    width: 100%;
    }
    
.bold-txt{
    font-weight: bold;
    }
    
.pointer {cursor: pointer;}

.align-left {float: left;}

.input-width {width: 9cm;}
"""

open_image_js1 = """
  <script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open('', '_blank', 'width=' + img.width + ',height=' + img.height);
    win.document.write('<html><head><title>' + title + '</title></head><body style="overflow: scroll; margin: 0;"><img src="' + url + '"></body></html>');
  };
}
  </script>

"""

open_image_js2 = """
<script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open('', '_blank', 'width=' + img.width + ',height=' + img.height);
    win.document.write('<html><head><title>' + title + '</title></head><body style="overflow: scroll; margin: 0;"><div style="height: ' + img.height + 'px;"></div><img src="' + url + '"></body></html>');
  };
}
</script>
"""

open_image_js3 = """
<script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open('', '_blank', 'width=' + img.width + ',height=' + img.height);
    var doc = win.document;
    doc.open();
    doc.write('<html><head><title>' + title + '</title><style>body { margin: 0; overflow: auto; }</style></head><body><img src="' + url + '" style="width: 100%; height: auto; display: block; max-width: none;"></body></html>');
    doc.close();
  };
}
</script>
"""

# js4 this version works the best except still has about:blank
# trying url as first parameter
open_image_js = """
<script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open('', '_blank', 'width=' + img.width + ',height=' + img.height);
    win.document.open();
    win.document.write('<html><head><title>' + title + '</title></head><body style="overflow: scroll; margin: 0;"><img src="' + url + '"></body></html>');
    win.document.close();
  };
}
</script>
"""

modal_html = """
<!-- The Modal -->
<div id="bmd-modal" class="modal">
    <!-- Modal content -->
    <div class="modal-content">
        <span class="modal-close">&times;</span>
        <p><input type="text" class="width-100" id="txt-name">
        <p><button id="btn-update">Update</button></p></p>
    </div>
</div>
"""

open_modal_js = """
<script>
    function showModal(input_id) 
    {
        // Get the modal
        var modal = document.getElementById("bmd-modal");
        
        // Prefill modal before opening
        var txt_name = document.getElementById("txt-name");
        txt_name.value = document.getElementById(input_id).value;
        
        // Open the modal
        modal.style.display = "block";
    
        // Get the <span> element that closes the modal
        var span = document.getElementsByClassName("modal-close")[0];
        
        // When the user clicks on <span> (x), close the modal
        span.onclick = function() {
            modal.style.display = "none";
        }
        
        // When the user clicks anywhere outside of the modal, close it
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
        
        // Get the button that updates the report html
        var btn_update = document.getElementById("btn-update");
        
        // When the user clicks the update button, modify report
        btn_update.onclick = function() {
            var corrected_text = document.getElementById("txt-name").value;
            document.getElementById(input_id).value = corrected_text;
            modal.style.display = "none";
        }
    }
</script>
"""

identical_values_js = """
<script>
    var name_dict = {};
    var value_dict = {};
    var input_text_list;
    var is_first_change = true;
    window.addEventListener('load',  
    function() 
    { 
        group_identical_input_values();
    });

    function group_identical_input_values() 
    {
        input_text_list = document.querySelectorAll('input[type="text"]');
              
        // Assign value_dict. Create value to list of tag names dict.
        for (let input of input_text_list) 
        {
            if (!(input.value in value_dict))
            {
                value_dict[input.value] = [input.name];
            }
            else
            {
                value_dict[input.value].push(input.name);
            }            
        }
        
        
        for (let input of input_text_list)
        {
            
            for (let value_list of Object.values(value_dict))
                if (value_list.includes(input.name))
                {
                    name_dict[input.name] = value_list;
                    break;
                }
        }
    }
    
    function updateText(input_name) 
    {
        // Get the input
        var input = document.getElementsByName(input_name)[0];
        var new_value = input.value
        
        // Get input text set
        input_list = name_dict[input_name];
        
        // Update value
        for (let i=0; i < input_list.length; i++)
        {
            document.getElementsByName(input_list[i])[0].value = new_value;
        }
    }
    
    function toggleLabel(checkbox)
    {
        var contest = checkbox.name.replace(/_checkbox$/, "");
        var label_id = contest.concat("_label")
        
        // Get corresponding label
        var label = document.getElementById(label_id);
        
        if (checkbox.checked == true)
        {
            label.innerHTML = "Verified";
        }        
        else
        {
            label.innerHTML = "";
        }
    }
    
    function checkboxListener(checkbox)
    {        
        if (is_first_change)
        {
            alert("Save report updates by clicking the submit button at the bottom of page.");
            is_first_change = false
        }
    }
    
</script>
"""

# this one opens the image as a file.
open_image_js5 = """
<script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open('', '_blank', 'width=' + img.width + ',height=' + img.height);
    var doc = win.document;
    doc.write('<html><head><title>' + title + '</title></head><body style="margin:0;"><iframe id="imageFrame" style="border:none;width:100%;height:100%;overflow:scroll;" allowfullscreen></iframe></body></html>');
    doc.close();
    doc.getElementById('imageFrame').src = url;
  };
}
</script>
"""


open_image_js6 = """
<script>
function openImage(url, title) {
  var img = new Image();
  img.src = url;
  img.onload = function() {
    var win = window.open(' ', '_blank', 'toolbar=0,location=0,menubar=0');
    win.document.write('<html><head><title>' + title + 
        '</title></head><body style="overflow: scroll; margin: 0;"><h1>' + title + 
        '<p><img src="' + url + 
        '" style="max-width: 100%; height: auto;"></body></html>');
    win.document.close();
  };
}
</script>
"""

old_open_image_js = """
<script>
function openImage(url) {
  var windowName = "imageWindow";
  var windowFeatures = "resizable=yes,scrollbars=yes,status=yes";
  var newWindow = window.open("", windowName, windowFeatures);
  
  // Create a new document object for the window
  var doc = newWindow.document;
  
  // Replace the default content with the full-size image
  doc.body.innerHTML = '<img src="' + url + '" style="max-width: 100%; max-height: 100%; object-fit: contain;">';
  
  // Set the window location to the URL of the full-size image
  doc.location.href = url;

  // Set the document title to the title parameter
  doc.title = title;
}
</script>
"""

