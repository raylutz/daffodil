# daf_pdf.py

"""

# Daf -- Daffodil -- python dataframes.

The Daf class provides a lightweight, simple and fast alternative to provide 
2-d data arrays with mixed types.

This file handles indexing with square brackets[] as functions that operate on
a daf instance 'self'.

"""

"""
    MIT License

    Copyright (c) 2024 Ray Lutz

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


"""
See README file at this location: https://github.com/raylutz/daffodil/blob/main/README.md
"""
import re
# import os
# import sys
# no longer need the following due to using pytest
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# from daffodil.lib.daf_types import T_df, T_dtype_dict #T_ls, T_li, T_doda, T_lb
                            # # T_lola, T_da, T_di, T_hllola, T_loda, T_dola, T_dodi, T_la, T_lota, T_buff, T_df, T_ds, 
                     
import pdfplumber

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Type, Callable ]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)   # pragma: no cover




#==== PDF
@classmethod
def _from_pdf(cls, filename, skip_to_header: int=0, skip_to_table: int=1):

    my_daf   = None
    precinct = ''
    category = ''
    type     = ''
    state    = 'looking'
    diagnose = False

    # Open the PDF file
    with pdfplumber.open(filename) as pdf:
        for page_number,page in enumerate(pdf.pages):
            print(f"Processing page {page_number}")
            # Extract all text from the page
            raw_text = page.extract_text()
            lines = raw_text.splitlines()
            
            # Skip the header (assumes the header is the first 4 lines)
            table_lines = lines[skip_to_table:]  # Adjust this based on actual header size
            
            top_line = lines[0]
            match   = re.search(r'County: ([\w\s]+) ELECTION PARTICIPATION DEMOGRAPHICS', top_line)
            if match:
                county = match[1]

            header = lines[skip_to_header]
            
            
            header = header.replace('65 & OVER', '65_&_OVER')
            header = header.replace('AGE UNKNOWN', 'AGE_UNKNOWN')

            header_ls = header.split()  # Adjust this based on actual header size
            
            header_ls = ['County', 'Precinct', 'Category', 'Type', 'Value']
            
            if not my_daf:
                my_daf = cls(cols=header_ls)
            
            state = 'looking'
            # Process the tabular lines
            for line_str in table_lines:
                
                #=================================================
                # line parser -- should be provided by a function??
                
                
                clean_line_str = line_str.replace(',', '')  # remove commas
                for (compound_word, safe_word) in [ ('REGISTERED',  'RV'), 
                                                    ('Democrat',    'DEM'), 
                                                    ('No Party',    'NP'),
                                                    ('Other',       'OTH'),
                                                    ('Republican',  'REP'),
                                                    ('TOTAL VOTED', 'AV'), 
                                                    ('ABSENTEE',    'MB')]:
                    if compound_word in clean_line_str:
                        clean_line_str = clean_line_str.replace(compound_word, safe_word)
                
                if diagnose:
                    print(f"line: {clean_line_str}")
                
                # Split the line into columns based on spacing
                line_la = clean_line_str.split()  # You may need to refine this based on column alignment
                
                if len(line_la) < 5 and line_la[0] != 'Page':
                    precinct = ' '.join(line_la)
                    if diagnose:
                        print(f"precinct: {precinct}")
                    continue
                
                if state == 'looking':
                    category = line_la[0]
                    if category in ['DEM', 'NP', 'OTH', 'REP', 'Total']:
                        state = 'gathering'
                        line_la.pop(0)          # pop category so type is consistently first.
                        if diagnose:
                            print(f"gathering {category}")
                    else:
                        continue
                        
                if state == 'gathering':
                    if line_la[0] == 'TURNOUT':
                        state = 'looking'
                        continue
                    value = line_la[-1]
                    type  = line_la[0]
                    table_la = [county, precinct, category, type, value]
                        
                    if diagnose:
                        print(f"table_la: {table_la}")    
                    my_daf.append(table_la)
                
    return my_daf