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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# from daffodil.lib.daf_types import T_df, T_dtype_dict #T_ls, T_li, T_doda, T_lb
                            # # T_lola, T_da, T_di, T_hllola, T_loda, T_dola, T_dodi, T_la, T_lota, T_buff, T_df, T_ds, 
                     
import pdfplumber

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Type, Callable ]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)   # pragma: no cover




#==== PDF
@classmethod
def _from_pdf(cls, filename, skip_to_header: int=0, skip_to_table: int=1):

    my_daf = None

    # Open the PDF file
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            # Extract all text from the page
            raw_text = page.extract_text()
            lines = raw_text.splitlines()
            
            # Skip the header (assumes the header is the first 4 lines)
            table_lines = lines[skip_to_table:]  # Adjust this based on actual header size

            header = lines[skip_to_header].split()  # Adjust this based on actual header size
            
            if not my_daf:
                my_daf = cls(cols=header)
            
            # Process the tabular lines
            for line_str in table_lines:
                
                clean_line_str = line_str.replace(',', '')  # remove commas 
                
                # Split the line into columns based on spacing
                line_la = clean_line_str.split()  # You may need to refine this based on column alignment
                
                my_daf.append(line_la)
                
    return my_daf