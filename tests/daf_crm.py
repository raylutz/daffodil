# daf_crm.py

"""
# Simple CRM based on Daffodil

This project was started by San Diego Python community as a learning tool, originally called "Rolodex"

## Uses:
- Keep track of friends and contacts. -- But everyone has a cell phone that does this pretty well.
- Track employers and job market in San Diego for those looking for work. Possibly useful for SDPUG community.
- Create html reports that can be hosted on website.
- Tiny CRM for customer relationship management, for small firms.
- Includes email blasting capability.

##Interface:
- CLI interface is simple but provides limited flexibility for entering and editing records, 
   - later can migrate to tkinter or django.
- Open default data file upon starting the program.
- **Import** -- import records from other files, poss. with field changes.
- **Select** -- multiple selections will continue to reduce the set, until "Clear" selection or "All". Can select records based on tags.
- **Manage** -- Simulate dialog box interface. User can enter or accept each field, can end early and maybe select field to jump to. Loop through records selected. 
- **Add** -- Add individual records using the same interface.
- **Link** -- connect to googledocs so the data file can be viewed or managed there.
- **Report** -- Create reports of records selected.
- **Template** -- Edit email template to provide field substitution. Show email formatted with current template.
- **Blast** -- Send emails to the selected records.
	
## Fields
The following fields are a starter set. These were found to be very commonly used among contact managers.

    Prefix
    FullName
    FirstName
    LastName
    Nickname
    Suffix
    Gender
    Phone1
    Phone2
    Email1
    Email2
    Org
    Employer
    JobTitle
    Industry
    Addr1	
    Addr2	
    City	
    State	
    Zip
    Country
    Birth Date
    Website
    SocialMedia
    Do Not Email
    Do Not Phone
    Do Not Sms
    Notes
    Keywords
    Alias



## Program flow:
	
    
### start program
		command line arg to select datafile
        
### operations:
    Select, Report, Manage, Add, Delete, Link, Template, Blast, Write, Import, Export, eXit, help.
			(s, r, m, a, d, l, t, b, i, e, x, h):
            
### Select:
	search for single or multiple records, or "All"
	Once selected, these are used in other operations.
	Multiple finds keep reducing the number found.
	Displays portion of the records selected.

### Report
	select columns to include
    select to screen or to file.
	if to file, enter file name
	format md or html
    
### Manage
	process each record:
			edit the fields
				Check format of phone, url, 
    delete records individually
    quit processing them.
    
### Delete:
	delete all records selected (after confirmation).

### Link:
    enter info to link to googledocs

### Template:
    edit email template to use with email blast

### Blast:
    use template selected or default and send emails to all records selected.

### Import:
    enter file name -- add records from import file.
	possibly do field name and record adjusting.

### Export:
	enter file name, export selected records
	possibly do field name and record adjusting.

### Help:
	provide longer help.

### Internal State:
	current data table.
	current selection.
	email template
	report columns
	current 
"""

import sys
import os
import json


from daffodil.daf import Daf
import daffodil.lib.daf_utils    as utils
from daffodil.lib.daf_types import T_daf
                     
# import daffodil.lib.daf_md       as md
# import daffodil.lib.daf_pandas   as daf_pandas

# from daffodil.keyedlist import KeyedList

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable, Iterable # Iterator
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Type, Callable, Iterable]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)       # pragma: no cover


contacts_dtypes = {
    'Prefix':       str,
    'FullName':     str,
    'FirstName':    str,
    'LastName':     str,
    'Nickname':     str,
    'Suffix':       str,
    'Gender':       str,
    'Phone1':       str,
    'Phone2':       str,
    'Email1':       str,
    'Email2':       str,
    'Org':          str,
    'Employer':     str,
    'JobTitle':     str,
    'Industry':     str,
    'Addr1':        str,	
    'Addr2':        str,	
    'City':         str,
    'County':       str,
    'State':        str,	
    'Zip':          str,
    'Country':      str,
    'DOB':          str,
    'Website':      str,
    'SocialMedia':  str,
    'NoEmail':      bool,
    'NoPhone':      bool,
    'NoSms':        bool,
    'Notes':        str,
    'Keywords':     str,
    'Alias':        str,
    }

contacts_formats = {
    'Gender':   {'kind': 'choice', 'choices': ['F','M','O']},
    'Phone1':   {'kind': 'phone'},
    'Phone2':   {'kind': 'phone'},
    'Email1':   {'kind': 'email'},
    'Email2':   {'kind': 'email'},
    'State':    {'kind': 'state_code'},	
    'Zip':      {'kind': 'postal_code'},
    'Country':  {'kind': 'country_code'},
    'DOB':      {'kind': 'date'},
    'Website':  {'kind': 'url'},
    }

	
def main():
    banner = f"{'=' * 25}\nWelcome to daffodil crm\n{'=' * 25}\n\n"

    print(banner)

    ops_daf = gen_ops_daf()
    
    ops_list = ops_daf[:,'Abr'].to_list()

    while True:
        opchr = get_single_character(f"Select Operation ({', '.join(ops_list)}): ")
    
        try:
            op_rec = ops_daf[opchr].to_dict()
        except KeyError:
            print(f"{opchr}: Operation not recognized")
            op_help(ops_daf)
            continue

        print(opchr + ': ' + op_rec['Description'])

            
        op_rec['func'](**op_rec['kwargs'])
            
            
    
def gen_ops_daf() -> Daf:

    ops_daf = Daf(
        dtypes = {'OpName': str, 'Abr': str, 'func': Callable, 'kwargs': dict, 'Description': str, 'Help': str},
        )
    ops_daf.append([{'OpName': 'Select',     'func': op_select},
                    {'OpName': 'Report',     'func': op_report},
                    {'OpName': 'Manage',     'func': op_manage},
                    {'OpName': 'Delete',     'func': op_delete},
                    {'OpName': 'Glink',      'func': op_glink},
                    {'OpName': 'Template',   'func': op_template},
                    {'OpName': 'Blast',      'func': op_blast},
                    {'OpName': 'Import',     'func': op_import},
                    {'OpName': 'Export',     'func': op_export},
                    {'OpName': 'Help',       'func': op_help,       'kwargs': {'ops_daf': ops_daf}},
                    {'OpName': 'Quit',       'func': op_quit},
                   ])
    
    # add Desciption and Help from docstring    
        
    for klist in ops_daf.iter_klist():
        klist['Abr'] = klist['OpName'][0].lower()
        klist['Description'], klist['Help'] = utils.extract_docstring_parts(klist['func'])
        klist['kwargs'] = klist['kwargs'] or {} 

    ops_daf.set_keyfield('Abr')

    return ops_daf
    

def op_select():
    """ Select Records.
    
        Select single or multiple records, or "All".
        Multiple selections keep reducing the records found.
        Displays portion of the records selected. 
        Once selected, these are used in other operations.
    """
    pass
    
def op_report():    
    """ Create Reports.
    
        Create markdown or html reports to screen or file.
        Includes records previously selected.
        Allows selection of columns to include
    """
    pass

def op_manage():
    """ Manage: Edit Individual Records
    
        process each record selected.
        edit the fields
        Check formats 
        delete records individually
    """
    pass
    

def op_delete():
    """ Delete Selection.
    
        Delete all records selected after confirmation.
    """
    pass
    

def op_glink():
    """ Link to googlesheet.
    
    Estalish linkage to googledocs spreadsheet """
    pass
    
def op_template():
    """ Template Edit.
    
    Edit the template used in email blast.
    Use {} to insert fields.    
    """
    pass
    
def op_blast():
    """ Blast to email list
    
    using email template, merge data fields and send emails to all records selected. 
    """
    pass

def op_import():
    """ Import Records.
    
    enter file name -- add records from import file.
    allows adjusting field names
    """
    default_fn = 'crm_data.csv'
    
    print("\nImport Records:")
    fn = input(f"Please enter file name to import (default is {default_fn}): ")
    if not fn:
        fn = default_fn
    
    try:
        with open(fn) as f:
            csv_buff = f.read()
    except Exception as err:
        print(f"Error reading the file: {err}")
        return 
        
    missing_list, extra_list = utils.precheck_csv_cols(csv_buff, list(contacts_dtypes.keys()))

    from_to_dict = {}
    if missing_list or extra_list:
        print(f"The following expected fields are missing from the import file:\n{missing_list}\n\n"
              f"The following fields are extra, and may need to be renamed:\n{extra_list}\n\n"
              "You can provide a from-to dict json file to convert the existing field names to the proper names.\n"
              )
        from_to_fn = input("If conversion is required, enter the file name of the from_to.json file (default = '')")
        if from_to_fn:
            from_to_dict = json.loads(from_to_fn)
            
    my_daf = Daf.from_csv_buff(
            csv_buff    = csv_buff,
            # dtypes      = contacts_dtypes,        # initially we don't set dtypes bc we're not sure of the columns.
            noheader    = False,
            user_format = True,                     # if True, preprocess the file and omit comment lines.
            # sep: str=',',                         # field separator.
            unflatten   = False,                    # unflatten fields that are defined as dict or list (none exist).
            # include_cols: Optional[T_ls]=None,    # include only the columns specified. noheader must be false.
            )
    
    if from_to_dict:
        my_daf.rename_cols(from_to_dict=from_to_dict)
    
    my_daf.apply_dtypes(enforce_cols = True)
        
    cache_and_write_daf(my_daf)


def op_export():
    """ Export Selected Records.
    
    enter file name, export selected records
    allows adjusting field names
    """
    pass
    
def op_help(ops_daf):
    """ Help
    
    Verbose help 
    """
    
    print("\n" + ops_daf[:,['OpName', 'Abr', 'Description']].to_md(just='^^<'))
    return
    

def op_quit():
    """ Quit.
    
    Exit daffodil crm 
    """
    
    print("\nGoodbye!\n")
    sys.exit(0)


#===================================
# support functions

def cache_and_write_daf(my_daf: T_daf):
    pass


def get_single_character(prompt):
    print(prompt, end='', flush=True)
    if os.name == 'nt':  # Windows
        import msvcrt
        char = msvcrt.getch().decode('utf-8')
    else:  # Unix-like (Linux, macOS)
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char
        
    
if __name__ == "__main__":
    main()