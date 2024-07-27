# daf_crm.py

"""
# Simple CRM based on Daffodil

This project was started by San Diego Python community as a learning tool, originally called "Rolodex"

## Uses:
- Keep track of friends and contacts. -- But everyone has a cell phone that does this pretty well.
- Track employers and job market in San Diego for those looking for work. Possibly useful for SDPUG community.
- Freate html reports that can be hosted on website.
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

### Write:
	write data file

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

from daffodil.daf import Daf


contacts_dtypes = {
    'Prefix': str,
    'FullName': str,
    'FirstName': str,
    'LastName': str,
    'Nickname': str,
    'Suffix': str,
    'Gender': str,
    'Phone1': str,
    'Phone2': str,
    'Email1': str,
    'Email2': str,
    'Org': str,
    'Employer': str,
    'JobTitle': str,
    'Industry': str,
    'Addr1': str,	
    'Addr2': str,	
    'City': str,	
    'State': str,	
    'Zip': str,
    'Country': str,
    'DOB': str,
    'Website': str,
    'SocialMedia': str,
    'NoEmail': bool,
    'NoPhone': bool,
    'NoSms': bool,
    'Notes': str,
    'Keywords': str,
    'Alias': str,
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
        print(opchr)
    
        try:
            op_rec = ops_daf[opchr].to_dict()
        except KeyError:
            print("Operation not recognized")
            op_help()
            continue
            
        op_rec['func'](**op_rec['kwargs'])
            
            
    
def gen_ops_daf() -> Daf:

    ops_daf = Daf(
        cols = ['OpName', 'Abr', 'Description', 'func', 'kwargs', 'Help'],
        )
    ops_daf.lol = [
            ['Select', 's', 'Select Records', op_select, {},
                    """ search for single or multiple records, or "All"
                    Once selected, these are used in other operations.
                    Multiple selections keep reducing the records found.
                    Displays portion of the records selected. """,
                ],
            ['Report', 'r', 'Create Reports', op_report, {},
                    """ Create markdown or html reports to screen or file.
                        select columns to include
                    """,
                ],    
            ['Manage', 'm', 'Edit Individual Records', op_manage, {}, 
                    """ process each record selected.
                        edit the fields
                        Check formats 
                        delete records individually
                    """,
                ],
            ['Delete', 'd', 'Delete Selection', op_delete, {},
                    """ Delete all records selected after confirmation.""",
                ],
            ['Glink',  'g', 'Link to Googledocs spreadsheet', op_glink, {},
                    """ Estalish linkage to googledocs spreadsheet"""
                ],
            ['Template', 't', 'edit email template to use with email blast', op_template, {},
                    """ Edit the template used in email blast. """
                ],
            ['Blast',   'b', 'blast to email list', op_blast, {},
                    """ using email template, merge data fields and send emails to all records selected. """
                ],
            ['Import',  'i', 'Import Records', op_import, {},
                    """ enter file name -- add records from import file.
                        provides field name and record adjusting
                    """,
                ],
            ['Export', 'e', 'Export Selection', op_export, {},
                    """ enter file name, export selected records
                        provides field name and record adjusting.
                    """,
                ],
            ['Help', 'h', 'Help', op_help, {'ops_daf': ops_daf}, "Verbose Help"],
            ['Quit', 'q', 'Quit', op_quit, {}, "Quit"],
        ]

    ops_daf.set_keyfield('Abr')

    return ops_daf
    

def op_select():
    """ search for single or multiple records, or "All"
        Once selected, these are used in other operations.
        Multiple selections keep reducing the records found.
        Displays portion of the records selected. 
    """
    pass
    
def op_report():    
    """ Create markdown or html reports to screen or file.
        select columns to include
    """
    pass

def op_manage():
    """ process each record selected.
        edit the fields
        Check formats 
        delete records individually
    """
    pass
    

def op_delete():
    """ Delete all records selected after confirmation."""
    pass
    

def op_glink():
    """ Estalish linkage to googledocs spreadsheet """
    pass
    
def op_template():
    """ Edit the template used in email blast. """
    pass
    
def op_blast():
    """ using email template, merge data fields and send emails to all records selected. """
    pass

def op_import():
    """ enter file name -- add records from import file.
        provides field name and record adjusting
    """
    pass


def op_export():
    """ enter file name, export selected records
        provides field name and record adjusting.
    """
    pass
    
def op_help(ops_daf):
    """ Verbose help """
    
    print("\n" + ops_daf[:,['OpName', 'Abr', 'Description']].to_md(just='^^<'))
    return
    

def op_quit():
    print("\nGoodbye!\n")
    sys.exit(0)


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