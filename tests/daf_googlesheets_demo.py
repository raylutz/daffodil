# pydf_googlesheets_demo.py
import os
import sys
import Pydf.Pydf as daf
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'src'))


def main():

    spreadsheet_id1 = '1'
    spreadsheet_id2 = '1'

    print(f"trying to load the googlesheet from '{spreadsheet_id1}'")

    my_daf = daf.Pydf.from_googlesheet(spreadsheet_id=spreadsheet_id1)
    
    print(my_daf.to_md())
    
    print(f"trying to write the googlesheet to '{spreadsheet_id2}'")

    my_daf.to_googlesheet(spreadsheet_id=spreadsheet_id2)

    print("written to the googlesheet")


if __name__ == '__main__':
    main()    