# pydf_googlesheets_demo.py

import Pydf.Pydf as Pydf


def main():

    my_pydf = Pydf.from_googlesheet(spreadsheet_id='1')
    
    print(my_pydf.to_md())
    
    my_pydf.to_googlesheet(spreadsheet_id='2')

    print("written to the googlesheet")


if __name__ == '__main__':
    main()    