# md_demo.py
# copyright (c) 2024 Ray Lutz

""" md_demo.py provides tools to create static markdown demonsrations of code.  """


import os, inspect, re
import functools


@functools.lru_cache(maxsize=1)
def read_file_cached(path) -> str:

    # Read the content of the file
    with open(path, 'r') as file:
        file_content = file.read()
        
    return file_content


def md_code_seg(
        label: str='',          # string that can be used as a header to the section.
        silent: bool=False,     # in addition to returning the section of code, print it to stdout.
        ) -> str:               # 
    '''
        create a markdown compatible string that will
        extract a section of the current file where this
        function is called. The call to md_code_seg() is not included in the output.
        

        usage without permanent report:
        
            from utilities.md_demo import md_code_seg
        
            md.md_code_seg('show this section of code')
            """ this comment in triple quotes will be placed in the markdown report
                right after the header is printed from the label. If you want to have
                just markdown text and no code in this block, then make sure the next call to the
                md_code_seg function is right after this first comment block.
            """
            ## double comment hash symbols will allow comments in the code but
            ## will not be placed in the code block. Everything after this and up to
            ## a possible closing comment and next md_code_seg function call will be
            ## treated as code. Please note, it will also be run in the code.            
        
            abc = 10
            abc += 20
            
            print(abc)
            
            """ This final closing comment block will be included in the markdown 
                report as non-code markdown.
            """
        
            md.md_code_seg('next section of code')
        
            def = 5
            result = def * abc
            
            print(result)
            
            md.md_code_seg('next section of code')
        
            # additional sections...
        
            md.md_code_seg()    # include this to terminate any section early and at the end.
            

         
        usage with permanent report:
        
            from utilities.md_demo import md_code_seg, pr
        
            md_report = md.md_code_seg('show this section of code')
        
            abc = 10
            abc += 20
            
            md_report += md.pr(abc)
        
            md_report += md.md_code_seg('next section of code')
        
            def = 5
            result = def * abc
            
            md_report += md.pr(result)
            
            md_report += md.md_code_seg('next section of code')
        
            # additional sections...
                
            md.md_code_seg()    # include this to terminate any section and at the end.
            
            with open('demo.md', 'w') as file:
                file.write(md_report)
    '''    

    if not label:
        return ''

    # Get the path of the top-level script (the caller of this function)
    top_level_file_path = _get_top_level_file()

    file_content = read_file_cached(top_level_file_path)
        
    # Define the regex pattern, must allow not using md. before function name.
    pattern1 = fr'.*md_code_seg\(\s*(?:label\s*=\s*)?[\'"]{label}[\'"][^)]*\)[^\n]*\n(.*?)\n[^\n]*md_code_seg'

    # Search for the pattern in the file content
    match = re.search(pattern1, file_content, re.DOTALL)

    if not match:
        message = f"\n\nCode segment with label '{label}' not found. Remove parenthesis in label and install null call at the end.\n\n"
        if not silent:
            print(message)
        return message
        
    code_segment = match.group(1)
    
    # if label == 'Select a record by the key:':
        # breakpoint() #temp
    
    # look for a first comment block in triple quotes
    first_comment = ''
    pattern2 = r'^[\n\s]*f?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(.*)$'
    match = re.search(pattern2, code_segment, re.DOTALL)
    if bool(match):
        # pull out the extracted comment text and remove it from code_segment.
        first_comment = match[1].strip()
        code_segment = match[2]

    # look for a last comment block in triple quotes
    last_comment = ''
    pattern3 = r'^(.*?)f?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')[\s\n]*$'
    match = re.search(pattern3, code_segment, re.DOTALL)
    if bool(match):
        # pull out the extracted comment text and remove it from code_segment.
        code_segment = match[1]
        last_comment = match[2].strip()
    
    # remove comment lines starting with ##
    code_segment = '\n'.join([line for line in re.split(r'\n', code_segment) if not re.match(r'^\s*##', line)])
    
    # strip lines before/after code section
    code_segment = re.sub(r'^(\s*\n)*', '', code_segment, re.DOTALL)         # preserve indentation
    code_segment = re.sub(r'[\s\n]*$', '', code_segment, re.DOTALL)
    
    formatted_code = f"\n## {label}\n\n{first_comment}\n\n```python\n{code_segment}\n```\n\n{last_comment}\n"
    if not silent:
        print(formatted_code)
    return formatted_code


def _get_top_level_file():
    frame = inspect.currentframe()
    while frame.f_back:
        frame = frame.f_back
    top_level_file = frame.f_globals['__file__']
    return os.path.abspath(top_level_file)
    

def pr(s: str) -> str:
        print(s)
        return s


