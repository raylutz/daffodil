
import os
import re
import shutil
import datetime
import logging

from Pydf.my_types import T_ls, T_da, T_lods

from typing import List, Dict, Any, Tuple, Optional
def fake_function(a: Optional[List[Dict[str, Tuple[int,str]]]] = None) -> Optional[Any]:
    return None

#from utilities import args, utils, zip_utils
import Pydf.pydf_utils as utils
#from models.DB import DB


""" This module deals with log file from lambda processes.

    Each lambda creates a logfile in /tmp because that is the only place
    files can be created in a lambda.

    Prior to shutting down the lambda, it copies the log file to 's3://{job_bucket}/.../{job_name}/{dirname}/logs'
    with filename "log!{chunk_name}.txt", where
    chunk_name is {group_root}!{dirname}!chunk_{chunk_idx}
    
    When logs are combined, they are placed in common folder so it can be easily zipped for later export.
    
    use stagedir which is defined for each stage. For each stage that does not use lambdas, the main log files
        are placed in {jobfolder}/logs/stagename/
        
    for stages that run lambdas, the logs are placed in the dirname folder under logs.
    The pipeline must include those separately.

    for gen_biabif:
        log!{archive_rootname}!bia_bif!chunk_NNNNN.txt
    for gentemplate:
        log_{style_num}_styles_chunk_NNN.txt
    for extractvote:
        log_{archive_rootname}_marks_chunk_NNN.txt
    for cmpcvr:
        log_{archive_rootname}_cmpcvr_chunk_NNN.txt



"""
global nologging
nologging = False

use_logger = False

global logsite
logsite = 'main'

openfile_handles: T_da = {}
openlogger_handles: T_da = {}

def set_logsite(name: str='main'):
    # logsite name can be 'main' or 'lambda'
    global logsite
    logsite = name

def is_lambda_logsite() -> bool:
    global logsite
    return bool(logsite == 'lambda')


def get_tmp_dirpath() -> str:
    if os.sep == '/':
        return '/tmp/'
    else:
        from models.DB import DB
        return DB.dirpath_from_dirname(dirname='logs', s3flag=False)   # this also creates the dir


def get_logfile_pathname(
        log_set: str='log',             # like 'log' or 'exc' for general logs and application exceptions.
        logsite: str='',                # like 'main' or 'lambda'
        extension: str='.txt',          # sometimes we log to csv files in the case of ballot exceptions.
        ) -> str:
        
    """ we have two log sets: main and lambda. Usage local to logs.py

        lambdas can only open files in /tmp, and must open lambda logs.
        If not running in lambda, may still want to
        1. act like we are running in a lambda and log to lambda logs,
        2. transfer lambda logs but still log in main.
        3. log and manipulate 'main' logs.
        
        log files are:
        log_main.txt
        exc_main.txt

    """
    from utilities import utils

    if utils.on_lambda():
        # if we are actually on a lambda, we only can manipulate that set, period.
        if is_lambda_logsite() or logsite == 'lambda':
            return f"/tmp/{log_set}!lambda{extension}"
        else:
            return ''     # logging turned off.
    else:
        from models.DB import DB
        dirpath = DB.dirpath_from_dirname(dirname='logs', s3flag=False)   # this also creates the dir

        if is_lambda_logsite() or logsite == 'lambda':
            # otherwise, we may want to log to lambda logs if we are running delegated functions locally,
            # or if we are transferring lambda logs from lambdas.
            return f"{dirpath}{log_set}!lambda{extension}"

        else:
            return f"{dirpath}{log_set}!main{extension}"

    
            

def append_report_by_pathname(pathname: str, string: str, end: str="\n", header: str='') -> None:
    """ Create report name of string in reports
        Used only within this module.
    """

    diagnose = False
    global use_logger

    if not pathname or not string:
        # log files are disabled in final writes of log files
        # when on lambda, but not if running locally and simulating
        # those files being written.
        return

    if not use_logger:
        global openfile_handles
        fh = openfile_handles.get(pathname)

        if fh:
            try:
                print(string, file=fh, end=end) #, flush=True)
            except Exception:
                fh.close()
                fh = None
                del openfile_handles[pathname]

        if not fh:

            fh = open(pathname, mode='ta+', buffering=1024, encoding="utf8")
            openfile_handles[pathname] = fh

            try:
                if header:
                    print(header, file=fh)
                if diagnose:
                    print(string, file=fh, end=end, flush=True)
                else:
                    print(string, file=fh, end=end) #, flush=True)
            except Exception:
                print(f"Failed to append to file: {pathname}") #, flush=True)
                raise RuntimeError
    else:
        # using logger
        # THIS NOT WORKING CORRECTLY.
        global openlogger_handles
        loggerhandle = openlogger_handles.get(pathname)

        if loggerhandle:
            loggerhandle.debug(string)
                
        else:
            loggerhandle = logging.getLogger(pathname)
            loggerhandle.setLevel(logging.DEBUG)
            filehandler = logging.FileHandler(pathname)
            filehandler.setFormatter(NoFormatFormatter())
            loggerhandle.addHandler(filehandler)
            
            openlogger_handles[pathname] = loggerhandle
            
            loggerhandle.debug(string)

            if header:
                loggerhandle.debug(header)
            loggerhandle.debug(string)


def close_open_logfile(pathname: str) -> None:

    global use_logger

    if not use_logger:
        global openfile_handles
        fh = openfile_handles.get(pathname)

        if fh is not None:
            try:
                fh.close()
                del openfile_handles[pathname]
            except Exception:
                pass
    else:
        global openlogger_handles
        logger_handle = openlogger_handles.get(pathname)

        if logger_handle is not None:
            logger_handle.flush()
            logger_handle.close()
            del openlogger_handles[pathname]   
    
    

def close_open_loggers() -> None:
    
    global use_logger

    if not use_logger:
        global openfile_handles
        for fh in openfile_handles.values():
            fh.close()
        openfile_handles = {}
    else:
        global openlogger_handles
        for logger_handle in openlogger_handles.values():
            logger_handle.flush()
            logger_handle.close()
        openlogger_handles = {}


def append_report(string: str, end: str='\n', log_set: str='log', header: str='') -> None:
    if not string: return
    logfile_path = get_logfile_pathname(log_set=log_set)
    append_report_by_pathname(logfile_path, string, end=end, header=header)


def sts(string: str, verboselevel: int=0, end: str='\n', enable: bool=True) -> str:
    """ Append string to logfile report.
        Also return the string so an interal version can be maintained
        for other reporting.
        The standard logger function could be used but we are interested
        in maintaining a log linked with each phase of the process.
        returns the string.
    """

    if string is None or not enable: return ''

    log_str = f"{get_datetime_str()}: {string}"

    _no_logging = nologging or not args.argsdict.get('job_name')

    if not _no_logging:
        try:
            append_report(log_str, end=end, log_set='log')
        except KeyError:
            # logs are not set up yet.
            pass
    if _no_logging or is_verbose_level(verboselevel):
        print(log_str, end=end, flush=True)
    return string+end


def stsa(string: str, verboselevel: int=0, end: str='\n') -> str:
    """ Append string to logfile report without any timestamp
        Also return the string so an interal version can be maintained
        for other reporting.
        The standard logger function could be used but we are interested
        in maintaining a log linked with each phase of the process.
    """

    if string is None: return

    if not nologging:
        append_report(string, end=end, log_set='log')
    if nologging or is_verbose_level(verboselevel):
        print(string, end=end, flush=True)
    return string+end


def exception_report(string: str, nobreak: bool=False) -> None:

    if not string: return
    log_str = f"{get_datetime_str()}: {string}"
    if not nologging:
        append_report(log_str, log_set='exc')
    sts(string, 3)
    if not nobreak and args.argsdict.get('CLI') and args.argsdict.get('break_on_exception_report') and not utils.on_lambda():
        import traceback; traceback.print_stack()
        error_beep()
        print("Use '--break_on_exception_report 0' to avoid this break when running in CLI mode.")
        import pdb; pdb.set_trace()     # permanent
        

def beep(freq: int=1080, ms: int=500):

    if not utils.is_linux():
        try:
            import winsound
        except ImportError:
            import os
            os.system('beep -f %s -l %s' % (freq, ms))
        else:
            winsound.Beep(freq, ms)
    
def error_beep():
    beep()
    

def notice_beep(freq: int=1080):
    beep(freq=freq, ms=250)
    beep(freq=freq, ms=250)
    



def get_datetime_str() -> str:

    return f"{datetime.datetime.now()}"
    #return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')


def get_YMDHMS_datetime_str() -> str:
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')


def is_verbose_level(verboselevel: int) -> bool:

    from utilities import args
    try:
        verbose_setting = args.argsdict['verbose']
    except KeyError:
        # argsdict not set up yet.
        verbose_setting = None
    if verbose_setting is None:
        verbose_setting = 3

    return bool(verbose_setting >= verbose_setting)


# def print_disagreements(string, end='\n'):
    # append_report(string, log_set='disagreements', end=end)


def merge_txt_dirname(dirname: str, destpath: str, file_pat: str, subdir: str='') -> int:
    """
    Local only.
    Consumes all .txt files in a given dirname and merges them into one
    :param dirname: name of dir of chunks to combine.
    :param destpath: path of file to create with combined files.
    :return:
    """
    from models.DB import DB
    file_list = DB.list_files_in_dirname_filtered(dirname, subdir=subdir, file_pat=file_pat, fullpaths=True, s3flag=False)[0]

    utils.remove_one_local_file(destpath)

    # the following concatenates the files in file_list into one file destpath
    file_cnt = 0
    with open(destpath, "a+b") as wfh:
        for txt_file in file_list:
            file_cnt += 1
            with open(txt_file, 'rb') as rfh:
                shutil.copyfileobj(rfh, wfh)

    return file_cnt


def report_main_logfile(log_set: str="log", stagename: str='undefined_stage'):
    """ copy main logfile to (job_name)/logs/(stagename)/log.txt
        and exceptions to    (job_name)/logs/(stagename)/exc.txt
        only if it exists and has nonzero size.
        do not use logs.sts() here!

        Note that s3 cannot accept any appending to files. So the
        log files are built locally, either on EC2 or local, and
        when the stage is done, the log file is copied to s3.

    """
    s3dirname = 'logs'
    
    if args.argsdict['CLI']:
        print("Not uploading logs because running from CLI")
        return

    # this generates the path to the local logfiles being built.
    logfile_pathname = get_logfile_pathname(log_set=log_set, logsite='main')  # this generates the path to the lambda or local folder for the logs.
    upload_name = f"{log_set}.txt"
    size = utils.safe_getsize(logfile_pathname)
    
    sts(f"Logfile {logfile_pathname} ({size} B), to upload to '(job_folder)/{s3dirname}/{stagename}/{upload_name}' ", 3)
    sts("Closing log file.", 3)

    close_open_logfile(logfile_pathname)
    buff = read_logfile(logfile_pathname)

    from models.DB import DB
    if buff:
        print(f"{len(buff):,} bytes read.", flush=True)
        print(f"Saving {logsite} logfile {logfile_pathname} to {s3dirname} as {upload_name}", flush=True)
        file_path = DB.save_data(data_item=buff, dirname=s3dirname, name=upload_name, subdir=stagename, s3flag=True)
        print(f"logfile {log_set}, {len(buff):,} characters saved to {file_path}", flush=True)
    else:
        print("Log file does not exist.", flush=True)
        # if nothing was logged, make sure the file does not exist, which is important for re-extractions of ballot exceptions.
        num_deleted = DB.delete_one_dirname_file(dirname=s3dirname, name=upload_name, subdir=stagename, s3flag=True, silent_error=True)
        if num_deleted:
            print(f"logfile {log_set} deleted, no new logfile exists.", flush=True)


def report_lambda_logfile(s3dirname: str, chunk_fn: str, log_set: str="log", subdir: str=''):
    """ copy lambda logfile at /tmp/log.txt to s3dirname
        only if it exists and has nonzero size.
        do not use logs.sts() here!

    """
    chunk_rn = utils.safe_splitext(chunk_fn)[0]
    
    logsite = 'lambda'
    logfile_pathname = get_logfile_pathname(log_set=log_set, logsite=logsite)  # this generates the path to the lambda or local folder for the logs.
    upload_name = f"{log_set}!{chunk_rn}.txt"
    print(f"Reading logfile {logfile_pathname}", flush=True)

    buff = read_logfile(logfile_pathname)
    from models.DB import DB
    if buff:
        print(f"Saving {logsite} logfile '{logfile_pathname}' to {s3dirname} as '{upload_name}'", flush=True)
        file_path = DB.save_data(data_item=buff, dirname=s3dirname, name=upload_name, format='.txt', subdir=subdir, s3flag=True)
        print(f"logfile {log_set}, {len(buff):,} characters saved to '{file_path}'", flush=True)
    else:
        # if nothing was logged, make sure the file does not exist, which is important for re-extractions of ballot exceptions.
        num_deleted = DB.delete_one_dirname_file(dirname=s3dirname, name=upload_name, subdir=subdir, s3flag=True, silent_error=True)
        if num_deleted:
            print(f"logfile {log_set} deleted, no new logfile exists.", flush=True)


def get_and_merge_s3_logs(
            dirname:        str,                    # work dirname
            subdir:         str='',                 # usually 'logs' where log chunks are placed.
            log_set:        str='log',              # log_set is either 'log', 'exc', or other prefix. (mixed in logs folder)
            chunk_pat:      str='',                 # log chunks
            dest_dirname:   str='',                 # defaults to work dirname
            dest_subdir:    str='',                 # leave as none to put logs in dirname root.
            discard_header: bool=False,
            ) -> Tuple[int, str] :                  # num_files, combined_log_s3filepath
    """
    Fetches all lambda logs from a job folder dirname/subdir on S3 that meet file_pat=fr"{log_set}!{chunk_pat}\.txt$"
    Note that chunk_pat does not include extension.
    combine into one file, write it to dirname/{log_set}!{dirname}.txt
    :param logs_folder: an S3 folder to fetch lambda logs from
    :returns:
        int: number of file combined
        destpath: combined_log_filepath
    log file name: f"log!{group_root}!{dirname}!chunk_{chunk_idx}.txt"
    """
    from models.DB import DB
    
    if not chunk_pat:
        chunk_pat = '.*'
    if not dest_dirname:
        dest_dirname = dirname

    sts(f"\nGetting the {log_set} files from s3 and combining")
    
    file_pat        = fr"{log_set}[!_]{chunk_pat}\.txt"

    # download all the log files
    sts(f"Downloading all {log_set} files, one per chunk", 3)
    # download according to matching pattern
    
    num_downloaded = DB.download_entire_dirname_v2(
                            dirname         = dirname,
                            subdir          = subdir,
                            file_pat        = file_pat,
                            #local_dirname   = subdir,                              # place in same subdir, prob logs/
                            empty_local_dir = False,                                # dirname should already be empty.
                            silent_error    = True, # bool(log_set == 'exc'),               # we may have no exceptions.
                            )

    if not num_downloaded:
        return 0, ''

    sts(f"Combining {num_downloaded} {log_set} files", 3)
    dest_name = f"{log_set}.txt"

    if not discard_header:
        local_dest_dirpath = DB.dirpath_from_dirname(dirname=dest_dirname, subdir=dest_subdir, s3flag=False)
        destpath = local_dest_dirpath + dest_name
        num_files = merge_txt_dirname(dirname=dirname, destpath=destpath, subdir=subdir, file_pat=file_pat)
    else:
        file_list = DB.list_files_in_dirname_filtered(dirname, subdir=subdir, file_pat=file_pat, s3flag=False)[0]
        num_files = len(file_list)
        destpath = DB.merge_csv_dirname_local(dirname=dirname, subdir=subdir, dest_dirname=dest_dirname, dest_subdir=dest_subdir, dest_name=dest_name, file_pat=file_pat)    

    if os.path.exists(destpath):
        if args.argsdict.get('zip_uploaded_log_files'):
            dest_name = zip_utils.zip_local_file(destpath)

        sts(f"Writing combined {log_set} file: {destpath} to s3 in dirname:logs/{dirname}'", 3)
        combined_log_s3filepath = \
            DB.upload_file_dirname(dirname=dest_dirname, subdir=dest_subdir, filename=dest_name)
        
    return num_files, combined_log_s3filepath


def read_logfile(logfile_path: str) -> Optional[str]:
    """
    Reads a logfile and returns it.
    If there is an error, returns None.
    """

    for i in range(2):
        try:
            with open(logfile_path, mode='rt', encoding="utf8") as lf:
                return lf.read()
        except FileNotFoundError:
            return None
        except UnicodeDecodeError:
        
            # we find sometimes that there is a decoding error, due to a dropped byte.
            # retry these without any delay but only one retry
            continue
        except Exception as err:
            print(f"An unexpected error '{err}' detected when trying to read the logfile {logfile_path}")
            return None
            
    else:
        print(f"Unicode Decode Error when reading the logfile {logfile_path}")
        return None
        
            


# def df_text_table(df, fields, sizes, limit=1000):
    # # DEPRECATED. Use md.py functions.

    # lod = df.to_dict(orient='records')

    # header_line = ''
    # under_line = ''
    # for idx, field in enumerate(fields):
        # field_size = abs(sizes[idx])
        # header_line += field.center(field_size) + ' '
        # under_line  += ('-' * field_size) + ' '

    # table_str = f"{header_line}\n{under_line}\n"

    # for rec in lod[0:limit]:
        # for idx, field in enumerate(fields):
            # field_size = sizes[idx]
            # field_len = abs(sizes[idx])
            # s = utils.shorten_str_keeping_ends(str(rec[field]), field_len)
            # if field_size < 0:
                # table_str += f"{'%*s' % (field_size, s)} "
            # else:
                # table_str += s.center(field_len) + ' '
        # table_str += "\n"

    # table_str += under_line
    # return table_str


def update_operation_status(
        argsdict:           T_da, 
        completion_percent: float=0.0, 
        status:             str='', 
        text_msg:           str='',
        ) -> None:

    """ Provide status for a given job to (job_bucket)/US/(job_name)/engine_status/engine_status.json
        status is one of ('Running', 'Completed', 'Failed')
    """

    op = argsdict.get('op')

    if argsdict['CLI']:
        sts(f"Job: {argsdict['job_name']} Operation '{op}' Progress: {round(completion_percent, 2)}%  {text_msg}", 3)
        return

    username = argsdict.get('username', 'username_not_set')

    if not status:
        status = f"Job: {argsdict['job_name']} Operation '{op}' {'Completed' if completion_percent == 100 else 'Running'}"

    status_block = {
        'username': username,
        'op': op,
        'status': status,
        'completion_percent': str(round(completion_percent, 2)),
        'text_msg': text_msg,
        }

    DB.save_data(status_block, dirname='engine_status', name='engine_status.json', s3flag=True)



def process_logs(
        dirname:            str, 
        subdir:             str='', 
        dest_dirname:       str='logs', 
        dest_subdir:        str='', 
        log_set_list:       Optional[T_ls]=None, 
        discard_header_list: Optional[T_ls]=None,
        ) -> T_lods:
    """ common section of code which may download all logs and combine them,
        and then produce result_lod to the combined logs.
        
        This is used when lambdas are utilized. Main logs are processed separately.

        For the cases in styles, when the logs are not tied to the archive
        Instead, need to download and combine:
            styles/{style_num}/exc!{style_num}!styles!chunk_\d+.txt
            styles/{style_num}/log!{style_num}!styles!chunk_\d+.txt

        Also need to create links to in style report:
            styles/{style_num}/{style_num}-redlined-1.png
            styles/{style_num}/{style_num}-redlined-2.png (if present)
            styles/{style_num}/{style_num}-template-1.png
            styles/{style_num}/{style_num}-template-2.png (if present)


            styles/{style_num}/{style_num}_rois.json
            styles/{style_num}/{style_num}_styles.json


        For extractvote:
            logs are like:
            marks/logs/exc!2019_Spring_Election_Ballot_Images1!marks!chunk_0.txt


    """

    result_lod = []
    
    if log_set_list is None:
        log_set_list = ['log', 'exc']
    if discard_header_list is None:
        discard_header_list = []
    if dest_subdir is None:
        dest_subdir = dirname

    for log_set in log_set_list:

        discard_header = bool(log_set in discard_header_list)
        # chunk_pat here does not include extension, but must match everything after log! or exc!
        num, path = get_and_merge_s3_logs(
            dirname=dirname,                    # dirname where lambdas ran
            subdir=subdir,                      # usually 'logs'
            log_set=log_set,                    # prefix of log files: 'log', 'exc', etc.
            chunk_pat=r'.*chunk_\d+',           # this is added to prefix
            discard_header=discard_header
            )
        if num:
            result_lod += [{'text': f"{dirname}_{log_set}", 'url': DB.path_to_url(path)}]

    return result_lod


def sane_str(string: str, limit: int=50) -> str:
    """ remove newlines and shorten to limit. """
    single_line = re.sub(r'[\n\r]', ' ', string)
    return utils.shorten_str_keeping_ends(single_line, limit)


def log_failed_ballot(
        ballot_id: str, 
        error_str: str, 
        ballot_image_path: str='', 
        style_num: str='', 
        precinct: str='', 
        archive_basename: str='',
        ) -> str:
    # this function only logs the ballot, it does not save the ballot file or images for later reporting.

    return log_ballot_exception(
        ballot_id=ballot_id,
        type='FAIL',
        error_str=error_str,
        style_num=style_num,
        ballot_image_path=ballot_image_path,
        precinct=precinct,
        archive_basename=archive_basename)

ballot_exceptions_header = 'timestamp|type|ballot_id|error_str|style_num|precinct|archive_basename|ballot_image_path'


def log_ballot_exception(
        ballot_id:          str='', 
        type:               str='', 
        error_str:          str='', 
        style_num:          str='', 
        ballot_image_path:  str='', 
        precinct:           str='', 
        archive_basename:   str='',
        ) -> str:
    # this function only logs the ballot, it does not save the ballot file or images for later reporting.
    # type should be FAIL if the ballot could not be processed.
    # WARN if the ballot was processed but there is an anomaly to be considered.

    ballot_id           = ballot_id or ''
    style_num           = style_num or ''
    ballot_image_path   = ballot_image_path or ''
    precinct            = precinct or ''
    archive_basename    = archive_basename or ''

    error_str_oneline   = re.sub(r'\n', '; ', error_str, flags=re.S)
    error_str_oneline   = re.sub(r'\|', '\\|', error_str_oneline)

    exception_report(f"EXCEPTION: ballot exception: {ballot_id} ({type}) {error_str}", nobreak=True)

    string = f"{get_datetime_str()}|{type}|{ballot_id}|{error_str_oneline}|{style_num}|{precinct}|{archive_basename}|{ballot_image_path}"
    append_report(string, end='\n', log_set='ballot_exceptions', header=ballot_exceptions_header)

    return string

# use this function to print the file and line number in any string to be printed.

def prog_loc() -> str:
    import inspect
    frame = inspect.currentframe()
    if frame is None:
        return ''
    try:
        frameinfo = inspect.getframeinfo(frame.f_back)          # type: ignore
        filename = re.split(r'[\\/]', frameinfo.filename)[-1]
        linenumber = frameinfo.lineno
    finally:
        del frame
    return f"[{filename}:{linenumber}]"