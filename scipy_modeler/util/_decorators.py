"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

wrappers.py (contains : 3 functions)

wrappers.py provides a collection of higher-order
functions for decorating other (passed) functions.

***********************
**** Functions (3) ****
***********************
[1] send_logs (higher-order): Decorator for writing
    log files.
       |-> logs_wrapper:

[2] time_me (higher-order):

[3] profile (higher-order):
------------------------------------------------------
"""
import os
from functools import wraps
from datetime import datetime
from time import time

import scipy_modeler.util._settings as settings

temp_file = settings._file_paths["main_file"]
project_file = settings._file_paths["project_file"]
log_file = settings._file_paths["log_major"]
file_name = settings._file_paths["log_file"]
log = settings._file_paths["log"]


def send_logs(func):
    """
    Decorator function for writing log  files of functions
    ran to log folder. Creates a log folder for current day if
    one does not already exist in directiry.

    Parameters
    ----------
    func: Function.
        A function to decorate.

    Returns
    -------
    Inner wrapper function.
    """
    @wraps(func)
    def logs_wrapper(*args, **kwargs):
        """
        Logging wrapper.

        Parameters
        ----------
        (*args), {**kwargs} :  any number of positional
            and keyword arguments.

        Returns
        -------
        Result of function passed to outter function.
        """
        directory = os.path.join(temp_file, project_file, log_file, file_name)

        # TODO: get current working directory, create log file in it, populate with logs
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, log), 'a') as f:
            f.write("----\n")
            f.write(f"Ran function: \"{func.__name__}\".\n")
            f.write(f"With parameters: \"{kwargs}\".\n")
            f.write(f"Description:\n")
            f.write(f"\t{func.__doc__}\n")
            f.write(f"Ran at: {datetime.now().time()}.\n")
            start = time()
            result = func(*args, **kwargs)
            stop = time()
            f.write(f"Completed at: {datetime.now().time()}.\n")
            run_time = round(stop - start, 2)
            f.write(f"Total run time: {run_time}.\n")
            f.write("----\n")
        return result
    return logs_wrapper


def assert_types(*types):
    """
    Asserts the types which can be passsed to a function

    """
    def assert_types_wrapper(func):
        @wraps(func)
        def inner(**kwargs):
            assert len(types) == len(kwargs), \
                ValueError("Number of data types passed must equal "
                           "number fo arguments of function decorated.")
            argtypes = tuple(map(type, kwargs.values()))
            if argtypes != types:
                raise TypeError("Passed improper types to function.")
            return func(**kwargs)
        return inner
    return assert_types_wrapper


def time_me(func):
    """*** TODO ***"""
    raise NotImplementedError

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass


def profile(func):
    """*** TODO ***"""
    raise NotImplementedError

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass
