import os
import json
import warnings
from datetime import datetime
from typing import Dict, Union, List, Tuple, Optional

# config json type (read in from config.json)
Nested_str_dict = Dict[str, Optional[Union[str, Dict[str, Optional[str]]]]]

def silent_mkdir(path: str):
    """
    Try to make directory - if exists, pass.
    """
    try: 
        os.makedirs(path)
    except FileExistsError:
        pass

def load_json_config(config_json_path: str) -> Nested_str_dict:
    """
    Load config file json into dict.
    """
    with open(config_json_path) as config_file:
        config_data: dict = json.load(config_file)

        return config_data

def impute_json_config(config_json: Nested_str_dict,
                       *args, warning: Optional[bool]=False
                            ) -> Nested_str_dict:
    """
    Replace null config values with *args.

    Parameters
    ----------
    config_json : dict
        A dictionary of configuration settings. 
    *args : str ... 
        Any number of positional arguments to 
        replace null values of a configuration
        dictionary in order.
    warning : bool, optional
        Whether or not to warn the user about a
        miss match in the number of null values
        in the configuration dict and the number
        of positional args passed for imputation
        (*args). Default = False.
    """
    num_args: int = len(args)
    total_null: int = 0
    
    for key in config_json.keys():
        if not config_json[key]:
            total_null += 1

    # if more null values than impute values, 
    # warn user and recycle the last arg passed
    if total_null != num_args:
        if warning:
            msg = (
                "Number of imputation args missmatch: "
                "The number of arguments to use for "
                "imputing null values must be the "
                "same length as the number of "
                "null values in the dictionary. "
                "Recycling last argument passed."
            )
            warnings.warn(msg)

        last_arg: str = args[-1]

        for i in range(total_null - num_args):
            args = args + (last_arg,)

    for key, imputer in zip(config_json.keys(), args):
        # if val falsey, impute it
        if not config_json[key]:
            config_json[key] = imputer
    
    return config_json

def try_make_paths(*args):
    """
    Attempt to make directory paths from an 
    arbitary number of paths passed by the user. 
    """
    for arg in args:
        assert isinstance(arg, (str, list)), \
            TypeError("""Must pass paths as strings.""")

    for arg in args:
        if isinstance(arg, list):
            for path in arg:
                silent_mkdir(path)
        else:
            silent_mkdir(arg)

def join_paths(*args) -> str:
    """
    Join path fragments into one long path.
    """
    for path in args:
        assert isinstance(path, str), \
            TypeError("""Must pass paths as strings.""")
    
    return os.path.join(*args)

def get_date_time() -> Tuple[str, str]:
    """
    Return todays date (year-month-day) and
    date time (hourminute-second)
    """
    today: str = f"{datetime.today().strftime('%Y-%m-%d')}"
    today_time: str = f"{datetime.today().strftime('%H%M-%S')}"
    
    return today, today_time

def assert_config_defaults(config_json: Nested_str_dict, 
                           *args):
    """
    Assert user has not left keys (specified in *args)
    as null.

    Parameters
    ----------
    config_json : dict
        A dictionary of configuration settings. 
    *args : str ... 
        Any number of positional arguments to 
        check for truthiness (not null in this case).
    """
    assert all(arg in config_json.keys() for arg in args), \
        KeyError("One or more keys provided are not "
                 "contained in the configuration file.")
    
    def _set_format_message(f):
        msg = (
            "Cannot pass a null value to "
           f"{f} key of the config file."
        )
        return msg

    for arg in args:
        if isinstance(config_json[arg], dict):
            if not all(config_json[arg].values()):
                if arg == "file_paths":
                    pass
                else:
                    raise ValueError(_set_format_message(arg))
        else:
            if not config_json[arg]:
                raise ValueError(_set_format_message(arg))



