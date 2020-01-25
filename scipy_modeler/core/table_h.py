"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

table_funcs.py (contains 10 functions)

Description
-----------
table_funcs.py contains mostly helper functions
for the SciTable class. A significant amount of the 
code in table.py should be refactored into this
script where appropriate. 


Functions (12)
--------------
[1] get_table_fields
[2] get_datetimes
[3] flatten_2dlist
[4] generate_struc
[5] _flag_switch
[6] _str_strip_all
[7] _str_case_strip
[8] _pandas_from_sql
[9] _gen_fquery
[10] _print_pretty
[11] permutate_data
[12] get_date_time

---------------------------------------------------
"""
from re import IGNORECASE
from datetime import datetime
from collections import OrderedDict
from typing import Union, Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
#from pandas import Series, DataFrame, read_sql #, read_sql_table, read_sql_query
from pandas.api.types import is_string_dtype

from scipy_modeler.core.db import Database, Cursor
from scipy_modeler.util._decorators import assert_types


def get_table_fields(schema: str, table: str, 
                sort: Optional[bool]=False) -> List[str]:
    """
    Return a list of field names contained in a given table.


    Parameters
    ---------
    schema : str
        A schema name.
    table : str
        A table name. 
    sort : bool
        Whether or not to sort the resulting list.

    
    Returns
    -------
    names : list of table field names.
    """
    assert schema in Database.schema.all, \
        ValueError("Schema not found in Database.")

    names = []
    with Cursor() as c:
        # append col names of taable to empty list
        for col in c.columns(table=f"{schema}.{table}"):
            names.append(col.column_name)
    if sorted:
        return sorted(names)
    return names

def get_datetimes(cols: List[str],
                  passthrough: List[str],
                  endswith: Optional[str]='_DT') -> List[str]:
    """
    Extract datetime columns from a list of columns. 

    
    Parameters
    ---------
    cols : list
        List of column names.
    passthrough : list
        List of columns to ignore. 
    endswith : bool, optional
        Optional string suffix to capture a
        naming convention for representing
        datetime columns. Default = '_DT'

    
    Returns
    -------
    dates_cols : list of datetime columns.
    """ 
    # '_DT' is a naming convention for datetime columns
    dates_cols = [
        i for i in cols if ( 
            i.endswith(endswith) ) and ( 
            i not in passthrough )
    ]             
    
    return dates_cols

def _flatten_2dlist(nested_list: List[List[str]]) -> List[str]:
    """
    Flatten a nested list of lists into a 
    single list. (2d array flattened into 1d array).
    """
    flattened = []
    for filters in nested_list:
        for i in filters:
            flattened.append(i)
    # with list comp:
    # [item for sublist in nested_list for item in sublist]
    return flattened 

@assert_types(str, str, list, list, list, list, list)
def generate_struc(schema: str, 
                   table: str, 
                   filter_out: Optional[List[str]] = [], 
                   str_cols: Optional[List[str]] = [], 
                   int_cols: Optional[List[str]] = [], 
                   float_cols: Optional[List[str]] = [], 
                   extra_date_cols: Optional[List[str]] = [],
                   dt_naming_convention: Optional[str] = '_DT'
                ) -> Dict[str, str]:
    """
    Generate a table struc (a dictionary mapping column 
    names to their desired pandas or numpy data types) for 
    a pandas dataframe.


    Parameters
    ----------
    schema : str
        A schema name.
    table : str
        A table name.
    filter_out : list, optional
        A list of table column names to
        filter out. These columns will
        never be read in from the database.
        Default = [] (empty list).
    str_cols : list, optional
        A list of column names from the
        database to be read into
        memory as strings. 
        Default = [] (empty list).
    int_cols : list, optional
        A list of column names from the
        database to be read into
        memory as integers. 
        Default = [] (empty list).
    float_cols : list, optional
        A list of column names from the
        database to be read into
        memory as floats. 
        Default = [] (empty list).
    extra_date_cols : list, optional
        A list of column names that
        represent additional datatime 
        columns not captured by
        the datetime naming convention
        passed. Default = [] (empty list).
    dt_naming_convention : str, optional
        A naming convention suffix 
        used for naming datetime columns.
        Default = '_DT'

    
    Notes
    -----
    1) categorical columns are deduced and thus do not 
    need to be passed. 

    2) Table strucs are used to pre-filter dataframes by 
    never reading in unwated columns from a table in a 
    database and for automatically  setting data types for 
    each column. This produces data quality gains and very 
    significant reduction in the memory useage of dataframes. 
    see :
     scipy_modeler/_presentables/memory_optimization_test1.ipynb
    """
    # get all col names of table
    cols: list = get_table_fields(schema, table, sort=False )
    # extract dt cols using naming convention
    date_cols: list = get_datetimes(
        cols=cols, passthrough=filter_out, 
        endswith = dt_naming_convention
    )
    # combine dt cols with extra dt cols (no name convention)
    date_cols: list = date_cols + extra_date_cols
    # combine non-nominal cols
    non_cats: list = str_cols + int_cols + float_cols + date_cols 
    # deduce nominal cols as those not provided nor filtered out
    cats = [
        i for i in cols if (
            i not in filter_out ) and (
            i not in non_cats
        )
    ]
    # return final list of all desired columns
    final_cols: list = cats + non_cats
    
    # construct 2d array to use as values for
    # the final struct (used for mapping data types)
    dtype_list = [
        ['category']*len(cats),
        ['object']*len(str_cols), 
        ['int64']*len(int_cols), 
        ['float64']*len(float_cols),
        ['datetime64']*len(date_cols)
    ]
    # flatten dtype 2d list into 1d list
    dtypes: list = _flatten_2dlist(dtype_list)
    # zip all columns with their respective dtypes
    final_struc: Dict[str, str] = {
        k:v for k,v in zip(final_cols, dtypes)
    }
    
    return final_struc
    
def _flag_switch(df: pd.DataFrame,  ref: str, 
                 val: Union[str,int,float], check: str,  
                 flag: int, fill: int, case: bool = True, 
                 regex: bool = False) -> pd.Series:
    """
    Returns a new series (flags) based on a value provided 
    and a check condition (i.e.: how the value should be checked 
    for - equality, greater, less-than). Rows in the series that 
    result in true will be imputed with the 'flag' value, else 
    (false), rows will be filled with the 'fill' value. 


    Parameters
    ----------
    df : pandas Dataframe
        A dataframe containing the reference column. 
    ref : str
        The column searched for the value to check for.
    val : str or int or float
        The value to check for in the reference column.
    check : {'equality', 'greater', 'less'}
        How to check for the value used to generate flags.
    flag : int or str
        A meaningful or arbitrary value used to impute rows  
        which meet the check criteria :
        (reference col->check conditon->value).
    fill : int or str
        A meaningful or arbitrary value used to impute rows  
        which do not meet the check criteria :
        (reference col->check->value).
    case : bool, optional
        Whether or not to be sensitive to case when checking
        for the value. Default = True.
    regex : bool, optional
        Whether or not the value to check for is a regular 
        expression. Default = False. 


    Returns
    -------
    flag_series - a Pandas series of flag and
        fill values. 
    """
    if regex:
        # if user pass regex, assert ref col is str
        if not is_string_dtype(df[ref]):
            raise TypeError(
                "Cannot use a regex pattern "
                "to match on a non-string column.")
        if case:
            # case sensitive
            flag_series = np.where(
                df[ref].str.contains(pat=val, regex=True,
                    na=False), 
                flag, fill
            )
        else:
            # ignore case
            flag_series = np.where(
                df[ref].str.contains(pat=val, regex=True,
                    flags=IGNORECASE, na=False), 
                flag, fill
            )
    else:
        # non-regex pattern match dict switch          
        switch = {
            "equality": lambda: np.where( 
                     df[ref] == val, flag, fill),
            "greater": lambda: np.where(
                      df[ref] > val, flag, fill),
            "less": lambda: np.where(
                      df[ref] < val, flag, fill)
        }
        if not case:
            # update to be case insensitive
            switch.update(
                {
                  "equality": lambda: np.where(
                      df[ref].str.strip().str.lower() == val, 
                            flag, fill
                    )
                }
            )
        # switch on check value
        flag_series = switch.get(
            check.lower(), ValueError(
                    "Check must be one of the following: "
                    "'equality', 'greater', 'less'.")
                )() # call lambda
    if ( isinstance(flag, int) and (isinstance(fill, int)) ):
        # explicitly coerce from float to int
        return flag_series.astype('int32')
    return flag_series


def _str_strip_all(df: pd.DataFrame, 
                   case: Optional[str] = None):
    """
    Apply string string to all columns in a pandas dataframe. 
    Optionally, convert the case of string columns. 

    Parameters:
    -----------
    df : pandas dataframe
        A dataframe to apply _str_strip_all
    case : {'lower', 'upper', 'title'}, optional
        Whether or not to convert the case
        of the string columns in the dataframe.
        Default = None.
    """
    if case:
        # switch on case and strip white space
        return {
            'lower':  df.applymap(
                lambda x: x.strip().lower()
                        if type(x) is str else x
                    ),
            'upper':  df.applymap(
                lambda x: x.strip().upper()
                        if type(x) is str else x
                    ),
            'title':  df.applymap(
                lambda x: x.strip().title()
                        if type(x) is str else x)
        }.get(case,
            ValueError(
                "case must be 'lower', 'upper', or 'title'.")
        )
    # no changes to case
    return df.applymap(
        lambda x: x.strip() if type(x) is str else x
        )

def _str_case_strip(df: pd.DataFrame, cols: List[str], 
                    case: Optional[str] = None) -> pd.DataFrame:
    """
    Apply string strip to select columns of a pandas 
    dataframe. Optionally, alter the case of string columns. 


    Parameters
    ----------
    df : pandas dataframe
        A dataframe to alter in place. 
    cols : list
        Select columns of a dataframe to strip.
    case : str, optional
        Whether or not to convert the case
        of the string columns in the dataframe.
        Default = None.


    Returns
    -------
    df : a pandas dataframe.
    """
    assert case in ('lower', 'upper', 'title'), \
        ValueError("'case' must be one of "
                   "'lower', 'upper', or 'title'")
    
    if case.lower() == 'lower':
        df[cols] = df[cols].apply(
            lambda x: x.str.strip().str.lower()
            )
    elif case.lower() == 'upper':
        df[cols] = df[cols].apply(
            lambda x: x.str.strip().str.upper()
            )
    else:
        df[cols] = df[cols].apply(
            lambda x: x.str.strip().str.title()
            )

    return df

def _pandas_from_sql(db_schema: str, 
                     table_name: str, 
                     columns: str,
                     dates: Optional[List[str]] = None, 
                     na_vals: Optional[Union[List[Any],Any]] = None, 
                     label: Optional[str] = None, 
                     date_format: Optional[str] = None, 
                     struc: 
                     
                      = None,  
                     chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Returns a pandas dataframe from an SQL select statement 
    including handling its column data types using a passed 
    struc, parsing any datetime columns which do not
    parse properly during a normal import, imputing missing values
     if required, and renaming the dependant variable.


    Parameters
    ----------
    db_schema : str
        A database schema name.
    table_name : str 
        A table name under the schema provided.
    columns : str
        A concatenated string of column names
        to read in to memory. This is the 
        'scope' varible in table._to_pandas().
    dates : optional, list
        A list of datetime columns in the table.
        Default = None.
    na_vals : list or str, optional
        Values of the table to replace with np.nan.
        Default = None.
    label : str, optional
        The dependant variable (if one exists).
        Deault = None.
    date_format : str, optional
        A string representing the date format.
        Default = None.
    struc : dict, optional
        A table structure of column names and types.
        Default = None.
    

    Returns
    -------
    A pandas dataframe.
    """
    # TODO: block access to information schema! I wish I had sqlalchemy :(
    query = f"SELECT {columns} FROM {db_schema}.{table_name};"
    
    if struc:
        assert isinstance(struc, dict), \
            TypeError("The struc passed must be a dictionary "
                      "of column names and their respective "
                      "datatypes.")
        if not dates:
            # look through struc and extract dates
            dates = []
            for k in struc.keys():
                if struc[k] == 'datetime64':
                    dates.append(k) 
        if date_format:
            # creat a date parse dict for read_sql
            fmt: List[Dict[str, str]] = [
                {'format': date_format}]*len(dates)
            date_parser: Dict[str, dict] = {
                k:v for k,v in zip(dates, fmt)
            }
            dataframe = pd.read_sql(query, parse_dates=date_parser, 
                chunksize=chunksize, con=Database.connect())
        else:
            #arg_dict = [{'errors': 'coerce'}]*len(dates)
            #date_dict = {k:v for k,v in zip(dates, arg_dict)}
            dataframe = pd.read_sql(query, parse_dates=dates, 
                chunksize=chunksize, con=Database.connect())
        
        dataframe.apply( # strip white space
            lambda x: x.strip() if type(x) is str else x
            )

        if label:
            assert label in struc.keys(), \
                KeyError("The label provided is not contained "
                         "in the table struc keys.")
            
            # copy dict key'LABEL' gets struc dependent var value
            struc_copy = struc.copy()
            struc_copy['LABEL'] = struc_copy.pop(label)
            
            dataframe.rename(
                columns={label: 'LABEL'}, inplace=True)
            dataframe = dataframe.astype( # set types
                struc_copy, errors='raise')
        else:
            dataframe = dataframe.astype( # set types
                struc, errors='raise')
    else:
        dataframe = pd.read_sql(query, 
            chunksize=chunksize, con=Database.connect())
        dataframe.applymap(
            lambda x: x.strip() if type(x) is str else x
            )

    if na_vals:
        # impute vals representing nas (eg: 'MI' or 99999)
        dataframe.replace(na_vals, np.nan, inplace=True)

    return dataframe


def _gen_fquery(filters: List[Tuple[str, str]], op: str) -> str:
    """
    Generates a WHERE clause of an SQL query via fstrings.
    Assumes first element of each tuple is a table name
    and second element is a filtering expressing.


    Parameters
    ----------
    filters : A list of tuples.
        Tuples contain table names and filtering expressions.
    op : str.
        An operator ('=' or 'LIKE').


    Returns
    -------
    Formatted string.
    """
    assert(isinstance(filters, list)
           ), TypeError(
               "Must pass a list to generate query string.")
    assert(isinstance(filters[0], tuple)
           ), TypeError(
               "Recieved a list of non-tuple objects. Must "
               "pass a list containing tuples.")

    if len(filters) > 1:
        query_str = [f"{i[0]} {op} {i[1]} AND" for i in filters]
        # join elements of list together into str
        # split and remove last element and re join.
        return " ".join(" ".join(query_str).split()[:-1])

    return f"{filters[0][0]} {op} {filters[0][1]}"


def _print_pretty(data: Union[dict, list, tuple], 
                  size: Optional[int] = 1):
    # TODO: make this work for other data types
    """
    Nicely prints passed data in reader-friendly
    way based on the type of the data passed.


    Parameters
    ----------
    data : type(data) == dict, list, or tuple
        A collection of data to display.
    size : int, optional
        Number of items in the collection passed.
        Default = 1.
    """
    assert(isinstance(data, (dict, list, tuple))
           ), TypeError(
               "Pretty print only supports dictionaries, "
               "lists, and tuples.")

    n_vars = len(data)

    if isinstance(data, (dict, OrderedDict)):
        print(f"Observations: {size}\nVariables: {n_vars}")
        for i, (key, value_list) in enumerate(data.items()):
            print(
                f'# col{i+1}:  <{key}>  {str(value_list)}'
            )
    elif isinstance(data, list):
        raise NotImplementedError
    elif isinstance(data, tuple):
        raise NotImplementedError
    else:
        raise TypeError(
            "Pretty print only supports dictionaries, "
            "lists, and tuples.")
        
def permutate_data(df, n: int) -> pd.DataFrame:
    """
    Returns a random permutation of the data size n.
    Can be used for training when time relationships
    may not be interesting. 
    """
    rows = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[rows].copy()
