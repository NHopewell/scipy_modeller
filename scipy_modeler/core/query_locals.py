"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

query_locals.py (contains : ___ functions)

Description
-----------
query_locals.py provides functions to insert into
query strings used to subset a Pandas dataframe 
(pandas.query()). These vectorized functions  
return boolean indicators for each row in a
PD dataframe, or contained to a specifiec 
column or columns, which meet some logical
expression (any expression which can evaluate to
true or false). Functions like these are included
in all SQL flavors to improve ease of querying.

Functions (__)
-------------


---------------------------------------------------
"""
from typing import Optional

import numpy as np

def NULL(col: str) -> bool: 
    """Return true if missing for each
        row in column of data frame"""
    return col.isnull()

def date_months_difference(date_col: str, 
                date_col2: str, value: int, 
                check: Optional[str]='less_than') -> bool:
    """
    Check if the difference between two date columns
    is greater than, less than, or equal to some
    number of months. 


    Parameters
    ----------
    date_col : string
        First datetime column to check.
    date_col2 : string
        Second datetime column to check.
    value : int
        Number of months difference between two
        columns to check for. 
    check : {'less_than','greater_than','equals'}, optional
        The logical condition to check the value for
        between  both columns. Default = 'less_than'.
    """
    assert check in (
        'less_than', 'greater-than', 'equals'), \
        ValueError(
            """'check' must be one of 'less_than' """
            """or 'greater-than'.""")
    
    if check == 'less_then':
        return ( 
        # take time diff between cols, divide into
        # months, check return value against condition
            (date_col - date_col2) / np.timedelta64(
                1, 'M') < value )
    elif check == 'greater-than':
        return ( 
            (date_col - date_col2) / np.timedelta64(
                1, 'M') > value )
    return (
        (date_col - date_col2) / np.timedelta64(
            1, 'M') == value )