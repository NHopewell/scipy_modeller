"""
---------------------------------------------------
Author: Jason Conte <Jason.Conte@cic.gc.ca>

test_h.py (contains 2 functions)

Description
-----------
test_h.py contains helper functions
for the modeller testing class

Functions (2)
--------------
[1] get_row_counts
[2] dataframes_are_equal
---------------------------------------------------
"""
from typing import Union, Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

def get_row_counts(df) -> Dict[np.array, int]:
    '''
    Generates a row counts for every row in the given Dataframe

    Returns
    -------
    Dictionary mapping rows (in the form of a tuple) to
    the number of instances of that row in the dataframe

    Examples
    --------
    df = 
            A   B
        0   a   b
        1   a   d
        2   a   b
    get_row_counts(df) ->
        {
            (a,b): 2,
            (a,d): 1
        }
    '''
    row_to_row_count = {}
    # Loops through each numpy row in the dataframe
    for row in list(df.values):
        # Converts to tuple so that it can be added to the dictionary
        row = tuple(row)
        # initializes new rows with a count of 1, increments rows already in the dictionary
        if row not in row_to_row_count:
            row_to_row_count[row] = 1
        else:
            row_to_row_count[row] += 1
    return row_to_row_count


def assert_dataframes_have_same_rows(df1, df2, check_column_order = True) -> bool:
    """
    Returns whether the given dataframes are equal
    Note: works even if rows and columns of each dataframe are in different orders

    Examples
    --------
    df1 = 
            A   B
        0   a   b
        1   c   d
    df2 = 
            A   B
        1   c   d
        0   a   b
    dataframes_are_equal(df1, df2) -> True
    """
    # Checks that the dataframes have the same columns
    if not check_column_order and set(df1.columns) != set(df2.columns):
        assert set(df1.columns) == set(df2.columns)
    elif check_column_order and list(df1.columns) != list(df2.columns):
        assert list(df1.columns) == list(df2.columns)
    else:
        # Reorders columns so that both dataframes have columns in the same order
        df2 = df2.reindex(columns = df1.columns)
        assert get_row_counts(df1) == get_row_counts(df2)