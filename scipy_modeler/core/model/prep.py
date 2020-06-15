"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>


---------------------------------------------------
"""
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import (
    train_test_split, StratifiedShuffleSplit)

def read_sample_csv(path: str, sample_proportion: float):
    """
    Return a sample of a dataframe of a csv file.


    Parameters
    ----------
    path : str
        Path to csv file.
    sample_proportion : float
        Percentage of datafile to read into memoery
        as a pandas dataframe.
    """
    return (
        pd.read_csv(path, skiprows=lambda x:
            np.random.rand() > sample_proportion and x > 0
            # x > 0 means always include first row
            )
    )

def random_sample(data, size):
    """ 
    Returns a random sample of n rows from df, 
    without replacement. 
    """
    rows = sorted(
        np.random.permutation(len(data))[:size])
    
    return data.iloc[rows].copy()

def random_train_test_split(df, percent_test):
    """
    Take a random train test split without replacement.


    Notes
    -----
    WARNING This is for prototyping only, 
    will generate a new split eveytime. Not
    to be used for repeated model training 
    and testing. This will not work when 
    building on new model on new data. 
    
    train, test = random_train_test_split(df, 0.2)
    """
    p = np.random.permutation(len(df)) # take a random permutation of the row indecides, of the length of the df - shuffle rows
    size_test = int(len(df) * percent_test) 
    test_data = p[:size_test]
    train_data = p[size_test:]
    
    return ( 
        df.iloc[train_data].copy(), df.iloc[test_data].copy() 
    )

def split_data(data, percent_train, 
                random_state:Optional[int]=None):
    # split into train, validation, test
    train, not_train = train_test_split(data,
                        test_size=1-percent_train, 
                        random_state=random_state
                    )
    validation, test = train_test_split(not_train, 
                        test_size=0.5,  
                        random_state=random_state
                    )
    return train, validation, test

def stratified_shuffle_split(data, strat_col, percent_train, 
                            random_state: Optional[int]=None):
    """
    Must choose nomial variable to shuffle split 
    on with. Often the label is used to strat shuffle 
    split, but not always.
    """
    assert strat_col in data.columns, \
        KeyError(
            "Column passed to use for generating "
            "a stratified sample does not exist "
            "in dataframe."
        )
    
    splitter = StratifiedShuffleSplit(
                    n_splits=1,  
                    test_size=1-percent_train, 
                    random_state=random_state)
    
    for train_idx, not_train_idx in splitter.split(
            data, data[strat_col]):
        strat_train = data.loc[train_idx]
        strat_not_train = data.loc[not_train_idx]
    
    test_splitter = StratifiedShuffleSplit(
                        n_splits=1, 
                        test_size=0.5, 
                        random_state=random_state
        )
    
    for validation_idx, test_idx in test_splitter.split(
            strat_not_train, strat_not_train[strat_col]):
        strat_validation = strat_not_train.loc[validation_idx]
        strat_test = strat_not_train.loc[test_idx]
        
    return strat_train, strat_validation, strat_test


def split_most_recent(data, percent_train, 
                        random_state:Optional[int]=None):

    if random_state:
        np.random.seed(random_state)

    size_train = int(len(data) * percent_train)
    train = data[:size_train]
    not_train = data[size_train:]
    p = np.random.permutation(len(not_train))
    size_validation = int(len(not_train) * 0.5)
    validation = p[:size_validation]
    test = p[size_validation:]

    return ( data.loc[train], 
            data.iloc[validation], data.iloc[test] )

def splits_to_csv(split, path):
    split.to_csv(path, index=False)




