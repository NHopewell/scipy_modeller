import pandas as pd
import pytest

import test_h

# Jason's Tests
# TODO: check
def test_get_row_counts(): 
    # Setup
    df = pd.DataFrame([[1,2],[3,4]])
    desired = {(1,2):1,(3,4):1}

    # Exercise
    actual = test_h.get_row_counts(df = df)

    # Verify
    assert actual == desired

def test_get_row_counts_multiple(): 
    # Setup
    df = pd.DataFrame([[1,2],[3,4],[1,2],[1,4]])
    desired = {(1,2):2,(3,4):1,(1,4):1}

    # Exercise
    actual = test_h.get_row_counts(df = df)

    # Verify
    assert actual == desired

def test_get_row_counts_empty(): 
    # Setup
    df = pd.DataFrame()
    desired = {}

    # Exercise
    actual = test_h.get_row_counts(df = df)

    # Verify
    assert actual == desired

def test_dataframes_have_same_rows(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]])
    df2 = pd.DataFrame([[1,2],[3,4]])
    desired = None

    # Exercise
    actual = test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2)

    # Verify
    assert actual == desired

def test_dataframes_dont_have_same_rows(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]])
    df2 = pd.DataFrame([[1,2],[5,4]])
    desired_error = AssertionError

    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2)
    

def test_dataframes_have_same_rows_diff_cols(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]],columns = ["A","B"])
    df2 = pd.DataFrame([[2,1],[4,3]],columns = ["B","A"])
    desired_error = AssertionError

    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2)

def test_dataframes_have_same_rows_column_order_doesnt_matter(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]],columns = ["A","B"])
    df2 = pd.DataFrame([[2,1],[4,3]],columns = ["B","A"])
    desired = None

    # Exercise
    actual = test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2, check_column_order = False)

    # Verify
    assert actual == desired

def test_dataframes_dont_have_same_rows_column_order_doesnt_matter(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]],columns = ["A","B"])
    df2 = pd.DataFrame([[2,1],[4,5]],columns = ["B","A"])
    desired_error = AssertionError

    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2, check_column_order = False)

def test_dataframes_have_same_rows_diff_cols_column_order_doesnt_matter(): 
    # Setup
    df1 = pd.DataFrame([[1,2],[3,4]],columns = ["A","B"])
    df2 = pd.DataFrame([[2,1],[4,3]],columns = ["C","A"])
    desired_error = AssertionError

    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert test_h.assert_dataframes_have_same_rows(df1 = df1, df2 = df2, check_column_order = False)