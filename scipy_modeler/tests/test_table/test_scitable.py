"""
Table unit tests. 
"""
import sys, os
import pytest
import pandas as pd

my_path = 'C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller'
sp_india_hist_path = my_path + "\\scipy_modeler\\tests\\test_case_data\\SP_INDIA_HIST.csv"
sp_india_hist_columns = ["UCI", "APPLICATION_TYPE", "APPLICATION_RECEIVED_DT", "FINAL_DECISION", "FINAL_DECISION_DT", "APPLICATION_NO", "APPLICATION_STATUS", "SOURCE", "PERMIT_START_DT", "PERMIT_EXPIRY_DT","VALID_TO_ON_COUNTERFOIL","VALID_FROM_ON_COUNTERFOIL","SUB_CATEGORY"]
sp_india_hist = pd.read_csv(sp_india_hist_path, names=sp_india_hist_columns)

if my_path not in sys.path:
    sys.path.append(my_path)

from scipy_modeler.tests import test_h
from scipy_modeler.core.structured import table
from unittest.mock import patch
from scipy_modeler.tests.mocks import mocked_table_h

# Jason's Tests
# TODO: check
@patch("scipy_modeler.core.structured.table_h._pandas_from_sql", new=mocked_table_h._pandas_from_sql)
def test_scitable(): 
    # Setup
    schema='SPSS_SCH'
    name='SP_INDIA_HIST'

    desired_data = sp_india_hist
    desired_nrow = 383183
    desired_ncol = 13
    desired_shape = (383183, 13)
    desired_col_names = ["UCI", "APPLICATION_TYPE", "APPLICATION_RECEIVED_DT", "FINAL_DECISION", "FINAL_DECISION_DT", "APPLICATION_NO", "APPLICATION_STATUS", "SOURCE", "PERMIT_START_DT", "PERMIT_EXPIRY_DT","VALID_TO_ON_COUNTERFOIL","VALID_FROM_ON_COUNTERFOIL","SUB_CATEGORY"]

    # Exercise
    actual = table.SciTable(schema=schema, name=name)
    actual_data = actual.data
    actual_nrow = actual.nrow
    actual_ncol = actual.ncol
    actual_shape = actual.shape
    actual_col_names = actual.col_names

    # Verify
    assert test_h.dataframes_are_equal(desired_data, actual_data)
    assert actual_nrow == desired_nrow
    assert actual_ncol == desired_ncol
    assert actual_shape == desired_shape
    assert actual_col_names == desired_col_names