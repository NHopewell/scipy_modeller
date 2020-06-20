"""
SQL unit tests. These are covered in their own file so as not
to slow down any of the other tests
"""
import sys, os
import pytest
import pandas as pd

my_path = 'C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller'
sp_india_hist_path = my_path + "\\scipy_modeler\\tests\\test_table_h\\SP_INDIA_HIST.csv"
sp_india_hist_columns = ["UCI", "APPLICATION_TYPE", "APPLICATION_RECEIVED_DT", "FINAL_DECISION", "FINAL_DECISION_DT", "APPLICATION_NO", "APPLICATION_STATUS", "SOURCE", "PERMIT_START_DT", "PERMIT_EXPIRY_DT","VALID_TO_ON_COUNTERFOIL","VALID_FROM_ON_COUNTERFOIL","SUB_CATEGORY"]
sp_india_hist = pd.read_csv(sp_india_hist_path, names=sp_india_hist_columns)

if my_path not in sys.path:
    sys.path.append(my_path)

from scipy_modeler.core.structured import table_h
from unittest.mock import patch
from scipy_modeler._table_strucs.sp_india_strucs import sp_india_main_struc
from scipy_modeler.tests import test_h

# Jason's Tests
# TODO: check
def test_pandas_from_sql(): 
    # Setup
    schema='SPSS_SCH'
    name='SP_INDIA_HIST'
    scope = ", ".join([i for i in sp_india_hist_columns])
    desired = sp_india_hist

    # Exercise
    actual = table_h._pandas_from_sql(db_schema = schema, table_name = name, columns = scope)

    # Verify
    assert test_h.dataframes_are_equal(desired, actual)