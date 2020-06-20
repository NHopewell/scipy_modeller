"""
Table struc unit tests. 

To test - get_table_fields(), get_datetimes(), flatten_list(), 
    generate_struc() from core/structured/table_h.py

Notes
-----
use get_table_fields() from the same file to help test.

The name of the test should tell me exactly what its doing so
if I run pytest from the terminal and it fails, I know
exactly what the test failed at doing. It is fine if the names
of these tests are very very long.
"""
import sys, os
import pytest

my_path = 'C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller'

if my_path not in sys.path:
    sys.path.append(my_path)

from scipy_modeler.core.structured import table_h
from unittest.mock import patch

# get_table_field() tests:
def test_len_get_table_fields():
    # test number of fields returned is correct
    ### Already tests that the exact fields returned are correct
    #raise NotImplementedError
    assert False

def test_get_table_fields_fails_with_schema_not_in_db():
    # Setup
    schema = 'a'
    table='SP_INDIA_HIST'
    desired_error = AssertionError
    desired_message = "Schema not found in Database."
    
    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert table_h.get_table_fields(schema = schema, table = table)
    assert str(e.value) == desired_message

def test_get_table_fields_sorted():
    # test fields are really sorted if sort=true
    ### See test_get_table_fields_sorted below
    raise NotImplementedError


# get_datetimes() tests:
def test_get_datetimes_returns_only_columns_with_endswith():
    # check you onl get columns that end with whatever is passed to 'ends_with'
    ### See test_get_datetimes_endswith below
    raise NotImplementedError

def test_get_datetimes_returns_all_dates():
    # test that all date columns (those who end in _dt - all should be grabbed) are returned
    ### See test_get_datetimes below
    raise NotImplementedError


# flatten_list() tests:
def test_flatten_list_returns_1d_list():
    # shouldnt be iterable and of type list
    ### See test_flatten_2dlist below
    raise NotImplementedError


# generate_struc() tests:
def test_filter_out_not_in_struc():
    ### See test_generate_struc_filter_out below
    raise NotImplementedError

def test_number_of_nominals_deduced():
    # test that the number of nominal values has been deduced properly (did it put the correct number as categorical?)
    ### Any generate struc test below will test this
    raise NotImplementedError

def test_extra_date_cols_properly_set_to_datetimes():
    # check dict that is returned - do the extra datecols have 'datetime' as their value? 
    ### See test_generate_struc_date_columns below
    raise NotImplementedError

def test_num_datetimes_in_struc():
    # check total number of datetimes (including _dt and extra date cols) is correct
    ### See test_generate_struc_date_columns below
    raise NotImplementedError

def test_num_ints_in_struc():
    # check num ints in returned dict is right
    ### See test_generate_struc_int_columns below
    raise NotImplementedError

def test_num_strings_in_struc():
    # check num string in returned dict is right
    ### See test_generate_struc_string_columns below
    raise NotImplementedError

def test_num_floats_in_struc():
    # check num floats in returned dict is right
    ### See test_generate_float_int_columns below
    raise NotImplementedError

def test_len_table_struc_keys_equals_total_without_filter_out():
    # make sure grand total of fields (keys of dict returned) = that passed + deduced - filtered_out
    ### test_generate_struc_filter_out tests that total = that passed - filtered_out
    ### Any other deduce test will test that total = that passed + deduced
    ### Together these tests cover this test
    raise NotImplementedError   

# Jason's Tests
# get_table_field() tests:
# TODO: check
def test_get_table_fields(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    desired = ['UCI',
        'APPLICATION_TYPE',
        'APPLICATION_RECEIVED_DT',
        'FINAL_DECISION',
        'FINAL_DECISION_DT',
        'APPLICATION_NO',
        'APPLICATION_STATUS',
        'SOURCE',
        'PERMIT_START_DT',
        'PERMIT_EXPIRY_DT',
        'VALID_TO_ON_COUNTERFOIL',
        'VALID_FROM_ON_COUNTERFOIL',
        'SUB_CATEGORY']
    
    # Exercise
    actual = table_h.get_table_fields(schema = schema, table = table)

    # Verify
    assert actual == desired

# TODO: check
def test_get_table_fields_sorted(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    sort = True
    desired = ['APPLICATION_NO',
        'APPLICATION_RECEIVED_DT',
        'APPLICATION_STATUS',
        'APPLICATION_TYPE',
        'FINAL_DECISION',
        'FINAL_DECISION_DT',
        'PERMIT_EXPIRY_DT',
        'PERMIT_START_DT',
        'SOURCE',
        'SUB_CATEGORY',
        'UCI',
        'VALID_FROM_ON_COUNTERFOIL',
        'VALID_TO_ON_COUNTERFOIL']
    
    # Exercise
    actual = table_h.get_table_fields(schema = schema, table = table, sort = sort)

    # Verify
    assert actual == desired

# get_datetimes() tests:
# TODO: check
def test_get_datetimes(): 
    # Setup
    cols = table_h.get_table_fields(schema = 'SPSS_SCH', table = 'SP_INDIA_HIST')
    passthrough = []
    desired = ['APPLICATION_RECEIVED_DT',
        'FINAL_DECISION_DT',
        'PERMIT_START_DT',
        'PERMIT_EXPIRY_DT']
    
    # Exercise
    actual = table_h.get_datetimes(cols = cols, passthrough = passthrough)

    # Verify
    assert actual == desired

# TODO: check
def test_get_datetimes_passthrough(): 
    # Setup
    cols = table_h.get_table_fields(schema = 'SPSS_SCH', table = 'SP_INDIA_HIST')
    passthrough = ['FINAL_DECISION_DT','PERMIT_START_DT']
    desired = ['APPLICATION_RECEIVED_DT',
        'PERMIT_EXPIRY_DT']
    
    # Exercise
    actual = table_h.get_datetimes(cols = cols, passthrough = passthrough)

    # Verify
    assert actual == desired

# TODO: check
def test_get_datetimes_endswith(): 
    # Setup
    cols = table_h.get_table_fields(schema = 'SPSS_SCH', table = 'SP_INDIA_HIST')
    passthrough = []
    endswith = "COUNTERFOIL"
    desired = ['VALID_TO_ON_COUNTERFOIL',
        'VALID_FROM_ON_COUNTERFOIL']
    
    # Exercise
    actual = table_h.get_datetimes(cols = cols, passthrough = passthrough, endswith = endswith)

    # Verify
    assert actual == desired

# flatten_list() tests:
# TODO: check
def test_flatten_2dlist(): 
    # Setup
    nested_list = [["a","b","c"],[],["c"]]
    desired = ["a","b","c","c"]
    
    # Exercise
    actual = table_h._flatten_2dlist(nested_list = nested_list)

    # Verify
    assert actual == desired

# TODO: check
def test_flatten_2dlist_empty(): 
    # Setup
    nested_list = [[],[]]
    desired = []
    
    # Exercise
    actual = table_h._flatten_2dlist(nested_list = nested_list)

    # Verify
    assert actual == desired

# generate_struc() tests:
# TODO: check
def test_generate_struc(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    desired = {'UCI': 'category',
        'APPLICATION_TYPE': 'category',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_filter_out(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    filter_out=['UCI', 'APPLICATION_TYPE']
    desired = {'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, filter_out = filter_out)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_string_columns(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    str_cols=['UCI', 'APPLICATION_TYPE']
    desired = {'UCI': 'object',
        'APPLICATION_TYPE': 'object',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, str_cols = str_cols)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_int_columns(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    int_cols=['UCI', 'APPLICATION_TYPE']
    desired = {'UCI': 'int64',
        'APPLICATION_TYPE': 'int64',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, int_cols = int_cols)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_float_columns(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    float_cols=['UCI', 'APPLICATION_TYPE']
    desired = {'UCI': 'float64',
        'APPLICATION_TYPE': 'float64',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, float_cols = float_cols)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_date_columns(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    extra_date_cols=['UCI', 'APPLICATION_TYPE']
    desired = {'UCI': 'datetime64',
        'APPLICATION_TYPE': 'datetime64',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'category',
        'VALID_FROM_ON_COUNTERFOIL': 'category',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'datetime64',
        'FINAL_DECISION_DT': 'datetime64',
        'PERMIT_START_DT': 'datetime64',
        'PERMIT_EXPIRY_DT': 'datetime64'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, extra_date_cols = extra_date_cols)

    # Verify
    assert actual == desired

# TODO: check
def test_generate_struc_date_naming_convention(): 
    # Setup
    schema = 'SPSS_SCH'
    table='SP_INDIA_HIST'
    dt_naming_convention="COUNTERFOIL"
    desired = {'UCI': 'category',
        'APPLICATION_TYPE': 'category',
        'FINAL_DECISION': 'category',
        'APPLICATION_NO': 'category',
        'APPLICATION_STATUS': 'category',
        'SOURCE': 'category',
        'VALID_TO_ON_COUNTERFOIL': 'datetime64',
        'VALID_FROM_ON_COUNTERFOIL': 'datetime64',
        'SUB_CATEGORY': 'category',
        'APPLICATION_RECEIVED_DT': 'category',
        'FINAL_DECISION_DT': 'category',
        'PERMIT_START_DT': 'category',
        'PERMIT_EXPIRY_DT': 'category'}

    # Exercise
    actual = table_h.generate_struc(schema = schema, table = table, dt_naming_convention = dt_naming_convention)

    # Verify
    assert actual == desired