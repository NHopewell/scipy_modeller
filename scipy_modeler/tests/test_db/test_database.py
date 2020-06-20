import sys, os
import pytest

my_path = 'C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller'

if my_path not in sys.path:
    sys.path.append(my_path)

from scipy_modeler.core.server import db
from scipy_modeler.core.server.db import Database
from unittest.mock import patch
from scipy_modeler.tests.mocks import mocked_db

def test_database_default_schema():

    desired_default = "SP_INDIA"
    actual_default = Database.get_current_schema()

    assert actual_default == desired_default

def test_database_default_table():

    desired_default = "APPS_VIEW_DEV"
    actual_default = Database.get_default_table()

    assert actual_default == desired_default

# Jason's Tests
# TODO: check
def test_database_default_name():
    # Setup
    desired = "PRD_EDW_AAL_DEV_QRY"
    
    # Exercise
    actual = Database.get_name()

    # Verify
    assert actual == desired

@patch("scipy_modeler.core.server.db.DatabaseSchema", new=mocked_db.MockedDatabaseSchema)
def test_set_current_schema():
    # Setup
    schema = "Mock Schema"
    desired = "Mock Schema"
    # Replaced with mock object
    db.Database.schema = db.DatabaseSchema(None)
    Database.set_current_schema(schema = schema)
    
    # Exercise
    actual = Database.get_current_schema()

    # Verify
    assert actual == desired

@patch("scipy_modeler.core.server.db.DatabaseSchema", new=mocked_db.MockedDatabaseSchema)
def test_set_current_schema_same_schema():
    # Setup
    schema = "Mock Schema"
    desired = "Mock Schema"
    # Replaced with mock object
    db.Database.schema = db.DatabaseSchema(None)
    Database.set_current_schema(schema = schema)
    # Once set to Mock, sets it again to Mock
    Database.set_current_schema(schema = schema)
    
    # Exercise
    actual = Database.get_current_schema()

    # Verify
    assert actual == desired

@patch("scipy_modeler.core.server.db.DatabaseSchema", new=mocked_db.MockedDatabaseSchema)
def test_set_current_schema_invalid_schema():
    # Setup
    schema = 'schema'
    db.Database.name = "name"
    # Replaced with mock object
    db.Database.schema = db.DatabaseSchema(None)
    desired_error = AssertionError
    desired_message = "'schema' is not a valid schema.\nSchemas in name: ['Mock Schema']"
    
    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert db.Database.set_current_schema(schema = schema)
    assert str(e.value) == desired_message

@patch("scipy_modeler.core.server.db.DatabaseSchema", new=mocked_db.MockedDatabaseSchema)
def test_set_default_table():
    # Setup
    table = "Mock Table"
    desired = "Mock Table"
    # Replaced with mock object
    db.Database.schema = db.DatabaseSchema(None)
    Database.set_default_table(table = table)
    
    # Exercise
    actual = Database.get_default_table()

    # Verify
    assert actual == desired

@patch("scipy_modeler.core.server.db.DatabaseSchema", new=mocked_db.MockedDatabaseSchema)
def test_set_default_table_invalid_table():
    # Setup
    table = 'Invalid Table'
    # Replaced with mock object
    db.Database.schema = db.DatabaseSchema(None)
    desired_error = AssertionError
    desired_message = "'Invalid Table' is not a valid table name.\nSee '.all_tables()'."
    
    # Exercise & Verify
    with pytest.raises(desired_error) as e:
        assert db.Database.set_default_table(table = table)
    assert str(e.value) == desired_message