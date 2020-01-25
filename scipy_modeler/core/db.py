"""
---------------------------------------------------
Author: Nick Hopewell <nicholas.hopewell@gmail.com>

database.py (contains : 1 function, 3 classes)

Description
-----------
Provides simple interaction with 
a database making calls to extract data
consistent and easily extensible. Only
used for reading data currently. May
extend Cursor class to allow inserting
data if needed. 

Functions (1)
-------------
[1] _con()

Classes (3)
-----------
[1] Cursor
[2] DatabaseSchema
[3] Database (static)
----------------------------------------------------
"""
import pyodbc as odbc
from json import dumps
from typing import Optional, Tuple, List, Dict

import scipy_modeller.scipy_modeler.util._settings as settings

def _con(dsn: str):
    """
    Initialise a simple connection to the DB
    using a data source name.
    
    
    Parameters
    ----------.
    dsn : string
        A users data source name.
        
        
    Returns
    -------
    con : pyodbc connection object.


    Raises
    ------
    ValueError
        - Data source name fails to connect to db.
    """
    try: 
        # pyodbc connect using individual data source name 'dsn' 
        con = odbc.connect(dsn=dsn, autocommit=False)
    except Exception:
        raise ValueError('Data source name (dns) is not valid.')
    return con

class Cursor:
    """
    Provides functionalty for using a cursor for 
    traversing records in a database. 
    
    
    Parameters
    ----------
    nill.
    
    
    Attributes
    ----------
    connection : None 
        Set as pyodbc connection object when __enter__ called.
    cursor : None
        Set as pyodbc connection.cursor object when __enter__ called.
    
        
    Notes
    ------
    This class only contains dunders because
    it should never be used outside of a 
    'with' clause.
    """
    def __init__(self):
        self.connection = None
        self.cursor = None

    def __enter__(self):
        """
        Defines execution entering `with` clause.
        

        Returns
        -------
        self.cursor : pyodbc cursor object
        """
        # initialize a simple connection (not a connection pool)
        self.connection = _con(dsn=settings.db_initalize['dsn'])
        # instantiate a cursor from the connection
        self.cursor = self.connection.cursor()

        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Defines execution exiting `with` clause.
        
        
        Notes
        -----
        If inserting into a db, this method should be extended
        to rollback if errors are thrown.
        """
        self.cursor.close()
    

class DatabaseSchema():
    """
    A container of table schemas under the passed
    database schema. Optionally models the entire 
    structure of all db schemas as nested json.
    Used as a class attribute of Database().

    
    Parameters
    ----------
    schema_name : string
        The default or current database reference schema.
    full_schema : bool, optional.
        whether the user wants to construct each db schema
        and table schema as a dict. Default = False
        
    
    Attributes
    ----------
    _all_tables : list[str]
        List of tables contained in the current schema 
        (defined in schema name).
    _table_count : int
        Count of all tables in current schema.
    
    
    Notes
    -----
    Instantiated ONLY as an attribute of the Database
    class. Not intended to be utilized stand alone.
    """
    def __init__(self, schema_name: str, full_schema: Optional[bool] = False):
        """
        Raises
        ------
        ValueError
            - schema_name not in all db schemas.
        """
        # all schema names of db
        self._all: list = ['AAL_COMMON_SCH', 'AAL_EXTERNAL_SCH', 'AAL_IDM_SCH',
                           'CIT_SCH', 'ETA_SCH', 'FAM_CLASS_SCH', 'SPSS_SCH',
                           'SP_CHINA', 'SP_INDIA', 'TRV_CHINA', 'TRV_INDIA',
                           'WP_INSP']
        # check if passed schema name is in all db schemas
        assert schema_name in self._all, \
            ValueError('schema_name is not contained in database.')
        self._schema_name: str = schema_name
        # create and store dict of all schemas and assocaited tables
        if full_schema:
            self._build_schema_dict()
        else:
            self._schema_dict: dict = {}
            
        self._all_tables = None
        self._table_count = None

    @property
    def all(self) -> List[str]:
        """Return all schema names in db."""
        return self._all

    @all.setter
    def all(self, new: list):
        """Sets all schema names in db."""
        assert isinstance(new, list), \
            TypeError("Must set '.all' with a list.")
        self._all = new
        
    @property
    def schema_name(self) -> str:
        """Return name of schema cursor is pointing at."""
        return self._schema_name
    
    @schema_name.setter
    def schema_name(self, new: str):
        """Sets name of schema cursor is pointing at."""
        assert isinstance(new, str), \
            TypeError("Must pass new schema as a string.")
        self._schema_name = new
        
    @property
    def schema_tree(self) -> Dict[str, List[str]]:
        """
        Return all schemas in db (as keys) and lists all
        table names of each schema (as values) as a dict.
        """
        if self._schema_dict:
            return self._show_schema_dict()
        # if schema_tree is falsey, build tree before returning
        self._build_schema_dict()
        return self._show_schema_dict()

    @schema_tree.setter
    def schema_tree(self, new: Dict[str, str]):
        """Sets all schemas in db."""
        assert isinstance(new, dict), \
            TypeError("Must pass schema_tree as a dict.")
        self.schema_tree = new
        
    @property
    def all_tables(self) -> List[str]:
        """Return all tables under current schema."""
        # if all_tables is falsey (None or empty list)
        if not self._all_tables:
            self._get_all_tables()
        return self._all_tables
    
    @all_tables.setter
    def all_tables(self, new: List[str]): 
        """Sets all tables under current schema."""
        assert isinstance(new, list), \
            TypeError("Must pass all_tables as a list.")
        self._all_tables = new
    
    @property
    def table_count(self) -> int:
        """
        Return count of total number of tables 
        under the current schema. 
        """
        # if count is falsey, get table count
        if not self.table_count:
            self._get_table_count()
        return self._table_count
    
    @table_count.setter
    def table_count(self, new: int):
        """
        Sets count of total number of tables 
        under the current schema. 
        """
        assert isinstance(new, int), \
            TypeError("Must pass table_count as an integer.")
        self._table_count = new

    def add_schema(self, schema: str, position: Optional[int] = None):
        """
        Add or append new schema to all (all schemas property).


        Parameters
        ----------
        schema : string
            name of schema to add or append.
        position : int, optional
            position in _all(all schemas) to add new.
            Default = None.


        Raises
        ------
        TypeError
            - Passed any type except a string.

        """
        assert isinstance(schema, str), \
            TypeError("Must pass new schema as a string.")
        # if position passed, insert at that position, else append
        if position:
            self._all.insert(position, schema)
        else:
            self._all.append(schema)
            
    def _get_all_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Parameters
        ----------
        schema : string, optional
            Schema to reference. Default = None.


        Returns
        -------
        List - all tables in referenced schema.
        """
        # if not none or not empty list, return
        if self._all_tables:
            return self._all_tables
        else:
            if schema is None:
                # use current (default) schema
                schema=self.schema_name
            # call _table_info(), this will set
            # .all_tables and .table_count properties
            self._table_info(schema) 
            return self._all_tables
    
    def _get_table_count(self, schema: Optional[str] = None) -> int:
        """
        Parameters
        ----------
        schema : string, optional
            Schema to reference. Default = None.
        

        Returns 
        -------
        Int - count of all tables in refernced schema.
        """
        # if count is 0, still returns 0 (not checking falseiness)
        if self._table_count is not None:
            return self.table_count
        # if .table_count is none, .all_tables will also be none
        # as they are both derived from calling '_table_info()'
        # Thus, cannot simply return length of .all_tables
        if schema is None:
            schema=self.schema_name
        self._table_info(schema) 
        return self._table_count
    
    def _table_info(self, schema: Optional[str] = None):
        """
        Sets .all_tables and .table_count properties
        based off of reference schema passed. Uses default
        schema is no schema is referenced by the user. 


        Parameters
        ----------
        schema : string, optional   
            Schema to reference. Default = None

        
        Raises
        ------
        ValueError
            - schema passed which is not contained in
              list of all db schemas.
        """
        #if a schema not passed, use current/default schema
        if schema is None:
            sch = self.schema_name
        else:
            msg = (
                    f"'{schema}' is not a valid schema.\n" 
                    f"All schemas in database: {self.all}."
                )
            assert(schema in self.all), ValueError(msg)
            sch = schema
        tables = []
        with Cursor() as c:
            # for each table, append table name to empty list 'tables'
            for table in c.tables(schema=sch):
                tables.append(table.table_name)
        # set properties, do not return        
        self.all_tables = tables
        self.table_count = len(tables)

    def _build_schema_dict(self):
        """
        Creates a dictionary of schema names as
        values and a list of table names under each
        schema as keys.

        Notes
        -----
        This is not evaluated by default when instantiating
        a DatabaseSchema object. 
        """
        schema_table_dict = {}
        # for each schema in all schemas, add schema as key
        # to dict, with empty lists as values for each
        for schema in self.all:
            schema_table_dict[schema] = []
            with Cursor() as c:
                # for each table in schema, append table name
                # to list at that key
                for table in c.tables(schema=schema):
                    schema_table_dict[schema].append(table.table_name)

        self._schema_dict = schema_table_dict

    def _show_schema_dict(self):
        """
        Prints schema tree in clean format.


        Raises
        ------
        AttributeError
            - Called when instance of DatabaseSchema
              has no schema_dict.
        """
        if self._schema_dict:
            print(dumps(self._schema_dict, indent=3,
                             sort_keys=False))
        else:
            msg = (
                "DatabaseSchema has no schema tree. A scheme tree "
                "(containing all DB schemas and associated tables) "
                "is only built when instantiating with full_schema=True."
            )
            raise AttributeError(msg)

class Database:
    """
    A static database class.

    
    Attributes
    ----------
    name : string
        Name of db.
    dsn : string
        Data Source Name from settings.
    current_schema : string
        Schema cursor will use by default use for iteration.
    default_table : string
        Table cursor will use ny default for iteration.
    Schema: DatabaseSchema
        The structure of the db schemas.

        
    Notes
    ------
    Static class -
        no constructor, never instantiated, used as a 
        namespace primarily.

    """
    name: str = settings.db_initalize["database"]
    dsn: str = settings.db_initalize["dsn"]
    current_schema: str = settings.db_initalize["default_schema"]
    default_table: str = settings.db_initalize["default_table"]
    # instantiate class defined above
    schema = DatabaseSchema(schema_name=current_schema) 
    
    @staticmethod
    def connect():
        """Manual conenct to db."""
        return _con(dsn=Database.dsn)

    @classmethod
    def get_name(cls) -> str:
        """Return name of database."""
        return cls.name

    @classmethod
    def set_name(cls, name: str):
        """Set name of database."""
        cls.name = name

    @classmethod
    def get_current_schema(cls) -> str:
        """Return current schema name."""
        return cls.current_schema

    @classmethod
    def set_current_schema(cls, schema: str):
        """Set current schema."""
        # checks if schema in schemas.all
        assert(schema in cls.schema.all
               ), ValueError(f"'{schema}' is not a valid schema.\n"
                             f"Schemas in {cls.get_name()}: {cls.schema.all}")

        if schema == cls.current_schema:
            pass 
        else:
            print(f"Changing schema reference...")
            # set current schema and default schema name
            cls.current_schema = schema
            cls.schema.schema_name = schema
            print(f"Using schema: {cls.schema.schema_name}.")
            cls.schema._table_info()

    @classmethod
    def get_default_table(cls) -> str:
        """Returns default table."""
        return cls.default_table

    @classmethod
    def set_default_table(cls, table: str):
        """Setter for default table."""
        assert(table in cls.schema.all_tables
               ), ValueError(f"'{table}' is not a valid table name.\n"
                              "See '.all_tables()'.")
        cls.default_table = table

