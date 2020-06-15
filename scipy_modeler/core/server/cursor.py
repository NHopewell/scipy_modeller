"""
---------------------------------------------------
Author: Nick Hopewell <nicholas.hopewell@gmail.com>

cursor.py (contains : 1 class, 1 function)

Description
-----------
Provides a simple cusor which is used as a python 
context manager to connect and retrieve observations 
from a database.
----------------------------------------------------
"""
import pyodbc as odbc
import scipy_modeler.util._settings as settings  

def _con(dsn: str):
    """
    Initialise a simple connection to the DB
    using a data source name.
    
    Parameters
    ----------.
    dsn : string
        A users odbc data source name.
        
        
    Returns
    -------
    con : pyodbc connection object.

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
        """
        if exc_val:  # same as.. "if exc_val is not None:"
            self.connection.rollback()
        else:
            self.cursor.close()
        # self.connection.commit()