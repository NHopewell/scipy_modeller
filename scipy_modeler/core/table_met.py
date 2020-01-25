"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

table.py (contains : 1 class, 2 functions)

table.py provides a model of a generic data table.
The contained classes are a part of the IMP pipeline
developed by the author of the current script.

*********************
**** Classes (1) ****
*********************
[1] Table : Provides model and properties of a data
    table.

    Properties
    ----------
    * Instance Properties:
        _database_schema (schema of table),
        _name (name of table),
        _nrow (number of rows),
        _ncol (number of columns),
        _shape (rows x columns),
        _col_names (column names).

    Methods
    -------
    * Dunders (magic methods)
        __str__, __repr__, __len__.

    * Instance Methods:
        build_data (higher-order) :
            |-> switch :

***********************
**** Functions (2) ****
***********************
[1] _gen_fquery : Helper function that generates and
    returns an SQL WHERE clause as a string based on
    whether and how many filters are passed. A quick
    and dirty alternative to using SQLAlchemy.

[2] _print_pretty : A function for cleanly printing
    data passed from other functions (such as
    functions which return a sample of data from a
    database).
---------------------------------------------------
"""
import os
import csv
from typing import Union, Optional
from sys import version
from collections import OrderedDict
import warnings

from scipy_modeler.util._decorators import send_logs
from scipy_modeler.util._settings import _cache
from scipy_modeler.core.db import Database, Cursor
from scipy_modeler.core.table_funcs import _print_pretty, _pandas_from_sql

_all_types = ['pandas', 'feather','csv', 'json', 'txt']

if version >= "3":
    long = int
    basestring = unicode = str
    
_pandas_types = [
    type(None),
    "bool",
    "int64",
    "float64",
    "str",
    "category",
    "object",
    "unicode",
    "long",
    "bytearray",
    "datetime",
    "decimal.Decimal",
    "datetime.date",
    "datetime.datetime",
    "datetime64",
    "datetime.time"
]

class TableSchema():
    """

    
    Parameters
    ----------

    
    Attributes
    ----------

        
    Notes
    ------
    If providing an initial data structure for a table taken
    directly from the database (before processing it), it is
    important to know that data quality errors may prevent
    Pandas from setting or coercing unclean columns to the
    datatypes specified. If the data taken from the db is
    not clean and these dtype errors occur when trying to
    build_data to pandas, simply do not provide and initial
    structure, and instead update the structure after
    cleaning

    """
    def __init__(self, structure: Optional[dict] = None):

        # TODO: handle how this will work with an infered schema
        if structure:
            assert isinstance(structure, dict), \
                TypeError("Must pass a dictionary of field names and "
                          "their associated data types to 'structure'.")
            assert all(isinstance(i, str) for i in structure.values()), \
                TypeError("Structure must contain strings for its values.")

            assert all(i in _pandas_types for i in structure.values()), \
                ValueError("Improper type passed for a Pandas dataframe.")
        else:
            structure = {}

        self.structure: dict = structure

    @classmethod
    def from_json(cls, json: dict):
        # return TableSchema(json, type_)
        return cls(json)

    def update_structure(self, fields: dict):
        """
        """
        assert isinstance(fields, dict), \
            TypeError("Must pass a dictionary of key-value pairs to update "
                      "TableSchema structure, with keys being column names "
                      "and values being data types.")
        assert all(i in _pandas_types for i in fields.values()), \
                ValueError("Improper type passed for a Pandas dataframe.")
        self.structure.update(fields)

    def alter_field_type(self, field_name: str, dtype: str):
        """
        """
        # pass

        try:
            self.structure[field_name]
        except KeyError:
            raise ValueError(f"{field_name} is not contained in the current "
                             "table structure.")
        assert dtype in _pandas_types, \
            ValueError("Improper type passed for a Pandas dataframe.")

        self.structure[field_name] = dtype

    def _swap_schema(self):
        """
        If and when AAL decides to adopt different frameworks
        this method can be used to swap between schemas
        using dictionaries where the keys are the data types of
        the current framework and the values are the equivalent
        values of the new framework. In other words, the current
        framework is treated as the origin, and the desired
        framework as the destination. These key-value
        pairs will then be used to update the values of the current
        table_struc (a seperate dict with keys being column names
        and values being data types). This new dict will be the
        table_struc.
        """
        pass

########################## > Super < ##########################  
            
class Table:
    """
    The Table super class provides a direct interface
    to the database (modelled by the Database class) via the Cursor class, 
    asks for different information ABOUT tables, and points a cursor
    to the schema and table of interest (automatically changing the 
    current reference behind the scenes). This also helps me generate 
    table struc dictionaries to use when reading into memory as a 
    pd.DataFrame.
    
    This is done without storing any actual table data. Instead, 
    it yields basic meta information and provides utilities to build 
    the actual data into different formats for child classes to utilize.
    The first child (AAL_Table) does contain real data and overloads 
    Table methods to break the direct interaction to the database, offering
    instead an encapsulated set of methods which interact with the API 
    of whichever framework AAL currently uses (whether that be Pandas, 
    Dask, DataTable etc.) Since AAL currently uses Pandas (but my want to 
    use more performant options later), AAL_Table interfaces with the 
    Pandas API.This can easily be made to allow for interaction with 
    multiple APIs by simply branching methods based on current framework. 
    
    This design emphasizes extensibility in terms of being
    compatible with new data science frameworks, libraries, and other 
    filetypes. While still allowing direct interaction with the DB without
    storing intermediate data structures where and when such utility
    is desired. 

    
    Parameters
    ----------

    
    Attributes
    ----------

        
    Notes
    ------


    """
    def __init__(self, schema=None, name: Optional[str] = None, 
                 struc: Optional[dict] = None, use_db_defaults: bool = False, 
                 nameless: bool = False):
        """
        Meta_Table constructor, uses 'AAL_IDM_SCH' as default
        schema if use_db_defaults=True.
        """
        self._database_schema: str = schema
        self._schema = TableSchema(structure=struc)
        self._name: str = name
        self._nrow: Optional[int] = None
        self._ncol: Optional[int] = None
        self._shape: Optional[tuple] = None
        self._col_names: Optional[list] = None
        
        if nameless:
            return
        else:
            if any((schema, name)) and use_db_defaults:
                raise ValueError(
                    "Must not provide schema name and table name when "
                    "using database default values. Set use_db_defaults=False.")
    
            if use_db_defaults:
                self._database_schema = Database.current_schema
                self._name = Database.default_table
            elif not all((schema, name)):
                raise ValueError("Must provide schema and table name.")
            else:
                assert(schema in Database.schema.all), \
                    ValueError(f"{schema} is not a valid schema.\n "
                               "Schemas in {Database.name}: "
                               "{Database.schema.all}")
                self._database_schema = schema
                Database.set_current_schema(schema)
                
                assert(name in Database.schema.all_tables), \
                       ValueError(f"{name} is not a valid table name.")
                self._name = name
    
            if schema is None:
                self._database_schema = Database.current_schema
            else:
                assert(schema in Database.schema.all
                       ), ValueError(
                    f"{schema} is not a valid schema.\n "
                    "Schemas in {Database.name}: {Database.schema.all}")
                self._database_schema = schema
    
            if name is None:
                self._name = Database.default_table
            else:
                assert(name in Database.schema.all_tables), \
                    ValueError(f"{name} is not a valid table name.")
                self._name = name

    def __str__(self):
        print(f"Table: {self.name} from schema: {self._database_schema}.")

    @property
    def database_schema(self):
        return self._database_schema

    @database_schema.setter
    def database_schema(self, new: str):
        self._database_schema = new

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, new: str):
        self._schema = new

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new: str):
        self._name = new

    @property
    def nrow(self):
        if self._nrow:
            return self._nrow

        with Cursor() as c:
            c.execute(f"SELECT COUNT(*) FROM {self.database_schema}.{self.name}")
            nrows = c.fetchall()
            self._nrow = nrows[0][0]
        return self._nrow

    @nrow.setter
    def nrow(self, new_num: int):
        self._nrow = new_num

    @property
    def ncol(self):
        if self._ncol:
            return self._ncol
        
        ncols = 0
        with Cursor() as c:
            for i in c.columns(table=self._name):
                ncols += 1
            self._ncol = ncols
        return self._ncol

    @ncol.setter
    def ncol(self, new_num):
        self._ncol = new_num

    @property
    def shape(self):
        if self._shape:
            return self._shape
        self._shape = (self.nrow, self.ncol)
        return self._shape

    @shape.setter
    def shape(self, new_tuple):
        self._shape = new_tuple

    @property
    def col_names(self):
        if not self._col_names:
            names = []
            with Cursor() as c:
                for row in c.columns(
                        table=f"{self.database_schema}.{self.name}"):
                    #print(row.column_name)
                    names.append(row.column_name)
            self._col_names = names
        return self._col_names

    @col_names.setter
    def col_names(self, new_list):
        self._col_names = new_list

    def load_table_sample(self, size=None, pretty_print=False):
       # TODO: make this look like %>% glimpse()
        """
        Return a sample of n(size) rows for each column of table
        (including header) in dict format. By default, does not
        print sample to user. If pretty_print=True, returns
        dictionary sample and prints sample in a viewer-friendly
        manner.

        Parameters
        ----------
        size : int, optional.
            A number (n) representing the size of the sample to return
            (number of rows).
        pretty_print : bool, default False.
            Whether or not to print nicely.

        Returns
        -------
        OrderedDict object containing sampled rows.
        """
        print(f"Loading sample from {Database.get_name()}."
             "{self.database_schema}.{self.name}...\n")

        sample_dict = OrderedDict()
        
        with Cursor() as c:
            c.execute(f"SELECT * FROM {self.database_schema}.{self.name};")

            if size:
                res = c.fetchmany(size)
                for i in res[0].cursor_description:
                    sample_dict[i[0]] = []
                for tuple_ in res:
                    for i, val in enumerate(tuple_):
                        sample_dict[list(sample_dict)[i]].append(val)
            else:
                res = c.fetchone()
                for tuple_, desc in zip(res, res.cursor_description):
                    sample_dict[desc[0]] = tuple_

        if pretty_print:
            _print_pretty(sample_dict, size=size)
        else:
            print(sample_dict)
            
    def build_data(self, to, dates=None, na_vals=None,
                   label=None, parser=None, infer_schema=False, 
                   chunksize=None):
        """
        Dict switch for how to output data. Must be one
        of '_all_types' defined in the global scope.
        Depending on user input, either builds a dataframe
        of the desired type or writes the data associated with
        the current table into a file.
    
        Parameters
        ----------
        df_type : string {'Pandas', 'PySpark', 'Feather', 'Datatable',
                   'csv', 'json', 'txt'}, default 'Pandas'.
            A string representing the desired type of the returned
            dataframe.
    
        Returns
        -------
        A dataframe of the table in the type specified by the user.
        Can also specify a file type to write to instead of a
        dataframe type.
        """
        assert to.lower() in _all_types, \
            ValueError("Can only output data as 'pandas', 'csv', \
                       'json', 'txt', or 'feather.'")
    
        return {'pandas': lambda: self._to_pandas(dates=dates,
                                                  na_vals=na_vals,
                                                  label=label,
                                                  parser=parser,
                                                  infer_schema=infer_schema,
                                                  chunksize=chunksize),
                'feather': lambda: self._to_feather(),
                'csv': lambda: self._to_csv(),
                'json': lambda: self._to_json(),
                'txt': lambda: self._to_txt()
                }.get(to.lower(),\
                      lambda: ValueError(f"'{to}' not supported.\n \t The following types are supported: {', '.join(_all_types)}."))()
        

    @send_logs
    def _to_pandas(self, dates=None, na_vals=None, label=None, parser=None,
                   infer_schema=True, chunksize=None):   # , regex=False, **kwargs):
        """

        Parameters
        ----------
        Dates : List.
            List of column names to parse to dates.

        Returns
        -------
        Pandas dataframe.

        """
        if not infer_schema:
            assert bool(self._schema.structure), \
                AssertionError(
                    "The current Table object has no schema "
                    "structure. Must either infer the schema or "
                    "provide one.")
                
            scope = ", ".join(
                    [i for i in self._schema.structure.keys()]
                )

            return _pandas_from_sql(self.database_schema, self.name,
                                    scope, dates=dates, na_vals=na_vals,
                                    label=label, parser=parser, 
                                    struc=self._schema.structure, 
                                    chunksize=chunksize)
        else:
            if bool(self._schema.structure):
                msg = (
                    "Infering Schema with Structure Warning: "
                    "Atteming to infer the table schema while a "
                    "schema structure has been passed by the user. "
                    "The passed schema will not be used. To use the "
                    "provided schema, set 'infer_schema' to False."
                )
                warnings.warn(msg)
                
            scope = "*"
            return _pandas_from_sql(self.database_schema, self.name,
                                    scope, na_vals=na_vals,
                                    label=label, chunksize=chunksize)

    @send_logs
    def _to_feather(self):
        """
        """
        # TODO
        raise NotImplementedError

    @send_logs
    def _to_csv(self):
        """
        """
        # TODO fix this
        if os.path.isfile(_data):
            try: 
                os.remove(_cache)
            except FileNotFoundError:
                pass
        with Cursor() as c:
            rows = c.execute(f"SELECT * FROM {self.database_schema}.{self.name};")
        with open (_data, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([i[0] for i in c.description])
            for i in rows:
                writer.writerow(i)         

    @send_logs
    def _to_json(self):
        """
        """
        # TODO        
        raise NotImplementedError

    @send_logs
    def _to_txt(self):
        """
        """
        # TODO        
        raise NotImplementedError

