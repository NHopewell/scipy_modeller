"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

table.py (contains : 2 classes)  

Description  ( >>> PLEASE READ <<< )
------------------------------------  
table.py provides a robust preprocessing api for
standardizing the maniputation of a Pandas 
dataframe. It is crucially important to standardize
data manipulation tasks across a team of individuals
and test/validate these tasks (for reasons that
I will partially detail below). 

Adding a layer on top of the data manipulation API to 
standardize your teams tasks is a common approach. 
One example of this is Shopify. 
The data engineers at Shopify have added a 
layer ontop of the Spark API to standardize
all of their data manipulation tasks to ensure 
valid, consistent, predictable results. 

Simply because an API itself is well-designed, 
consistent, and well-written (such as Pandas or 
Spark DF), does not mean that the way each of us 
interacts with that API is consistent. Therefore, 
it is crucial to use the abstraction built in the 
SciTable class to ensure quality, consistent results 
the first time when preprocessing many lines of 
business.  

Furthermore, it is crucial for efficiency sake to write
processing methods one time and test them very well, 
rather than rewrite these methods every time a new 
line of buisness comes down the pipeline. Preprocessing
structured (and even unstructured data) reqires similar
steps that need to be repeated time and time again. So
these steps need to be programmed, tested to be valid,
and reused as much as possible.


Classes (2)
-----------
[1] SciTable
[2] EncodedTable
---------------------------------------------------
"""
import os
import warnings
from typing import Optional, Union, List, Dict, Any
from IPython.display import display
# from datetime import date, time, datetime

import numpy as np
# from numpy import int64, float64, datetime64
import pandas as pd
from pandas.api.types import CategoricalDtype, DatetimeTZDtype
from pandas.api.types import is_string_dtype, is_numeric_dtype, \
    is_categorical_dtype

# from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

# import aal_encode as encode
from scipy_modeler.core.server.query_locals import *
from scipy_modeler.util._settings import _cache
from scipy_modeler.util._decorators import send_logs
from scipy_modeler.core.structured.table_met import Table, _all_types
from scipy_modeler.core.structured.table_h import  (
    _str_strip_all, _str_case_strip, _flag_switch,
    _make_categorical_numeric )

########################## > Table Child < ##########################  
class SciTable(Table):
    """
    Parameters
    ----------
    schema : str, optional
        A schema name. Default = None. 
    name : str, optional
        A table name. Default = None.
    struc : Dict[str, str], optional.
        A table struc specifying fields to retain, their
        respective datatypes, and how NA values are
        represented. Default = None.
    use_db_defaults : bool, optional
        Whether or not to read data using the database
        class default schema and table. Default = False.
    na_vals : Any, optional
        How NA values are represented in the database
        being read from. Default = None. 
    label : str, optional
        The dependant (target) variable. Default = None.
    date_parser_format : str, optional
        A string representing the datetime format to
        use when parsing datetime fields. Default = None.
    nameless : bool, optional
        Wether or not to instantiate a nameless table 
        (a table not contained in the database and only used
        temporarily as an intermediate table, for instance,
        before joining to another table). Default = False.
    backup : bool, optional
        Whether or not to make a checkpoint backup of the 
        data on read in before any manipulation takes
        place. Default = False. 

    
    Attributes
    ----------
    _data : pd.DataFrame
        An accessor directly to the Pandas dataframe.
    _checkpoint : pd.DataFrame, optional
        A backup of the data made at any point during
        preprocessing. Called 'revert_to_checkpoint()'
        to replace _data with _checkpoint and clear
        _checkpoint (reversing any changes made to the 
        data after the checkpoint was made). 
    _nrow, _ncol : ints
        Number of rows and columns of _data.
    _shape : Tuple[int]
        Tuple or nrows and ncols of _data.
    _colnames : List[str]
        All column names of _data.

    
    Notes
    -----
    This is a very heavy class which contains the 
    preprocesing API. 
    
    Methods should continue to be refactored and 
    chunked out into table_h.py (helper file) where
    appropriate. 
    
    This class also requires the most comprehensive
    testing and validation.
    """
    def __init__(self, 
                 schema: Optional[str] = None, 
                 name: Optional[str] = None, 
                 struc: Optional[Dict[str,str]] = None, 
                 use_db_defaults: Optional[bool] = False, 
                 na_vals: Optional[Any] = None, 
                 label: Optional[str] = None,
                 date_parser_format: Optional[str] = None, 
                 nameless: Optional[bool] = False,
                 backup: Optional[bool] = False):

        super().__init__(schema, name, struc, use_db_defaults, 
                    nameless)
        
        # merged table, not directly from db
        if nameless:
            self.data = None
        else:
            if struc:
                self._data: pd.DataFrame = self.build_data(
                        to='Pandas', na_vals=na_vals,
                        label=label, parser=date_parser_format,
                        infer_schema=False # will use self.struc
                    )
            else:
                if date_parser_format:
                    msg = (
                        "Providing Date Parser Format Without Specifying "
                        "Date Types Warning: Passed a date formate without "
                        "providing a structure to specify date type columns. "
                        "Will not use date parser format on read-in. Please "
                        "do not specify a value for 'date_parser_format' "
                        "next time unless a structure is provided."
                    )
                    warnings.warn(msg)

                self._data: pd.DataFrame = self.build_data(
                    to='Pandas', na_vals=na_vals,
                    label=label, infer_schema=True
                    )
            
        if backup:
            self._checkpoint: pd.DataFrame =  self.data.copy()
        else:
            self._checkpoint: pd.DataFrame = None

    @property
    def nrow(self):
        return len(self.data.index)            

    @nrow.setter
    def nrow(self, new_num):
        self._nrow = new_num

    @property
    def ncol(self):
        return len(self.data.columns)   
        
    @ncol.setter
    def ncol(self, new_num):
        self._ncol = new_num
    
    @property
    def shape(self):
        return self.data.shape
    
    @shape.setter
    def shape(self, tuple_):
        self._shape = tuple_
        
    @property
    def col_names(self):
        return list(self.data.columns)
    
    @col_names.setter
    def col_names(self, new_list):
        self._col_names = new_list
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data
        
    @property
    def checkpoint(self):
        return self._checkpoint
        
    @checkpoint.setter
    def checkpoint(self, new_checkpoint):
        self._checkpoint = new_checkpoint
        
    def copy(self):
        """return a copy of the table data."""
        return self.data.copy()
        
    def load_table_sample(self, size=None):
        """
        """
        return self.data.head(size)

    def list_by_dtype(self, dtype: str, action: str = 'include'):
        """
        Return a list of columns of a certain data type or all
        columns excluding a certain data type. 


        Parameters
        ----------
        dtype : str
            A data type.
        actions : {include, exclude}, optional.
            Whether to return a list of columns
            including those only of the passed data
            type, or excluding said data type. 
            Default = 'include'
        """
        assert action in ("include", "exclude"), \
            ValueError("action must be 'include' or 'exclude'.")
        if action == "include":
            return tuple(
                self.data.select_dtypes(include=dtype).columns)
        return tuple(
                self.data.select_dtypes(exclude=dtype).columns)
    
    def save_struc(self) -> Dict[str,str]:
        """
        Save the current table struc. 


        Notes
        -----
        Use this as a final step after preprocessing data
        to reuse when reading in processed data
        during a follow-up step in the current session. 
        """
        # TODO: MAKE THIS WRITE OUT TO FILE 

        dtypes = ["category", "object", "int", "float64", "datetime64"]
        struc_order = []
        dtype_dicts = []
        complete_dtype_dict = {}
        struc = {}
        # append cols of each dtype in order,
        
        for i in range(len(dtypes)):
            struc_order.append(self.list_by_dtype(dtypes[i])) 
            dtype_dicts.append(
                dict( # zip into dict with their respective dtypes as str
                    zip( 
                        struc_order[i], [ dtypes[i] * len(struc_order[i]) ] 
                    )
                )
            ) 
            # update final dict which each dtype dict
            complete_dtype_dict.update(dtype_dicts[i])

        for col in list(self.data.columns):
            struc[col] = complete_dtype_dict[col]
        
        return struc
        
    def expanded_view(self, nrow: Optional[int] = 100, 
                       ncol: Optional[int] = 100):
        """
        Display more rows and columns than allowed by default in
        environments such as jupyter notebooks.
    
        Parameters
        ----------
        nrow : int, optional 
            max number of rows to display
            default = 1000.
        ncol : int, optional 
            max number of cols to display.
            default = 1000.
        """
        with pd.option_context("display.max_rows", nrow):
            with pd.option_context("display.max_columns", ncol):
                display(self.data)
    
    def info(self, memory: Optional[str] = 'deep', 
                complete: Optional[bool] = True):
        """
        Display dummary information about the data.
        """
        if complete:
            self.data.info(memory_usage=memory)
        else:
            print(self.data.memory_usage(deep=True))
    
    @send_logs    
    def write_to_cache(self, file_type: Optional[str] = 'csv', 
                        na_rep: Optional[Any] = 'U/I'):
        """
        Write data to a csv or json file.

        Parameters
        ----------
        file_type : {'csv', 'json'}, optional.
            The filetype to write the data to. Default = 'csv'.
        na_rep : Any, optional
            How NA values are represented. Default = 'U/I'
        """
        assert file_type in ('csv', 'json'), ValueError(
            "Must choose to write data to csv or json.")
        # replace existing
        if os.path.isfile(_cache):
            os.remove(_cache)
        if file_type.lower() == 'csv':
            self.data.to_csv(_cache, na_rep=na_rep)
        else:
            self.data.to_json(_cache)
        
    def read_from_cache(self, file_type: str = 'csv'):
        """
        Read data from a csv or json file. 

        Parameters
        ----------
        file_type : {'csv', 'json'}, optional.
            The filetype to write the data to. Default = 'csv'.
        """
        assert file_type in ('csv', 'json'), ValueError(
            "Must choose to write data to csv or json.")
        try:
            if file_type.lower() == 'csv':
                self.data = pd.read_csv(_cache)
            else:
                self.data = pd.read_json(_cache)
        except FileNotFoundError:
            msg = (
                "There is no cached data to read. Check your "
                "cache path to make sure it is current and that "
                "there is data in your cache folder."
            )
            raise FileNotFoundError(msg)
    
    @send_logs
    def make_checkpoint(self):
        """
        Make a checkpoint of the current state of the data.
        May be reverted back to later. 

        Notes
        -----
        If the data is large or memory is pressed, simply
        write the table to a file and use that as a checkpoint.
        """
        self.checkpoint: pd.DataFrame = self.data.copy()
        print("Checkpoint successfully made.")
    
    
    @send_logs
    def revert_to_checkpoint(self, drop_after: bool = False):
        """
        Revert data to a previous checkpoint.


        Parameters
        ----------
        drop_after : bool, optional
            Whether or not to remove the checkpoint
            after reverting to it. Default = False. 
        """
        if not self.checkpoint:
            msg = ("No checkpoint data to revert to. "
                   "You can specify a checkpoint at any point "
                   "by calling '.make_checkpoint()'."
            )
            raise AssertionError(msg)

        self.data = self.checkpoint.copy()
        if drop_after:
            print("NOTICE: deleted checkpoint of data.")
            del self.checkpoint
        print("Reverted data to previous checkpoint.")
            
        
    def downsize_to(self, n: int, checkpoint_before: bool = False):
        """
        Downsize data to include only nrows.


        Parameters
        ----------
        n : int
            The number of rows to downsize to.
        checkpoint_before : bool, optional
            Whether or not to make a checkpoint
            of the current data before downsizing.
            Default = False.
        """
        if checkpoint_before:
            self.make_checkpoint()
            self.data = self.data[:n]
            msg = (
                f"NOTICE: Made a checkpoint before downsizing to {n} "
                "rows. Previous checkpoint will be overwritten. To "
                "revert back afterwards, call '.revert_to_checkpoint()'."
            )
            print(msg)
        else:
            self.data = self.data[:n]
            
    def distinct(self, columns: List[str], 
                checkpoint_before: bool = False, 
                keep: str = "first"):
        """
        Drop duplicate rows in the data (based on a specified
        subset of columns).


        Parameters
        ----------
        columns : list
            A list of columns used to check for duplicate
            rows.
        checkpoint_before : bool, optional
            Whether or not to make a checkpoint
            of the current data before downsizing.
            Default = False.
        """
        assert keep in ("first", "last"), ValueError(
            "Must choose to keep 'first' or 'last' unique record.")

        if checkpoint_before:
            self.make_checkpoint()
            self.data.drop_duplicates(
                subset=columns, keep=keep, inplace=True)
            msg = (
                f"NOTICE: Made a checkpoint before deduplicating data. "
                "Previous checkpoint will be overwritten. "
                "To revert back afterwards, call '.revert_to_checkpoint()'."
            )
            print(msg)
        else:
            self.data.drop_duplicates(
                subset=columns, keep=keep, inplace=True)
     
    def random_permutation(self, size: int, seed: Optional[int] = None):
        """
        Returns a random permutation of the data of nrows = size.
        Use seed for reproducibility of resulting permutation.
        """
        if seed:
            np.random.seed(seed)
        rows = sorted(
            np.random.permutation(len(self.data))[:size]
            )
        return self.data.iloc[rows].copy()
            
    def clean_col_names(self, 
                        to_replace: Optional[str] = '', 
                        replace_with: Optional[str] = '', 
                        case: Optional[str] = None, 
                        strip: Optional[bool] = False):
        """
        Replace characters, alter case, and strip white space
        of all columns of the table. 


        Parameters
        ----------
        to_replace : str, optional. 
            A character or exact patter to replace
            in the column name. Default = ''.
        replace_with : str, optional. 
            A character or patern to use for replacement.
            Default = ''.
        case : {'upper', 'lower', 'title'}, optional. 
            Whether, and how, to convert the case
            of the column names. Default = None.
        strip : {'leading', 'trailing', 'all'}, optional. 
            Whether, and how, to stripe whitespace
            from the column names. Default = None.
        """
        if not case and not strip:
            self.data.columns = self.data.columns.str.replace(
                to_replace, replace_with)
        if case:
            # convert col name cases
            assert case.lower() in ('upper', 'lower', 'title'), \
                ValueError(
                    "case must be one of 'upper', 'lower', or 'title'.")
            if case.lower() == 'upper':
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.upper()
            elif case.lower() == 'lower':
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.lower()   
            else:
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.title()
        if strip:
            # remove white spaces
            assert strip.lower() in ('leading', 'trailing', 'all'), \
                ValueError("strip must be one of 'leading', 'trailing', "
                           "or 'all'.")
            if strip.lower() == 'leading':
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.lstrip()
            elif case.lower() == 'trailing':
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.rstrip()  
            else:
                self.data.columns = self.data.columns.str.replace(
                    to_replace, replace_with).str.strip()
                    
    def rename_all_columns(self, new_names: List[str]):
        """
        Rename all columns of the table. 
        """
        assert len(self.data.columns) == len(new_names), ValueError(
                "Must provide the same number of new "
                "colunmn names as there are present in the table.")

        self.data.rename(columns = new_names, inplace = True)
    
    def rename_columns(self, old_names: List[str], 
                        new_names: List[str]):
        """
        Rename specific columns of the table.
        """
        assert len(old_names) == len(new_names), ValueError(
            "Must provide the same number of new "
            "colunmn names and old column names.")
  
        rename_dict = {
            k:v for k,v in zip( self.data.columns, new_names ) 
            }

        self.data.rename(rename_dict, axis='columns', inplace=True)
        
    def add_prefix(self, 
                   prefix: str, 
                   include_dtype: Optional[str] = None,
                   columns: Optional[Union[str, List[str]]] = 'all',
                   ignore: Optional[Union[str, List]] = None):
        """
        Add a prefix to all or some column names.


        Parameters
        ----------
        prefix : str
            A prefix to add to the begining of all or 
            specified column names.
        include_dtype : {'category', 'numeric', 'datetime', 'object'}, optional
            A datatype. Only include columns
            of this type when adding the specified prefix.
            Default = None.
        columns : str or list
            Column names to add prefix to. Others will be
            ignored. Default = 'all'.  
        ignore : str ot list, optional.  
            Column names to ignore when adding prefixes to
            all other columns. Default = None.
        """
        assert isinstance(columns, (str, list)), TypeError(
            "Must pass a string or list to 'ignore'.")
        if columns != 'all':
            assert all(col in self.data.columns for col in columns), ValueError(
                "One of more columns passed is not contained "
                "in the current table.")
        assert not all( (columns, include_dtype) ), AssertionError(
            "Cannot specify a data type to select "
                           "columns based off of and include specific "
                           "column names. Must specify one or the other")
        new_names = []

        # TODO :  make this work for columns of different data types
        if columns != 'all' and ignore: 
            raise ValueError(
                    "Only specifiy columns to ignore when choosing"
                    " to add a prefix to all columns (columns=all)."
            )
        if columns == 'all':
            # add prefix to all columns
            if ignore:
                if isinstance(ignore, list):
                    new_names = []
                    for col in self.data.columns:
                        if col not in ignore:
                            new_names.append(f"{prefix}{col}")
                        else:
                            new_names.append(col)
                else:
                    for col in self.data.columns:
                        if col != ignore:
                            new_names.append(f"{prefix}{col}")
                        else:
                            new_names.append(col)
            else:
                new_names = [
                        f"{prefix}{col}" for col in self.data.columns ]
        elif include_dtype:
            # add prefix to only cols of that dtype
            if include_dtype == "category":
                for col in self.data.columns:
                    if self.data[col].dtype == 'category':
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)
            elif include_dtype == "numeric":
                for col in self.data.columns:
                    if self.data[col].dtype in ('int', 'float'):
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)
            elif include_dtype == "datetime":
                for col in self.data.columns:
                    if self.data[col].dtype == 'datetime64[ns]':
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)
            else:
                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)
        else:
            # add prefix only to specified columns or single col
            if isinstance(columns, list):
                for col in columns:
                    if col in self.data.columns:
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)
            else:
                for col in self.data.columns:
                    if col == columns:
                        new_names.append(f"{prefix}{col}")
                    else:
                        new_names.append(col)

        # dictionary to map to pd.rename of existing and new col names
        c_names = dict(
            zip(list(self.data.columns), new_names)
            )
        self.data.rename(columns=c_names, inplace=True)


    def add_suffix(self, 
                   suffix: str, 
                   include_dtype: Optional[str] = None,
                   columns: Optional[Union[str, List[str]]] = 'all'):
        """
        Add a suffix to all or some column names.


        Parameters
        ----------
        suffix : str
            A suffix to append to the ending of all or 
            specified column names.
        include_dtype : {'category', 'numeric', 'datetime', 'object'}, optional
            A datatype. Only include columns
            of this type when adding the specified suffix.
            Default = None.
        columns : str or list
            Column names to add suffix to. Others will be
            ignored. Default = 'all'.
        """
        assert isinstance(columns, (str, list)), TypeError(
            "Must pass a string or list to 'ignore'.")
        assert all(col in self.data.columns for col in columns), ValueError(
            "One of more columns passed is not contained "
                      "in the current table.")
        assert not all(columns, include_dtype), AssertionError(
            "Cannot specify a data type to select "
                           "columns based off of and include specific "
                           "column names. Must specify one or the other")
        new_names = []

        # TODO :  make this work for columns of different data types
        
        if columns == 'all':
            # add suffix to all columns
            new_names = [f"{col}{suffix}" for col in self.data.columns]
        elif include_dtype:
            # add suffix to only cols of that dtype
            if include_dtype == "category":
                for col in self.data.columns:
                    if self.data[col].dtype == 'category':
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)
            elif include_dtype == "numeric":
                for col in self.data.columns:
                    if self.data[col].dtype in ('int', 'float'):
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)
            elif include_dtype == "datetime":
                for col in self.data.columns:
                    if self.data[col].dtype == 'datetime64[ns]':
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)
            else:
                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)
        else:
            # add suffix only to specified columns or single col
            if isinstance(columns, list):
                for col in columns:
                    if col in self.data.columns:
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)
            else:
                for col in self.data.columns:
                    if col == columns:
                        new_names.append(f"{col}{suffix}")
                    else:
                        new_names.append(col)

        # dictionary to map to pd.rename of existing and new col names
        c_names = dict(zip(list(self.data.columns), new_names))
        self.data.rename(columns=c_names, inplace=True)
     
            
    def join(self, 
             to, 
             left_on : str, 
             right_on: str, 
             how: Optional[str] = 'inner', 
             validate: Optional[str] = None, 
             indicator: Optional[bool] = False):
        """
        Join 2 tables together based on a unique key identifier.


        Parameters
        ----------
        to : SciTable
            A table to join to. 
        left_on : str
            A key column in the joining table to
            use for merging rows.
        right_on : str
            A key column in the table to be joined to.
        how : {'inner', 'outer', 'left', 'right', 
                'anti', 'partial-outer'}, optional
            What type of join to perform.  
            Default = 'inner'.
        validate : {'1:1', '1:m', 'm:1', 'm:m'}, optional
            Validate the type of relation you expect
            as result of merging to a new table. 
            Default = None.
        indicator : bool, default
            Whether or not to append an indicator column
            labeled '_merge' to the end of the resulting
            table. This column labels where the row
            existed prior to the join (left, right, or both).
            Default = False.


        Returns
        -------
        self or new_table : SciTable


        Notes
        -----
        Performing an anti merge will filter the rows of the
        current table in place, thus, no assignment should
        be done. Any other type of merge will produce a new
        SciTable object which must be assigned to a variable. 
        """
        # TODO: implement a fast ordered join
        assert how in ('inner', 'outer', 'left', 'right', 'anti', 
                       'partial-outer'), ValueError(
            "Join type can only be one of the following: "
            "'inner', 'outter', 'left', 'right', 'anti', "
            "'partial-outter'.")   
        assert left_on in self.data.columns, KeyError(
            f"Must join on a unique column of the current table. "
            f"'{left_on}' does not exist in the current table.")
        assert right_on in to.data.columns, KeyError(
            f"Must join to a unique column of the joining table. "
            f"'{right_on}' does not exist in the joining table.")
        
        if how == 'anti':
            # union with second table ( A ∪ B )
            anti_dat = pd.merge(
                self.data, to.data, left_on=left_on, right_on=right_on, 
                how='outer', indicator=True)
            # filter outer join rows based on set difference ( U \ A ) 
            self.data = anti_dat[anti_dat['_merge'].isin(
                    ['right_only', 'left_only'])].copy()
            # drop merge indicator col
            self.data.drop(columns='_merge', inplace=True)
            # not returning a table here because anti join essentially is 
            # filtering rows in place with some set theory.
        elif how == 'partial-outer':
            # union with second table ( A ∪ B )
            po_dat = pd.merge(
                self.data, to.data, left_on=left_on, right_on=right_on, 
                how='outer', indicator=True)
            new_table = SciTable(nameless=True) 
            # filter outer join rows based on partial set union ( A ∪ (A ∩ B) ) 
            new_table.data = po_dat[po_dat['_merge'].isin(
                    ['both', 'left_only'])].copy()
            # drop merge indicator col
            new_table.data.drop(columns='_merge', inplace=True)
            return new_table
        else:
            new_table = SciTable(nameless=True) 
            # any other type of merge pd supports with basic API useage:
            new_table.data = self.data.copy().merge(
                to.data.copy(), left_on=left_on, right_on=right_on, 
                how=how, validate=validate, indicator=indicator)
            return new_table
    
    def subset_columns(self, columns: Union[str, List[str]], 
                       action: Optional[str] = "exclude", 
                       checkpoint_before: Optional[bool] = False):
        """
        Drop excluded columns or retain only included
        columns of a table. 


        Parameters
        ----------
        columns : str or list
            Column(s) to include ot exclude from the
            table when subsetting. 
        action : {'exclude', 'include'}, optional
            Whether to include or exclude the specified
            columns when subsetting. Default = 'exclude'.
        checkpoint_before : bool, optional
            Whether or not to make a checkpoint
            of the current data before downsizing.
            Default = False.
        """
        assert action.lower() in ('exclude', 'include'), ValueError(
            "Must specify to either 'exclude' or 'include' "
            "columns passed.")
        if checkpoint_before:
            self.make_checkpoint()
            msg = (
                "NOTICE: Made a checkpoint before subsetting column(s). "
                 "Previous checkpoint will be overwritten. "
                 "To revert back afterwards, call '.revert_to_checkpoint()'."
            )
            print(msg)
            
        if action.lower() == 'exclude':  
            self.data.drop(columns, axis=1, inplace=True)
        else:
            if isinstance(columns, list):
                columns = [
                    i for i in self.data.columns if i not in columns]
            else:
                columns = [
                    i for i in self.data.columns if i != columns]
            self.data.drop(columns, axis=1, inplace=True)

        
    def select(self, to_select: Dict[str, Union[str,List[str]]], 
                action: Optional[str] = 'exclude', 
                checkpoint_before: bool = False):
        """
        Select rows from a table based on values of a given
        column or columns. 'select' and 'filter'are often
        used interchangibly. 


        Parameters
        ----------
        to_select : dict
            A dictionary of column names and a one or
            more values (keys being a single string or
            a list of string values) which specify the
            inclusion or exclusion criteria when selecting
            rows.
        action : {'include', 'exclude'}, optional
            Whether to include or exclude rows which meet
            the critera specified in 'to_select'.
            Default = 'exclude'.
        checkpoint_before : bool, optional
            Whether or not to make a checkpoint
            of the current data before downsizing.
            Default = False.
        """
        for column in to_select.keys():
            assert column in self.data.columns, KeyError(
                f"{column} not contained in table.")
            assert(is_categorical_dtype(self.data[column]) or 
                   is_string_dtype(self.data[column])), TypeError(
                       "Filter table can only be passed columns which "
                        "are categorical or strings. To filter based on "
                        "a number or a date, see 'filter_numeric' or "
                        "'filter-date.'")
        assert action.lower() in ('exclude', 'include'), ValueError(
                "Must specify filtering action to be one of 'exclude' "
                "(to filter data out based on the value passes), or "
                "'include' (to retain data based on the value passed, "
                "ie: filter all other data out.).")
        if checkpoint_before:
            print(
                "NOTICE: Making a checkpoint before filtering data. "
                "Previous checkpoint will be overwritten. "
                "To revert back afterwards, call '.revert_to_checkpoint()'.")
            self.make_checkpoint()
            
        for i, v in to_select.items():
            column = i # col name
            values = v # either a str or list of vals
            
            # if the column is categorical, cat pointers will have to 
            # be dropped once a category is mapped, even if no rows have 
            # a value after preprocessing, that pointer still exists.
            if is_categorical_dtype(self.data[column]):
                if action.lower() == 'exclude':
                    if type(values) == list:
                        # TODO: test this now
                        self.data[column].cat.remove_categories(values, 
                                 inplace=True) # drop cat level
                        self.data = self.data[~self.data[column].isin(values)]
                    else:
                        self.data[column].cat.remove_categories(values, 
                                 inplace=True)
                        self.data = self.data[~self.data[column] == values]
                else:
                    # inclusion criteria
                    all_cats = self.data[column].cat.categories
                    to_remove = [i for i in all_cats if i not in values]

                    if type(values) == list:
                        self.data[column].cat.remove_categories(to_remove, 
                                 inplace=True)
                        self.data = self.data[self.data[column].isin(values)]
                    else:
                        self.data[column].cat.remove_categories(to_remove, 
                                 inplace=True)
                        self.data = self.data[self.data[column] == values]
            else:
                if action.lower() == 'exclude':
                    if type(values) == list:
                        self.data = self.data[~self.data[column].isin(values)]
                    else:
                        self.data = self.data[~self.data[column] == values]
                else:
                    # inclusion criteria
                    if type(values) == list:
                        self.data = self.data[self.data[column].isin(values)]
                    else:
                        self.data = self.data[self.data[column] == values]
                    
    def filter_numeric(self):
        """
        """
        # TODO
        # implement filtering for numerics (> >= < <=)
        raise NotImplementedError
        
    def select_by_date(self, 
                        column: str, 
                        dates: Union[str, List[str]], 
                        action: Optional[str] = 'between', 
                        errors: Optional[str] = 'raise', 
                        checkpoint_before: Optional[bool] = False, 
                        inclusive: Optional[bool] = False):
        """
        Select all instances from a table where values of a 
        datetime column are 'earlier' or 'later' than a given date 
        or 'between' two dates.


        Parameters
        ----------
        column : str
            A datetime column to use to check values.
        dates : str or list
            A date or list of two dates.
        action: {'between', 'earlier', 'later'}, optional
            How to select rows in reference to the given
            date(s). Default = 'between'.
        errors : {'ignore', 'raise', 'coerce'}, optional
            How to handle errors (records which cannot be parsed
            to datetime). Default = 'raise'.
        checkpoint_before : bool, optional
            Whether or not to make a checkpoint of the current 
            data before downsizing.Default = False.
        inclusive : bool, optional
            Whether or not to include or exclude the exact date
            passed when selecting rows (i.e.: action='later' and
            inclusive = True, would include rows with a value in 
            the column of interest equal to or later than the date). 
            Default = False.
        """
        # TODO: make this work for all numerics, not jsut dates
        assert action.lower() in ('between', 'before', 'after'), ValueError(
            "action must be set to 'between', 'before', or 'after'.")
        assert isinstance(dates, (str, list)), TypeError(
            "Must either pass one date as a string or multiple "
            "dates in a list.")

        if (type(dates) == list) and (len(dates) > 1):
            assert action.lower() == 'between', ValueError(
                "When passing multiple dates, filter action may "
                "only be set to 'between'.")
        if (type(dates) == list) and (len(dates) > 2):
            raise ValueError("When passing multiple dates in a list, may "
                             "only pass a max of 2 dates.")
        if type(dates) == list:
            for d in dates:
                if not np.issubdtype(d, np.datetime64):
                    try:
                        pd.to_datetime(d, infer_datetime_format=True, 
                                       errors='raise')
                    except ValueError:
                        msg = (
                            "One of the dates passed to use for filtering "
                            "is not in a proper date format. Please use "
                            "the following format: 2016-01-11 0:00:00."
                        )
                        raise ValueError(msg)
        else:
            if not np.issubdtype(dates, np.datetime64):
                try:
                    pd.to_datetime(dates, infer_datetime_format=True, 
                                       errors='raise')
                except ValueError:
                    msg = (
                        "Date passed to use for filtering "
                        "is not in a proper date format. Please use the "
                        "following format: 2016-01-11 0:00:00."
                    )
                    raise ValueError(msg)
                    
        if not np.issubdtype(column.dtype, np.datetime64):
            warnings.warn("Passed a column not in datetime format. Tried to "
                          "convert to datetime before filtering table.", 
                          UserWarning)
            self.data[column] = pd.to_datetime(self.data[column], 
                     infer_datetime_format=True, errors=errors)
        if checkpoint_before:
            self.make_checkpoint()
            print("NOTICE: Made a checkpoint before filtering data. "
                  "Previous checkpoint will be overwritten. "
                  "To revert back afterwards, call '.revert_to_checkpoint()'.")
            
        if inclusive:
            if action.lower() == 'between':
                self.data = self.data.loc[(self.data[column]>=dates[0]) & 
                                          (self.data[column]<=dates[1])]
            elif action.lower == 'before':
                self.data = self.data.loc[(self.data[column]<=dates)]
            else:
                self.data = self.data.loc[(self.data[column]>=dates)]
        else:
            if action.lower() == 'between':
                self.data = self.data.loc[(self.data[column]>dates[0]) & 
                                          (self.data[column]<dates[1])]
            elif action.lower == 'before':
                self.data = self.data.loc[(self.data[column]<dates)]
            else:
                self.data = self.data.loc[(self.data[column]>dates)]
            
    def reindex_column_order(self, column_order: List[str]):
        """
        Reindex column ordering of table.

        Parameters
        ----------
        column_order : List[str]
            A list of column names in their desired order.
        """
        self.data = self.data.reindex(columns = column_order)
    
    def string_split_mutate(self, reference_column: str, split_on: str, 
                            new_columns: Union[List[str], str], 
                            keep_index: Optional[Union[int, str]]='all'):
        """
        Split a string column based on some 'split_on' character.
        User can choose to expand each split subsection into a 
        new distinct column (by default with keep_index='all') or 
        keep a specific split subsection by specifying its index after
        splitting. I.e.: if the user split 'NYC-New York" on '-' and 
        only wanted to keep 'NYC' she/he would specify keep_index=0.


        Parameters
        ----------
        reference_column : str
            A column containing strings to split into subsections.
        split_on : str
            A character to split on.
        new_columns : list or str
            A list of new column names with a length matching the number
            of resulting string subsections after splitting 
            the reference column values, or a str representing the
            the single new column name in cases where a keep_index is 
            provided.
        keep_index : {'all'} or int, optional
            An index of a specific subsection of the split string
            which the user wants to keep. Default = 'all'
        """
        if keep_index.lower() == 'all':
            self.data[new_columns] = self.data[
                reference_column].str.split(
                    split_on, expand=True)
        else:
            assert isinstance(keep_index, int), TypeError(
                "Must provide an integer index to keep from the "
                "resulting list after spliting the string in the "
                "reference column."
            )
            assert isinstance(new_columns, str), TypeError(
                "When providing a keep_index, may only pass one "
                "new column to create."
            )
            self.data[new_columns] = self.data[
                reference_column].str.split(
                    split_on, expand=True)[keep_index] # indexed here
            
    def strip_cols(self, columns: Optional[Union[List[str], str]] = 'all', 
                    case: Optional[str] = None):
        """
        Strip all whitespace from columns and optionally
        convert their casing
    
        Parameters
        ----------
        columns : A string or list, optional.
            A single column name collection of column anmes to
            strip/convert. Default = 'all'.
        case : {'lower', 'upper', 'title'}, optional.
            Whether or not to change case of string columns.
            Default = None.
        """
        if columns.lower() == 'all':
            # see table_h
            self.data = _str_strip_all(self.data, case)
        else:
            if type(columns) == list:
                assert all(i in self.data.columns for i in columns), KeyError(
                    "One or more columns specified are not contained "
                    "in the table.")
            else: 
                assert columns in self.data.columns, KeyError(
                    f"{columns} not contained in table.")
            for col in columns:
                if not is_string_dtype(self.data[col]):
                    raise TypeError(f"Column '{col}' is not a string. "
                                    "Must only pass strings to strip.")

            if case:
                assert case.lower() in ('lower', 'upper', 'title'), ValueError(
                    "case must be 'lower', 'upper', or 'title'.")
                    # see table_h
                self.data = _str_case_strip(self.data, columns, case)
            else:
                self.data[columns] = self.data[columns].apply(
                        lambda x: x.str.strip())
            
    
    def fill_all_missing(self, fill_with: Optional[Any] = 'MI'):
        """
        Impute all missing values of a table with the value
        passed to 'fill_with'. Missing vals default to
        np.nan.
        """
        self.data.apply(lambda x: x.fillna(fill_with))
        
    def impute_missing_numeric(self, columns, how='median'):
        """
        """
        raise NotImplementedError
        # TODO: fix this method 


        assert isinstance(columns, (str, list)), TypeError(
            "Can only pass a single column as a string or a list "
            "of columns to fill.")
        assert how.lower() in ('median', 'mean'), ValueError(
            "Must choose to replace numeric columns with "
            "either meadian or mean. Median recommended.")

        if (type(columns) == list) and (len(columns) > 1):
            missing_indicator = {}
            for col in columns:
                if is_numeric_dtype(col):
                    if pd.isnull(col).sum() >= 1 or (col in missing_indicator):
                        self.data[f"{col}_na"] = pd.isnull(col)
                        if how.lower == 'median':
                            fill_with =  missing_indicator[col] if col in \
                                missing_indicator else col.median()
                        else:
                            fill_with =  missing_indicator[col] if col in \
                                missing_indicator else col.mean()
                        self.data[col] = fill_with
                else:
                    raise TypeError(f"{col} is not a numeric column.")
            return missing_indicator
        else:
            missing_indicator = {}
            if is_numeric_dtype(columns):
                if pd.isnull(columns).sum() >= 1 or (
                        columns in missing_indicator):
                    self.data[f"{columns}_na"] = pd.isnull(columns)
                    if how.lower == 'median':
                        fill_with =  missing_indicator[columns] if col in \
                            missing_indicator else col.median()
                    else:
                        fill_with =  missing_indicator[columns] if col in \
                            missing_indicator else col.mean()
                    self.data[col] = fill_with
            else:
                raise TypeError(f"{columns} is not a numeric column.")
        return missing_indicator
              
    def replace(self, to_replace: Any, 
                replace_with: Any, 
                columns: Optional[Union[str, List[str]]] = 'all', 
                regex: Optional[bool]=False):
        """
        Replace a value or list of values with a new value or
        new list of values in a specified column or list of columns.


        Parameters
        ----------
        to_replace : Any
            Values to replace in specified columns.
        replace_with : Any
            Values to use for replacement in specified columns.
        columns : str or list or 'all', optional.
            The columns in which to do the replacement.
            Default = 'all'.
        regex : bool, optional
            Whether the replacement strings are regex patterns.
            Default = False.
        """
        assert isinstance(columns, (str, list)), TypeError(
            "Must pass a column name as a string or a list "
            "of columns.")

        if ( isinstance(to_replace, list) and isinstance(replace_with, list) ):
            assert len(to_replace) == len(replace_with), AssertionError(
                    "If passing lists of values to both 'to_replace' "
                    "and 'replace_with', these lists must be over equal "
                    "length. Values of the shorter length list will not be "
                    " recycled during replacement.")

        if isinstance(columns, str):
            if columns.lower() == 'all':
                self.data.replace(to_replace, replace_with, regex=regex, 
                                  inplace=True)
            else:
                replace_dict = { # replace dict
                    columns:{to_replace: replace_with}
                    }

                self.data.replace(replace_dict, regex=regex, inplace=True)
        else:
            assert all (
                col in self.data.columns for col in columns), KeyError(
                    "One or more columns specified are not contained "
                    "in table.")

            replace_dict = { # replace dict, each key is a column
                col:{to_replace: replace_with} for col in columns
                }

            self.data.replace(replace_dict, regex=regex, inplace=True)
            
    def replace_exact_idx(self, col_name: str, row_index: int, 
                        new_value: Union[str, int, float]):
        """
        Replace a value at an exact location given a column and
        a row indexindex.
        """
        self.data.loc[self.data.index[row_index], col_name] = new_value
            
    def convert_to_numeric(self, columns: Union[str, List[str]],
                            errors: Optional[str] = 'coerce'):
        """
        Converts specified column(s) to numeric. By default, coerces
        all invalid records (ones which cannot be cast to numeric) 
        to np.NaN. 

        Parameters
        ----------
        columns : str or list or 'all', optional.
            The columns in which to do the replacement.
            Default = 'all'.
        errors : {'ignore', 'raise', 'coerce'}, optional
            How to handle errors (records which cannot be parsed
            to datetime). Default = 'coerce'.
        """
        if isinstance(columns, list):
            assert all(i in self.data.columns for i in columns), KeyError(
                "One or more columns specified are not "
                "contained in the table.")
        else: 
            assert columns in self.data.columns, KeyError(
                f"{columns} not contained in table.")
        
        print(
            "NOTICE: Converting specified column(s) to numeric. "
            "Any data which cannot be converted will be replaced with NaN.")
        
        if isinstance(columns, list):
            self.data[columns] = self.data[columns].apply(pd.to_numeric, 
                    errors=errors)
        else:
            self.data[columns] = self.data[columns].to_numeric( 
                    errors=errors)

        
    def convert_to_datetime(self, columns: Union[str, List[str]],
                            dt_format: Optional[str] = "%Y-%m-%d", 
                            errors: Optional[str] = 'raise', 
                            infer_format: Optional[bool] = False):
        """
        Converts specified columns to datetime with the given date
        format provided. By default, does not coerces all 
        invalid records (ones which cannot be cast to numeric) 
        to np.NaN, instead raises an input error to the user.
        
        
        Parameters
        ----------
        columns : str or list
            The column(s) to convert to datetime.
        dt_format : str, optional
            A datetime format to use to parse the provided
            column(s) to datetime[ns] type.
            Default = '%Y-%m-%d'
        errors : {'ignore', 'raise', 'coerce'}, optional
            How to handle errors (records which cannot be parsed
            to datetime). Default = 'raise'.
        infer_format : bool, optional
            Whether or not to let Pandas attempt to infer the
            datetime format for parsing. Default = False.
        """
        # TODO:  this has to be fixed and tested

        assert not all( (dt_format, infer_format) ), ValueError(
            "Cannot provide a datetime format while "
            "specifying infer_format = True.")
        
        if isinstance(columns, list):
            assert all(i in self.data.columns for i in columns), KeyError(
                "One or more columns specified are not "
                "contained in the table.")
            for col in columns:
                c_dtype = col.dtype
                if isinstance(c_dtype, DatetimeTZDtype):
                    c_dtype = np.datetime64
                if not np.issubdtype(c_dtype, np.datetime64):
                    if infer_format:
                        self.data[columns] = self.data[columns].apply(
                                pd.to_datetime, infer_datetime_format=True, 
                                errors=errors)
                    else:
                        self.data[columns] = self.data[columns].apply(
                                pd.to_datetime, format=dt_format, errors=errors)
                else:
                    msg = (
                        "It seems one or more columns passed are "
                        "already in datetime format. Consider double "
                        "checking current by calling '.info()'. Converting "
                        "the rest to datetime."
                    )
                    raise TypeError(msg)
        else: 
            assert columns in self.data.columns, KeyError(
                f"{columns} not contained in table.")
            c_dtype = columns.dtype    
            if isinstance(c_dtype, DatetimeTZDtype):
                c_dtype = np.datetime64
            if not np.issubdtype(c_dtype, np.datetime64):
                if infer_format:
                    self.data[columns] = pd.to_datetime(self.data[columns], 
                        infer_datetime_format=True, errors=errors)
                else:
                    self.data[columns] = pd.to_datetime(self.data[columns], 
                             format=dt_format, errors=errors)
            else:
                raise TypeError("Passed a column already in datetime "
                                "format. No action was taken.")
                          
    def convert_to_string(self, columns: Union[List[str], str]):
        """
        Converts specified column(s) to string.
        """
        if type(columns) == list:
            assert all(
                i in self.data.columns for i in columns), KeyError(
                    "One or more columns specified are not contained "
                    "in the table.")
        else: 
            assert columns in self.data.columns, \
                KeyError(f"{columns} not contained in table.")
        if isinstance(columns, list):
            self.data[columns] = self.data[columns].apply(pd.to_string)
        else:
            self.data[columns] = self.data[columns].to_string()
        
                    
    def _cat_from_levels(self, columns : Union[str, List[str]], 
                ordered: Optional[bool] = False) -> pd.Series:
        """
        Returns
        -------
        Series : unique levels of a nominal column as a 
        CategoricalDtype object
        """
        if ordered:
            # get unique nominal levels, set as a catDtype object
            levels = CategoricalDtype(self.data[columns].unique(), 
                    ordered=True)
        else:
            levels = CategoricalDtype(self.data[columns].unique(), 
                    ordered=False)
        
        return self.data[columns].astype(levels)

    def convert_to_categorical(self, columns : Union[str, List[str]], 
                        ordinal: Optional[bool] = False):
        """
        Convert column(s) to categorical.

        Parameters
        ----------
        columns : str or list
            The column(s) to convert to categorical.
        ordinal : bool, optional
            Whether or not the categorical levels of
            the passed column(s) should exist on an ordinal
            scale, rather than an unordered nominal one.
            Default = False.
        """
        if isinstance(columns, list):
            assert all(i in self.data.columns for i in columns), KeyError(
                "One or more columns specified are not contained "
                "in the table.")
        else: 
            assert columns in self.data.columns, KeyError(
                f"{columns} not contained in table.")
        
        if not isinstance(columns, str):
            if ordinal:
                self.data[columns] = self._cat_from_levels(
                    columns, ordered=True)
            else: 
                self.data[columns] = self._cat_from_levels(
                    columns, ordered=False)
        else:
            for col in columns:
                if ordinal:
                    self.data[col] = self._cat_from_levels(
                        col, ordered=True)
                else: 
                    self.data[col] = self._cat_from_levels(
                        col, ordered=False)
            
            
    def _all_strings_to_cats(self, ordinal: bool=False):
        """Convert strs to cats"""
        if ordinal:
            for col, val in self.data.items():
                if is_string_dtype(val):
                    self.data[col] = val.astype('category').cat.as_ordered()
        else:          
            for col, val in self.data.items():
                if is_string_dtype(val):
                    self.data[col] = val.astype('category')
                    
    def levels(self, column: str):
        """
        Return levels of a str or categorical column.
        """
        if isinstance(column.dtype, CategoricalDtype):
            return self.data[column].cat.categories
        else:
            levels = list(self.data[column].unique())
            return [i for i in levels if str(i) != 'nan']
        
    def rename_all_categories(self, column: str, new_levels: List[str]):
        """
        Rename levels of a categorical column (CategoricalDtype).
        """
        assert is_categorical_dtype(self.data[column]), TypeError(
            "Must pass a column with the datatype 'category'.")
        assert isinstance(new_levels, list), TypeError(
            "Must pass a list of new levels.")
        assert isinstance(column, str), TypeError(
            "Must only pass a single column as a string.")

        # will return np.nan as a unqiue value
        cols = list(self.data[column].unique())
        old_levels = [i for i in cols if str(i) != 'nan']

        assert len(old_levels) == len(new_levels), AssertionError(
                "Must provide the same number of levels as currently "
                "present in the column to rename.")

        rename_map = {k:v for k, v in zip(old_levels, new_levels)}

        self.data[column].cat.rename_categories(
            rename_map, inplace=True)
        
    def reorder_cat_levels(self, column, new_order):
        """Change the ordering of a categorical column.
        """
        self.data[column].cat.reorder_categories(
            new_order, inplace=True)
        
        
    def bin_continous(self, 
                      reference_column: str, 
                      bins: List[int], 
                      levels: List[str], 
                      new_col_name: Optional[str] = None, 
                      ordinal: Optional[bool] = False):
        """
        Bins a continous column into categories. Will replace 
        continuous column with categorical by default unless a 
        new_col_name is given. If a new column is given, the 
        original continuous column will remain and a new categorical 
        column will be added to the table.


        Parameters
        ---------- 
        reference_column : str 
            A column containing values on a continuous scale to
            bin to discrete categories.
        bins : list
            An array of integers to use as thresholds for
            cutting the reference column into discrete buckets.
        levels : list
            A list of names to label each category resulting from
            binning the reference column.
        new_col_name : str, optional
            Name of a new discrete column to append to the table.
            Default = None.
        ordinal : bool, optional
            Whether or not the values of the resulting discrete 
            column should be considered ordered. Default = False.


        Notes
        -----
        If the user specifies an ordinal column to be created, 
        the values of 'levels' will be treated as ordered based on
        the natural ordering entered by the user. If the user does 
        not ask for an ordinal column to be created, 'levels will be 
        used for nominal labels of the new column. 
        """
        assert ( len(levels) == (len(bins)+1) ), ValueError(
            "The number of levels (labels) provided does not match "
            "the number of bins + 1."
        )

        if ordinal:
            cats = CategoricalDtype(levels, ordered=True)
        else:
            cats = CategoricalDtype(levels, ordered=False)

        if new_col_name:
            self.data[new_col_name] = pd.cut(self.data[reference_column], 
                        bins=bins, labels=levels).astype(cats)
        else:
            self.data[reference_column] = pd.cut(
                    self.data[reference_column], 
                        bins=bins, labels=levels).astype(cats)


    def set_to_flag(self, 
                    referrence_col: str, 
                    value: Any, 
                    check: Optional[str] = 'equality', 
                    flag_value: Optional[Union[str, int]] = 1, 
                    fill_value: Optional[Union[str, int]] = 0, 
                    case: Optional[bool] = True, 
                    regex: Optional[bool] = False, 
                    drop_original: Optional[bool] = False):
        """
        Create a new binary flag column consisting of a flag value
        (given to instances which contain a certain value of interest
        in the reference column) and a fill value (given to all remaining
        instances which do not contain the value of interest). In other
        words, this method produces a one-hot encoded column based
        on a given criteria.

        Parameters
        ----------
        reference_col : str
            An existing column in the table to search for a value
            of interest within.
        value : str or int or float or list
            A single value or list of values contained in the reference
            column to use for flagging instances. Instances of the table
            containing the value (or one of the values if passed a list)
            are given the 'flag value' in the newly generate column(s). 
            otherwise, instances not containg the value (or one of the values)
            will be given the 'fill_value'. 
        check : {'equality', 'greater', 'less'}, optional
            How to check for the value used to generate flags.
            ie: check that instances equal a value, are greater than a value,
            or are lower than a value. Default = equality.
        flag_value : int or str, optional
            A meaningful or arbitrary value used to impute rows  
            which meet the check criteria :
            (reference col->check conditon->value). Default = 1.
        fill_value : int or str, optional
            A meaningful or arbitrary value used to impute rows  
            which do not meet the check criteria :
            (reference col->check->value). Default = 0.
        case : bool, optional
            Whether or not to be sensitive to case when checking
            for the value. Default = True.
        regex : bool, optional
            Whether or not the value to check for is a regular 
            expression. Default = False. 
        drop_original : bool, optional
            Whether or not to remove the reference column after generating
            the single or multiple flag columns. Default = False.
 
    
        Notes
        -----
        Embed functions found in query_locals.py into query strings
        to augement functionality.
        """
        assert isinstance(value, (str, int, float, list)), TypeError(
            "The value of interest used to generate a new flag "
            "column must be a string, integer, float, or "
            "list of values.")
            
        if regex and check.lower() != 'equality':
            raise AssertionError("Cannot check for anything but equality when "
                                 "using a regex value.")

        cast_series_as_int = False
        if ( isinstance(flag_value, int) and isinstance(fill_value, int) ):

            cast_series_as_int = True
            
        if ( isinstance(value, list) ) or ( value.lower() == "all" ):
            if isinstance(value, list):
                assert all(
                    isinstance(i, (str, int, float)) for i in value), TypeError(
                        "The reference value used to generate a new flag "
                          "column must be a string, integer, or float.")
            else:
                # return array of all unique levels
                value = list(self.data[referrence_col].unique())
            for v in value:
                new_col = f"{referrence_col}_{v}_flag".upper().replace(
                        ' ', '_') #name of new col
                # see table_h.py _flag_switch
                self.data[new_col] = _flag_switch(
                    self.data, referrence_col, v,
                    check, flag_value, fill_value, case, regex
                )
                    
                if cast_series_as_int:
                    # explicitly coerce from float to int
                    self.data[new_col] = self.data[new_col].astype(int)

            if drop_original:
                self.data.drop(referrence_col, axis=1, inplace=True)
        else:    
            new_col = f"{referrence_col}_{value}_flag".upper().replace(
                    ' ', '_')

            self.data[new_col] = _flag_switch(
                self.data, referrence_col, value,
                check, flag_value, fill_value, case, regex
            )

            if cast_series_as_int:
                # explicitly coerce from float to int
                self.data[new_col] = self.data[new_col].astype(int)

            if drop_original:
                self.data.drop(referrence_col, axis=1, inplace=True)
    
    def multi_set_to_flag(self):
        """
        """
        #TODO implement a multi condition flag 
        raise NotImplementedError
        
    def derive_new_count(self, new_col_name: str, increment_with: str, 
            increment_when: str):
        """
        Create a new grouped count column which provides a sum of 
        how many times unique value occurs in given field of the table,
        grouped by a unique identifier (key) column.


        Parameters
        ----------
        new_col_name : str
            A name to assign the new count column.
        increment_with : str
            A key column to use for generating grouped counts
            of unique values.
        increment_when : str
            A query string providing a condition (or several
            conditions) used to count occurences where
            instances of the table meet said condition.

        Notes
        -----
        nans are filled with 0 after grouping, counting, and
        joining back to the main table.
        """
        # TRIPLE CHECK THIS
        # take a value count of the with column based on the when condition
        # rename the with column to the new column name
        count = ( 
            pd.DataFrame(
                self.data.query(increment_when
                    )[increment_with].value_counts()
                        ).rename(columns={increment_with:new_col_name})
        )
            #.groupby(increment_with).agg('count').iloc[:,0])
            # .rename(columns={self.data.columns[0]:new_col_name}))
            #count.reset_index(level=0, inplace=True)

        # left join back to main table right on index
        self.data = self.data.merge(
            count, left_on=increment_with, right_index=True, 
                        how='left')
        # fill nans with 0
        self.data[new_col_name] = self.data[new_col_name].fillna(
            0).astype(int)
    
    def derive_new_conditional(self, new_col_name, referrence_col, 
                               increment_when, increment_by, reset_when, 
                               initial_value=0):
        """
        """
        # TODO
        raise NotImplementedError
    
    def derive_new_flag(self, new_col_name: str, 
                        flag_when: str, 
                        flag_value: Optional[Union[str, int]] = 1, 
                        fill_value: Optional[Union[str, int]] = 0):
        """
        Derive a new binary flag feature based on a given condition.


        Parameters
        ----------
        new_col_name : str
            The name of the newly generated binary flag
            column.
        flag_when : str
            A query string providing a condition (or several
            conditions) used to provide a flag value to
            instances of the table meet said condition.
        flag : int or str, optional
            A meaningful or arbitrary value used to impute rows  
            which meet the check criteria :
            (reference col->check conditon->value). Default = 1.
        fill : int or str, optional
            A meaningful or arbitrary value used to impute rows  
            which do not meet the check criteria :
            (reference col->check->value). Default = 0.

        
        Notes
        -----
        Embed functions found in query_locals.py into query strings
        to augement functionality.
        """
        # arbitrary range index temp col
        self.data['temp'] = range(1, len(self.data) + 1)
        # select only indecies which meet condition
        to_flag = pd.DataFrame(
            self.data.query(flag_when, engine="python")['temp']
            )
        # left join indecies back to main table with indicator
        self.data = self.data.merge(
            to_flag, left_on='temp', right_on='temp', how='left', 
                indicator=True
            )
        # bool flag table where instaces found in both tables, fill
        self.data[new_col_name] = np.where(
            self.data['_merge'] == 'both', flag_value, fill_value
            )
        # cast explicitly 
        self.data[new_col_name] = self.data[new_col_name].astype(int)
        # drop temp and indicator cols
        self.data.drop(columns=['temp', '_merge'], inplace=True)

    
    def expand_date(self, columns: List[str], 
                    drop_original: Optional[bool] = True, 
                    include_time: Optional[bool] = False, 
                    all_info: Optional[bool] = False, 
                    errors: Optional[str] = "raise"):
        """
        Expand a date column into its contained parts such as
        year, month, week, day, day of week, day of year. 
        

        Parameters
        ----------
        columns : str or list
            The datetime column(s) to expand.
        drop_original : bool, optional
            Whether or not to remove the original column(s) after 
            expanding it. Default = False.


        Notes
        -----
        This method is practical in cases where dates and times may
        have predictive power over the dependant variable, but
        that relationship is not directly known and instead needs
        to be mined through feature expansion followed by
        best-subset selection (for models which require it). 
        """
        self.convert_to_datetime(columns, errors=errors, 
            infer_format=True)
        
        expand_into = [
            'Year', 'Month', 'Week', 'Day', 
            'Dayofweek', 'Dayofyear'
            ]
        if include_time:
            # append more attribs
            expand_into = expand_into + [
                'Hour', 'Minute', 'Second'
            ]
        if all_info:
            # append more attribs
            expand_into = expand_into + [
                'Is_month_end', 'Is_month_start', 
                'Is_quarter_end', 'Is_quarter_start', 
                'Is_year_end', 'Is_year_start'
            ]        
        if isinstance(columns, list):
            for col in columns:
                # replace naming convention if exists
                field = col.replace("_DT", "")
                for i in expand_into:
                    # https://www.geeksforgeeks.org/python-getattr-method/
                    # these elements are properties of a datetime col.dt
                    self.data[field + i] = getattr(col.dt, i.lower())
                if drop_original:
                    self.data.drop(col, axis=1, inplace=True)
        else:
            field = columns.replace("_DT", "")
            for i in expand_into:
                self.data[field + i] =  getattr(columns.dt, i.lower())
            if drop_original:
                self.data.drop(columns, axis=1, inplace=True)

    
    def make_numeric(self, columns: Union[str, List[str]], 
                    max_levels: Optional[int] = None, 
                    inplace: Optional[bool] = False):
        """
        Convert all categorical columns to their numeric codes. 
        Also useful making an ordinal column numeric and the 
        deafult level (0) ought to be 1. Can be paired with 
        cat_convert_all() to first convert strs to cats, convert
        cats to numeric, resulting in an entirely numericalize 
        feature space. 

        Parameters
        ----------
        columns : str or list
            The column(s) to convert to numeric.
        max_levels : int, optional




        """
        #TODO this method needs fixing?

        assert isinstance( columns, (list, str) ), TypeError(
            "Must pass column(s) as a string or list."
        )
        assert isinstance( max_levels, int ), TypeError(
            "'max_levels' must be an integer."
        )

        if ( (isinstance(columns, list) ) and (len(columns) > 1) ):
            for col in columns:
                _make_categorical_numeric(
                    self.data, col, max_levels, inplace )
        else:
            _make_categorical_numeric(
                self.data, columns, max_levels, inplace )
    
    def reindex_categorical_columns(self, label: str, move_to: Optional[str] = 'left', 
                                    checkpoint_before: bool = False):
        """
        Moves catagorical columns to front of datafront and one hot encodes 
        them. Skips label column passed. 
        
        Should pass LABEL COLUMN to ignore
        """
        assert move_to.lower() in ('left', 'right'), ValueError(
            "move_to must be either 'left' or 'right'.")
        
        assert isinstance(label, str), ValueError(
            "Must pass a string representing "
                       "the label (target) column to 'label'.")
        assert label in self.data.columns, KeyError(
            "label Column passed  does not exist in data.")
        
        if checkpoint_before:
            self.make_checkpoint()
            print("Made a checkpoint before one hot encoding data. "
                  "Previous checkpoint will be overwritten. "
                  "To revert back afterwards, call '.revert_to_checkpoint()'.")
            
        cats = []
        everything_else = []

        for col in self.data.columns:
            if col == label:
                pass
            elif is_categorical_dtype(col):
                cats.append(col)
            else:
                everything_else.append(col)
        
        if move_to.lower() == 'left':
            new_index = cats + everything_else + label
        else: 
            new_index = everything_else + cats + label
            
        self.data.reindex(new_index, axis="columns")

            
########################## > Grand Children < ##########################
class EncodedTable(SciTable):
    """

    
    Parameters
    ----------

    
    Attributes
    ----------

        
    Notes
    ------


    """
    def __init__(self, schema=None, name=None, struc=None, 
                 use_db_defaults=False, nameless=False):
        super().__init__(schema, name, struc, use_db_defaults, nameless)
    
    def encode_data(self):
        """
        """
        raise NotImplementedError
    
    def decode_data(self):
        """
        """
        raise NotImplementedError
    
    def _strings_to_cats(self, ordinal=False):
        """
        """
        if ordinal:
            for col, val in self.data.items():
                if is_string_dtype(val):
                    self.data[col] = val.astype('category').cat.as_ordered()
    
        for col, val in self.data.items():
            if is_string_dtype(val):
                self.data[col] = val.astype('category')
    # or
    # self.data.applymap(lambda x: x.astype('category') if is_string_dtype(x) else x)
    
    def _cat_convert_all(self):
        """
        """
        return self.data.apply(lambda x: x.astype('category'))
    
    def _cat_to_code(self):
        """
        """
        for col, val in self.data.items():
            if is_categorical_dtype(val):
                self.data[col] = self.data[col].cat.codes
        # or
        # self.data.applymap(lambda x: x.cat.codes if is_categorical_dtype(x) else x)
    
    def _all_codes(self):
        """
        """
        return self.data.apply(lambda x: x.cat.codes)
    
    def _build_col_dict(self):
        """
        """
        col_dict = {}
        rev_col_dict = {}
    
        for num, val in enumerate(self.data.columns):
            col_dict[val] = f"Column_{num}"
    
        rev_col_dict = {v: k for k, v in col_dict.items()}
    
        return col_dict, rev_col_dict
    
    def _build_value_dict(self):
        """
        """
        code_dict = {}
        rev_code_dict = {}
        cat_cols = []
        vals = []
    
        for col, val in self.data.items():
            if is_categorical_dtype(val):
                cat_cols.append(col)
                vals.append(dict(enumerate(self.data[col].cat.categories)))
        for col, val in zip(cat_cols, vals):
            code_dict[col] = val
        for key, dic in code_dict.items():
            dic = {v: k for k, v in dic.items()}
            rev_code_dict[key] = dic
    
        return code_dict, rev_code_dict
    
    def _reverse_coded_values(self, d):
        """
        """
        cat_cols = []
    
        for col, val in self.data.items():
            if is_categorical_dtype(val):
                cat_cols.append(col)
        for col, dic in zip(cat_cols, d.values()):
            self.data[col] = self.data[col].map(dic)
    
    def _see_map(self, d, vals=True):
        """
        """
        if vals:
            for key, val in d.items():
                print(f"{key}:\n")
                for i, v in val.items():
                    print(f"\t{i} : {v}")
                    print()
        else:
            for key, val in d.items():
                print(f"\t{key} : {val}\n")


       