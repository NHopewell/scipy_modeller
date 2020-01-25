"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

aal_encode.py (Contains: 11 functions)

Description
-----------
aal_encode.py provides a collection of 
functions which are intended to work together to 
endcode and decode sensitive data to allow data 
and ML rules to be hosted off-site (or be cherry 
picked where needed as stand-alone functions 
depending on needs). 

Column names  and values will be mapped to masked
values and stored  in python dictionaries. These
dictionaries can be used to reverse the mapping 
when needed (after data is sent off site and 
returns on-site).

Essentially, sensitivity of the data will be
encoded in-memory, abstracting secrecy away from 
the data for use on a server. 


Functions (11)
--------------
[1] make_fake_data
[2] strings_to_cats
[3] cat_convert_all
[4] cat_to_code
[5] all_codes
[6] build_col_dict
[7] build_value_dict
[8] reverse_coded_values
[9] see_map
[10] pred_coded_row
[11] rev_dummies
---------------------------------------------------
"""
from pandas.api.types import (
    is_string_dtype, is_categorical_dtype)

from pandas import DataFrame, Series, get_dummies
from numpy import where
from typing import Tuple, Dict, List


def make_fake_data() -> DataFrame:
    """
    Generates a toy dataset of vertebrate data for
    demonstration purposes.

    Returns
    -------
    A Pandas dataframe with 8 independaant variables
    and a target called 'Class Label'.
    """
    cols = ['Vertebrate Name', 'Body Temperature', 'Skin Cover',
            'Gives Birth', 'Aquatic Creature', 'Aerial Creature',
            'Has Legs', 'Hibernates', 'Class Label']
    string_data = [
        'human python salmon whale frog komodo-dragon bat \
         pigeon cat leopard turtle penguin porcupine eel \
         salamander',
        'warm-blooded cold-blooded cold-blooded warm-blooded \
         cold-blooded cold-blooded warm-blooded warm-blooded \
         warm-blooded cold-blooded cold-blooded warm-blooded \
         warm-blooded cold-blooded cold-blooded',
        'hair scales scales hair none scales hair feathers \
         fur scales scales feathers quills scales none',
        'yes no no yes no no yes no yes yes no no yes no no',
        'no no yes yes semi no no no no yes semi semi no yes \
         semi',
        'no no no no no no yes yes no no no no no no no',
        'yes no no no yes yes yes yes yes no yes yes yes no yes',
        'no yes no no yes no yes no no no no no yes no yes',
        'mammal non-mammal non-mammal mammal non-mammal non-mammal \
         mammal non-mammal mammal non-mammal non-mammal non-mammal \
         mammal non-mammal non-mammal'
    ]
    # split into list of lists
    vals = [row.split() for row in string_data]
    # comprehend into a dict, pass to pd.DataFrame
    dat = {k: v for k, v in zip(cols, vals)}
    df = DataFrame(data=dat, columns=dat.keys())

    return df


def strings_to_cats(df, ordinal: bool = False):
    """
    Finds string columns in dataframe and converts
    them to categorical dtypes.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame

    Returns
    -------
    a Pandas dataframe with strings converted to
    categories.
    """
    if ordinal:
        for col, val in df.items():
            if is_string_dtype(val):
                df[col] = val.astype('category').cat.as_ordered()
    else:
        for col, val in df.items():
            if is_string_dtype(val):
                df[col] = val.astype('category')

    # or, more cryptically:
    # df.apply(lambda x: x.astype('category') if is_string_dtype(x) else x)


def cat_convert_all(df):
    """
    Converts all columns to categorical dtypes.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Assumes all columns are nominal strings.

    Returns
    -------
    a Pandas dataframe with strings converted to
        categories.
    """
    return df.apply(lambda x: x.astype('category'))


def cat_to_code(df):
    """
    Finds categorical columns in dataframe and converts
    their nominal values to categorical codes.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame

    Returns
    -------
    a Pandas dataframe where categorical dtypes
    have been converted to their categorical codes
    (numbers).
    """
    for col, val in df.items():
        if is_categorical_dtype(val):
            df[col] = df[col].cat.codes
    # or
    # df.applymap(lambda x: x.cat.codes if is_categorical_dtype(x) else x)


def all_codes(df):
    """
    Converts all columns to codes - assumes all column
    dtypes are categorical.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Assumes all columns categorical.

    Returns
    -------
    a Pandas dataframe where categorical dtypes
    have been converted to their categorical codes.
    """
    return df.apply(lambda x: x.cat.codes)


def build_col_dict(df) -> Tuple[Dict[str, str], 
                                Dict[str, str]]:
    """
    Builds a dictionary of column names (as they appear
    in the dataframe passed) as keys and masked column
    names (generated via enumerated format strings) as
    values. Also generates a reversed dictionary of this
    column dictionary for backwards mapping capabilities.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame

    Returns
    -------
    a tuple of dictionaries (col_dict, rev_col_dict).
        -  col_dict: dict where columns have been
            renamed based on their order in dataframe.
        -  rev_col_dict: dict mapping the keys and values
            of col_dict in reverse pairings.

    Examples
    --------
    Given the column name "Skin Cover", col_dict generates:

    >>>
    >>>

            #>  Skin Cover : Column_2

    rev_col_dict generates:

            #>  Column_2 : Skin Cover
    """
    col_dict = {}
    rev_col_dict = {}

    for num, val in enumerate(df.columns):
        col_dict[val] = f"Column_{num}"

    rev_col_dict = {v: k for k, v in col_dict.items()}

    return col_dict, rev_col_dict


def build_value_dict(df) -> Tuple[Dict[str, str], 
                                  Dict[str, str]]:
    """
    For each column, builds a sub dictionary of dataframe
    values (as they appear in the dataframe passed) mapped
    to an enumerated value as the nested values of a super
    dictionary where each key is a column name. For each
    nested value dict, builds a reversed nested dict within
    a seperate super dict as this seperate dicts values.
    This can be used to map values to codes, see the mapping
    for each column, and map the codes back to their orgional
    values.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame

    Returns
    -------
    a tuple of dictionaries (code_dict, rev_code_dict).
        - code_dict: each column mapped to a key, values = nested
          dict where all category codes mapped to their nominal
          string values.
        - rev_code_dict: code_dict with keys and values of nested
          value dicts reversed.

    Examples
    --------
    Given the following dataframe:

    >>>

          Vertebrate Name Body Temperature Skin Cover Gives Birth \\
    #> 1            human     warm-blooded       hair         yes \\
    #> 2           python     cold-blooded     scales          no \\
    #> 3           salmon     cold-blooded     scales          no \\
    #> 4            whale     warm-blooded       hair         yes \\
    #> 5             frog     cold-blooded       none          no \\
          Aquatic Creature Aerial Creature Has Legs Hibernates Class Label
    #> 1                no              no      yes         no      mammal
    #> 2                no              no       no        yes  non-mammal
    #> 3               yes              no       no         no  non-mammal
    #> 4               yes              no       no         no      mammal
    #> 5              semi              no      yes        yes  non-mammal

    Builds a tuple with the following two dicts:
        1. Below is ONE key (representing a column) of Code_dict:

    #> Skin Cover:
    #>      0 : feathers
    #>      1 : fur
    #>      2 : hair
    #>      3 : none
    #>      4 : quills
    #>      5 : scales


       2. Below is the same key (representing a column) of
           rev_code_dict:

    #> Skin Cover:
    #>      feathers : 0
    #>      fur : 1
    #>      hair : 2
    #>      none : 3
    #>      quills : 4
    #>      scales : 5
    """
    code_dict = {}
    rev_code_dict = {}
    cat_cols = []
    vals = []

    for col, val in df.items():
        if is_categorical_dtype(val):
            cat_cols.append(col)
            vals.append(dict(enumerate(df[col].cat.categories)))
    for col, val in zip(cat_cols, vals):
        code_dict[col] = val
    for key, dic in code_dict.items():
        dic = {v: k for k, v in dic.items()}
        rev_code_dict[key] = dic

    return code_dict, rev_code_dict


def reverse_coded_values(df, d: Dict[str, str]):
    """
    Takes a dataframe with values which have already been
    mapped to their categorical codes and a reference dictionary
    (generated from build_value_dict()) and uses the dictionary
    key-value pairs to map the keys of the dictionary
    (representing the encoded values of the dataframe passed)
    to the values of the reference dictionary (representing the
    original, pre-encoded, values of the passed dataframe).

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    d : Dictionary.
        A reference dictionary to map.

    Returns
    -------
    a Pandas dataframe with encoded column values mapped back to
    their original values.
    """
    cat_cols = []

    for col, val in df.items():
        if is_categorical_dtype(val):
            cat_cols.append(col)
    for col, dic in zip(cat_cols, d.values()):
        df[col] = df[col].map(dic)


def see_map(d, vals: bool = True):
    """
    Prints out an easy-to-read format of the key-value
    pairs of a passed dictionary. Assumes the dict
    passed is a column or value reference dict generated
    from build_col_dict() or build_value_dict().

    Parameters
    ----------
    d : a dictionary.
        A reference dictionary to view.
    vals : whether the dictionary is a value or column
        dictionary. Defaults to value dict (which contains
        a nested dict as values).

    Returns
    -------
    ~None.

    Notes
    -----
    Prints formated text for dictionary mapping.    """
    if vals:
        for key, val in d.items():
            print(f"{key}:\n")
            for i, v in val.items():
                print(f"\t{i} : {v}")
                print()
    else:
        for key, val in d.items():
            print(f"\t{key} : {val}\n")


def pred_coded_row(d: Dict[str, str], df) -> DataFrame:
    #TODO

    raise NotImplementedError
    #global cat_convert_all

    d = DataFrame(d); d[:] = 1
    d = cat_convert_all(d)
    col_dict, _ = build_col_dict(d)
    del _
    d = d.rename(columns=col_dict)
    r_coded = get_dummies(d)
    r_coded = r_coded.reindex(columns=df.columns, fill_value=0)

    return r_coded


def rev_dummies(df):
    """
    """
    raise NotImplementedError