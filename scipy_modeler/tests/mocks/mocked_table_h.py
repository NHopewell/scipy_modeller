from typing import Union, Optional, List, Dict, Tuple, Any
import pandas as pd

def get_table_fields(schema: str, table: str, 
                sort: Optional[bool]=False) -> List[str]:
    pass

def get_datetimes(cols: List[str],
                  passthrough: List[str],
                  endswith: Optional[str]='_DT') -> List[str]:
    pass

def _flatten_2dlist(nested_list: List[List[str]]) -> List[str]:
    pass

#@assert_types(str, str, list, list, list, list, list)
#@assert_types(str, str, Optional[List[str]], Optional[List[str]], Optional[List[str]], Optional[List[str]], Optional[List[str]], Optional[str])
def generate_struc(schema: str, 
                   table: str, 
                   filter_out: Optional[List[str]] = [], 
                   str_cols: Optional[List[str]] = [], 
                   int_cols: Optional[List[str]] = [], 
                   float_cols: Optional[List[str]] = [], 
                   extra_date_cols: Optional[List[str]] = [],
                   dt_naming_convention: Optional[str] = '_DT'
                ) -> Dict[str, str]:
    pass
    
def _flag_switch(df: pd.DataFrame,  ref: str, 
                 val: Union[str,int,float], check: str,  
                 flag: int, fill: int, case: bool = True, 
                 regex: bool = False) -> pd.Series:
    pass

def _str_strip_all(df: pd.DataFrame, 
                   case: Optional[str] = None):
    pass

def _str_case_strip(df: pd.DataFrame, cols: List[str], 
                    case: Optional[str] = None) -> pd.DataFrame:
    pass

def _pandas_from_sql(db_schema: str, 
                     table_name: str, 
                     columns: str,
                     dates: Optional[List[str]] = None, 
                     na_vals: Optional[Union[List[Any],Any]] = None, 
                     label: Optional[str] = None, 
                     date_format: Optional[str] = None, 
                     struc: Optional[dict]= None,  
                     chunksize: Optional[int] = None) -> pd.DataFrame:
    # Returns SP_INDIA_HIST.csv
    my_path = 'C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller'
    sp_india_hist_path = my_path + "\\scipy_modeler\\tests\\test_case_data\\SP_INDIA_HIST.csv"
    sp_india_hist_columns = ["UCI", "APPLICATION_TYPE", "APPLICATION_RECEIVED_DT", "FINAL_DECISION", "FINAL_DECISION_DT", "APPLICATION_NO", "APPLICATION_STATUS", "SOURCE", "PERMIT_START_DT", "PERMIT_EXPIRY_DT","VALID_TO_ON_COUNTERFOIL","VALID_FROM_ON_COUNTERFOIL","SUB_CATEGORY"]
    sp_india_hist = pd.read_csv(sp_india_hist_path, names=sp_india_hist_columns)
    return sp_india_hist

def _gen_fquery(filters: List[Tuple[str, str]], op: str) -> str:
    pass

def _print_pretty(data: Union[dict, list, tuple], 
                  size: Optional[int] = 1):
    pass
        
def permutate_data(df, n: int) -> pd.DataFrame:
    pass

def _make_categorical_numeric(df : pd.DataFrame,
                    col: Union[str, List[str]], 
                    max_levels: Optional[int] = None, 
                    inplace: Optional[bool] = False):
    pass