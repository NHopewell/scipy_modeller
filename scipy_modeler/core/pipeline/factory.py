"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

facotry.py  (contains : 2 classes)

Description
-----------
factory.py implements a pipeline factory - a way to
dynamically generate appropriate transformation 
pipelines on the fly based on specification sent
from the pipeline manager.

Classes (2)
-----------
[1] SciLabelEncoder
[2] PipelineFactory


Notes
-----
Ensuring the proper creation and execution of 
valid transformation pipelines is as important as 
valid data preprocessing for ensuring accurate
results during training, testing, and validation of
any model. This script ought to be tested to complete
coverage.

SciLabel encoder is not implemented yet, but the 
idead is to implement a custom transformer 
(such as those used for generating flags) if 
it is decided that the dependant variable requires
any special handling (beyond simply ordinal 
endcoding).
---------------------------------------------------
"""
from typing import Optional, Union, List, Tuple, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler,
    LabelEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from pandas.api.types import (
    is_numeric_dtype, is_categorical_dtype, 
    is_string_dtype)
     

class SciLabelEncoder(BaseEstimator, TransformerMixin):
    """


    """
    def __init__(self):
        raise NotImplementedError

        self.encoder = LabelEncoder()

    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=None):
        return self.encoder.transform(x)


class PipelineFactory():
    """
    Dyanamically constructs a unique column transformer
    pipeline (containing several smaller pipelines and 
    column transformers). This factory may be extended to
    incorporate custom transformers writen for each line
    of business. These pipelines dictate how data are
    transformed before fitting a statistically-learned
    classifier or regressor.
    
    Parameters
    ----------
    encoder : str
    cat_imputer_strat : str
    cat_imputer_val : Any
    num_imputer_strat : str, optional
    num_imputer_val : float, optional
    scaler : str, optional
    matrix_sparsity_threshold : float, optional
    ignore_cols : List[str] or str, optional
    
    Attributes
    ----------
    encoder : 
    cat_imputer : 
    scaler : 
    num_imputer : 
    transformer :
    + bindings for other params in signature
        
    Notes
    ------
    Does not return a numpy array-like object.
    Categorial pipelines will produce a sparse 
    matrix object (only positions of non-zero values 
    stored) to conserve memory, while numeric pipelines 
    produce a dense matrix. When combining these 
    pipelines into one column transformer pipeline, 
    the density of the final matrix will be  estimated 
    (the ratio of non-zero cells) and it returns a 
    sparse matrix if the estimated density is lower 
    than a default  threshold of 0.3, otherwise, it 
    will produce a dense martrix object. 
    """
    def __init__(self, encoder: str = 'onehot',
                 cat_imputer_strat: str = 'constant',
                 cat_imputer_val: Any = 'missing',
                 num_imputer_strat: Optional[str] = None,
                 num_imputer_val: Optional[float] = None,
                 scaler: Optional[str] = None,
                 matrix_sparsity_threshold: float = 0.3, 
                 ignore_cols: Optional[Union[List[str], str]] = None):
        
        
        self.encoder = OneHotEncoder(handle_unknown='ignore')  #MAKE SURE YOU WANT THIS
        self.matrix_sparsity_threshold = matrix_sparsity_threshold
        
        if cat_imputer_strat:
            assert cat_imputer_strat in (
                'most_frequent', 'constant'), ValueError(
                    "Imputation strategy must be one of 'mean', "
                    "'median', or 'most_frequent'.")
            
            if cat_imputer_strat == 'constant':
                assert bool(cat_imputer_val), ValueError(
                    "Must pass a value to fill missing categoricals with.")
                
                self.cat_imputer = SimpleImputer(strategy=cat_imputer_strat, 
                                                 fill_value=cat_imputer_val)
            else:
                self.cat_imputer = SimpleImputer(strategy=cat_imputer_strat)
        else:
            self.cat_imputer = None
        
        if scaler: 
            assert scaler in ('standard', 'min-max'), ValueError(
                "Numeric scaler must be either standard scaling or "
                "min-max scaling.")
                
            if scaler == 'standard':
                self.scaler = StandardScaler()
            else: 
                self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        if num_imputer_strat:
            assert num_imputer_strat in (
                'mean', 'median','most_frequent', 'constant'), ValueError(
                "Imputation strategy must be one of 'mean', "
                           "'median', or 'most_frequent'.")

            self.num_imputer_strat = num_imputer_strat
                
            if self.num_imputer_strat == 'constant':
                assert bool(num_imputer_val), ValueError(
                    "Must pass a value to fill missing numerics with.")
                
                self.num_imputer = SimpleImputer(strategy=num_imputer_strat, 
                                                 fill_value=num_imputer_val)
            else:
                self.num_imputer = SimpleImputer(strategy=num_imputer_strat)
        else:
            self.num_imputer = None
        if ignore_cols:
            assert isinstance(ignore_cols, (list, str)), \
                TypeError("'ignore_cols' must be past a list of columns to "
                          "ignore or a single column (as a string) to ignore.")
            if not isinstance(ignore_cols, list):
                self.ignore_cols = list(ignore_cols)
            else:
                self.ignore_cols = ignore_cols
        else:
            self.ignore_cols = None

        self.transformer = None

    def _parse_types(self, Table, label: str) -> List[str]:
        """
        """
        cats = []
        nums = []
        if self.ignore_cols:
            for k,v in Table.items():
                if ( (is_categorical_dtype(v) or is_string_dtype(v)) 
                        and (k not in self.ignore_cols) ): 
                    cats.append(k)
            if self.num_imputer:
                for k,v in Table.items():
                    if (is_numeric_dtype(v)) and (k not in self.ignore_cols):
                        nums.append(k)
        else:
            for k,v in Table.items():
                if is_categorical_dtype(v) or is_string_dtype(v): 
                    cats.append(k)
            if self.num_imputer:
                for k,v in Table.items():
                    if is_numeric_dtype(v):
                        nums.append(k)            
        try:
            cats.remove(label)
        except ValueError:
            pass

        return cats, nums

    def _build_cat_pipeline(self) -> Pipeline:
        """
        """

        if self.cat_imputer:
            categorical_steps = [
                ('imputer', self.cat_imputer),
                ('ohe', self.encoder),
            ]
        else:
            categorical_steps = [
                ('ohe', self.encoder)
            ]
            
        categorical_pipeline = Pipeline(categorical_steps)
        
        return categorical_pipeline

    def _build_num_pipeline(self) -> Pipeline:
        """
        """
        numeric_steps = [
                ('imputer', self.num_imputer),
                ('ss', self.scaler),
            ]
        numeric_pipeline = Pipeline(numeric_steps)

        return numeric_pipeline

    def _create_transformer_steps(self, how: Optional[str] = 'cats_only',
                    cat_pipeline: Optional[Pipeline] = None, 
                    num_pipeline: Optional[Pipeline] = None,
                    cat_columns: Optional[List[str]] = None,
                    num_columns: Optional[List[str]] = None
                ) -> List[Tuple[str, Pipeline, List[str]]]:

        assert how in ("cats_only", "nums_only", "both"), \
            ValueError(
                '"how" must be one of: "cats_only", "nums_only", "both"')
        
        if how == "cats_only":
            transformers = [
                ('cats', cat_pipeline, cat_columns),
            ]
        elif how == "nums_only":
            transformers = [
                ('nums', num_pipeline, num_columns),
            ]
        else:
            transformers = [
                ('cats', cat_pipeline, cat_columns),
                ('nums', num_pipeline, num_columns),
            ]

        return transformers

    def _build_column_transformer(self, 
                    transformers: List[Tuple[str, Pipeline, List[str]]],
                    remainder: Optional[str] ='passthrough'
                        ) -> ColumnTransformer:
        """
        """
        col_trans = ColumnTransformer(
                transformers, 
                remainder=remainder,
                sparse_threshold=self.matrix_sparsity_threshold
            )
        return col_trans

    def build_transformer_pipeline(self, Table, label: str):
        """
        Builds a series of transformations as a pipeline
        of column transformers
        """
        cats, nums = self._parse_types(Table, label)

        categorical_pipeline =  self._build_cat_pipeline()

        if self.num_imputer:
            numeric_pipeline =  self._build_num_pipeline()

            transformers = self._create_transformer_steps(
                how='both', 
                cat_pipeline=categorical_pipeline,
                num_pipeline=numeric_pipeline,
                cat_columns=cats,
                num_columns=nums
                            )
        else:
            transformers = self._create_transformer_steps(
                how='cats_only', 
                cat_pipeline=categorical_pipeline,                        
                cat_columns=cats
                            )
        # ColumnTransformer expects a list of tuples(name, transformer, columns to apply to)
        self.transformer = self._build_column_transformer(transformers, 
                                remainder='passthrough')
        
        return self.transformer
    

"""
class ModelPipeline():
    
    # TODO: make this work for many models 
    
    def __init__(self, json):
        args = ['transformer_pipeline', 'model', 'paramgrid',
                'stratified', 'crossval_folds', 'n_jobs',
                'eval_best_estimators', 'persist_best']
        assert all(i in args for i in json.keys()), \
            ValueError("Did not provide proper key-value pairs to "
                       "PipelineGenerator.")
        
        self.transformer_pipeline=json['transformer_pipeline']
        self.model=json['model']
        self.paramgrid=json['paramgrid']
        self.stratified=json['stratified']
        self.crossval_folds=json['crossval_folds']
        self.n_jobs=json['n_jobs']
        self.eval_best_estimators=json['eval_best_estimators']
        self.persist_best=json['persist_best']
        

        self.best_estimators=[]
        
        self.gen_model_pipe()
        
    def gen_model_pipe(self):
        model_pipe = Pipeline([('transformer', self.transformer_pipeline),
                               ('model', self.model)])
        grid_search = GridSearchCV(model_pipe, self.paramgrid, 
                                   self.crossval_folds)
        
        self.model_pipe=model_pipe
        self.grid_search=grid_search
        
    
    def fit(self):
        
        if self.eval_best_estimators:
            raise NotImplementedError
        
        if self.persist_best:
            raise NotImplementedError
            
        raise NotImplementedError
        
    def best_estimator_validation_params(self):
        raise NotImplementedError
        
    def best_estimator_validation_performance(self):
        raise NotImplementedError
    
    def plot_best_estimator_validation_performance(self):
        raise NotImplementedError
        
    def all_estimators_validation_performance(self):
        raise NotImplementedError
        
    def plot_all_estimator_validation_performance(self):
        raise NotImplementedError
        
    def eval_best_estimators_on_test(self):
        raise NotImplementedError
        
    def final_test_performance(self):
        raise NotImplementedError
        
    def plot_final_test_performance(self):
        raise NotImplementedError
        
    def persist_best_fitting_model(self):
        raise NotImplementedError
        
    def model_file_location(self):
        raise NotImplementedError
    
"""   

        
        
    
    
    
    
    
    
    
    
