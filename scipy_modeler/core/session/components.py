"""
---------------------------------------------------
Author: Nick Hopewell <nicholas.hopewell@gmail.com>

components.py  (contains : 3 classes)

Description
-----------
components.py contains the things that makeup the
tangible components of a scipy session. This includes
the data, the model, and the pipeline. 


Classes (3)
-----------
[1] SessionData
[2] SessionModel
[3] SessionPipeline


Notes
-----
Some of thesevclasses can be implemented as dataclasses 
but I chose not to do so as not everyone in the Python 
community implements data classes. I prefer the more 
verbose traditional class. 

Feb 26, 2020 - this is mainly signatures at the 
moment, its a work in progress to be complete.
----------------------------------------------------
"""
import warnings
import json, os
from typing import (
    Optional, Union, Tuple, Dict, List, Any)

import joblib 
from pandas import read_csv, DataFrame
import numpy as np

from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
        classification_report, confusion_matrix)

from scipy_modeler.util._settings_h import get_date_time
from scipy_modeler.core.model.prep import ( 
        split_data, stratified_shuffle_split, 
        split_most_recent, splits_to_csv
)
from scipy_modeler.core.pipeline.factory import PipelineFactory
from scipy_modeler.util._settings import (
        training_path, validation_path, 
        test_path, project_path, model_path,
        complete_data_path
)

class SessionData():
    """
    Used like a data class to store data, paths, 
    and the column transformer pipeline object. 
    Should not have methods which manipulate the
    actual data.


    Parameters
    ----------
    struc : Dict
        Columns (keys) and associated datatypes (values).
        Used for setting dataframe dtypes on read-in for
        data which has been segregated. 
    train_path : string
        File path to location of training data
    validate_path : string
        File path to location of validation data
    testing_path : string 
        File path to location of testing data

    Notes
    -----
    The main goal of this class ought to be to assert
    expected setter behaviour.
    """
    def __init__(self, struc : Dict[str, str],
                complete_path : str, 
                train_path: str, 
                validate_path: str, 
                testing_path: str ):
        # columns and dtypes dict
        self._struc: Dict[str, str] = struc
        # paths to segregated data
        self._complete_path: str = complete_path
        self._train_path: str = train_path
        self._validate_path: str = validate_path
        self._testing_path: str = testing_path
        # complete data to retrain on once final model selected
        self.X_complete = None
        self.y_complete = None
        self.X_complete_transformed = None
        # segregated training data (IVs, DV, IVs transformed)
        self.X_train = None
        self.y_train = None
        self.X_train_transformed = None
        # segregated validation data (IVs, DV, IVs transformed)
        self.X_valid = None
        self.y_valid = None
        # segregated test data (IVs, DV, IVs transformed)
        self.X_test = None
        self.y_test = None
        
        # class label for inference 
        self.label: str = None

    # getters and setters for struc and paths
    @property
    def struc(self) -> Dict[str, str]:
        """Returns table structure."""
        return self._struc
    @struc.setter
    def struc(self, new: Dict[str, str]):
        """
        Notes
        -----
        This must assert the user behaviour or the
        entire process will be flawed. 

        Raises
        ------
        TypeError
            - Passed any data type except for a dictionary.
            - Passed a dictionary with keys of any type except
               string. 
            - Passed a dictionary with values of any type except
               string.
        """
        # user must pass a dict
        assert isinstance(new, dict), TypeError(
            "Must pass a dictionary to struc in order to set "
            "data types.")
        # dict keys and values must be strings
        assert all( 
            isinstance(i, str) for i in list(new.keys()) 
            ), TypeError(
                "Keys of dictionary passed must be strings.")
        assert all( 
            isinstance(i, str) for i in list(new.values()) 
            ), TypeError(
                "Values of dictionary passed must be strings.")
        self._struc = new

    @property
    def complete_path(self) -> str:
        """Returns path to complete (non-segregated) data"""
        return self._complete_path
    @complete_path.setter
    def complete_path(self, new: str):
        """Sets path to complete (non-segregated) data"""
        self._complete_path = new

    @property
    def train_path(self) -> str:
        """Returns training data path."""
        return self._train_path
    @train_path.setter
    def train_path(self, new: str):
        """Sets training data path."""
        self._train_path = new
        
    @property
    def validate_path(self) -> str:
        """Returns validation data path."""
        return self._validate_path
    @validate_path.setter
    def validate_path(self, new: str):
        """Sets validation data path."""
        self._validate_path = new
        
    @property
    def testing_path(self) -> str:
        """Returns testing data path."""
        return self._testing_path
    @testing_path.setter
    def testing_path(self, new: str):
        """Sets testing data path."""
        self._testing_path = new

class SessionModel():
    """
    Used like a data class to store information
    about the model such as the algorithm,
    hyperparameters, a path to a file where it
    is stored as a binary object, and the modelling
    artifact (the model object itself).
    """
    def __init__(self, hyperparameters: 
                        Optional[Dict[str, Any]]=None, 
                path: Optional[str]=None):

        self._hyperparameters: Dict[str, Any] = hyperparameters
        self._path: str = path
        # algorithm = string representing algo name. eg "KNN"
        self._algorithm = None
        # artifact = trained sklearn model object        
        self._artifact = None 

    @property
    def algorithm(self):
        """Return classification or regression algo."""
        return self._algorithm
    @algorithm.setter
    def algorithm(self, new):
        """Set classification or regression algo."""
        self._algorithm = new

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return self._hyperparameters
    @hyperparameters.setter
    def hyperparameters(self, new: Dict[str, Any]):
        """Sets model hyperparameters."""
        assert isinstance(new, dict), TypeError(
            "Must pass a dictionary of hyperparameters.")
        self._hyperparameters = new

    @property
    def artifact(self):
        """Return model artifact."""
        return self._artifact
    @artifact.setter
    def artifact(self, new):
        """Sets model artifact."""
        self._artifact = new

    @property
    def path(self) -> str:
        """Return path where model artifact 
        is stored as binary."""
        return self._path
    @path.setter
    def path(self, new: str):
        """Sets path where model artifact 
        is stored as binary."""
        self._path = new

class SessionPipeline():
    """
    NOTE:  THIS OUGHT TO INHERIT AND EXTEND THE 
    SKL PIPELINE OBJECT  !!!


    models_path : string
        Path to binary model files.
    """
    def __init__(self, models_path: Optional[str]=model_path, 
                transformer: ColumnTransformer = None):
        
        self._complete_pipeline: Pipeline = None
        self.transformer: ColumnTransformer = transformer
        self.model = SessionModel(path=models_path)

    @property
    def complete_pipeline(self):
        """ """
        return self._complete_pipeline
    
    @complete_pipeline.setter
    def complete_pipeline(self, new: Pipeline):
        """ """
        self.complete_pipeline: Pipeline = new

    def all_steps(self):
        """ """
        return self.complete_pipeline.steps

    def build_pipeline(self):
        """
        Put the transformation and model together
        into a pipeline
        """
        raise NotImplementedError

    def persist(self):
        """
        """
        raise NotImplementedError
    
