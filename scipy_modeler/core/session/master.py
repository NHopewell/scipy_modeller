"""
---------------------------------------------------
Author: Nick Hopewell <nicholas.hopewell@gmail.com>

master.py (contains : 1 class)

Description
-----------
master.py exposes methods neccessary for automating 
data transformations (after data has been 
preprocessed via the core API), the construction 
of data transformation pipelines (that are sensitive 
to data types), data segregation and resampling, 
model training, validation, persistence, and model
evaluation via inference on untransformed test
data. Transformation pipelines are built on the
training set and applied on validation and test
sets before machine learning algorithms are
fit to the data. 

Learned models are persisted to disk as a bin
file which, when called on for inference on unseen
data, are read back into RAM as a Python object. 
* see notes about future bin dumping.


Classes (1)
-----------
[1] ScipySession  


Notes
-----
For future production, this model and pipeline 
persistence should be done into an AWS S3 bin or 
something similar. The bin files are only dropped to
disk for local production testing of the pipeline.
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
from scipy_modeler.core.session.components import (
    SessionData, SessionModel, SessionPipeline
)
from scipy_modeler.core.pipeline.manager import PipelineManager
from scipy_modeler.core.model.prep import ( 
    split_data, stratified_shuffle_split, 
    split_most_recent, splits_to_csv
)

from scipy_modeler.util._settings import (
    training_path, validation_path, 
    test_path, project_path, model_path,
    complete_data_path
)
    

class SciPySession():
    """
    Controls a "modelling session" the user can start
    after data exploration and preprocessing when
    data needs to be split, transformed, modelled, 
    and used for validation and test inference. 

    Unique sessions are identified with a session name.
    Each sessions' settings are stored in json files to
    load the sessions back again later or  in a new 
    notebook or workspace after splitting data and 
    storing their associated paths. 

    Trained models are dumped to binary files and 
    unpicked with joblib during model inference on
    test data. 


    Parameters
    ----------
    session_name : string
        A unique name the user gives to the session.
    train_path : string
        Path to segregated training data.
    validate_path : string
        Path to segregated validation data. 
    testing_path : string
        Path to segregated testing data. 
    jobs_ran : list, optional
        Collection of __name__(s) of jobs ran.
        Default = None.
    sessions_datetime : string, optional.
        String representing sessions creation which
        can be parsed to datetime.
        Default = None.
    

    Notes
    -----
    To extend a sessions, first save_session() and
    then load_session() in new workspace/notebook. 
    """
    
    # TODO: handle this different 
    # path to project main folder
    _project_path=project_path
    
    def __init__(self, session_name: str, 
                struc: Dict[str, str], 
                complete_path: Optional[str]=complete_data_path,
                train_path: Optional[str]=training_path, 
                validate_path: Optional[str]=validation_path, 
                testing_path: Optional[str]=test_path, 
                models_path: Optional[str]=model_path,
                jobs_ran: Optional[List[str]]=None,
                session_datetime=None):
        
        self.session_name = session_name
        # instantiate SessionData with struc/paths, SessionModel with model path 
        self.data = SessionData(struc, complete_path, train_path, 
                                validate_path, testing_path)
        self.pipeline = SessionPipeline(models_path=models_path)

        # if not passed by user, initialize as empty list
        if not jobs_ran:
            self._jobs_ran=[]
        else:
            self._jobs_ran=jobs_ran
        self.session_datetime=session_datetime

    @classmethod
    def build_session(cls, session_name):
        """
        Return a new SciPySession with the settings
        defined in a past session (identified with
        the session name). 


        Parameters
        ----------
        session_name : string
            The name of the session to build back
            into memory.


        Raises
        ------
        FileNotFoundError
            - path to session not found. 
        """
        try:
            # open the session folder, load the json file into a dict
            with open(f"{SciPySession._project_path}{session_name}", 'r') as f:
                session_details: dict = json.load(f) 
            return cls(**session_details) # **kwargs to instantiate new session
        except FileNotFoundError:
            msg = ("A session with that name does not exist. "
                   "Check that the name is correct and remember "
                   "to save your previous session before trying "
                   "to build it again.")
            raise FileNotFoundError(msg)
    
    def save_session(self, path=None):
        """
        Saves the current session as a json file. 


        Parameters
        ----------
        path : string, optional
            Path to dump the sessions settings as json.
            If not specified, will defaul to the main
            project folder + the session name. 
            Default = None.

        
        Raises
        ------
        FileNotFoundError
            - path specified does not exist. 


        Notes
        -----
        The session name will be used as an identifier. 
        Load the session back with 'build_session()'
        """
        if not path:
            # deafult to project path + session name
            path = f"{SciPySession._project_path}{self.session_name}"
        try:
            with open(path, 'w') as f:
                # append job ran, date session was saved.
                today, today_time = get_date_time()
                self.jobs_ran.append(self.save_session.__name__)
                self.session_datetime: str = f"{today}{today_time}"
                # dump settings to specified path
                json.dump({'session_name': self.session_name,
                           'struc': self.data.struc,
                           'complete_path': self.data.complete_path,
                           'train_path': self.data.train_path,
                           'validate_path': self.data.validate_path,
                           'testing_path': self.data.testing_path,
                           'models_path': self.pipeline.model.path,
                           'jobs_ran': self.jobs_ran,
                           'session_datetime': self.session_datetime}, f)
        except FileNotFoundError:
            msg = ("Path to file not found. Check path is correct "
                   " and remember that if the path is unspecified "
                   "the default project main path will be used.")
            raise FileNotFoundError(msg)


    def flush_session(self):
        """
        Flush all files (inclusing the session settings file)
        from their respective storage folders. 
        """
        except_msg = "file does not exist."
        path_names = ['complete', 'training', 'testing', 
                      'validation', 'model']

        paths = [
            self.data.complete_path, self.data.train_path, 
            self.data.testing_path, self.data.validate_path, 
            self.pipeline.model.path
            ]
        # zip into a collection of tuples
        for path, name in zip(paths, path_names):
            if os.path.exists(path):
                os.remove(path) # remove file
            else:
                warnings.warn(f"{name.title()} {except_msg}.",
                    UserWarning)
        try:
            # remove session json file
            os.remove(f"{SciPySession._project_path}{self.session_name}")
        except FileNotFoundError:
            raise FileNotFoundError("Could not find session file to remove.")
        finally:
            # update jobs ran
            self.jobs_ran.append(self.flush_session.__name__)

        
    def load_training_data(self, path: Optional[str]=None) -> DataFrame:
        """
        Reads training data in csv format from training
        data path and returns a pandas dataframe. 
        If a path is not specified, will use the default
        training path.


        Returns
        -------
        Pandas dataframe.
        """
        if not path:
            # use stored path
            path = self.data.train_path

        return read_csv(path, dtype=self.data.struc)  
    
    @property
    def jobs_ran(self) -> List[str]:
        """
        Returns a list of all jobs which have been
        run under the current session name.
        """
        return self._jobs_ran

    @jobs_ran.setter
    def jobs_ran(self, new: List[str]):
        """Set jobs ran."""
        self._jobs_ran = new
    
    def segregate_data(self, table: DataFrame, 
                      percent_train: float, 
                      method:str = 'random', 
                      strat_col:Optional[str] = None,
                      random_state:Optional[int] = None):
        """
        Splits data into train, validation, and tests sets.
        Updates the respective paths with the current date
        and time of segreation (to ensure a unique path in 
        cases where data is split multiple times). Finally, 
        each segment of the data is writen to its respective
        path as a csv file.


        Parameters
        ----------
        table : pandas.core.frame.DataFrame
            A pandas dataframe you would like to segregate
            for modelling.
        percent_train : float
            The percentage of the complete data set to 
            use for training. The remaining percentage
            not used for training will be split equally 
            between validation and training sets.
            e.g.: specifying percent_train = 0.7 (70%)
                  will result in 0.15 and 0.15 (15%)
                  for validation and testing respectively. 
        method : {'random', 'stratified', 'test_most_recent'}, 
                    optional.
            Method to use for segregating the data 
            (random permutations or a stratified sample).
            If a stratified sample is chosen, must specify
            a column to use for stratified sampling with 
            the parameter 'strat_col'. Default = 'random'
        strat_col : string, optional. 
            The column name used for stratified sampling.
            Must be specified if a stratified sample is
            desired. Default = None.
        random_state : int, optional.
            A random state to use for replication of
            results. Default = None.


        Notes
        -----
        Each path is used to read segregated data into
        memory when the session is built again. 
        """
        assert method.lower() in ('random', 'stratified', 'test_most_recent'), \
            ValueError("""Splitting method must be one of 'random', """
                       """'stratified', of 'test_most_recent'""")
        
        # all data to retrain on when optimal model is found
        complete = table.data
        # segregated data
        if method.lower() == 'random':
            # permute data with given percent train
            train, validation, test = split_data(table.data, 
                    percent_train=percent_train, random_state=random_state)
        elif method.lower == 'stratified':
            msg = ("""When performing a stratified shuffle split, """
                   """must pass a column to use for stratified sampling.""")
            # assert user passed a column to use for strat sampling
            assert strat_col, ValueError(msg)
            train, validation, test = stratified_shuffle_split(table.data, strat_col,
                     percent_train=percent_train, random_state=random_state)
        else: 
            # keep most recent data out for testing
            train, validation, test = split_most_recent(table.data, 
                    percent_train=percent_train, random_state=random_state)

        today, today_time = get_date_time()
        # update paths 
        self.data.complete_path = f"{self.data.complete_path}{today}{today_time}.csv"
        self.data.train_path = f"{self.data.train_path}{today}{today_time}.csv"
        self.data.validate_path = f"{self.data.validate_path}{today}{today_time}.csv"
        self.data.testing_path = f"{self.data.testing_path}{today}{today_time}.csv"
        
        paths = ( self.data.complete_path, self.data.train_path, 
                  self.data.validate_path, self.data.testing_path )
        # write each path to csv
        for split, path in zip( (complete, train, validation, test), paths ):
            splits_to_csv(split, path)

        self.jobs_ran.append(self.segregate_data.__name__)
        
    
    def run_transformation_job(self, transform_params: Dict[str, Any]):
        """
        Fits a transformer to the training data (specified
        in the input data path) without the label. Then, 
        a transformation pipeline is build with the steps
        specified in the transform_params. This pipeline then
        transforms the training data and stores the transformed
        data as well as the transformation pipeline to use
        for validation and testing transformations. 

        Parameters
        ----------
        transform_params : dict
            A dictionary specifying the parameters used
            for transforming the training data. These parameters
            must fit the SciKit learn API. 
        """
        # instantiate a new manager 
        manager = PipelineManager()
        # get manager to order a pipeline and return it
        pipeline_factory = manager.order_new_factory(
            factory_specifications = transform_params)
        # read in training data
        dat = read_csv(transform_params["Input_data_config"]["path"],
                       dtype=self.data.struc)
        # store label
        self.data.label = transform_params["Input_data_config"]["label"]
        # seperate DVs from IV
        self.data.X_train = dat.drop(self.data.label, axis=1)
        self.data.y_train = dat[self.data.label].copy()
        # build and store a transformer pipeline
        self.pipeline.transformer = pipeline_factory.build_transformer_pipeline(
            self.data.X_train, label=None)

        # transform training data and store prepared data  
        prepared =  self.pipeline.transformer.fit_transform(self.data.X_train)
        self.data.X_train_transformed = prepared

        self.jobs_ran.append(self.run_transformation_job.__name__)

 
    def run_training_job(self, training_job_params: Dict[str, Any], 
                        deploy: bool = False):
        """
        Train a model based on a dictionary of training
        parameters. This dictionary of parameters contains
        the desired classification or regression algorithm, 
        its associated hyperparameters (including only thiose
        the user wants to set, leaving the remainder as 
        SKLearn defaults), and the path to the data the
        user wishes to train on. 


        Parameters
        ----------
        training_job_params : dict
            A dictionary specifying the parameters used
            for model training. These parameters
            must fit the SciKit learn API. The 
            hyperparameters passed must be associated 
            with the algorithm of choice. For instance, 
            the user cannot specifiy a 'max_depth' when
            fitting a logistic regression model to the 
            transformed training data. 
        deploy : bool, optional
            Whether or not to deploy the model object to a 
            binary file after training is complete. 
            Default = False.


        Notes
        -----
        - A transformation job must be run on the training data
          before a training job is ran (if not, SKlearn will 
          throw errors). 
        - The input data "path" can simply be the transformed 
          data stored in the current session. Otherwise, 
          the transformed training data must be written to
          a csv file.
            e.g.: 
            "Input_data_config":
                    {"path": xyz_session.transformed_data}
        - if 'deply_model' is set to equal True, the model
          induced from the transformed training data will be
          written to a binary file which can be reused for 
          inference on new data at a future time. 
        """
        # i.e.: "CART"
        self.pipeline.model.algorithm = training_job_params["Algorithm"]
        
        if self.pipeline.model.algorithm == 'CART':
            # instantiate a sklearn tree with hyperparams unpacked as kwargs -> fit tree.
            model = tree.DecisionTreeClassifier(
                **training_job_params["Hyperparameters"]
                )
            model = model.fit(
                training_job_params["Input_data_config"]["path"], self.data.y_train
                )
        # save model    
        self.pipeline.model.artifact = model

        if deploy: 
            self.deploy_model(complete_retrain=True)
    
    def retrain_on_all_data(self, deploy: bool=False):
        """
        Once optimal (or simply desirable) hyperparameters
        have been determined, retrain the model on the
        entire dataset (including the segregated train, 
        test, and validation data). 


        Parameters
        ----------
        deploy : bool, optional
            Whether or not to deploy the model object to a 
            binary file after training is complete. 
            Default = False.
        
        
        Notes
        -----
        This step should almost always be taken after optimal 
        parameters have been found and before launching a 
        model into production. 
        """
        dat = read_csv(self.data.complete_path,
                       dtype=self.data.struc)

        self.data.X_complete = dat.drop(self.data.label, axis=1)
        self.data.y_complete = dat[self.data.label].copy()

        # transform training data and store prepared data  
        complete_prepared = self.pipeline.transformer.fit_transform(
            self.data.X_complete)
        self.data.X_complete_transformed = complete_prepared
        
        if self.pipeline.model.algorithm == 'CART':
            # instantiate a sklearn tree with hyperparams unpacked as kwargs
            model = tree.DecisionTreeClassifier(
                self.pipeline.model.artifact.get_params()
                )
            model = model.fit(
                self.data.X_complete_transformed, self.data.y_complete
                )
        # save final model    
        self.pipeline.model.artifact = model

        if deploy: 
            self.deploy_model()

        self.jobs_ran.append(self.retrain_on_all_data.__name__)

    
    def deploy_model(self, complete_retrain: Optional[bool]=True):
        """
        Deploy a model by dumping it as a binary object
        to a file for later use during inference. 


        Parameters
        ----------
        complete_retrain: bool, optional 
            Whether the current model artifact should
            be retrained (using its stored hyperparameters)
            on the complete data set (including those data
            used for training, validation, and testing).


        Notes
        -----
        It is imperative that the model deployed has been 
        trained on the complete data set unless there is 
        some valid justification to do otherwise or this 
        retraining process was done manually. 
        """
        if complete_retrain:
            self.retrain_on_all_data()

        # dump the trained model to a binary file to unpickle 
        # later for inference uses
        filename = os.path.join(self.pipeline.model.path, 
                        "testmodel.joblib")  #TODO make this work somehow !!!!!!!!!!!!!!!!!!!!
        joblib.dump(self.pipeline.model.artifact, 
                    f"{self.session_name}")

    def deploy_pipeline(self):
        """
        """
        pass
    
    def best_estimator(self):
        # retrain model with best params on complete data, stores it
        # should be a property
        pass
    
    def deploy_best_estimator(self):
        pass
    
    def validate_model(self, how: Optional[str]="confusion_matrix"):
        """
        Validate the model induced from the training set 
        on the validation set. 


        Parameters
        ----------
        how : {"confusion_matrix", "classification_report"},
            optional How the model performance on the 
            validation set will be quantified. 
            Default = "confusion_matrix".        


        Notes
        -----
        A significant drop in performance of the model
        between training and validation sets indicates
        possible overfitting of the training data. The
        model which is persisted for production should
        have its hyperparameters determined based on 
        performance on the validation set. Overfitting
        the validation set is also a risk if too many,
        too specific, iterations of hyperparameter tuning
        are complete. Hence, before productionalizing a 
        model, it is beneficial to test its true 
        performance on the testing set (along with,
        potentially, a couple other configurations of 
        parameters which performed very well on the 
        validation set).           
        """
        assert how in ("confusion_matrix", "classification_report"), \
            ValueError("Model performance can only be quantified "
                       "as either a confusion matrix or a "
                       "classification report.")
        # read in validation data, split into IVs and DV
        valid_dat = read_csv(self.data.validate_path, dtype=self.data.struc)
        self.data.X_valid = valid_dat.drop(self.data.label, axis=1)
        self.data.y_valid = valid_dat[self.data.label].copy()
        
        # transform validation data 
        self.data.X_valid_prepared = self.pipeline.transformer.transform(
            self.data.X_valid)
        # make predictions on transformed validation data
        predictions: np.array = self.pipeline.model.artifact.predict(
            self.data.X_valid_prepared)
        # return performance (perdictions vs real labels)
        if how == "confusion_matrix":
            return confusion_matrix(self.data.y_valid, predictions)
        elif how == "classification_report":
            return classification_report(self.data.y_valid, predictions)
        else:
            raise NotImplementedError
        
   
    def test_model(self, how: Optional[str]="confusion_matrix"):
        """
        Test the model induced from the training set 
        on the testing set. 


        Parameters
        ----------
        how : {"confusion_matrix", "classification_report"}, optional
            How the model performance on the testing
            set will be quantified. Default = "confusion_matrix".        
        """
        assert how in ("confusion_matrix", "classification_report"), \
            ValueError("Model performance can only be quantified "
                       "as either a confusion matrix or a "
                       "classification report.")       
        # load the model from its binary object into memory
        self.pipeline.model.artifact = joblib.load(f"{self.session_name}")
        
        # read in test data, split into IVs and DV
        test_dat = read_csv(self.data.testing_path, dtype=self.data.struc)
        self.data.X_test= test_dat.drop(self.data.label, axis=1)
        self.data.y_test = test_dat[self.data.label].copy()
        
        # transform test data 
        self.data.X_test_transformed = self.pipeline.transformer.transform(
            self.data.X_test) 
        # make predictions on the transformed testing data
        predictions: np.array = self.pipeline.model.artifact.predict(
            self.data.X_test_transformed)
        # return performance (perdictions vs real labels)
        if how == "confusion_matrix":
            return confusion_matrix(self.data.y_test, predictions)
        elif how == "classification_report":
            return classification_report(self.data.y_test, predictions)
        else:
            raise NotImplementedError
            
    def predict_test_cases(self, n_predictions: int) -> Tuple[
                        List[str], List[str], int]:
        """
        Print class predictions of n rows of the test 
        data compared to n corresponding real labels 
        of the test data.


        Parameters
        ----------
        n_predictions : int
            The number of test set row labels to predict.


        Returns
        -------
        Tuple - containing 2 lists (each of length 
            n_predictions) of predicted and actual test 
            set class labels, and the number of 
            misclassified labels (of only the n_predictions
            made).
        """
        # lists of predicted and true labels
        preds: List[float] = list(self.pipeline.model.artifact.predict(
            self.data.X_test_transformed))[:n_predictions]
        actual: List[float] = list(self.data.y_test)[:n_predictions]
        # count of difference between predicted and true labels
        diff: int = 0
        for pred, true in zip(preds, actual):
            if pred != true:
                diff += 1
 
        print("Predicted Labels:")
        print(preds )
        print()
        print("Actual Labels:")
        print(actual)
        print()
        msg = (
            f"""Of {n_predictions} predictions made, """
            f"""the model misclassified {diff} labels."""    
        )
        print(msg)
        
        return preds, actual, diff
        
        

