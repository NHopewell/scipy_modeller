# -*- coding: utf-8 -*-
"""
@author: Nicholas.Hopewell
"""
from typing import Dict, Any

from sklearn.pipeline import Pipeline

from scipy_modeler.core.pipeline.factory import PipelineFactory 

class PipelineManager:
    """
    Manages execution of different elements of a pipeline.
    
    Parameters
    ----------
    None.

    
    Attributes
    ----------

        
    Notes
    -----
    This class can be used as a decorator when defining 
    functions to create a basic workflow.
    """
    def __init__(self):
        self.all_tasks = []
        
    def new_task(self, do_before=None, depends_on=None,
                 first=False, last=False):
        """
        Wrapper function for adding a new task to the pipeline
        inserting a task at the index +1 of the task it depends
        on before it should be executed
        """
        assert sum(
            ( bool(do_before), bool(depends_on), first, last )
            ) <=1, ValueError(
                "May only specify one parameter of the following: "
                "'do_before', 'do_after'. 'first', or 'last'."
                )

        task_index = 0
        if depends_on:
            task_index = self.all_tasks.index(depends_on) + 1
        if do_before: 
            task_index = self.all_tasks.index(do_before) - 1
        if first:
            task_index = 0
        if last:
            task_index = len(self.all_tasks)
            
        def inner(func):
            self.all_tasks.insert(task_index, func)
            return func
        return inner
    
    @property
    def task_report(self, print_: bool = False, 
                    doc_string: bool = False) -> list:
        """
        Prints  a report of the  dunder names of each 
        task in the pipeline of taks (and doc string if desired).
        Returns a list of functions in pipeline.
        """
        if print_:
            if doc_string:
                for task in self.all_tasks:
                    print(task.__name__)
                    print(f"\t{task.__doc__}", end='\n\n')
            else:
                for task in self.all_tasks:
                    print(task.__name__)
        
        self._task_report = [i.__name__ for i in self.all_tasks]
        return self._task_report
    
    @task_report.setter
    def task_report(self, new: list):
        self._task_report = new
    
    def execute_all_tasks(self, in_):
        """
        Execute pipeline given the dependencies specified in the
        decorator argument "depends_on"
        """
        out = in_
        for func in self.all_tasks:
            out = func(in_)
        return out

    def order_new_factory(self, factory_specifications: Dict[str, Any]) -> Pipeline:
        """
        Grab a new PipelineFactory to apply to the data. 
        The factory builds a custom pipeline on the fly whose specific
        application is not 'learned' until it is actually applied to the data.  

        Parameters
        ----------
        factory_specifications : dict
            A dictionary specifying the parameters used
            for transforming the training data. These parameters
            must fit the SciKit learn API. 
        """

        assert isinstance(factory_specifications, dict), TypeError(
            "Transformation parameters specify how the transformation "
            "pipeline will be constructed and must be passed as a dictionary."
        )

        pipe_factory = PipelineFactory(
                encoder=factory_specifications[
                    "categorical_config"]["encoder"
                    ],
                cat_imputer_strat=factory_specifications[
                    "categorical_config"]["cat_imputer_strat"
                    ],
                cat_imputer_val=factory_specifications[
                    "categorical_config"]["cat_imputer_val"
                    ],
                num_imputer_strat=factory_specifications[
                    "numerical_config"]["num_imputer_strat"
                    ],
                num_imputer_val=factory_specifications[
                    "numerical_config"]["num_imputer_val"
                    ],
                scaler=factory_specifications[
                    "numerical_config"]["scaler"]
                    ,
                matrix_sparsity_threshold=factory_specifications[
                    "matrix_sparsity_threshold"
                    ], 
                ignore_cols=factory_specifications["ignore_cols"
                ]
        )

        return pipe_factory
    

    

