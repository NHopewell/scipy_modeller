"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>


---------------------------------------------------
"""
import os
import joblib 
from typing import (
    Optional, Union, Tuple, Dict, List, Any)

from sklearn import tree
import matplotlib.pyplot as plt
from pandas.api.types import (
        is_numeric_dtype, is_categorical_dtype, 
        is_string_dtype
)

from scipy_modeler.util._settings_h import get_date_time

def _get_transformer_cols(dataframe) -> Tuple[List[str], List[str]]:
    """
    Return nominal column names (including strings)
    and numeric column names of a pandas dataframe.


    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame


    Returns
    -------
    Tuple - (categorical columns, numeric columns)
    """
    cats = []
    nums = []
    # TODO - this cant include all strings
    for k,v in dataframe.items():
        if is_categorical_dtype(v) or is_string_dtype(v): 
            cats.append(k)
        elif is_numeric_dtype(v):
            nums.append(k)

    return ( cats, nums )

def plot_risk_buckets(session, 
                    title: str="SP India Risk Buckets",
                    x_lab: str="""\nRisk Buckets """
                """\nDeterminded by probability of """
                """being classified as 'Approved'""",
                    y_lab: str="n test_cases\n",
                    num_buckets: int=5):
    """
    Create and visualize a bar plot of risk buckets.
    Risk buckets are determined by the probability 
    of being classified as the positive class 
    ('Approved' for instance).


    Parameters
    ----------
    title :  string
        The main plot title.
    x_lab : string
        The x-axis label.
    y_lab : string
        The y-axis label.
    num_buckets : int
        The number of risk buckets to generate
        between 0-1 (0% risk - 100% risk). 


    Notes
    -----
    Risk bucket generation requires a model to be
    trained and test data to be transformed before
    calling plot_risk_buckets().
    """
    from pandas import cut, Series

    # get class probabilities if model loaded, if not, load model into mem
    if session.model.artifact: 
        class_probs = session.model.artifact.predict_proba(
            session.data.X_test_transformed)
    else: 
        session.model.artifact = joblib.load(f"{session.session_name}")
        class_probs = session.model.artifact.predict_proba(
            session.data.X_test_transformed)
    
    # proabability of row being classified as 'approved'
    prob_approved = Series( [i[0] for i in class_probs] )
    
    start: int = 0
    step: float = 1 / num_buckets
    bins: List[int] = []
    num_loops: int = num_buckets - 1

    while num_loops:
        bins.append(start + step)
        start += step
        num_loops -= 1

    bins = [round(bin, 2) for bin in bins]
    bins.insert(0, 0)
    bins.append(1)
    
    bin_buckets: List[float] = []

    for i in range(1, len(bins)-1):
        bin_buckets.append(bins[i])
        bin_buckets.append(bins[i]+0.01)

    bin_buckets = [round(bin, 2) for bin in bin_buckets]
    bin_buckets.insert(0, 0)
    bin_buckets.append(1)

    # zip list in strides of 2 starting at odd/even indicies
    bucket_tup: List[Tuple[int, int]] = list( 
        zip( bin_buckets[::2], bin_buckets[1::2] ) 
    ) 
    # convert to list of strings 
    bucket_labels = [f'{i}' for i in bucket_tup]
    probs = cut(prob_approved, bins=bins, include_lowest=True)
    
    ax = probs.value_counts(sort=False).plot.bar(
                        rot=0, 
                        color="darkorange", 
                        figsize=(14,8), 
                        edgecolor='grey')

    ax.set_xticklabels(bucket_labels, fontsize=14)
    
    ax.set_title(f"{title}\nAlgorithm: {session.model.algorithm}", 
                fontsize=18)
    ax.set_xlabel(x_lab, fontsize=14)
    ax.set_ylabel(y_lab, fontsize=14)
    
    plt.show()

    
def plot_tree_model(session, class_labels: 
                    Optional[List[str]]=['Approved', 'Refused']):
    """
    Visualize a decision tree model. Most
    useful within a Jupyter Notebook.


    Parameters
    ---------
    class_labels : list, optional
        A list of class labels.
        Default = ['Approved', 'Refused']
    """
    # get lists of categorical and numeric columns
    cats, nums = _get_transformer_cols(session.data.X_train)
    # get one hot encoded categorical columns with proper labels,
    # append with nums -> list of transformed column space
    cols = list( session.data.transformer.transformers_[0][1].\
        named_steps['ohe'].get_feature_names(
            input_features=cats) ) + nums

    plt.figure(figsize=(70, 30),dpi=150)
    
    ax=plt.subplot()
    
    _ = tree.plot_tree(session.model.artifact, 
                    feature_names = cols, 
                    class_names = class_labels, 
                    filled=True, 
                    rotate=True,
                    fontsize=15, 
                    ax=ax,rounded=True)
    
    plt.show()
    
def tree_rules(session, write_to_txt: Optional[bool]=False,
                    out_path: Optional[str]=None,
                    out_file_name: Optional[str]=None):
    """
    Return if-then rules for classification based
    on a decision-tree model. Optionally, write the
    decision tree rules to a text file. 


    Parameters
    ----------
    write_to_txt : bool, optional
        Whether or not to write the decision tree
        rules to a .txt file. Default = False.
    out_path : string, optional
        The desired path to write the tree rules
        to. If write_to_text = True and an out_path
        is not specified, tree rules will be written
        to the current working directory. 
        Default = None. 
    out_file_name : string, optional
        The desired name of the text file to write
        the tree rules into. If write_to_txt = True
        and an out_file_name is not specified, the
        tree rules file will be given a default unique
        name. 
    """
    from sklearn.tree.export import export_text
    # get lists of categorical and numeric columns
    cats, nums = _get_transformer_cols(session.data.X_train)
    # get one hot encoded categorical columns with proper labels,
    # append with nums -> list of transformed column space
    cols = list( session.data.transformer.transformers_[0][1].\
        named_steps['ohe'].get_feature_names(
            input_features=cats) ) + nums
    
    tree_rules = export_text(session.model.artifact, 
        feature_names=cols)

    # write to test file in passed path or defaul working dir    
    if write_to_txt:
        if not out_path:
            out_path = os.getcwd()
        if not out_file_name:
            today, today_time = get_date_time()
            out_file_name = f"tree_rules_{today}{today_time}.txt"
        # complete path to write rules to
        out = os.path.join(out_path, out_file_name)
        
        with open(out, 'w') as f:
            f.write(tree_rules)

    return tree_rules

def plot_confusion_matrix(session):
    raise NotImplementedError
    






 



