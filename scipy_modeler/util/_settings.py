"""
---------------------------------------------------
Author: Nick Hopewell <nicholashopewell@gmail.com>

---------------------------------------------------

"""
import os
import json
from datetime import datetime

import scipy_modeler.util._settings_h as settings 
#from dotenv import load_dotenv
#load_dotenv()

####### TODO: CHANGE THIS BACK TO SIMPLY "config.json"
try:
    config_data = settings.load_json_config("config.json")
except:
    try:
        config_data = settings.load_json_config("C:\\Users\\jason.conte\\Desktop\\AASC\\AASC\\scipy_modeller\\config.json")
    except:
        config_data = settings.load_json_config("C:\\Users\\Nicholas.Hopewell\\Desktop\\AASC\\scipy_modeller\\config.json")

today, today_time = settings.get_date_time()

# assert user has not left these keys null in config file
settings.assert_config_defaults(config_data, 
    "odbc", "project_defaults", "file_paths"
)

# extract config settigs from json
database = config_data["odbc"]["database"]
dsn = config_data["odbc"]["dsn"]
main_out_path = config_data["file_paths"]["main_file"]

project_name = config_data["project_defaults"]["project_file"]
default_schema = config_data["project_defaults"]["default_schema"]
default_table = config_data["project_defaults"]["default_table"]

if not all(
    config_data["file_paths"]["log_paths"].values()):
    logs_path = settings.impute_json_config(
                    config_data["file_paths"]["log_paths"], 
                    "_logs/", 
                    f"{today} logs/", 
                    f"log_{today_time}.txt"
    )
else:
    logs_path = config_data["file_paths"]["log_paths"]

if not all (
    config_data["file_paths"]["cache_paths"].values()):
    cache_path = settings.impute_json_config(
                config_data["file_paths"]["cache_paths"], 
                    "_cache/", 
                    f"{today} cache/", 
                    f"cache_{today_time}.txt"
    )
else:
    cache_path = config_data["file_paths"]["cache_paths"]

data_path = config_data["file_paths"]["data_paths"]
model_major = config_data["file_paths"]["model_major"]
master_major = config_data["file_paths"]["master_major"]

_general = {
    "project": project_name
}

db_initalize = {
    "database": database,
    "dsn": dsn,
    "default_schema": default_schema,
    "default_table": default_table,
}

settings.try_make_paths(main_out_path)

_file_paths = {
    "main_file": main_out_path,
    "project_file": project_name,
    "log_major": logs_path['log_major'],
    "log_file": logs_path['log_file'],
    "log": logs_path['log'],
    "cache_major": cache_path['cache_major'],
    "cache_file": cache_path['cache_file'],
    "cache": cache_path['cache'],
    "data_major": data_path['data_major'],
    "all_data": data_path['all_data'],
    "training_data": data_path['training_data'],
    "validation_data": data_path['validation_data'],
    "testing_data": data_path['testing_data'],
    "transformed_data": data_path['transformed_data'],
    "model_major": model_major,
    "master_major": master_major
}

_logs = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["log_major"],
    _file_paths["log_file"]
)

_cache = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["cache_major"],
    _file_paths["cache_file"],
    _file_paths["cache"]
)

complete_data_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["data_major"],
    _file_paths["all_data"]
)                      

training_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["data_major"],
    _file_paths["training_data"]
)

validation_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["data_major"],
    _file_paths["validation_data"]
)

test_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["data_major"],
    _file_paths["testing_data"]
)

transformed_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["data_major"],
    _file_paths["transformed_data"]
)

model_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["model_major"]
)

project_path = settings.join_paths(
    _file_paths["main_file"],
    _file_paths["project_file"],
    _file_paths["master_major"]
)

scipy_paths = [_logs, _cache, complete_data_path, training_path,
               validation_path, test_path, transformed_path, 
               model_path, project_path]

settings.try_make_paths(scipy_paths)

# _model_object = where the binary file containing the latest trained model will be (modelling artificact).
# Calling predict should look into this folder, bring the object back to a python
# object, and use that for inference.
_model_object = []

# No longer need to write an algorithm to traverse our trees and organize them into
# rules. New SKLearn API does this.
_model_rule_sets = []

_model_diagrams = []

_models = ['CART', 'Bagged Trees', 'Random Forest', 'Extremely Random Trees',
           'voting classifier', 'AdaBoost', 'XGBoost', 'catBoost',
           'Logistic Regression', 'Linear SVM', 'Nonlinear SVM', 'KNN']

_performance_eval = ['hold out set', 'k-fold cross validation',
                     'stratified k-fold cross validation', 'out of bag score']

_classification_metrics = ['full report', 'confusion matrix', 'accuracy',
                           'balanced accuracy', 'AUC', 'AUC ROC', 'PRC',
                           'precision', 'recall', 'f1 score']
