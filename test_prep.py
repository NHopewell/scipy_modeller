# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text
from sklearn import tree

train_set, test_set = train_test_split(sp_india_final.data, test_size=0.2, random_state=42)

sp_india_data = train_set.drop('LABEL', axis=1)
sp_india_label = train_set['LABEL'].copy()

# instantiate Transformer class, pass params for cat encoding
CatTransformer = Transformer(encoder='onehot',
                             cat_imputer_strat='constant',
                             cat_imputer_val='missing')

# build a transformer pipeline based on sp data
SPTransformer = CatTransformer.build_transformer_pipeline(sp_india_data, label=None)

# transform the data before modeling
#x = SPTransformer.fit(sp_india_data)
prepared = SPTransformer.fit_transform(sp_india_data)

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, \
     is_string_dtype

cats = []
nums = []
for k,v in sp_india_data.items():
    if is_categorical_dtype(v) or is_string_dtype(v): 
        cats.append(k)
for k,v in sp_india_data.items():
    if is_numeric_dtype(v):
        nums.append(k)

cols = list(SPTransformer.transformers_[0][1].named_steps['ohe'].get_feature_names(input_features=cats)) + nums


clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(prepared, sp_india_label)

SPTransformer.transformers_[0][1].named_steps['ohe'].get_feature_names(input_features=cats)

import matplotlib.pyplot as plt

plt.figure(figsize=(70, 30),dpi=150)
ax=plt.subplot()
_ = tree.plot_tree(clf, feature_names = cols, class_names = ['Approved', 'Refused'], 
                   filled=True, rotate=True,fontsize=15, ax=ax,rounded=True)
plt.show()

tree.plot_tree(clf)

from sklearn.tree.export import export_text
r = export_text(clf, feature_names=cols)
print(r)



some_data = sp_india_data.iloc[:10]
some_labels = sp_india_label.iloc[:10]
some_prepared_data = SPTransformer.transform(some_data)

print("Predictions:", clf.predict(some_prepared_data))

print("Labels:", list(some_labels))

X_test = test_set.drop('LABEL', axis=1)
y_test = test_set['LABEL'].copy()
X_test_prepared = SPTransformer.transform(X_test)

final_predictions = clf.predict(X_test_prepared)
accuracy_score(y_test, final_predictions)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, final_predictions))

confusion_matrix(y_test, final_predictions)