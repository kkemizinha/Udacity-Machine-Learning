#!/usr/bin/python


# This code was based on Feature Selection lesson
# under Udacity sample code in feature_selection/find_signature.py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from tester import dump_classifier_and_data

#Load the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

#remove outlier
data_dict.pop('TOTAL')

# All the features were tested. After the First round of test, 
# the features that score was 0 (did not contribute) were ripped off.

#Features tested without the new variables
#features_old_to_be_selected = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
#                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
#                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
#                 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
#                 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

features_to_be_selected = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
                 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
                 'poi_total','ratio_messages_from_poi', 'ratio_messages_to_poi', 'ratio_messages_shared']


# Creating new variable - poi_total
for person in data_dict.values():
    if person['from_this_person_to_poi'] != 'NaN':
        to_poi = person['from_this_person_to_poi']
    else:
        to_poi = 0.
    if person['from_poi_to_this_person'] != 'NaN':
        from_poi = person['from_poi_to_this_person']
    else:
        from_poi = 0.
    person['poi_total'] = to_poi + from_poi

#This function is responsible to create new features
def new_features_ratio(key,key2):
    new_feat=[]

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][key2] == "NaN":
            new_feat.append(0.)
        elif data_dict[i][key] >= 0:
            new_feat.append(float(data_dict[i][key])/float(data_dict[i][key2]))
    return new_feat

# Creating new variable
ratio_messages_from_poi = new_features_ratio('from_poi_to_this_person','to_messages')
ratio_messages_to_poi = new_features_ratio('from_this_person_to_poi','from_messages')
ratio_messages_shared = new_features_ratio('shared_receipt_with_poi','to_messages')

# New features 
features_new_to_be_selected = []

# Insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]['ratio_messages_from_poi'] = ratio_messages_from_poi[count]
    data_dict[i]['ratio_messages_to_poi'] = ratio_messages_to_poi[count]
    data_dict[i]['ratio_messages_shared'] = ratio_messages_shared[count]
    count +=1

data = featureFormat(data_dict, features_to_be_selected)

labels, features = targetFeatureSplit(data)

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,
                                                           labels,test_size=0.1, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred = clf.predict(features_test)
print 'accuracy', score


importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(len(importances)):
    print "{} feature {} ({})".format(i+1,features_to_be_selected[i+1],importances[indices[i]])


# function for Precision Score TP/(TP+FP)
print 'Precision = ', precision_score(labels_test,pred)

# function for Recall Score TP/(TP+FN)
print 'recall = ', recall_score(labels_test,pred)



# Classifier to feature selection
clf_test = DecisionTreeClassifier()

# Select K best - performance test 2 to 10
for k in range(2, 10):
    data = featureFormat(data_dict, features_to_be_selected, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    selector = SelectKBest(k=k)
    selector.fit(features, labels)
    selected_features_list = ['poi']
    # Printing the selected features
    print '::: PERFORMANCE WITH {} FEATURES :::'.format(k)
    for elem in zip(features_to_be_selected[1:], selector.get_support()):
        if elem[1] == True:
            print elem[0]
            selected_features_list.append(elem[0])

    data = featureFormat(data_dict, selected_features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    print 'Precision :', np.mean(cross_val_score(clf_test, features, labels, scoring='precision', cv=4))
    print 'Recall :', np.mean(cross_val_score(clf_test, features, labels, scoring='recall', cv=4))


def value_scaling(value, values_list):
     if value == 'NaN':
         value = 0.0
     else:
         value = float(value)
     return (value - min(values_list)) / (max(values_list) - min(values_list))


# Scaling features
salary = []
exercised_stock_options = []
bonus = []
total_stock_value = []
ratio_messages_to_poi = []

for scaling_features in data_dict.values():
    salary.append(float(scaling_features['salary'] if scaling_features['salary'] != 'NaN' else 0))
    exercised_stock_options.append(float(scaling_features['exercised_stock_options']
                                   if scaling_features['exercised_stock_options'] != 'NaN' else 0))
    bonus.append(float(scaling_features['bonus'] if scaling_features['bonus'] != 'NaN' else 0))
    total_stock_value.append(float(scaling_features['total_stock_value']
                                   if scaling_features['total_stock_value'] != 'NaN' else 0))
    ratio_messages_to_poi.append(float(scaling_features['ratio_messages_to_poi']
                                   if scaling_features['ratio_messages_to_poi'] != 'NaN' else 0))

for scaling_features in data_dict.values():
    scaling_features['scaled_salary'] = value_scaling(scaling_features['salary'], salary)
    scaling_features['scaled_exercised_stock_options'] = value_scaling(
             scaling_features['exercised_stock_options'], exercised_stock_options)
    scaling_features['scaled_bonus'] = value_scaling(scaling_features['bonus'], bonus)
    scaling_features['scaled_total_stock_value'] = value_scaling(scaling_features['total_stock_value'], total_stock_value)
    scaling_features['scaled_ratio_messages_to_poi'] = value_scaling(scaling_features['ratio_messages_to_poi'], ratio_messages_to_poi)

# THe chosen parameters were
#salary
#exercised_stock_options
#bonus
#total_stock_value
#ratio_messages_to_poi
features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus',
                 'total_stock_value', 'ratio_messages_to_poi']

data = featureFormat(data_dict, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaled_features_list = ['poi', 'scaled_salary', 'scaled_exercised_stock_options', 'scaled_bonus',
                        'scaled_total_stock_value', 'scaled_ratio_messages_to_poi']

# Separating features and labels scaled
data_scaled = featureFormat(data_dict, scaled_features_list, sort_keys = True)
scaled_labels, scaled_features = targetFeatureSplit(data_scaled)


### Task 4: Try a variety of classifiers

# Testing Decision Tree Classifier
param_grid_tree = {"criterion": ["gini", "entropy"],
                   "min_samples_split": np.arange(2, 5, 1),
                   "max_depth": np.arange(1, 5, 1),
                   "min_samples_leaf": np.arange(1, 5, 1),
                   "max_leaf_nodes": np.arange(2, 7, 1),
                   "class_weight": ['balanced'],
                   "random_state": [42]}

clf_tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(clf_tree, param_grid_tree, scoring='precision')
grid_search_tree.fit(features, labels)

print '# DECISION TREE #'
print 'PARAMETERS SEARCHED:'
print 'Criterion: gini, entropy'
print 'Min sample split: 2, 3, 4'
print 'Max depth: 1, 2, 3, 4'
print 'Min samples leaf: 1, 2, 3, 4'
print 'Max leaf node: 2, 3, 4, 5, 6'
print 'Scoring Method: Precision'
print 'Best score: ', grid_search_tree.best_score_
print 'Best params: ', grid_search_tree.best_params_
print

# Testing AdaBoost Classifier
param_grid_ada = {'algorithm': ['SAMME', 'SAMME.R'], 'random_state': [42], 'n_estimators': [30, 50, 70, 90]}

clf_ada = AdaBoostClassifier()
grid_search_ada = GridSearchCV(clf_ada, param_grid_ada, scoring='precision')
grid_search_ada.fit(features, labels)

print '# ADA BOOST CLASSIFIER #'
print 'PARAMETERS SEARCHED:'
print 'n_estimator: 30, 50, 70, 90'
print 'Algorithm: SAMME, SAMME.R'
print 'Best params: ', grid_search_ada.best_params_
print 'Best precision score: ', grid_search_ada.best_score_
print

# Testing Naive Bayes Classifier
clf_nb = GaussianNB()

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf_nb.fit(features_train, labels_train)
pred = clf_nb.predict(features_test)
print '# GAUSSIAN NB #'
print 'PARAMETERS SEARCHED: no parameters to search'
print 'Precision Score: ', precision_score(labels_test, pred)
print

# Testing SVC with scaled features
clf_svc = SVC()

C_rbf = 10. ** np.arange(-10, 11, 1)
gamma_rbf = 10. ** np.arange(-10, 11, 1)

gamma_sigmoid = 10. ** np.arange(-20, 21, 1)
coef0_sigmoid = np.arange(-40, 41, 10)

param_grid_svc = [{'kernel': ['rbf'], 'gamma': gamma_rbf, 'C': C_rbf, 'class_weight': ['balanced'],
                   'random_state':[42]},
                  {'kernel' : ['sigmoid'], 'gamma' : gamma_sigmoid, 'coef0' : coef0_sigmoid,
                   'class_weight': ['balanced'], 'random_state': [42]}]


grid_search_svc = GridSearchCV(clf_svc, param_grid_svc, scoring='precision')
grid_search_svc.fit(scaled_features, scaled_labels)

print '# SVC + SCALED FEATURES #'
print 'PARAMETERS SEARCHED:'
print 'Kernel: Rbf, gamma: [10^-10, 10^-9, ..., 10^10], C: [10^-10, 10^-9, ..., 10^10]'
print 'Kernel: Sigmoid, gamma: [10^-20, 10^-19, ..., 10^20], coef0: [-40, -30, ..., 40]'
print 'Scoring method: Precision'
print 'Best score: ', grid_search_svc.best_score_
print 'Best params: ', grid_search_svc.best_params_
print

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
cross_val_prec_nb = cross_val_score(clf_nb, features, labels, scoring='precision', cv=4)
cross_val_rec_nb = cross_val_score(clf_nb, features, labels, scoring='recall', cv=4)

cross_val_prec_dt = cross_val_score(clf_tree, features, labels, scoring='precision', cv=4)
cross_val_rec_dt = cross_val_score(clf_tree, features, labels, scoring='recall', cv=4)

cross_val_prec_ada = cross_val_score(clf_ada, features, labels, scoring='precision', cv=4)
cross_val_rec_ada = cross_val_score(clf_ada, features, labels, scoring='recall', cv=4)

cross_val_prec_svc = cross_val_score(clf_svc, scaled_features, scaled_labels, scoring='precision', cv=4)
cross_val_rec_svc = cross_val_score(clf_svc, scaled_features, scaled_labels, scoring='recall', cv=4)

print '### CLASSIFIERS PERFORMANCE ###'

print 'Algorithm:  GaussianNB'
print 'Precision: ', round(np.mean(cross_val_prec_nb), 3)
print 'Recall: ', round(np.mean(cross_val_rec_nb), 3)
print
print '######################'
print
print 'Algorithm:  Decision Tree'
print 'Precision: ', round(np.mean(cross_val_prec_dt), 3)
print 'Recall: ', round(np.mean(cross_val_rec_dt), 3)
print
print '######################'
print
print 'Algorithm:  Ada Boost Classifier'
print 'Precision: ', round(np.mean(cross_val_prec_ada), 3)
print 'Recall: ', round(np.mean(cross_val_rec_ada), 3)
print
print '######################'
print
print 'Algorithm:  SVC with scaled features'
print 'Precision: ', round(np.mean(cross_val_prec_svc), 3)
print 'Recall: ', round(np.mean(cross_val_rec_svc), 3)


# That's the Classifier with best performance to test with tester.py
clf = clf_nb


dump_classifier_and_data(clf, data_dict, features_list)

print 'Done!'