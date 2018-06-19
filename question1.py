#!/usr/bin/python

import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

#load dictionary
data_enron = pickle.load(open("final_project_dataset.pkl", "r"))

print "Number of executives in Enron Dataset: ", len(data_enron)

print "Number of features: ",len(data_enron['SKILLING JEFFREY K'].keys())

n_poi = 0
for entry in data_enron:
    if data_enron[entry]['poi'] == 1:
        n_poi += 1
print "Number of Person of Interest (POI): " + str(n_poi)

print "Name of executives: ",data_enron.keys()

missing_values = defaultdict(int)
for person in data_enron.values():
    for feature, value in person.items():
        if value == 'NaN':
            missing_values[feature] += 1
        else:
            missing_values[feature] += 0

# Summarising missing data
missing_list = []
for feature, missing in missing_values.items():
    missing_list.append((feature, round(float(missing)/len(data_enron)*100, 2)))

objects = []
keys_y = []
print 'Percentage of missing values: '
for elem in sorted(missing_list, key = lambda x : x[1], reverse=True):
    objects.append(elem[0])
    keys_y.append(float(elem[1]))
    print '{0:25s} - {1:.2f} %'.format(elem[0], elem[1])

y_pos = range(len(objects))
plt.bar(y_pos, keys_y, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Proportion (%)')
plt.title('Missing Data')

plt.show()


