import numpy as np
import pandas as pd

#Loading the PlayTennis data
PlayTennis = pd.read_csv("4.csv")
print("Given dataset:\n", PlayTennis,"\n")
targetnames=np.unique(PlayTennis.play)

#It is easy to implement Decision Tree with numerical values.
#We can convert all the non numerical values into numerical values
#using LabelEncoder
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['wind'] = Le.fit_transform(PlayTennis['wind'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])
print("the encoded dataset is:\n",PlayTennis)

X = PlayTennis.drop(['play'],axis=1) #X- attributes.
y = PlayTennis['play'] #y - holds target values.

# Fitting the model
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
class_names=targetnames)
graph = graphviz.Source(dot_data)

# To Classify the new example
print("\n Classifying new ex:[rainy,cool,normal,strong] as:")
new_data = [[1,1,1,0]]
X_pred = clf.predict(new_data)
if (X_pred==1):
    print("Yes")
else:
    print("No")
    
print("\n To see the decision tree type 'graph' in console window \n")