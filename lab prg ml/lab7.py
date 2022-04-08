import matplotlib.pyplot as plt# plot the graph
from sklearn import datasets # import inbuilt dataset
from sklearn.cluster import KMeans #Kmeans is imported
import sklearn.metrics as sm # accuracy is used
from sklearn import preprocessing #preprocessing
from sklearn.mixture import GaussianMixture #GaussiaMixture
import pandas as pd # for row manipultion
import numpy as np # random variable
iris = datasets.load_iris() #load iris dataset
X = pd.DataFrame(iris.data)#dataset iris
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
# name of the columns
y = pd.DataFrame(iris.target)
y.columns = ['Targets']#object is created
#plt.figure(figsize=(7,7))# inch 7 by 7 graph
#colormap = np.array(['red', 'lime', 'black'])# three color are selected

model = KMeans(n_clusters=3)# three cluster are created
model.fit(X)# object is created
score1=sm.accuracy_score(y, model.labels_)# score1 accuracy
print("Accuracy of KMeans=",score1)
plt.figure(figsize=(7,7))#plot is done
colormap = np.array(['red', 'lime', 'black'])# red , lime and black
plt.subplot(1, 2, 1)#first plot 1, 2, 1 graph
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification') # title of the graph
scaler = preprocessing.StandardScaler()# object is created
scaler.fit(X)
xsa = scaler.transform(X)#object is created
xs = pd.DataFrame(xsa, columns = X.columns)#columns nameare displayed
gmm = GaussianMixture(n_components=3)# three components are created
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)#ycluster consists of three cluster values
score2=sm.accuracy_score(y, y_cluster_gmm)# accuracy is score
print("Accuracy of EM=",score2)
plt.subplot(1, 2, 2)#plot second graph in two
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40) # grapgh is drawn
plt.title('EM Classification')# title of the graph