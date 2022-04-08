from sklearn import datasets # import the dataset
from sklearn import metrics # import metrics
from sklearn.naive_bayes import GaussianNB # naive_bayes import GaussianNB

dataset=datasets.load_diabetes()# diabetic dataset is taken
model=GaussianNB()# object is created
model.fit(dataset.data,dataset.target)#object is created
expected=dataset.target #expected output
predicted=model.predict(dataset.data)
print("The confusion Matrix\n",metrics.confusion_matrix(expected,predicted))
print("The Accuracy",metrics.accuracy_score(expected,predicted))