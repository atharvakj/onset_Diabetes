import pandas as pd
import numpy as np
data = pd.read_csv('/home/atharva/Downloads/diabetes2.csv')
import seaborn as sns


from cassandra.cluster import Cluster
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('miniproject')

rows = session.execute('SELECT * FROM diabetes;')

Pregnancies = []
Glucose = []
BloodPressure= []
SkinThickness= []
Insulin= []
BMI= []
DiabetesPedigreeFunction= []
Age= []
Outcome= []

for ans in rows:
    Pregnancies.append(ans.pregnancies)
    Glucose.append(ans.glucose)
    BloodPressure.append(ans.bloodpressure)
    SkinThickness.append(ans.skinthickness)
    Insulin.append(ans.insulin)
    BMI.append(ans.bmi)
    DiabetesPedigreeFunction.append(ans.diabetespedigreefunction)
    Age.append(ans.age)
    Outcome.append(ans.outcome)

df =pd.DataFrame([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome])
df = df.transpose()

df.columns = ['Pregnancies',"Glucose","BloodPressure","SkinThickness",'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

data = df

data= data.ix[:,'Pregnancies':'Outcome']
data['Outcome'] = data['Outcome'].astype('category')
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1,random_state =42,test_size = 0.2 )

for train_index,test_index in split.split(data,data['Outcome']):
    train = data.loc[train_index]
    test = data.loc[test_index]

trainX = train.ix[:,'Pregnancies':'Age']
trainY = train.ix[:,'Outcome']
testX = test.ix[:,'Pregnancies':'Age']
testY = test.ix[:,'Outcome']

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        ]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    ]

from sklearn.model_selection import cross_val_score

results={}
for name, clf in zip(names,classifiers):
    scores = cross_val_score(clf,trainX,trainY,cv = 5)
    results[name] = scores

from sklearn.model_selection import cross_val_predict
result2 = {}
for name,clf in zip(names,classifiers):
    scores2 = cross_val_predict(clf,trainX,trainY,cv = 5)
    result2[name] = scores2
