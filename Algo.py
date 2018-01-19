import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

def visualize(data, no, title):
    plt.subplot(no)
    data.plot(kind='bar').set_title(title)

originalDataset = pd.read_csv('data.csv')

dataset = originalDataset.drop('Unnamed: 32', axis=1)

dataset['diagnosis'] = dataset['diagnosis'].map({'B':0, 'M':1})

datasetCorr = dataset.corr()

print("Corelations:- \n")
print(datasetCorr['diagnosis'].sort_values(ascending=False))
print("\n")

#Select top 14 features
selectedFeatures = ['concave points_worst', 'perimeter_worst', 
                    'concave points_mean', 'perimeter_worst', 
                    'concave points_mean', 'radius_worst',
                    'perimeter_mean', 'area_worst', 'radius_mean', 
                    'area_mean', 'concavity_mean', 'concavity_worst', 
                    'compactness_mean', 'compactness_worst']

dataset = dataset[selectedFeatures + ['diagnosis']]

dataset[selectedFeatures] = scale(dataset[selectedFeatures])

xtrain, xtest, ytrain, ytest = train_test_split(dataset.drop('diagnosis', axis=1), dataset['diagnosis'])

clf = RandomForestClassifier()

clf.fit(xtrain, ytrain)

ypred = pd.Series(clf.predict(xtest))

print("Accuracy: " + str(accuracy_score(ytest, ypred)))

plt.figure()
visualize(dataset.diagnosis.map({1:'Malignant', 0:'Benign'}).value_counts(), 221, 'Values in dataset')
visualize(ytest.map({1:'Malignant', 0:'Benign'}).value_counts(), 222, 'Test Actual')
visualize(ypred.map({1:'Malignant', 0:'Benign'}).value_counts(), 223, 'Test Predicted')
 

       