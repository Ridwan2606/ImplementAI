import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift


#Setting the seed of the random generator to 0
print("\nInitialising ... ")
np.random.seed(0)

#Reads data from csv file
input_csv = 'CAE_dataset.csv'
print("Gathering raw data from csv file: " + input_csv + "... ")
input_data = pd.read_csv('CAE_dataset.csv')

#Transform the csv file by dropping first column, rows with missing data and grouping by the "Id" column
print("Pre-processing training data ... ")
input_data = input_data.drop(columns=['Unnamed: 0'])
input_data = input_data.dropna(axis=0)
input_data = input_data.groupby(input_data['Id'])

#Find the mean of each feature per group sorted by 'Id"
feat_new = input_data.mean()

#Assign all the features columns to X
X = feat_new.iloc[:, 0:10] 

#Assign the competency flag to y
y = feat_new['label']

print("Training the Model ... ")

#Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
sss_train = KFold(n_splits=5, shuffle=False, random_state=0)

#Dividing the data set into the training and test sets
for train_index, test_index in sss_train.split(X, y):
    X_train, X_test = np.asarray(X)[train_index], np.asarray(X)[test_index]
    y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]

    #Initialising clusters paramaters for 3 classifying techniques (Kmeans, db, meanshift)
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20000000)
    knn_classifier_kmeans = KNeighborsClassifier(5)
    knn_classifier_meanshift = KNeighborsClassifier(5)
    knn_classifier_dbscan = KNeighborsClassifier(5)
    meanShift_clustering = MeanShift()
    dbscan_cluster = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto')

    #Fitting the training datasets for each classifying techniques (Kmeans, db, meanshift)
    kmeans.fit(X_train)
    dbscan_cluster.fit(X_train)
    meanShift_clustering.fit(X_train)

    #Carrying out KNN to 3 classifying techniques (Kmeans, db, meanshift) to improve accuracy of data
    knn_classifier_kmeans.fit(X_train, kmeans.labels_)
    knn_classifier_meanshift.fit(X_train, meanShift_clustering.labels_)
    knn_classifier_dbscan.fit(X_train, dbscan_cluster.labels_)

print("Model is fully trained.\n")

#if (input("Any test data to predict from? [Y/N]")=="")

test_csv = ""

#Prompts for a correct test dataset csv name
while True:
    test_csv=input("\nPlease type name of csv file (in the same folder) to predict: ")
    if (os.path.exists(test_csv)):
        break
    else:
        print("No file with such a name exists ")

#Reads the csv test dataset file
test_final = pd.read_csv(test_csv)

#Pre-processing the dataset file
print("\nPre-processing the test data...")
test_final = test_final.drop(columns=['Unnamed: 0'])
test_final = test_final.dropna(axis=0)
test_final = test_final.groupby(test_final['Id'])

#Find the mean of each feature per group sorted by 'Id"
test_final = test_final.mean()
#Assigning all feature columns to X_test_final, label column to y_test_final
X_test_final = test_final.iloc[:, 0:10]


#Predicts the competency for the test dataset based on our different machine learning models
print("Predicting the test data...")
kmeans_prediction = kmeans.predict(X_test_final)
knn_predict = knn_classifier_kmeans.predict(X_test_final)
meanShift_predict = knn_classifier_meanshift.predict(X_test_final)

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 
  
#Create the final output csv file "HackerBoises"
print("Creating the output file...") 
with open('HackerBoises.csv', 'w') as f:
        writer=csv.writer(f, lineterminator='\n')
        writer.writerow(("Id","Label Predicted"))

#For each Id/Pilot
for k in range(X_test_final.shape[0]):
    voting_score = [0, 0 , 0]
    voting_score[0] = knn_predict[k]
    voting_score[1] = meanShift_predict[k]
    voting_score[2] = kmeans_prediction[k]
    out_score = most_frequent(voting_score)
    with open('HackerBoises.csv', 'a') as f:
        writer=csv.writer(f, lineterminator='\n')
        writer.writerow((k, out_score))


print("Created/Updated output csv file 'HackerBoises.csv' \n") 
print("END\n")

