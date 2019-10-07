# Implement AI 2019
CAE Challenge
Team Name: HackerBoises
Team Members: Ridwan Kurmally, Achraf Essemlali, Jean Marie Cimula and Gohar Saqib Fazal

Problem:
CAE trains more than 120,000 people in civil aviation, defence and security every year. Due to the safety risks associated with the training of the pilots the company checks whether a pilot owns, or not, a specific competency after every 6 months. During a training session the instructor tags the competency if the pilot does not comply with all requirements after each maneuver. Since manual labour is involved there are chances that some maneuvers are mislabelled. This project aims to use machine learning algorithms to predict the relationship between the features and the labels. It then uses the predicted values using a given dataset to return if the competency has to be flagged.  

Vision:  
Aviation industry is on the rise and pilot’s jobs are expected to grow at a rate of about 12 percent per year. The increase in demand of pilots means that there is an added responsibility on the instructors to make sure that flight rules are followed to ensure the safety of passengers. This project prevents mistakes caused by human errors affecting the outcome of the training session and therefore, makes the process more transparent.

Strategy:
The solution of the program grouped the data by id to predict competency. It was ensured throughout the course of the project that the workload was divided equally between the members of the team to maximize efficiency.


Timeline:
Team strategy and division of workload - 2 hours
Research and Mathematical Analysis - 6 hours
Data Visualization and Preprocessing - 2 hours
Choosing machine learning technique - 10 hours
Time series analysis - 2 hours
Reporting results - 2 hours
Total time of project - 24 hours


Methodology:
1. Data Visualization and Preprocessing
   * Visualization
      * Orange Biolab
   * Data cleaning
   * Data reduction
   * Data transformation
2. Choosing machine learning technique
   * K-mean clustering algorithm
   * Mean shift algorithm
   * Density-based spatial clustering of applications with noise(DBSCAN) algorithm
   * K-nearest neighbours algorithm
   * Cross-validation
3. Reporting results
   * Competency prediction 

Source Code and related files: 

- CAE_Challenge_Specs : Description of the challenge proposed by CAE
- CAE_dataset.csv : Training Sample to make the unsupervised machine learning model
- CAE_test_dataset.csv : Test Sample to use after the model has been trained
- HackerBoises.csv: Result file with the values predicted on the CAE_test_dataset using the machine learning model
- HackerBoises.py : Python code which creates and train the model using the CAE_dataset and test the model using CAE_test_dataset

